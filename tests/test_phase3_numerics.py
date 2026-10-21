"""Synthetic controller tests only; these are not GPU correctness evidence."""

import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/phase3"))
import correctness
import probe_numerics
from benchmark import validate_gate


def test_profiles_are_explicit_and_default_is_unchanged(monkeypatch):
    state = {"flash": True, "mem": True, "math": True, "cudnn": True, "low_math": False}
    cuda = SimpleNamespace(matmul=SimpleNamespace(allow_fp16_reduced_precision_reduction=True))
    for flag, key in (("flash_sdp", "flash"), ("mem_efficient_sdp", "mem"),
                      ("math_sdp", "math"), ("cudnn_sdp", "cudnn")):
        setattr(cuda, f"enable_{flag}", lambda value, key=key: state.__setitem__(key, value))
    cuda.allow_fp16_bf16_reduction_math_sdp = lambda value: state.__setitem__("low_math", value)
    monkeypatch.setattr(correctness.torch.backends, "cuda", cuda)
    assert correctness.configure_experiment("default") is None
    assert cuda.matmul.allow_fp16_reduced_precision_reduction
    assert correctness.configure_experiment("sdpa-full-reduction") == "sdpa"
    assert not cuda.matmul.allow_fp16_reduced_precision_reduction
    assert state == {"flash": True, "mem": True, "math": True, "cudnn": True, "low_math": False}
    assert correctness.configure_experiment("math-full-reduction") == "sdpa"
    assert state == {"flash": False, "mem": False, "math": True, "cudnn": False, "low_math": False}
    assert correctness.configure_experiment("eager-full-reduction") == "eager"
    with pytest.raises(ValueError, match="Unknown"):
        correctness.configure_experiment("invented")


def test_experiments_cannot_authorize_benchmarks():
    for marker in ({"experiment_only": True}, {"numerical_experiment": "sdpa-full-reduction"}):
        with pytest.raises(ValueError, match="cannot authorize"):
            validate_gate({"status": "pass", **marker}, {}, [1, 2, 4])


def prior_fixture():
    return {"status": "fail", "model": "Qwen/Qwen2.5-0.5B", "dtype": "float16",
            "revision": "unit-test-only", "source_file_sha256": {"src/example.py": "same"},
            "workload": [{"id": "synthetic-only", "max_new_tokens": 32}],
            "environment": {"versions": {"torch": "test", "transformers": "test"},
                            "gpus": [{"name": "SYNTHETIC TEST FIXTURE"}]}}


def test_probe_rejects_runtime_or_environment_drift():
    prior = prior_fixture()
    source = {"source_file_sha256": dict(prior["source_file_sha256"])}
    assert probe_numerics.check_prior(prior, source, prior["environment"]) == []
    source["source_file_sha256"]["src/example.py"] = "changed"
    with pytest.raises(ValueError, match="Unexpected source"):
        probe_numerics.check_prior(prior, source, prior["environment"])
    source["source_file_sha256"]["src/example.py"] = "same"
    changed_env = {**prior["environment"], "versions": {"torch": "changed"}}
    with pytest.raises(ValueError, match="torch changed"):
        probe_numerics.check_prior(prior, source, changed_env)
    with pytest.raises(ValueError, match="GPU inventory"):
        probe_numerics.check_prior(prior, source, {**prior["environment"], "gpus": []})
    prior["source_file_sha256"]["scripts/phase3/correctness.py"] = "old"
    source["source_file_sha256"]["scripts/phase3/correctness.py"] = "new"
    assert probe_numerics.check_prior(prior, source, prior["environment"]) == ["scripts/phase3/correctness.py"]


def test_controller_retains_failed_profiles_and_runs_full_gate_each_time(tmp_path):
    prior, commands = prior_fixture(), []

    def execute(command, log, label):
        commands.append(command)
        profile = command[command.index("--numerical-experiment") + 1]
        output = Path(command[-1])
        status = "pass" if profile == "math-full-reduction" else "fail"
        output.write_text(json.dumps({"status": status, "experiment_only": True,
            "dtype": "float16", "diagnostic_fp32": False,
            "numerical_experiment": profile, "workload": prior["workload"],
            "error": None if status == "pass" else "ValueError: Internal cached/batched parity failed",
            "continuous": [{"capacity": c, "exact_token_parity": status == "pass"} for c in (1, 2, 4)],
            **({"http_sse": {"synthetic": True}} if status == "pass" else {})}))
        if status == "fail":
            raise RuntimeError("unit-test expected nonzero worker")

    rows = probe_numerics.run_profiles(tmp_path, prior, execute)
    assert [row["status"] for row in rows] == ["fail", "pass", "fail"]
    assert rows[1]["http_sse_passed"] and rows[1]["passing_capacities"] == [1, 2, 4]
    assert len(commands) == 3
    for command in commands:
        offset = command.index("--concurrencies")
        assert command[offset + 1:offset + 4] == ["1", "2", "4"]
        assert "benchmark.py" not in " ".join(map(str, command))


def test_unexpected_worker_failure_stops_and_exports_partial_evidence(monkeypatch, tmp_path):
    prior = prior_fixture()
    prior_path, output = tmp_path / "prior.json", tmp_path / "probe"
    prior_path.write_text(json.dumps(prior))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(sys, "argv", ["probe", "--prior", str(prior_path), "--output", str(output)])
    monkeypatch.setattr(probe_numerics, "source_provenance", lambda: {
        "source_commit": "unit-test-only", "source_file_sha256": prior["source_file_sha256"]})
    prior["source_commit"] = "unit-test-only"
    prior_path.write_text(json.dumps(prior))
    monkeypatch.setattr(probe_numerics, "environment", lambda: prior["environment"])
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic missing driver, not a parity result")
    monkeypatch.setattr(probe_numerics, "run_profiles", fail)
    with pytest.raises(RuntimeError, match="synthetic missing driver"):
        probe_numerics.main()
    report = json.loads((output / "experiments.json").read_text())
    assert report["status"] == "error" and not report["benchmark_valid"]
    assert (tmp_path / "phase3-colab-probe.zip").is_file()


def test_fp32_control_is_one_strict_explicitly_ineligible_worker(tmp_path):
    prior, commands = prior_fixture(), []
    def execute(command, log, label):
        commands.append(command)
        assert "--diagnostic-fp32" in command
        assert command[command.index("--numerical-experiment") + 1] == "math-full-reduction"
        Path(command[command.index("--output") + 1]).write_text(json.dumps({
            "status": "pass", "experiment_only": True, "dtype": "float32", "diagnostic_fp32": True,
            "numerical_experiment": "math-full-reduction", "workload": prior["workload"],
            "continuous": [{"capacity": c, "exact_token_parity": True} for c in (1, 2, 4)],
            "http_sse": {"synthetic_unit_test_only": True}}))
    rows = probe_numerics.run_profiles(tmp_path, prior, execute, fp32_control=True)
    assert len(commands) == 1 and rows[0]["dtype"] == "float32"
    assert rows[0]["http_sse_passed"] and rows[0]["passing_capacities"] == [1, 2, 4]
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_gate(json.loads((tmp_path / "math-fp32-control.json").read_text()), {}, [1, 2, 4])


def test_fp32_cannot_silently_replace_the_default_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["correctness", "--revision", "unit-test-only",
        "--output", str(tmp_path / "unused.json"), "--diagnostic-fp32"])
    with pytest.raises(SystemExit):
        correctness.main()
    assert not (tmp_path / "unused.json").exists()


def test_fp32_control_promotes_only_the_loaded_fp16_parameter_values(monkeypatch, tmp_path):
    model = torch.nn.Linear(2, 2, bias=False).half()
    original_values = model.weight.detach().float().clone()
    model.config = SimpleNamespace(model_type="qwen2", _commit_hash="unit-test-only",
                                   _attn_implementation="sdpa", max_position_embeddings=2048)
    tokenizer = SimpleNamespace(pad_token_id=0)
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return model, tokenizer
    def check(*args):
        assert model.weight.dtype == torch.float32
        assert torch.equal(model.weight, original_values)
        raise RuntimeError("synthetic stop before GPU inference")
    output = tmp_path / "control.json"
    monkeypatch.setattr(sys, "argv", ["correctness", "--revision", "unit-test-only", "--output", str(output),
        "--numerical-experiment", "math-full-reduction", "--diagnostic-fp32"])
    monkeypatch.setattr(correctness, "source_provenance", lambda: {"source_commit": "synthetic-unit-only"})
    monkeypatch.setattr(correctness, "environment", lambda: {"synthetic_unit_only": True})
    monkeypatch.setattr(correctness, "numerical_settings", lambda: {})
    monkeypatch.setattr(correctness, "configure_experiment", lambda profile: "sdpa")
    monkeypatch.setattr(correctness, "load_model_and_tokenizer", load)
    monkeypatch.setattr(correctness, "eos_token_ids", lambda *args: ())
    monkeypatch.setattr(correctness, "workload", lambda tokenizer: ({}, []))
    monkeypatch.setattr(correctness.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(correctness.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(correctness, "internal_checks", check)
    with pytest.raises(RuntimeError, match="synthetic stop"):
        correctness.main()
    assert calls[0]["dtype"] == "float16" and calls[0]["attn_implementation"] == "sdpa"
    report = json.loads(output.read_text())
    assert report["status"] == "fail" and report["experiment_only"]
    assert report["dtype"] == "float32" and report["promote_fp16_weights_for_control"]


def test_one_cell_numerics_probe_is_syntax_valid_and_offline():
    doc = (Path(__file__).parents[1] / "docs/phase3-numerics-probe.md").read_text(encoding="utf8")
    cells = re.findall(r"```python\n(.*?)```", doc, re.S)
    assert len(cells) == 1
    compile(cells[0], "numerics-probe-cell", "exec")
    assert 'HF_HUB_OFFLINE="1"' in cells[0] and "pip install" not in cells[0]
    assert '"--fp32-control"' in cells[0]

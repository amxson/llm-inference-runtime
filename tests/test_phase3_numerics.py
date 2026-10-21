"""Synthetic controller tests only; these are not GPU correctness evidence."""

import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

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
    def fail(*args):
        raise RuntimeError("synthetic missing driver, not a parity result")
    monkeypatch.setattr(probe_numerics, "run_profiles", fail)
    with pytest.raises(RuntimeError, match="synthetic missing driver"):
        probe_numerics.main()
    report = json.loads((output / "experiments.json").read_text())
    assert report["status"] == "error" and not report["benchmark_valid"]
    assert (tmp_path / "phase3-colab-probe.zip").is_file()


def test_one_cell_numerics_probe_is_syntax_valid_and_offline():
    doc = (Path(__file__).parents[1] / "docs/phase3-numerics-probe.md").read_text(encoding="utf8")
    cells = re.findall(r"```python\n(.*?)```", doc, re.S)
    assert len(cells) == 1
    compile(cells[0], "numerics-probe-cell", "exec")
    assert 'HF_HUB_OFFLINE="1"' in cells[0] and "pip install" not in cells[0]

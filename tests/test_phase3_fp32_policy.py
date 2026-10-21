"""CPU/synthetic policy checks, never GPU comparison evidence."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/phase3"))
import correctness
import policies
import resume_colab
from benchmark import server_command, validate_gate
from test_phase3_workflow import unit_artifact
from validate import validate_artifact


def fp32_record():
    return {"execution_policy": "fp32-math", "dtype": "float32", "checkpoint_load_dtype": "float32",
            "parameter_dtypes": ["torch.float32"], "attention_implementation": "sdpa",
            "numerical_settings": {"flash_sdp": False, "mem_efficient_sdp": False, "cudnn_sdp": False,
                "math_sdp": True, "math_sdp_reduced_precision": False,
                "matmul_allow_tf32": False, "fp16_reduced_precision_reduction": False}}


def test_fp32_policy_is_explicit_shared_and_does_not_change_fp16_default(monkeypatch):
    values = {}
    cuda = SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True, allow_fp16_reduced_precision_reduction=True))
    for name in ("flash_sdp", "mem_efficient_sdp", "cudnn_sdp", "math_sdp"):
        setattr(cuda, f"enable_{name}", lambda value, name=name: values.__setitem__(name, value))
    cuda.allow_fp16_bf16_reduction_math_sdp = lambda value: values.__setitem__("reduced_math", value)
    monkeypatch.setattr(torch.backends, "cuda", cuda)
    assert policies.apply_policy("fp16-sdpa")["dtype"] == "float16" and not values
    assert cuda.matmul.allow_tf32
    assert policies.apply_policy("fp32-math")["dtype"] == "float32"
    assert values == {"flash_sdp": False, "mem_efficient_sdp": False, "cudnn_sdp": False,
                      "math_sdp": True, "reduced_math": False}
    assert not cuda.matmul.allow_tf32 and not cuda.matmul.allow_fp16_reduced_precision_reduction
    selected = policies.policy("fp32-math")
    selected["dtype"] = "wrong"
    assert policies.policy("fp32-math")["dtype"] == "float32"


def test_fp32_gate_requires_native_loading_and_full_policy_metadata():
    data, gate, provenance = unit_artifact()
    gate.update(fp32_record(), http_sse={"synthetic_unit_only": True}, forward={"logits_dtype": "torch.float32"})
    validate_gate(gate, provenance, [1])
    for field, value in (("dtype", "float16"), ("checkpoint_load_dtype", "float16"),
                         ("parameter_dtypes", ["torch.float16"]), ("promote_fp16_weights_for_control", True),
                         ("numerical_settings", {}), ("http_sse", {})):
        with pytest.raises(ValueError):
            validate_gate({**gate, field: value}, provenance, [1])


def test_both_server_commands_use_matching_fp32_without_patching_vllm():
    args = SimpleNamespace(runtime_python=Path("runtime-python"), vllm_python=Path("production-python"), concurrencies=[1, 2, 4])
    gate = {"model": "Qwen/Qwen2.5-0.5B", "revision": "unit-test-only", "execution_policy": "fp32-math"}
    ours = server_command("runtime", args, gate, 8766)
    theirs = server_command("vllm", args, gate, 8767)
    assert ours[ours.index("--execution-policy") + 1] == "fp32-math"
    assert theirs[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert theirs[theirs.index("--dtype") + 1] == "float32"
    assert json.loads(theirs[theirs.index("--attention-config") + 1]) == {"backend": "TRITON_ATTN"}
    assert "--enforce-eager" in theirs and "--no-enable-prefix-caching" in theirs
    assert theirs[theirs.index("--host") + 1] == "127.0.0.1"
    old = server_command("vllm", args, {**gate, "execution_policy": "fp16-sdpa"}, 8767)
    assert old[old.index("--dtype") + 1] == "half" and "--enforce-eager" not in old


def test_validator_rejects_server_policy_drift_after_a_passing_fp32_gate():
    data, gate, provenance = unit_artifact()
    gate.update(fp32_record(), http_sse={"synthetic_unit_only": True}, forward={"logits_dtype": "torch.float32"})
    data.update(dtype="float32", execution_policy="fp32-math")
    args = SimpleNamespace(runtime_python=Path("runtime"), vllm_python=Path("vllm"), concurrencies=[1])
    for session in data["sessions"]:
        if session["engine"] == "runtime":
            session["environment"].update(fp32_record())
        else:
            session["command"] = server_command("vllm", args, gate, 8767)
    assert validate_artifact(data, gate, provenance)["status"] == "pass"
    bad = copy.deepcopy(data)
    bad["sessions"][0]["environment"]["checkpoint_load_dtype"] = "float16"
    with pytest.raises(ValueError, match="FP32"):
        validate_artifact(bad, gate, provenance)
    bad = copy.deepcopy(data)
    command = bad["sessions"][1]["command"]
    command[command.index("--dtype") + 1] = "half"
    with pytest.raises(ValueError, match="vLLM FP32"):
        validate_artifact(bad, gate, provenance)


def test_native_fp32_correctness_does_not_promote_an_fp16_checkpoint(monkeypatch, tmp_path):
    class Model(torch.nn.Linear):
        def float(self):
            raise AssertionError("Native loading must not promote rounded FP16 values")
    model = Model(2, 2)
    model.config = SimpleNamespace(model_type="qwen2", _commit_hash="unit-test-only",
                                   _attn_implementation="sdpa", max_position_embeddings=2048)
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["dtype"] == "float32"
        return model, SimpleNamespace(pad_token_id=0)
    def stop(*args):
        raise RuntimeError("unit-test stop before CUDA inference")
    output = tmp_path / "control.json"
    monkeypatch.setattr(sys, "argv", ["correctness", "--revision", "unit-test-only", "--execution-policy", "fp32-math", "--output", str(output)])
    monkeypatch.setattr(correctness, "source_provenance", lambda: {"source_commit": "synthetic-unit-only"})
    monkeypatch.setattr(correctness, "environment", lambda: {"synthetic_unit_only": True})
    monkeypatch.setattr(correctness, "numerical_settings", lambda: fp32_record()["numerical_settings"])
    monkeypatch.setattr(correctness, "apply_policy", lambda name: None)
    monkeypatch.setattr(correctness, "load_model_and_tokenizer", load)
    monkeypatch.setattr(correctness, "eos_token_ids", lambda *args: ())
    monkeypatch.setattr(correctness, "workload", lambda tokenizer: ({}, []))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(correctness, "internal_checks", stop)
    with pytest.raises(RuntimeError, match="unit-test stop"):
        correctness.main()
    report = json.loads(output.read_text())
    assert calls and not report["experiment_only"] and report["dtype"] == "float32"
    assert report["checkpoint_load_dtype"] == "float32" and not report["promote_fp16_weights_for_control"]


def test_resume_failure_exports_and_never_installs(monkeypatch, tmp_path):
    root, output, prior = tmp_path / "environments", tmp_path / "run", tmp_path / "prior.json"
    for name in ("runtime", "vllm"):
        target = root / name / "bin/python"
        target.parent.mkdir(parents=True)
        target.touch()
    prior.write_text(json.dumps({"model": "Qwen/Qwen2.5-0.5B", "revision": "a" * 40}))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["resume", "--env-root", str(root), "--prior", str(prior), "--output", str(output)])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(resume_colab, "source_provenance", lambda: {"source_commit": "unit-test-only"})
    commands = []
    def fail(command, log, label):
        commands.append(command)
        Path(log).write_text("synthetic failure, not GPU evidence")
        raise RuntimeError("synthetic preflight failure")
    monkeypatch.setattr(resume_colab, "run_command", fail)
    with pytest.raises(RuntimeError, match="synthetic preflight"):
        resume_colab.main()
    assert len(commands) == 1 and "preflight.py" in commands[0][1]
    report = json.loads((output / "run-status.json").read_text())
    assert report["status"] == "fail" and not report["environment_reinstalled"]
    assert (tmp_path / "phase3-colab-run.zip").is_file()

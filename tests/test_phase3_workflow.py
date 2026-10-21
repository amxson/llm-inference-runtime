"""Synthetic protocol/unit checks only. These never produce GPU evidence."""

import asyncio
import copy
import json
import re
import socket
import threading
import time
from pathlib import Path
import sys

import httpx
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/phase3"))
from benchmark import aggregate, check_runtime_outputs, live_group_members, validate_gate
from common import child_environment, workload
from setup_colab import install_commands, resolved_constraints
import setup_colab
import run_colab
from stream_client import cohort, distribution, first_divergence, sse_events, stream_request
from validate import validate_artifact


def spec():
    return {"id": "test-only", "prompt": "1 2", "prompt_token_ids": [1, 2], "max_new_tokens": 3}


def mock_client(text, status=200):
    return httpx.AsyncClient(base_url="http://test.local", transport=httpx.MockTransport(
        lambda request: httpx.Response(status, text=text, headers={"Content-Type": "text/event-stream"})))


def test_real_sse_field_parsing_comments_and_multiline_data():
    async def run():
        async with mock_client(': comment\r\nevent: token\r\ndata: {"a":\r\ndata: 1}\r\n\r\ndata: [DONE]\n\n') as client:
            async with client.stream("GET", "/") as response:
                return [(kind, data) async for kind, data in sse_events(response)]
    assert asyncio.run(run()) == [("token", '{"a":\n1}'), ("message", "[DONE]")]


@pytest.mark.parametrize("engine", ["runtime", "vllm"])
def test_stream_client_uses_ids_not_text_and_records_actual_chunk_sizes(engine):
    if engine == "runtime":
        text = 'event: token\ndata: {"token_id": 3, "generated_text": "three"}\n\nevent: done\ndata: {"prompt_token_ids": [1,2], "generated_token_ids": [3], "stop_reason": "max_new_tokens"}\n\n'
    else:
        text = 'data: {"choices": [{"prompt_token_ids": [1,2], "token_ids": [3,4], "text": "three four", "finish_reason": "length"}]}\n\ndata: {"choices": [], "usage": {"completion_tokens": 2}}\n\ndata: [DONE]\n\n'
    async def run():
        async with mock_client(text) as client:
            return await stream_request(client, engine, spec(), eos_ids=(47,))
    result = asyncio.run(run())
    assert result["success"] and result["prompt_ids_verified"]
    assert len(result["arrival_events"]) == 1  # Do NOT invent per-token timestamps.
    assert result["generated_token_ids"] == ([3] if engine == "runtime" else [3, 4])
    assert result["latency_ms"] >= result["ttft_ms"] >= 0
    assert result["mean_itl_ms"] is None if engine == "runtime" else result["mean_itl_ms"] == 0


@pytest.mark.parametrize("text", [
    'data: {"choices": [{"token_ids": [3], "text": "x", "finish_reason": "length"}]}\n\ndata: [DONE]\n\n',
    'data: {"choices": [{"prompt_token_ids": [99], "token_ids": [3], "text": "x", "finish_reason": "length"}]}\n\ndata: [DONE]\n\n',
    'data: {"choices": [{"prompt_token_ids": [1,2], "token_ids": [3], "text": "x", "finish_reason": "length"}]}\n\n',
    'data: {"error": {"message": "test failure"}}\n\n',
])
def test_malformed_or_incomplete_production_streams_are_failures(text):
    async def run():
        async with mock_client(text) as client:
            return await stream_request(client, "vllm", spec(), eos_ids=(47,))
    result = asyncio.run(run())
    assert not result["success"] and result["error"]


def test_http_errors_preserve_failure_instead_of_success_metrics():
    async def run():
        async with mock_client("no", status=500) as client:
            return await stream_request(client, "runtime", spec(), eos_ids=())
    result = asyncio.run(run())
    assert not result["success"] and result["ttft_ms"] is None


def test_drift_or_missing_gpu_gate_is_rejected():
    provenance = {"source_commit": "a" * 40, "source_file_sha256": {"test": "digest"}}
    gate = {**provenance, "status": "pass", "environment": {"cuda_available": True},
            "continuous": [{"capacity": n, "exact_token_parity": True} for n in [1, 2, 4]],
            "dtype": "float16", "model": "Qwen/Qwen2.5-0.5B"}
    validate_gate(gate, provenance, [1, 2, 4])
    for key, value in [("status", "fail"), ("source_commit", "b" * 40), ("dtype", "bfloat16")]:
        with pytest.raises(ValueError):
            validate_gate({**gate, key: value}, provenance, [1, 2, 4])
    with pytest.raises(ValueError):
        validate_gate(gate, provenance, [8])
    with pytest.raises(ValueError, match="parity"):
        check_runtime_outputs({"requests": [{"id": "A", "generated_token_ids": [4]}]}, {"A": [3]})


def test_comparison_records_external_divergence_instead_of_rejecting_it():
    rows = []
    for engine in ("runtime", "vllm"):
        for repetition in range(2):
            rows.append({"engine": engine, "concurrency": 1, "repetition": repetition,
                         "wall_time_ms": 10 if engine == "runtime" else 5,
                         "request_throughput": 100 if engine == "runtime" else 200,
                         "generated_token_throughput": 200 if engine == "runtime" else 400,
                         "requests": [{"id": "A", "success": True, "latency_ms": 10 if engine == "runtime" else 5,
                                       "ttft_ms": 2, "mean_itl_ms": 1,
                                       "generated_token_ids": [1, 2] if engine == "runtime" else [1, 3]}],
                         "engine_stats": {"torch_memory": {"peak_allocated_bytes": 100, "peak_reserved_bytes": 200}}})
    result = aggregate(rows, [1])[0]
    assert result["comparison"]["exact_output_parity_rate"] == 0
    assert result["comparison"]["outputs"][0]["first_divergence_position_zero_based"] == 1
    assert result["comparison"]["vllm_over_runtime_token_throughput"] == 2
    assert result["vllm"]["torch_peak_allocated_bytes"] is None
    assert first_divergence([1, 2], [1]) == 1
    assert first_divergence([1], [1]) is None
    assert distribution([1, 2, 3])["p95"] == 2.9


def test_installation_commands_isolate_binary_versions_and_keep_base_env(monkeypatch):
    commands = install_commands(Path("/content/test-environments"))
    assert commands["runtime"][0] != commands["vllm"][0]
    assert all("torch==2.10.0+cu129" in command for command in commands.values())
    assert "transformers==5.17.0" in commands["runtime"]
    assert "transformers==4.57.6" in commands["vllm"]
    assert "vllm==0.17.1" in commands["vllm"]
    driver_paths = "/test-only/nvidia/driver:/test-only/cuda/lib64"
    monkeypatch.setenv("LD_LIBRARY_PATH", driver_paths)
    assert child_environment()["LD_LIBRARY_PATH"] == driver_paths
    import os
    assert os.environ["LD_LIBRARY_PATH"] == driver_paths
    monkeypatch.delenv("LD_LIBRARY_PATH")
    assert "LD_LIBRARY_PATH" not in child_environment()
    assert resolved_constraints({"install": [
        {"metadata": {"name": "torch", "version": "2.10.0+cu129"}},
        {"metadata": {"name": "llm-inference-runtime", "version": "0.1.0"}}
    ]}) == "torch==2.10.0+cu129\n"


def test_fixed_workload_is_varied_and_shared():
    from test_cached_continuous_batcher import TinyTokenizer
    class TestTokenizer(TinyTokenizer):
        def __call__(self, prompt, return_tensors):
            return super().__call__(" ".join(["1"] * len(prompt.split())), return_tensors)
    config, specs = workload(TestTokenizer())
    assert len(specs) == 24 and config["concurrencies"] == [1, 2, 4]
    assert {s["max_new_tokens"] for s in specs} == {32, 64, 128}
    assert len({len(s["prompt_token_ids"]) for s in specs}) > 4
    assert len(workload(TestTokenizer(), smoke=True)[1]) == 4


def test_colab_bootstrap_avoids_ensurepip_and_explicitly_targets_only_new_venv(monkeypatch, tmp_path):
    builder = MagicMock()
    calls = []
    monkeypatch.setattr(setup_colab.venv, "EnvBuilder", builder)
    monkeypatch.setattr(setup_colab, "logged", lambda command, path: calls.append((command, path)))
    target = tmp_path / "runtime"
    setup_colab.bootstrap_venv(target, tmp_path)
    builder.assert_called_once_with(with_pip=False, system_site_packages=False)
    builder.return_value.create.assert_called_once_with(target)
    command, log = calls[0]
    assert command[:6] == [sys.executable, "-m", "pip", "--python", str(target), "install"]
    assert "pip==26.2.2" in command and log == tmp_path / "bootstrap-runtime.txt"
    assert "--break-system-packages" not in command


def unit_artifact():
    """Entirely synthetic, in-memory validator fixture; NEVER GPU evidence."""
    provenance = {"source_commit": "a" * 40, "source_file_sha256": {"test-only": "digest"}}
    env = {"cuda_available": True, "gpus": [{"name": "SYNTHETIC UNIT FIXTURE"}],
           "versions": {"torch": "unit-test"}, "cuda_runtime": "unit-test"}
    gate = {**provenance, "status": "pass", "environment": env,
            "continuous": [{"capacity": 1, "exact_token_parity": True}],
            "dtype": "float16", "model": "Qwen/Qwen2.5-0.5B", "revision": "b" * 40,
            "references": [{"id": "A", "generated_token_ids": [1, 2]}]}
    sessions, warmups, rows = [], [], []
    for index, engine in enumerate(("runtime", "vllm", "vllm", "runtime")):
        sessions.append({"index": index, "engine": engine,
                         "environment": {"environment": env} if engine == "runtime" else env})
        warmups.extend({"session": index, "concurrency": 1, "failed_requests": 0} for _ in range(2))
        rows.append({"engine": engine, "concurrency": 1, "repetition": 0 if index < 2 else 1,
                     "successful_requests": 1, "failed_requests": 0, "wall_time_ms": 5,
                     "request_throughput": 200, "generated_token_throughput": 400,
                     "requests": [{"id": "A", "success": True, "latency_ms": 5, "ttft_ms": 1,
                                   "mean_itl_ms": 2, "generated_token_ids": [1, 2], "generated_tokens": 2,
                                   "tokens_per_second": 400, "arrival_events": [
                                       {"arrival_ms": 1, "token_ids": [1]}, {"arrival_ms": 3, "token_ids": [2]}]}],
                     "engine_stats": {"model_forward_count": 2, "forwarded_token_positions": 3,
                                      "forwards": [{"phase": "prefill", "input_shape": [1, 2]},
                                                   {"phase": "decode", "input_shape": [1, 1]}],
                                      "torch_memory": {"peak_allocated_bytes": 64, "peak_reserved_bytes": 128}}})
    data = {**provenance, "schema_version": 1, "status": "pass", "final_eligible": False,
            "revision": gate["revision"], "model": gate["model"], "dtype": gate["dtype"],
            "hardware": env, "concurrencies": [1], "workload": [{"id": "A", "max_new_tokens": 2}],
            "sessions": sessions, "warmups": warmups, "raw_repetitions": rows,
            "aggregates": aggregate(rows, [1])}
    return data, gate, provenance


def test_validator_recomputes_a_unit_fixture_and_rejects_mutations():
    data, gate, provenance = unit_artifact()
    assert validate_artifact(data, gate, provenance)["status"] == "pass"
    cases = []
    bad = copy.deepcopy(data)
    bad["final_eligible"] = True
    cases.append(bad)
    for field, value in (("ttft_ms", 0), ("mean_itl_ms", 0), ("generated_tokens", 3), ("success", False)):
        bad = copy.deepcopy(data)
        bad["raw_repetitions"][0]["requests"][0][field] = value
        cases.append(bad)
    bad = copy.deepcopy(data)
    bad["aggregates"][0]["runtime"]["request_throughput"]["median"] = 10000
    cases.append(bad)
    bad = copy.deepcopy(data)
    bad["warmups"].clear()
    cases.append(bad)
    for bad in cases:
        with pytest.raises(ValueError):
            validate_artifact(bad, gate, provenance)


def test_proc_cleanup_waits_for_owned_live_workers_not_zombies_or_other_groups(tmp_path):
    for pid, state, group in ((100, "S", 42), (101, "Z", 42), (102, "S", 50)):
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "stat").write_text(f"{pid} (python worker) {state} 1 {group} 42 0 0 0")
    assert live_group_members(42, tmp_path) == [100]


def test_notebook_is_output_free_syntax_valid_and_matches_documented_cells():
    root = Path(__file__).parents[1]
    book = json.loads((root / "notebooks/phase3_colab.ipynb").read_text(encoding="utf8"))
    code = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
    documented = re.findall(r"```python\n(.*?)```", (root / "docs/phase3-colab.md").read_text(encoding="utf8"), re.S)
    assert len(code) == 7 and code == documented
    for index, source in enumerate(code):
        compile(source, f"phase3-cell-{index + 1}", "exec")
    assert all(cell.get("outputs", []) == [] and cell.get("execution_count") is None
               for cell in book["cells"] if cell["cell_type"] == "code")


def test_one_cell_launcher_is_fail_fast_and_exports_failure_logs(monkeypatch, tmp_path):
    output, environments = tmp_path / "evidence", tmp_path / "environments"
    monkeypatch.setattr(sys, "argv", ["run_colab", "--output", str(output), "--env-root", str(environments)])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(run_colab, "source_provenance", lambda: {"source_commit": "unit-test-only"})
    calls = []
    def fail(command, log, label):
        calls.append(label)
        Path(log).write_text("synthetic unit-test failure, not GPU evidence")
        raise RuntimeError("test-only preflight failure")
    monkeypatch.setattr(run_colab, "run_command", fail)
    with pytest.raises(RuntimeError, match="test-only preflight"):
        run_colab.main()
    assert len(calls) == 1
    status = json.loads((output / "run-status.json").read_text())
    assert status["status"] == "fail" and status["failed_stage"].startswith("1/7")
    assert (tmp_path / "phase3-colab-evidence.zip").exists()
    assert not environments.exists()


def test_one_cell_launcher_checks_pins_before_installing(tmp_path):
    for name, packages in {"runtime": {"torch": "2.10.0+cu129", "transformers": "5.17.0"},
                           "vllm": {"torch": "2.10.0+cu129", "transformers": "4.57.6", "vllm": "0.17.1"}}.items():
        (tmp_path / f"resolve-{name}.json").write_text(json.dumps({
            "install": [{"metadata": {"name": k, "version": v}} for k, v in packages.items()]}))
    run_colab.verify_resolution(tmp_path)
    (tmp_path / "resolve-vllm.json").write_text('{"install": []}')
    with pytest.raises(RuntimeError, match="before installing"):
        run_colab.verify_resolution(tmp_path)


def test_one_cell_notebook_matches_documentation_and_has_no_execution_output():
    root = Path(__file__).parents[1]
    book = json.loads((root / "notebooks/phase3_colab_one_cell.ipynb").read_text(encoding="utf8"))
    code = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
    documented = re.findall(r"```python\n(.*?)```", (root / "docs/phase3-colab-one-cell.md").read_text(encoding="utf8"), re.S)
    assert len(code) == 1 and code == documented
    compile(code[0], "phase3-one-cell", "exec")
    assert all(not cell.get("outputs") for cell in book["cells"])


def test_new_measurement_client_against_actual_loopback_qwen_cpu_sse():
    """Real TCP/SSE and tiny CPU weights, explicitly not a GPU benchmark."""
    import uvicorn
    from llm_runtime.server import create_app
    from test_model_adapter import tiny_qwen
    from test_cached_continuous_batcher import TinyTokenizer
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    app = create_app(tiny_qwen(), TinyTokenizer(), model_name="unit-test-tiny-qwen2-cpu")
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(.01)
        specs = [{**spec(), "id": f"test-only-{i}"} for i in range(3)]
        result = asyncio.run(cohort(f"http://127.0.0.1:{port}", "runtime", specs, 2, ()))
        assert result["successful_requests"] == 3 and result["failed_requests"] == 0
        assert all(r["prompt_ids_verified"] and len(r["generated_token_ids"]) == 3 for r in result["requests"])
        assert all(len(r["arrival_events"]) == 3 for r in result["requests"])
    finally:
        server.should_exit = True
        thread.join(15)
        listener.close()
        assert not thread.is_alive()

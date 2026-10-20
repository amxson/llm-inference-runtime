"""Download-free checks for sweep math, pairing, parity and raw-artifact integrity."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from llm_runtime import benchmark_cache_sweep as sweep
from runtime_fakes import FakeModel, FakeTokenizer


def test_distributions_use_linear_quantiles_and_sample_stdev():
    result = sweep.distribution([5, 1, 4, 2, 3])
    assert result == {"count": 5, "mean": 3, "median": 3, "min": 1, "max": 5,
                      "p25": 2, "p75": 4, "p90": 4.6, "p95": 4.8,
                      "sample_stdev": pytest.approx(1.58113883)}
    two = sweep.distribution([0, 10])
    assert two["p25"] == 2.5 and two["p75"] == 7.5 and two["p95"] == 9.5
    assert sweep.distribution([7])["sample_stdev"] is None
    empty = sweep.distribution([])
    assert empty["count"] == 0 and all(v is None for k, v in empty.items() if k != "count")


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [None], ["1"], [True]])
def test_distribution_rejects_invalid_values(values):
    with pytest.raises(ValueError, match="finite"):
        sweep.distribution(values)


def test_pairwise_ratios_are_not_ratio_of_unrelated_means():
    ratios = sweep.paired_ratios([10, 100, 8], [2, 50, 2])
    assert ratios["values"] == [5, 2, 4]
    assert ratios["summary"]["median"] == 4
    assert ratios["summary"]["mean"] == pytest.approx(11 / 3)
    assert ratios["summary"]["p25"] == 3 and ratios["summary"]["p75"] == 4.5


@pytest.mark.parametrize("a,b", [([], []), ([1], []), ([1], [0]), ([0], [1]), ([-1], [1]), ([float("nan")], [1])])
def test_pairwise_invalid_inputs(a, b):
    with pytest.raises(ValueError):
        sweep.paired_ratios(a, b)


def test_position_counts_and_order():
    assert sweep.position_counts(5, 10) == {"uncached_positions": 95, "cached_positions": 14,
        "position_reduction_fraction": pytest.approx(1 - 14 / 95),
        "position_ratio_uncached_over_cached": pytest.approx(95 / 14)}
    for n in (32, 128, 512):
        counts = sweep.position_counts(5, n)
        assert counts["uncached_positions"] == sum(range(5, 5 + n))
        assert counts["cached_positions"] == 5 + n - 1
    assert sweep.position_counts(5, 0)["position_ratio_uncached_over_cached"] is None
    assert [sweep.execution_order(i) for i in range(4)] == [
        ["uncached", "cached"], ["cached", "uncached"],
        ["uncached", "cached"], ["cached", "uncached"]]
    for args in ((0, 1), (5, -1), (5, 1.5)):
        with pytest.raises(ValueError):
            sweep.position_counts(*args)
    with pytest.raises(ValueError):
        sweep.execution_order(-1)


def test_fake_sweep_excludes_warmups_and_records_exact_order(monkeypatch):
    calls = []
    original = {"uncached": sweep.timed_greedy_decode, "cached": sweep.timed_cached_greedy_decode}
    for mode, name in (("uncached", "timed_greedy_decode"), ("cached", "timed_cached_greedy_decode")):
        def record(*args, mode=mode):
            calls.append((mode, args[-1]))
            return original[mode](*args)
        monkeypatch.setattr(sweep, name, record)
    levels = sweep.run_sweep(FakeModel(), FakeTokenizer(), budgets=(2, 4), progress=lambda *a, **kw: None)
    for offset, budget in ((0, 2), (24, 4)):
        expected = ["uncached", "cached"] * 2 + [m for i in range(10) for m in sweep.execution_order(i)]
        assert calls[offset:offset + 24] == [(mode, budget) for mode in expected]
    for level in levels:
        assert level["measured_pair_count"] == 10 and level["exact_token_id_parity"]
        assert level["all_runs_reached_budget"]
        assert len(level["pairs"]) == len(level["paired_latency_ratios"]["values"]) == 10
        for mode in ("uncached", "cached"):
            assert all(s["count"] == 10 for s in level["aggregates"][mode].values())
    json.dumps(levels, allow_nan=False)
    bad = copy.deepcopy(levels[0]["pairs"])
    bad[0]["cached"]["generated_token_ids"][0] = 99
    with pytest.raises(ValueError, match="parity"):
        sweep.aggregate_level(2, bad)
    bad = copy.deepcopy(levels[0]["pairs"])
    bad[0]["uncached"]["total_input_token_positions_processed"] += 1
    with pytest.raises(ValueError, match="position"):
        sweep.aggregate_level(2, bad)
    bad = copy.deepcopy(levels[0]["pairs"])
    bad[0]["execution_order"].reverse()
    with pytest.raises(ValueError, match="order"):
        sweep.aggregate_level(2, bad)


def test_parity_failure_stops_execution_immediately(monkeypatch):
    calls = []
    def fake(*args):
        calls.append(True)
        return {"generated_token_ids": [len(calls)]}
    monkeypatch.setattr(sweep, "timed_greedy_decode", fake)
    monkeypatch.setattr(sweep, "timed_cached_greedy_decode", fake)
    with pytest.raises(ValueError, match="parity"):
        sweep.run_sweep(FakeModel(), FakeTokenizer(), budgets=(2,), progress=lambda *a, **kw: None)
    assert len(calls) == 6  # Four warmups, first pair; no second pair.


def test_eos_is_recorded_and_invalid_configuration_is_rejected():
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = 4
    levels = sweep.run_sweep(FakeModel(), tokenizer, budgets=(32,), progress=lambda *a, **kw: None)
    assert levels[0]["observed_generated_tokens"] == [2]
    assert not levels[0]["all_runs_reached_budget"]
    assert all(p["cached"]["stop_reason"] == "eos" for p in levels[0]["pairs"])
    for kwargs in ({"warmups": 1}, {"pairs": 9}, {"budgets": ()}, {"budgets": (32, 32)}, {"budgets": (1024,)}):
        with pytest.raises(ValueError):
            sweep.run_sweep(FakeModel(), tokenizer, **kwargs)
    with pytest.raises(ValueError):
        sweep.aggregate_level(32, [])


def test_cli_requires_confirmation_before_loading_or_writing(monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", ["sweep", "--output", str(output)])
    monkeypatch.setattr(sweep, "load_model_and_tokenizer", lambda *a: pytest.fail("Must not load"))
    with pytest.raises(SystemExit):
        sweep.main()
    assert not output.exists()


def test_cli_writes_nothing_when_measured_parity_fails(monkeypatch, tmp_path):
    output = tmp_path / "invalid.json"
    monkeypatch.setattr("sys.argv", ["sweep", "--confirm-ac-quiet", "--output", str(output)])
    monkeypatch.setattr(sweep, "load_model_and_tokenizer", lambda *a: (FakeModel(), FakeTokenizer()))
    monkeypatch.setattr(sweep, "host_observation", lambda: {})
    monkeypatch.setattr(sweep.torch, "set_num_threads", lambda n: None)
    def mismatch(*args):
        raise ValueError("Token parity failed; no speedup may be published.")
    monkeypatch.setattr(sweep, "run_sweep", mismatch)
    with pytest.raises(ValueError, match="parity"):
        sweep.main()
    assert not output.exists()


def test_committed_sweep_can_be_recomputed_from_raw_pairs():
    path = Path(__file__).parents[1] / "benchmarks/results/cache_length_sweep_cpu.json"
    data = json.loads(path.read_text(encoding="utf8"))
    assert data["schema_version"] == 1 and data["benchmark"] == "cache_length_sweep_cpu"
    assert data["environment"]["device"] == "cpu"
    assert data["config"]["warmup_runs_per_mode_per_length"] >= 2
    assert data["config"]["measured_pairs_per_length"] >= 10
    assert data["exact_token_id_parity"]
    assert len(data["source_revision"]) == 40
    assert data["host"]["user_confirmed_ac_quiet"]
    for name, expected_hash in data["source_file_sha256"].items():
        source = path.parents[2] / "src/llm_runtime" / name
        assert hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected_hash
    assert [level["requested_tokens"] for level in data["levels"]] == [32, 128, 512]
    for level in data["levels"]:
        assert len(level["pairs"]) >= 10
        assert sweep.aggregate_level(level["requested_tokens"], level["pairs"]) == level

import copy
import json
from pathlib import Path

import pytest

from llm_runtime.benchmark_continuous_cache import (
    aggregate_pairs, assert_parity, batch_summary, run_benchmark, run_cohort, trace_accounting,
)
from test_cache_batch import tiny_gpt2
from test_cached_continuous_batcher import TinyTokenizer


@pytest.fixture(scope="module")
def pair():
    model, tokenizer = tiny_gpt2(), TinyTokenizer()
    workload = [("1 2 3", 2), ("4 5 6 7 8", 4), ("9 10", 3)]
    return {"pair_index": 0, "execution_order": ["uncached", "cached"],
            **{mode: run_cohort(model, tokenizer, mode, 2, workload) for mode in ("uncached", "cached")}}


def test_aggregation_parity_position_accounting_and_batch_sizes(pair):
    summary = aggregate_pairs(2, [pair])
    u, c = pair["uncached"], pair["cached"]
    assert summary["exact_token_id_parity"]
    assert summary["paired_wall_time_ratios"]["summary"]["median"] == u["wall_ms"] / c["wall_ms"]
    assert c["cached_decode_input_positions"] == 6  # (2-1)+(4-1)+(3-1)
    assert c["uncached_input_positions"] == 0
    assert u["total_input_positions"] == u["uncached_input_positions"]
    assert c["total_input_positions"] < u["total_input_positions"]
    assert c["peak_active"] == u["peak_active"] == 2
    assert c["request_count"] == c["successful_count"] == 3
    assert summary["aggregates"]["cached"]["prefill_batch_sizes"]["counts"] == {1: 1, 2: 1}
    assert summary["aggregates"]["cached"]["request_latency_ms"]["count"] == 3
    assert batch_summary(iter([2, 2, 1]))["counts"] == {1: 1, 2: 2}
    assert json.loads(json.dumps(summary))["max_active"] == 2


def test_invalid_parity_pairing_and_schema_are_rejected(pair):
    bad = copy.deepcopy(pair)
    bad["cached"]["results"][0]["generated_token_ids"].append(99)
    with pytest.raises(ValueError, match="parity"):
        assert_parity(bad["uncached"], bad["cached"])
    for field, value in [("pair_index", 9), ("execution_order", ["cached", "uncached"])]:
        bad = {**pair, field: value}
        with pytest.raises(ValueError, match="order"):
            aggregate_pairs(2, [bad])
    bad = copy.deepcopy(pair)
    bad["cached"]["total_input_positions"] += 1
    with pytest.raises(ValueError, match="accounting"):
        aggregate_pairs(2, [bad])
    with pytest.raises(ValueError):
        aggregate_pairs(2, [])
    with pytest.raises(ValueError):
        run_benchmark(None, None, pairs=6)
    with pytest.raises(ValueError):
        run_cohort(None, None, "invalid", 2)


def test_cached_decode_width_must_be_one():
    frame = {"tick": 1, "events": [{"phase": "decode", "model_input_width": 5, "batch_size": 1}]}
    with pytest.raises(ValueError, match="one newest token"):
        trace_accounting([frame], True)


def test_recorded_artifact_math_and_schema_if_present():
    path = Path(__file__).resolve().parents[1] / "benchmarks/results/cached_continuous_batching_cpu.json"
    if not path.exists():
        pytest.skip("Measured artifact not recorded yet")
    data = json.loads(path.read_text(encoding="utf8"))
    assert data["schema_version"] == 1 and data["exact_token_id_parity"]
    assert data["config"]["warmup_cohorts_per_mode_per_level"] == 2
    assert data["config"]["measured_pairs_per_level"] >= 7
    assert len(data["config"]["workload"]) >= 8
    assert [level["max_active"] for level in data["levels"]] == [1, 2, 4]
    for level in data["levels"]:
        recalculated = aggregate_pairs(level["max_active"], level["pairs"])
        assert json.loads(json.dumps(recalculated)) == level
        assert len(level["pairs"]) == data["config"]["measured_pairs_per_level"]

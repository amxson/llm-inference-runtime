import json
from pathlib import Path

import pytest

from llm_runtime.benchmark_summary import SOURCES, build_summary, canonical_sha256

RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results"


def test_committed_summary_exactly_reproduces_sources():
    summary = build_summary(RESULTS)
    assert summary == json.loads((RESULTS / "m17_summary.json").read_text(encoding="utf-8"))
    for milestone, filename in SOURCES.items():
        original = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
        entry = summary["experiments"][milestone]
        assert entry["source_canonical_sha256"] == canonical_sha256(original)
        if "comparison" in original:
            assert entry["comparison"] == original["comparison"]
        assert entry["config"] == original["config"]
        assert entry["environment"] == original["environment"]


def test_hash_ignores_json_whitespace_and_key_order():
    assert canonical_sha256({"a": 1, "b": [2]}) == canonical_sha256(json.loads('{ "b": [2], "a": 1 }'))


def test_missing_sources_are_not_replaced_with_invented_measurements(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_summary(tmp_path)

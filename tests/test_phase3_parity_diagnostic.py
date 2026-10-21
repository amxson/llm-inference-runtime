"""Download-free diagnostic plumbing; synthetic old failures, never GPU evidence."""

import json
import re
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/phase3"))
from diagnose_parity import (analyse_capture, clone_cache, failed_targets,
                             logit_summary, observe_engine, reference_at_prefix)
from llm_runtime.cached_decode import cached_greedy_decode
from llm_runtime.cache_batch import cache_tensors
from test_model_adapter import tiny_qwen
from test_cached_continuous_batcher import TinyTokenizer


def test_teacher_forced_cached_reference_and_clone_ownership():
    model, tokenizer = tiny_qwen(), TinyTokenizer()
    ref = cached_greedy_decode(model, tokenizer, "1 2 3", 3)
    logits, before, after = reference_at_prefix(model, [1, 2, 3], ref["generated_token_ids"][:2])
    assert int(logits.argmax()) == ref["generated_token_ids"][2]
    assert before.get_seq_length() == 4 and after.get_seq_length() == 5
    copied = clone_cache(before)
    assert cache_tensors(before)[0][0].data_ptr() != cache_tensors(copied)[0][0].data_ptr()


def test_capture_analyses_same_prefix_without_replacing_engine_tokens():
    model, tokenizer = tiny_qwen(), TinyTokenizer()
    specs = [{"id": "unit-A", "prompt": "1 2 3", "max_new_tokens": 4},
             {"id": "unit-B", "prompt": "4 5 6 7 8", "max_new_tokens": 4}]
    references = {s["id"]: cached_greedy_decode(model, tokenizer, s["prompt"], 4) for s in specs}
    original = references["unit-A"]["generated_token_ids"][1]
    # A deliberately wrong old result triggers the observer in this CPU unit
    # test only. The independently computed real logits must NOT be replaced.
    references["unit-A"]["generated_token_ids"][1] = (original + 1) % 48
    captures, missing, tail = observe_engine(model, tokenizer, specs, references, {"unit-A": 1}, 2)
    assert not missing and len(captures) == 1
    frame = captures[0]
    assert int(frame["original_logits"].argmax()) == original
    report = analyse_capture(model, tokenizer, frame, references["unit-A"])
    assert report["generated_token_number_one_based"] == 2
    assert report["observed_selected_id"] == original
    assert report["prefix_matches_prior_reference"]
    assert all(report["invariants"].values())
    assert report["logits"]["independent_cached"]["argmax_id"] == original
    assert report["cache_before_vs_independent"]["same_layer_count"]
    assert report["cache_after_vs_independent"]["same_layer_count"]
    assert all(layer["key"]["exact_equal"] and layer["value"]["exact_equal"]
               for layer in report["collation_vs_request_cache"]["layers"])
    assert set(report["logits"]) == {
        "original_batch", "independent_cached", "independent_cached_repeat", "original_batch_repeat",
        "single_uncached", "batch_uncached", "single_with_observed_cache", "batch_with_reference_cache"}
    json.dumps(report, allow_nan=False)


def test_nonreproducing_case_is_labelled_not_claimed_fixed():
    model, tokenizer = tiny_qwen(), TinyTokenizer()
    ref = cached_greedy_decode(model, tokenizer, "1 2 3", 3)
    captures, missing, _ = observe_engine(model, tokenizer,
        [{"id": "unit", "prompt": "1 2 3", "max_new_tokens": 3}], {"unit": ref}, {"unit": 1}, 1)
    assert not missing and captures[0]["trigger"] == "prior_divergence_step_not_reproduced"


def test_top_logit_ties_and_failed_target_positions_remain_explicit():
    summary = logit_summary(torch.tensor([1., 2., 2., 0.]), TinyTokenizer(), [0])
    assert summary["argmax_id"] == 1 and summary["exact_top_tie_count"] == 2
    assert summary["top_two_margin"] == 0
    nonfinite = logit_summary(torch.tensor([float("nan"), 1., float("inf")]), TinyTokenizer())
    assert not nonfinite["finite"]
    json.dumps(nonfinite, allow_nan=False)
    gate = {"references": [{"id": "unit", "generated_token_ids": [1, 2]}],
            "continuous": [{"capacity": 2, "results": [{"request_id": "unit", "generated_token_ids": [1, 3]}]}]}
    _, targets = failed_targets(gate, 2)
    assert targets == {"unit": 1}
    gate["continuous"][0]["results"][0]["generated_token_ids"] = [1, 2]
    with pytest.raises(ValueError, match="No failed request"):
        failed_targets(gate, 2)


def test_documented_diagnostic_cell_is_self_contained_and_syntax_valid():
    text = (Path(__file__).parents[1] / "docs/phase3-parity-diagnostic.md").read_text(encoding="utf8")
    cells = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(cells) == 1
    compile(cells[0], "colab-parity-diagnostic-cell", "exec")
    assert 'HF_HUB_OFFLINE="1"' in cells[0] and "pip install" not in cells[0]

"""Observe a failed FP16 gate, not a benchmark or an automatic correctness fix.

Capture the actual batch before its first disagreement, then teacher-force the
same prefix through independent cached/uncached forwards. No precision/backend
settings are changed and no gate is relaxed. Only JSON statistics are exported.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import torch
from transformers import DynamicCache

from common import environment, save_json, source_provenance, utc_now
from run_colab import export_evidence
from stream_client import first_divergence
from llm_runtime.cache_batch import cache_tensors, collate_caches, split_cache
from llm_runtime.cached_continuous_batcher import CachedContinuousBatcher
from llm_runtime.model_adapter import model_adapter
from llm_runtime.model_loader import load_model_and_tokenizer
from llm_runtime.request import RequestState


def clone_cache(cache):
    if cache is None:
        return None
    return DynamicCache(ddp_cache_data=[(key.detach().clone(), value.detach().clone())
                                       for key, value in cache_tensors(cache)])


def clone_inputs(inputs):
    return {name: (clone_cache(value) if name == "past_key_values" else
                   value.detach().clone() if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()}


def vector_stats(left, right):
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    # FP32 here measures differences; model weights/forwards stay unchanged.
    a, b = left.detach().float(), right.detach().float()
    delta = (a - b).abs()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    return {"same_shape": True, "finite": finite,
            "exact_equal": bool(torch.equal(left, right)),
            "max_abs_difference": float(delta.max()) if finite else None,
            "mean_abs_difference": float(delta.mean()) if finite else None}


def cache_stats(left, right):
    if left is None or right is None:
        return {"both_empty": left is None and right is None}
    a, b = cache_tensors(left), cache_tensors(right)
    if len(a) != len(b):
        return {"same_layer_count": False}
    return {"same_layer_count": True, "layers": [
        {"layer": i, "key": vector_stats(ka, kb), "value": vector_stats(va, vb)}
        for i, ((ka, va), (kb, vb)) in enumerate(zip(a, b))]}


def logit_summary(vector, tokenizer, candidates=()):
    vector = vector.detach().float().cpu()
    values, indices = vector.topk(min(8, vector.numel()))
    best = int(vector.argmax())
    ids = set(indices.tolist()) | set(candidates) | {best}
    margin = values[0] - values[1]
    return {"argmax_id": best, "argmax_text": tokenizer.decode([best]),
            "finite": bool(torch.isfinite(vector).all()),
            "top_two_margin": float(margin) if bool(torch.isfinite(margin)) else None,
            "exact_top_tie_count": int((vector == vector.max()).sum()),
            "top_ids": indices.tolist(), "candidate_logits": [
                {"id": token, "text": tokenizer.decode([token]),
                 "logit": float(vector[token]) if bool(torch.isfinite(vector[token])) else str(float(vector[token]))}
                for token in sorted(ids)]}


def failed_targets(gate, capacity):
    refs = {row["id"]: row for row in gate["references"]}
    group = next(row for row in gate["continuous"] if row["capacity"] == capacity)
    targets = {}
    for row in group["results"]:
        reference = refs[row["request_id"]]["generated_token_ids"]
        offset = first_divergence(reference, row["generated_token_ids"])
        if offset is not None:
            targets[row["request_id"]] = offset
    if not targets:
        raise ValueError("No failed request in the selected prior cohort")
    return refs, targets


@torch.inference_mode()
def observe_engine(model, tokenizer, specs, references, targets, capacity):
    """Wrap only this diagnostic engine's call boundary; never replace logits."""
    engine = CachedContinuousBatcher(model, tokenizer, capacity)
    for spec in specs:
        engine.add_request(RequestState(spec["prompt"], spec["max_new_tokens"], spec["id"]))
    pending = set(targets)
    context, captures = {}, []
    original_forward = engine._forward

    def forwarding(states, phase):
        context["states"], context["phase"] = states, phase
        return original_forward(states, phase)

    engine._forward = forwarding

    def before(module, args, kwargs):
        states = context["states"]
        context["snapshot"] = None
        if any(state.request_id in pending for state in states):
            context["snapshot"] = {
                "inputs": clone_inputs(kwargs), "phase": context["phase"],
                "request_caches_before": [clone_cache(s.kv_cache) for s in states],
                "rows": [{"id": s.request_id, "prompt_ids": list(s.prompt_token_ids),
                          "generated_prefix": list(s.generated_token_ids)} for s in states]}

    def after(module, args, kwargs, output):
        snapshot = context.get("snapshot")
        if snapshot is None:
            return
        for index, row in enumerate(snapshot["rows"]):
            name, step = row["id"], len(row["generated_prefix"])
            if name not in pending:
                continue
            expected = references[name]["generated_token_ids"]
            selected = int(output.logits[index, -1].argmax())
            mismatch = step >= len(expected) or selected != expected[step]
            if mismatch or step == targets[name]:
                lengths = [len(r["prompt_ids"]) + len(r["generated_prefix"]) for r in snapshot["rows"]]
                captures.append({**snapshot, "row_index": index, "request_id": name, "step": step,
                                 "observed_selected_id": selected,
                                 "trigger": "observed_divergence" if mismatch else "prior_divergence_step_not_reproduced",
                                 "original_logits": output.logits[index, -1].detach().float().cpu(),
                                 "cache_after": split_cache(output.past_key_values, lengths)[index],
                                 "trace_before": list(engine.trace)[-8:]})
                pending.remove(name)

    hooks = [model.register_forward_pre_hook(before, with_kwargs=True),
             model.register_forward_hook(after, with_kwargs=True)]
    try:
        while engine.has_work and pending:
            engine.step()
            if engine.tick % 64 == 0:
                print(f"Observed tick {engine.tick}; waiting for targets {sorted(pending)}", flush=True)
        trace_tail = list(engine.trace)[-8:]
    finally:
        for hook in hooks:
            hook.remove()
        for name in list(engine.requests):
            engine.cancel_request(name)
    return captures, sorted(pending), trace_tail


@torch.inference_mode()
def reference_at_prefix(model, prompt, generated):
    """Independent prefill plus forced one-token cached steps to THIS prefix."""
    adapter, device = model_adapter(model), next(model.parameters()).device
    cache, before = None, None
    for step in range(len(generated) + 1):
        ids = prompt if step == 0 else [generated[step - 1]]
        inputs = {"input_ids": torch.tensor([ids], device=device),
                  "attention_mask": torch.ones(1, len(prompt) + step, dtype=torch.long, device=device)}
        if step == len(generated):
            before = clone_cache(cache)
        output = model(**adapter.forward_inputs(inputs), past_key_values=cache, use_cache=True)
        cache = output.past_key_values
    return output.logits[0, -1].detach().float().cpu(), before, cache


def full_prefix_inputs(rows, tokenizer, device):
    prefixes = [r["prompt_ids"] + r["generated_prefix"] for r in rows]
    width = max(map(len, prefixes))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(rows), width), pad, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, prefix in enumerate(prefixes):
        ids[i, -len(prefix):] = torch.tensor(prefix, device=device)
        mask[i, -len(prefix):] = 1
    return {"input_ids": ids, "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)}


@torch.inference_mode()
def analyse_capture(model, tokenizer, frame, reference):
    index, rows, inputs = frame["row_index"], frame["rows"], frame["inputs"]
    row, device, adapter = rows[index], next(model.parameters()).device, model_adapter(model)
    prompt, prefix = row["prompt_ids"], row["generated_prefix"]
    independent, ref_before, ref_after = reference_at_prefix(model, prompt, prefix)
    repeated, _, _ = reference_at_prefix(model, prompt, prefix)
    vectors = {"original_batch": frame["original_logits"], "independent_cached": independent,
               "independent_cached_repeat": repeated}
    vectors["original_batch_repeat"] = model(**clone_inputs(inputs)).logits[index, -1].detach().float().cpu()
    single_full = full_prefix_inputs([row], tokenizer, device)
    vectors["single_uncached"] = model(**adapter.forward_inputs(single_full), use_cache=False).logits[0, -1].detach().float().cpu()
    batch_full = full_prefix_inputs(rows, tokenizer, device)
    vectors["batch_uncached"] = model(**adapter.forward_inputs(batch_full), use_cache=False).logits[index, -1].detach().float().cpu()
    before_stats = {"both_empty": True}
    collation_stats = {"both_empty": True}
    if frame["phase"] == "decode":
        lengths = [len(r["prompt_ids"]) + len(r["generated_prefix"]) - 1 for r in rows]
        caches = split_cache(inputs["past_key_values"], lengths)
        observed = caches[index]
        collation_stats = cache_stats(observed, frame["request_caches_before"][index])
        before_stats = cache_stats(observed, ref_before)
        single = {"input_ids": inputs["input_ids"][index:index + 1].clone(),
                  "attention_mask": torch.ones(1, lengths[index] + 1, device=device, dtype=torch.long),
                  "position_ids": inputs["position_ids"][index:index + 1].clone()}
        vectors["single_with_observed_cache"] = model(**adapter.forward_inputs(single),
            past_key_values=clone_cache(observed), use_cache=True).logits[0, -1].detach().float().cpu()
        caches[index] = clone_cache(ref_before)
        collated = collate_caches(caches)
        replacement = clone_inputs(inputs)
        replacement["past_key_values"] = collated.cache
        vectors["batch_with_reference_cache"] = model(**replacement).logits[index, -1].detach().float().cpu()
        reconstructed_mask_matches = bool(torch.equal(collated.attention_mask, inputs["attention_mask"]))
        reconstructed_positions_match = bool(torch.equal(collated.position_ids, inputs["position_ids"]))
    else:
        reconstructed_mask_matches = reconstructed_positions_match = None
    candidates = {int(v.argmax()) for v in vectors.values()}
    if frame["step"] < len(reference["generated_token_ids"]):
        candidates.add(reference["generated_token_ids"][frame["step"]])
    current_length = len(prompt) + len(prefix)
    expected_positions = list(range(len(prompt))) if frame["phase"] == "prefill" else [current_length - 1]
    return {"request_id": frame["request_id"], "generated_token_number_one_based": frame["step"] + 1,
            "observed_selected_id": frame["observed_selected_id"],
            "trigger": frame["trigger"], "phase": frame["phase"], "row_index": index,
            "rows": rows, "prompt": reference["prompt"],
            "prefix_matches_prior_reference": prefix == reference["generated_token_ids"][:len(prefix)],
            "input_ids": inputs["input_ids"].tolist(), "attention_mask": inputs["attention_mask"].tolist(),
            "position_ids": inputs["position_ids"].tolist(),
            "invariants": {"attended_positions_match_logical_length": int(inputs["attention_mask"][index].sum()) == current_length,
                           "input_token_ids_match_prefix": inputs["input_ids"][index, -len(prompt):].tolist() == prompt if frame["phase"] == "prefill" else inputs["input_ids"][index].tolist() == prefix[-1:],
                           "logical_positions_match": inputs["position_ids"][index, -len(expected_positions):].tolist() == expected_positions,
                           "reconstructed_mask_matches": reconstructed_mask_matches,
                           "reconstructed_positions_match": reconstructed_positions_match},
            "logits": {name: logit_summary(v, tokenizer, candidates) for name, v in vectors.items()},
            "logit_differences_from_independent": {name: vector_stats(v, independent) for name, v in vectors.items()},
            "cache_before_vs_independent": before_stats,
            "collation_vs_request_cache": collation_stats,
            "cache_after_vs_independent": cache_stats(frame["cache_after"], ref_after),
            "trace_before": frame["trace_before"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh diagnostic directory; original evidence is never edited")
    args.output.mkdir(parents=True)
    report = {"status": "running", "kind": "parity_diagnostic_not_benchmark", "benchmark_valid": False,
              "started_at_utc": utc_now(), "frames": []}
    try:
        report.update(source_provenance())
        gate = json.loads(args.prior.read_text(encoding="utf8"))
        report["prior_source_commit"] = gate["source_commit"]
        report["prior_correctness_sha256"] = hashlib.sha256(args.prior.read_bytes()).hexdigest()
        if gate["status"] != "fail" or gate["dtype"] != "float16":
            raise ValueError("Expected the original failed FP16 correctness artifact")
        for name, expected in gate["source_file_sha256"].items():
            if report["source_file_sha256"].get(name) != expected:
                raise ValueError(f"Prior runtime source changed: {name}; do not compare silently")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise ValueError("Enable HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1; reuse the cached checkpoint")
        report["environment"] = environment()
        for package in ("torch", "transformers"):
            if report["environment"]["versions"][package] != gate["environment"]["versions"][package]:
                raise ValueError(f"The installed {package} differs from the failed run")
        refs, targets = failed_targets(gate, 2)
        report["targets"] = targets
        report["model"], report["revision"], report["dtype"] = gate["model"], gate["revision"], gate["dtype"]
        model, tokenizer = load_model_and_tokenizer(gate["model"], "cuda", dtype="float16", revision=gate["revision"])
        if next(model.parameters()).dtype != torch.float16 or model.config._commit_hash != gate["revision"]:
            raise ValueError("Model precision/revision does not match the original gate")
        if model.config._attn_implementation != gate["attention_implementation"]:
            raise ValueError("Attention implementation differs from the original failed run")
        report["attention_implementation"] = model.config._attn_implementation
        report["fp16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        print(f"Observe original cohort, targets={targets}; FP16/backend unchanged.", flush=True)
        with torch.inference_mode():
            captures, missing, tail = observe_engine(model, tokenizer, gate["workload"], refs, targets, 2)
            report["uncaptured_targets"], report["trace_tail"] = missing, tail
            for frame in captures:
                print(f"Probe {frame['request_id']} at token {frame['step'] + 1}: {frame['trigger']}", flush=True)
                report["frames"].append(analyse_capture(model, tokenizer, frame, refs[frame["request_id"]]))
        report["status"] = "diagnostic_complete"
        report["note"] = "Comparisons are evidence, not an automatic root-cause verdict or a passed correctness gate. No benchmark ran."
    except BaseException as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished_at_utc"] = utc_now()
        save_json(args.output / "diagnostic.json", report)
        export_evidence(args.output)


if __name__ == "__main__":
    main()

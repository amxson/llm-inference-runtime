"""CPU-only dynamic INT8 experiment on GPT-2 attention/MLP projections."""

import argparse
import copy
import json
from pathlib import Path
import platform
import time

import torch
import transformers
from transformers.pytorch_utils import Conv1D

from llm_runtime.benchmark_naive import summarize_runs, timed_greedy_decode, validate_benchmark_parameters
from llm_runtime.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer


def convert_conv1d(model) -> list[str]:
    """Replace GPT-2 affine Conv1D layers in-place; Linear stores W transposed."""
    names = [name for name, module in model.named_modules() if isinstance(module, Conv1D)]
    for name in names:
        old = model.get_submodule(name)
        linear = torch.nn.Linear(old.weight.shape[0], old.weight.shape[1], bias=True,
                                 device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(old.weight.T)
            linear.bias.copy_(old.bias)
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, linear)
    return names


def weight_payload_bytes(model) -> int:
    """Unique parameter + unpacked quantized-weight storage, not RSS/packed overhead."""
    tensors = list(model.parameters())
    for module in model.modules():
        if isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
            tensors.append(module.weight())
            if module.bias() is not None:
                tensors.append(module.bias())
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def prepare_quantized_model(model, tokenizer, prompt: str):
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("This experiment supports CPU models only.")
    adapted = copy.deepcopy(model).eval()
    names = convert_conv1d(adapted)
    if not names:
        raise ValueError("This experiment requires GPT-2 Conv1D projections.")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        original = model(**inputs, use_cache=False).logits
        converted = adapted(**inputs, use_cache=False).logits
        torch.testing.assert_close(original, converted, rtol=1e-4, atol=1e-4)
        adapter_error = float((original - converted).abs().max())
    # Name-based selection excludes the tied FP32 lm_head; do not quantize all Linear.
    quantized = torch.ao.quantization.quantize_dynamic(
        adapted, set(names), dtype=torch.qint8, inplace=True)
    if not all(isinstance(quantized.get_submodule(name), torch.ao.nn.quantized.dynamic.Linear)
               for name in names):
        raise RuntimeError("Not all selected projections were quantized.")
    return quantized, {"quantized_module_names": names, "quantized_module_count": len(names),
                       "fp32_adapter_max_abs_logit_error": adapter_error}


def run_experiment(model, tokenizer, prompt="The capital of France is", max_new_tokens=10,
                   warmup_runs=1, runs=3, model_name=DEFAULT_MODEL_NAME) -> dict:
    validate_benchmark_parameters(max_new_tokens, warmup_runs, runs)
    start = time.perf_counter()
    quantized, details = prepare_quantized_model(model, tokenizer, prompt)
    setup_ms = (time.perf_counter() - start) * 1000
    models = {"fp32": model, "int8": quantized}
    for _ in range(warmup_runs):
        for candidate in models.values():
            timed_greedy_decode(candidate, tokenizer, prompt, max_new_tokens)
    measured = {name: [] for name in models}
    order = []
    for index in range(runs):
        pair = ["fp32", "int8"] if index % 2 == 0 else ["int8", "fp32"]
        order.append(pair)
        for name in pair:
            measured[name].append(timed_greedy_decode(models[name], tokenizer, prompt, max_new_tokens))
    summaries = {name: summarize_runs(values) for name, values in measured.items()}
    sizes = {name: weight_payload_bytes(candidate) for name, candidate in models.items()}
    reference = measured["fp32"][0]["generated_token_ids"]
    parity = all(run["generated_token_ids"] == reference for values in measured.values() for run in values)
    float_ms = summaries["fp32"]["mean_total_latency_ms"]
    int8_ms = summaries["int8"]["mean_total_latency_ms"]
    return {
        "benchmark": "cpu_dynamic_int8_gpt2", "status": "measured",
        "environment": {"model_name": model_name, "device": "cpu",
                        "python_version": platform.python_version(), "platform": platform.platform(),
                        "pytorch_version": str(torch.__version__), "transformers_version": transformers.__version__,
                        "torch_thread_count": torch.get_num_threads(),
                        "quantized_engine": torch.backends.quantized.engine},
        "config": {"prompt": prompt, "max_new_tokens": max_new_tokens,
                   "warmup_runs_per_mode": warmup_runs, "measured_runs_per_mode": runs,
                   "measured_execution_order": order, "use_cache": False,
                   "timing": "M5 boundaries: before tokenization to final selection; load/setup and final text decode excluded",
                   "method": "torch.ao.quantization.quantize_dynamic; Conv1D-to-Linear; qint8 projection weights",
                   "precision": "INT8 projection weights and dynamic activation quantization; FP32 biases/outputs/other modules",
                   "unquantized": "embeddings, layer norms, tied lm_head",
                   "storage_metric": "unique parameter + unpacked quantized weight payload; excludes buffers, qparams, packing overhead, activations, KV and RSS"},
        "setup": {"quantization_setup_ms": setup_ms, **details},
        **{name: {"runs": measured[name], "summary": summaries[name], "weight_payload_bytes": sizes[name]}
           for name in models},
        "comparison": {"exact_token_id_parity": parity,
                       "weight_payload_reduction_bytes": sizes["fp32"] - sizes["int8"],
                       "weight_payload_reduction_fraction": 1 - sizes["int8"] / sizes["fp32"],
                       "latency_delta_int8_minus_fp32_ms": int8_ms - float_ms,
                       "latency_speedup_fp32_over_int8": float_ms / int8_ms if int8_ms > 0 else None,
                       "quality_scope": "greedy token/text sanity on one prompt, not a quality evaluation"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        validate_benchmark_parameters(args.max_new_tokens, args.warmup_runs, args.runs)
    except ValueError as error:
        parser.error(str(error))
    start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args.model, "cpu")
    load_ms = (time.perf_counter() - start) * 1000
    result = run_experiment(model, tokenizer, args.prompt, args.max_new_tokens,
                            args.warmup_runs, args.runs, args.model)
    result["setup"]["fp32_model_tokenizer_load_ms"] = load_ms
    result["setup"]["load_scope"] = "one FP32 load (cache/download conditions affect it); deepcopy+adapter+probe+quantization measured separately"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"setup": result["setup"], "comparison": result["comparison"],
                      "summaries": {name: result[name]["summary"] for name in ("fp32", "int8")}}, indent=2))


if __name__ == "__main__":
    main()

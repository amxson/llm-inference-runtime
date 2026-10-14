# LLM Inference Runtime

An LLM inference runtime built from first principles to explore and implement autoregressive decoding, KV caching, batching, scheduling, streaming, and inference optimizations.

The project focuses on correctness, benchmarking, systems understanding, and progressive optimization. It draws inspiration from systems such as vLLM, Hugging Face Text Generation Inference (TGI), and llama.cpp, but is not intended to replace them.

## Status

**M8 — Request Lifecycle**

The runtime can load a Hugging Face tokenizer and decoder-only causal language model, select CPU or CUDA, tokenize a prompt, execute one forward pass, and inspect the output logits and final-position next-token candidates. The default model for local experiments is `distilgpt2`.

The runtime also supports manual greedy decoding for one request: select the next token with argmax, append it to the sequence, and repeat until EOS or the requested token limit.

Stochastic decoding now supports temperature scaling, top-k and top-p (nucleus) filtering, multinomial sampling, and reproducible local seeds.

The benchmark harness measures the existing single-request uncached greedy decoder, with per-request timing, repeated-work counts, and JSON results.

Cached greedy decoding now prefills the prompt once, then explicitly reuses the returned transformer K/V state while forwarding only the newest token.

## Expected progression

The project is expected to grow through these topics:

1. Manual decoding
2. Sampling
3. Benchmarking
4. KV caching
5. Request scheduling
6. Batching
7. Continuous batching
8. Serving
9. Memory-management experiments

Each topic will be introduced only when its milestone is reached so the implementation stays understandable.

## Setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Inspect a model

Load the default model and tokenizer, then print their metadata:

```powershell
python -m llm_runtime.model_loader
```

Pass a different Hugging Face model identifier or choose a device explicitly:

```powershell
python -m llm_runtime.model_loader distilgpt2 --device cpu
```

Model files are downloaded to the normal Hugging Face cache, not this repository. An explicit CUDA request fails with a clear error when CUDA is unavailable.

## Inspect one forward pass

Run one forward pass for the default prompt, `The capital of France is`:

```powershell
python -m llm_runtime.forward_inspect --device cpu
```

Supply a different prompt, model, or device when needed:

```powershell
python -m llm_runtime.forward_inspect --prompt "The tallest mountain is" --model distilgpt2 --device cpu
```

The report shows token IDs, token pieces, tensor shapes, output logits, and the top five raw-logit candidates from the final prompt position. It identifies what greedy decoding would select but does not append that token or run the model again.

## Manual greedy decoding

```powershell
python -m llm_runtime.greedy_decode --device cpu --prompt "The capital of France is" --max-new-tokens 10
```

The decoder tokenizes once, selects each new token using `torch.argmax`, manually appends it, and extends the attention mask. It stops at EOS or `max_new_tokens`; a limit of zero performs no forward passes. The result includes generated text, token IDs, and a short trace of input sequence lengths.

`model.generate()` is not used. This path remains deliberately uncached: every step recomputes the complete sequence. For a prompt of length T, successive forward calls process lengths T, T+1, T+2, and so on. The separate cached greedy decoder below reuses K/V state.

## Sampling

```powershell
python -m llm_runtime.sample_decode --device cpu --prompt "The capital of France is" --max-new-tokens 10 --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42
```

Greedy selection uses `logits -> argmax`. Sampling uses `logits -> divide by temperature -> top-k -> top-p -> softmax -> multinomial`. Temperature rescales preferences; top-k and top-p remove candidates; softmax produces probabilities; `torch.multinomial` draws the token.

Temperature must be positive and finite. Top-k is disabled by default; enabled values must be integers >= 1 and are clamped to the vocabulary size. Top-p is disabled by default; enabled values must satisfy `0 < top_p <= 1`. Nucleus filtering retains the token that first reaches or crosses the cumulative probability threshold. Temperature zero is invalid; use the separate greedy decoder for argmax.

An optional seed initializes a local `torch.Generator` on the model's device, without resetting global random state. Repeating a seed and configuration is reproducible in the same model/environment; exact results can differ across devices or library versions. Both decoders manually append tokens and stop at EOS or the token limit.

`model.generate()` is not used. Sampling remains uncached with `use_cache=False`, recomputing the full sequence each generation step. Cached decoding is currently available only for greedy selection.

## KV-cache greedy decoding

```powershell
python -m llm_runtime.cached_decode --device cpu --prompt "The capital of France is" --max-new-tokens 10
```

Uncached decoding forwards the full context, appends a token, then forwards the larger context again. Cached decoding prefills the full prompt once with `use_cache=True`, stores `outputs.past_key_values`, and explicitly passes it into subsequent calls with only the newest token. Each call's returned cache becomes the next call's cache. `model.generate()` and generation helpers are not used.

The trace separates prefill from decode and distinguishes actual model input length from logical context length. Attention masks cover the full logical sequence even when model input length is one. In the inspected Transformers 5.17.0 GPT-2 API, `DynamicCache.get_seq_length()` supplies the position offset; explicit position IDs are unnecessary for this ordinary unpadded single-request path.

The real `distilgpt2` probe returned six dynamic cache layers, with K/V shape `[1, 12, 5, 64]` per layer after the five-token prompt. Correctness checks compare fresh cached and uncached greedy token IDs; the paired M7 benchmark below measures performance separately.

## Naive baseline benchmark

```powershell
python -m llm_runtime.benchmark_naive --device cpu --prompt "The capital of France is" --max-new-tokens 10 --warmup-runs 1 --runs 3 --output benchmarks/results/m5_naive_cpu.json
```

Model/tokenizer loading is excluded. Warmups execute the same path before measured requests and do not enter the statistics. Timing uses `time.perf_counter()` and synchronizes CUDA before timing boundaries; CPU synchronization is a no-op.

Request timing begins immediately before tokenization. TTFT ends when the first token is selected; total latency ends when the final generated token is selected. ITLs are differences between consecutive token-selection timestamps, including intervening token appending and mask updates. Final text decoding and result formatting are excluded. Zero-token requests report zero latency/throughput and null TTFT/mean ITL; a one-token request has no ITL.

Tokens/sec is generated tokens divided by request latency: single-request throughput, not concurrent server throughput. Summary mean ITL pools all measured intervals. Three runs do not establish production tail percentiles. Input token positions processed is the sum of full-sequence forward input lengths, not a FLOP count. The complete sequence is recomputed for every token with `use_cache=False`.

The representative machine/environment-specific result is saved in [benchmarks/results/m5_naive_cpu.json](benchmarks/results/m5_naive_cpu.json). It is a baseline measurement, without a cache comparison or speedup claim. Other local result files under `benchmarks/results/` are ignored.

Recorded environment: Windows, Python 3.11.9, PyTorch 2.14.0+cpu, Transformers 5.17.0, four PyTorch threads. Model: `distilgpt2`; one warmup and three measured requests for the prompt above. Values are rounded here; JSON preserves the raw measurements.

| Device | Prompt tokens | Generated tokens/request | Mean total latency | Mean TTFT | Mean ITL | Mean tokens/sec | Input positions/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 5 | 10 | 343.123 ms | 24.579 ms | 35.394 ms | 29.212 | 95 |

All measured runs produced identical greedy token IDs. These results describe this environment and request, not production serving performance.

## Cached vs uncached

```powershell
python -m llm_runtime.benchmark_cache --device cpu --prompt "The capital of France is" --max-new-tokens 10 --warmup-runs 1 --runs 3 --output benchmarks/results/m7_cached_vs_uncached_cpu.json
```

This comparison loads one model/tokenizer, warms up each mode, then alternates which mode runs first in each measured pair. Both modes use the M5 timing boundaries and synchronization helper. All measured token IDs must match before comparison ratios or JSON are produced. The uncached measurements are fresh; the historical M5 result is not the comparison denominator.

KV caching avoids recomputing prior attention K/V state during incremental decoding. Prefill still processes the whole prompt, so a major TTFT improvement is not promised. The cache grows with sequence length: less recomputation does not make attention state or its memory free. There is no paged KV allocator.

The representative [M7 JSON](benchmarks/results/m7_cached_vs_uncached_cpu.json) retains all raw runs, logical context lengths, work counts, per-request GPU peak allocations (null on CPU), and clearly named ratios. This is a short, environment-specific, single-request CPU experiment, not production serving throughput or a tail-latency study.

Fresh results for `distilgpt2`, a five-token prompt, ten generated tokens, one warmup per mode, and three measured requests per mode (Windows, Python 3.11.9, PyTorch 2.14.0+cpu, Transformers 5.17.0, four PyTorch threads):

| Metric | Uncached | Cached | Comparison |
| --- | ---: | ---: | --- |
| Input positions/request | 95 | 14 | Cached/uncached: 0.1474 |
| Mean total latency | 699.031 ms | 381.035 ms | Uncached/cached: 1.835x |
| Mean TTFT | 50.271 ms | 49.693 ms | Nearly unchanged |
| Mean ITL | 72.084 ms | 36.816 ms | Uncached/cached: 1.958x |
| Mean tokens/sec | 14.345 | 26.293 | Cached/uncached: 1.833x |

All six measured requests produced exactly the same token IDs. Uncached input lengths were `[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`; cached input lengths were `[5, 1, 1, 1, 1, 1, 1, 1, 1, 1]`, while both logical contexts grew from 5 to 14. Cached decoding was faster in this measured experiment, primarily after prefill. The position count is not a FLOP count, and these results do not establish a universal speedup or production capacity.

## Request lifecycle

`RequestState` owns an ID, prompt, token limits, selected IDs, request-local runtime inputs/cache, execution count, and stop reason. Its explicit lifecycle is `WAITING -> PREFILL -> DECODING -> FINISHED`, with EOS, token-limit, zero-token, cancellation, and error completion. Finished requests release runtime tensors/cache; small result dictionaries preserve tokens and text without serializing model state.

## Current limitations

M8 adds request lifecycle state while preserving earlier decoders and benchmarks. No paged/block cache allocator, request batching, scheduler, or server exists. M2's separate inspection command still executes exactly one forward pass.

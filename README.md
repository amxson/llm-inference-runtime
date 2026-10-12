# LLM Inference Runtime

An LLM inference runtime built from first principles to explore and implement autoregressive decoding, KV caching, batching, scheduling, streaming, and inference optimizations.

The project focuses on correctness, benchmarking, systems understanding, and progressive optimization. It draws inspiration from systems such as vLLM, Hugging Face Text Generation Inference (TGI), and llama.cpp, but is not intended to replace them.

## Status

**M3 — Manual Greedy Autoregressive Decoding**

The runtime can load a Hugging Face tokenizer and decoder-only causal language model, select CPU or CUDA, tokenize a prompt, execute one forward pass, and inspect the output logits and final-position next-token candidates. The default model for local experiments is `distilgpt2`.

The runtime also supports manual greedy decoding for one request: select the next token with argmax, append it to the sequence, and repeat until EOS or the requested token limit.

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

`model.generate()` is not used. This is a deliberately naive uncached decoder: every step recomputes the complete sequence. For a prompt of length T, successive forward calls process lengths T, T+1, T+2, and so on. Cache creation is disabled; KV-cache optimization is not implemented yet. No performance improvement is claimed.

## Current limitations

M3 supports one request with deterministic greedy decoding. Sampling, KV caching, request batching, scheduling, and serving are not implemented yet. M2's separate inspection command still executes exactly one forward pass.

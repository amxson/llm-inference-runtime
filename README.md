# Educational LLM Inference Runtime

This repository is an educational, engineering-focused project for learning how an LLM inference runtime works. It will explore the mechanisms behind high-throughput LLM serving in small, inspectable milestones.

The project draws inspiration from systems such as vLLM, Hugging Face Text Generation Inference (TGI), and llama.cpp, but it is not a clone or a replacement for any of them.

## Status

**M1 — Model & Tokenizer Loading**

The runtime can now load a Hugging Face tokenizer and decoder-only causal language model, select CPU or CUDA, place the model in evaluation mode, and report configuration metadata. The default model for local experiments is `distilgpt2`.

No forward pass, token generation, or decoding has been implemented yet.

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

## Current limitations

M1 only loads and inspects model and tokenizer objects. It does not tokenize prompts, run forward passes, inspect logits, generate or sample tokens, manage inference memory, batch requests, schedule work, or expose a server.

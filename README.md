# Educational LLM Inference Runtime

This repository is an educational, engineering-focused project for learning how an LLM inference runtime works. It will explore the mechanisms behind high-throughput LLM serving in small, inspectable milestones.

The project draws inspiration from systems such as vLLM, Hugging Face Text Generation Inference (TGI), and llama.cpp, but it is not a clone or a replacement for any of them.

## Status

**M0 — Repository Bootstrap**

The repository currently contains only the Python package foundation and a package-import smoke test. No model or tokenizer loading, inference, generation, or serving functionality has been implemented.

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

## Current limitations

M0 is intentionally only a bootstrap. It does not load models or tokenizers, run forward passes, generate tokens, use accelerators, manage inference memory, batch requests, schedule work, or expose a server.

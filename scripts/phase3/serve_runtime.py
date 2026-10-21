"""Loopback-only GPU measurement server; probes are NOT in the public app."""

import argparse

import torch
import uvicorn
from fastapi import HTTPException

from common import environment, torch_memory
from policies import POLICIES, apply_policy, numerical_settings
from llm_runtime.model_loader import load_model_and_tokenizer
from llm_runtime.server import create_app


def gpu_app(model, tokenizer, *, model_name, revision, execution_policy="fp16-sdpa"):
    app = create_app(model, tokenizer, model_name=model_name)
    forwards = []

    def observe(module, args, kwargs):
        shape = list(kwargs["input_ids"].shape)
        forwards.append({"input_shape": shape,
                         "phase": "decode" if kwargs.get("past_key_values") is not None else "prefill"})

    model.register_forward_pre_hook(observe, with_kwargs=True)

    def idle():
        runtime = app.state.runtime
        if runtime.in_flight or runtime.engine.has_work or not runtime.commands.empty() or runtime.channels:
            raise HTTPException(409, "Wait until all requests and cleanup commands finish.")
        return runtime

    @app.get("/phase3/environment")
    async def env():
        return {"environment": environment(), "model": model_name, "revision": revision,
                "dtype": str(next(model.parameters()).dtype),
                "execution_policy": execution_policy,
                "checkpoint_load_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
                "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                "numerical_settings": numerical_settings(),
                "attention_implementation": model.config._attn_implementation}

    @app.post("/phase3/reset")
    async def reset(max_active: int = 4):
        if max_active not in (1, 2, 4, 8):
            raise HTTPException(422, "Unsupported concurrency")
        runtime = idle()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        forwards.clear()
        runtime.engine.max_active = max_active
        runtime.engine.trace.clear()
        runtime.recent_traces.clear()
        return {"status": "reset", "max_active": max_active}

    @app.get("/phase3/stats")
    async def stats():
        runtime = idle()
        return {"torch_memory": torch_memory(), "forwards": list(forwards),
                "model_forward_count": len(forwards),
                "forwarded_token_positions": sum(f["input_shape"][0] * f["input_shape"][1] for f in forwards),
                "peak_active_requests": max((len(f["active_request_ids"]) for f in runtime.engine.trace), default=0),
                "traces_tail": list(runtime.recent_traces),
                "trace_note": "Tail is bounded to 128 ticks; full work counts come from the forward observer."}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--execution-policy", choices=tuple(POLICIES), default="fp16-sdpa")
    args = parser.parse_args()
    environment()  # No CPU fallback.
    selected = apply_policy(args.execution_policy)
    options = {} if selected["attention"] is None else {"attn_implementation": selected["attention"]}
    model, tokenizer = load_model_and_tokenizer(args.model, "cuda", dtype=selected["dtype"], revision=args.revision, **options)
    app = gpu_app(model, tokenizer, model_name=args.model, revision=args.revision, execution_policy=args.execution_policy)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

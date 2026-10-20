"""Public deployment entrypoint for the real runtime, not an inference proxy/fake.

Vercel's read-only bundle uses /tmp for model downloads. Cold starts can be slow.
Worker state is per function instance, not global across autoscaled replicas.
"""
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HOME", "/tmp/llm-runtime-hf")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from llm_runtime.model_loader import load_model_and_tokenizer
from llm_runtime.server import create_app


def create_public_app(*, loader=load_model_and_tokenizer, engine=None):
    # One CPU row for this small public demo; engine algorithms are unchanged.
    runtime_app = create_app(loader=loader, model_name="distilgpt2", device="cpu", max_active=1, engine=engine)
    runtime_app.root_path = "/api"
    return runtime_app


torch.set_num_threads(1)
app = create_public_app()

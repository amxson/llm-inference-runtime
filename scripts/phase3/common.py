"""Shared evidence helpers. No GPU work happens at import time."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = Path(__file__).with_name("workload.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def command_output(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "returncode": None, "error": str(error)}


def source_provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked checkout must be clean before GPU evidence collection.")
    paths = git("ls-files", "src", "scripts/phase3", "pyproject.toml").splitlines()
    return {"source_commit": git("rev-parse", "HEAD"), "source_file_sha256": {
        name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in paths}}


def environment(require_cuda=True):
    import torch
    available = torch.cuda.is_available()
    if require_cuda and not available:
        raise RuntimeError("REAL CUDA GPU REQUIRED; refusing CPU fallback.")
    versions = {}
    for package in ("torch", "transformers", "vllm", "fastapi", "uvicorn", "httpx", "flashinfer-python"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append({"index": index, "name": props.name, "total_vram_bytes": props.total_memory,
                     "compute_capability": list(torch.cuda.get_device_capability(index))})
    return {"recorded_at_utc": utc_now(), "python": platform.python_version(),
            "os": platform.platform(), "cpu": platform.processor(), "versions": versions,
            "system_ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None,
            "cuda_runtime": torch.version.cuda, "cuda_available": available,
            "gpu_count": len(gpus), "gpus": gpus,
            "nvidia_smi": command_output(["nvidia-smi"]),
            "nvcc": command_output(["nvcc", "--version"]),
            "gpu_observation": gpu_observation(),
            "loader_library_policy": "Child processes remove LD_LIBRARY_PATH; use wheel-provided CUDA libraries, not Colab system CUDA overrides."}


def gpu_observation():
    return command_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,temperature.gpu,power.draw",
                           "--format=csv,noheader,nounits"])


def torch_memory():
    import torch
    torch.cuda.synchronize()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "metric_scope": "PyTorch allocator in OUR server process, not physical/process VRAM"}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as target:
        target.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def workload(tokenizer, smoke=False):
    config = json.loads(WORKLOAD.read_text(encoding="utf8"))
    prompts = config["prompts"][:4] if smoke else config["prompts"]
    requests = []
    for index, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        budget = 4 if smoke else config["budgets"][index % len(config["budgets"]) ]
        if not ids or len(ids) + budget > config["context_limit"]:
            raise ValueError("Workload exceeds the shared context limit.")
        requests.append({"id": f"request-{index:02d}", "prompt": prompt,
                         "prompt_token_ids": ids, "max_new_tokens": budget})
    return config, requests


def child_environment():
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["DO_NOT_TRACK"] = "1"
    return env

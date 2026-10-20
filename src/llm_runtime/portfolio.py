"""Small, read-only portfolio payloads. No tensor/cache serialization."""

import json
from pathlib import Path

from fastapi import HTTPException

ENGINE_COMMIT = "59a9c8885e579815cd5f1193c45035fba5e95fab"
BENCHMARK_FILES = (
    "m5_naive_cpu.json", "m7_cached_vs_uncached_cpu.json", "m13_serving_cpu.json",
    "m15_quantization_cpu.json", "m16_speculative_cpu.json", "m17_summary.json", "final_cpu.json",
    "cache_length_sweep_cpu.json", "cached_continuous_batching_cpu.json",
)


def read_benchmark(directory: Path, name: str) -> dict:
    # Exact allowlist, never join arbitrary user paths or serve local model files.
    if name not in BENCHMARK_FILES:
        raise HTTPException(404, "Unknown benchmark artifact.")
    try:
        return json.loads((directory / name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(404, "Benchmark artifact is not installed.") from None


def benchmark_index(directory: Path) -> dict:
    return {"engine_commit": ENGINE_COMMIT, "artifacts": [
        {"name": name, "benchmark": read_benchmark(directory, name)["benchmark"]}
        for name in BENCHMARK_FILES if (directory / name).is_file()
    ]}

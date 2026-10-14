"""Aggregate representative JSON artifacts; no inference or new measurements."""

import argparse
import hashlib
import json
from pathlib import Path

SOURCES = {
    "m5": "m5_naive_cpu.json",
    "m7": "m7_cached_vs_uncached_cpu.json",
    "m13": "m13_serving_cpu.json",
    "m15": "m15_quantization_cpu.json",
    "m16": "m16_speculative_cpu.json",
}


def canonical_sha256(data: dict) -> str:
    """Hash JSON content independently of whitespace/platform line endings."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_summary(results_directory: Path) -> dict:
    artifacts = {key: json.loads((results_directory / name).read_text(encoding="utf-8"))
                 for key, name in SOURCES.items()}
    # Explicit keys fail loudly if a measurement disappears; never fabricate defaults.
    entries = {key: {"source_file": SOURCES[key], "source_canonical_sha256": canonical_sha256(data),
                     "benchmark": data["benchmark"], "environment": data["environment"],
                     "config": data["config"]} for key, data in artifacts.items()}
    m5, m7, m13, m15, m16 = (artifacts[key] for key in SOURCES)
    entries["m5"]["summary"] = m5["summary"]
    entries["m5"]["input_positions_per_run"] = [run["total_input_token_positions_processed"] for run in m5["runs"]]
    for mode in ("uncached", "cached"):
        entries["m7"][mode] = m7[mode]["summary"]
    entries["m7"]["comparison"] = m7["comparison"]
    entries["m13"]["levels"] = [{"concurrency": level["concurrency"], "summary": level["summary"],
                                 "server_forward_calls_during_level": level["server_forward_calls_during_level"],
                                 "server_max_observed_batch_size_since_start": level["server_max_observed_batch_size_since_start"]}
                                for level in m13["levels"]]
    for mode in ("fp32", "int8"):
        entries["m15"][mode] = {"summary": m15[mode]["summary"],
                                "weight_payload_bytes": m15[mode]["weight_payload_bytes"]}
    entries["m15"]["comparison"] = m15["comparison"]
    entries["m15"]["setup"] = {key: m15["setup"][key] for key in
                               ("quantization_setup_ms", "quantized_module_count", "fp32_model_tokenizer_load_ms")}
    for mode in ("ordinary", "speculative"):
        entries["m16"][mode] = m16[mode]["summary"]
    entries["m16"]["comparison"] = m16["comparison"]
    entries["m16"]["compatibility"] = m16["compatibility"]
    entries["m16"]["speculative_counters_per_run"] = [
        {key: run[key] for key in ("proposed_draft_tokens", "accepted_draft_tokens", "rejected_draft_tokens",
                                  "target_verification_calls", "draft_forward_calls", "acceptance_rate")}
        for run in m16["speculative"]["runs"]]
    return {"benchmark": "representative_milestone_summary", "schema_version": 1,
            "scope": "copied measurements only; artifacts are separate experiments, not a cross-milestone leaderboard",
            "caveats": ["M7 uses fresh paired modes, not historical M5 as denominator",
                        "M13 concurrency-one phase was anomalously slow; cause is not established",
                        "M15 outputs differed; storage payload is not RSS",
                        "M16 has exact greedy parity but no accepted draft tokens on this workload",
                        "Small CPU samples do not establish production capacity or reliable tails"],
            "experiments": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-directory", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/m17_summary.json"))
    args = parser.parse_args()
    result = build_summary(args.results_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Aggregated {len(SOURCES)} representative experiments into {args.output}")


if __name__ == "__main__":
    main()

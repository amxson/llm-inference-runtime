"""Record the actual environment and fail before model download if CUDA fails."""

import argparse
import json

from common import environment, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--report-only", action="store_true", help="Inventory base Colab; don't require our pinned stack")
    args = parser.parse_args()
    result = environment()
    import torch
    if not args.report_only:
        if not str(torch.__version__).startswith("2.10.0") or torch.version.cuda != "12.9":
            raise RuntimeError("Expected isolated torch 2.10.0+cu129, not Colab's base packages")
        if torch.cuda.get_device_capability() < (7, 5):
            raise RuntimeError("This reviewed vLLM/T4 workflow requires compute capability >=7.5")
        # Actually execute FP16 CUDA, not just an availability boolean.
        with torch.inference_mode():
            matrix = torch.ones((32, 32), device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            output = matrix @ matrix
            torch.cuda.synchronize()
            if not bool(torch.isfinite(output).all()) or not bool((output == 32).all()):
                raise RuntimeError("FP16 CUDA matrix multiplication failed")
        result["fp16_cuda_matmul"] = "pass"
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

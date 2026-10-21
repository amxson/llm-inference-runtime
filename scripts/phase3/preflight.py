"""Record the actual environment and fail before model download if CUDA fails."""

import argparse
import json

from common import environment, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--report-only", action="store_true", help="Inventory base Colab; don't require our pinned stack")
    parser.add_argument("--check-fp32-triton", action="store_true", help="Verify installed vLLM backend declarations, not inference success")
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
        with torch.inference_mode():
            fp32 = torch.ones((32, 32), device="cuda", dtype=torch.float32)
            output32 = fp32 @ fp32
            torch.cuda.synchronize()
            if not bool(torch.isfinite(output32).all()) or not bool((output32 == 32).all()):
                raise RuntimeError("FP32 CUDA matrix multiplication failed")
        result["fp32_cuda_matmul"] = "pass"
    if args.check_fp32_triton:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm.platforms.interface import DeviceCapability
        backend = AttentionBackendEnum.TRITON_ATTN.get_class()
        capability = DeviceCapability(*torch.cuda.get_device_capability())
        if not backend.supports_dtype(torch.float32) or not backend.supports_compute_capability(capability):
            raise RuntimeError("Installed vLLM TRITON_ATTN does not declare FP32/GPU support")
        result["vllm_fp32_triton_declaration"] = "supported; actual inference remains required"
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

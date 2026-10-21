"""Explicit, shared Phase 3 execution policies; no public-demo default changes."""

POLICIES = {
    "fp16-sdpa": {"dtype": "float16", "attention": None,
                  "vllm_dtype": "half", "vllm_attention": "FLASHINFER", "vllm_eager": False},
    "fp32-math": {"dtype": "float32", "attention": "sdpa",
                  "vllm_dtype": "float32", "vllm_attention": "TRITON_ATTN", "vllm_eager": True},
}


def policy(name):
    if name not in POLICIES:
        raise ValueError(f"Unknown Phase 3 execution policy: {name}")
    return dict(POLICIES[name])


def numerical_settings():
    import torch
    cuda = torch.backends.cuda
    return {"fp16_reduced_precision_reduction": cuda.matmul.allow_fp16_reduced_precision_reduction,
            "flash_sdp": cuda.flash_sdp_enabled(), "mem_efficient_sdp": cuda.mem_efficient_sdp_enabled(),
            "math_sdp": cuda.math_sdp_enabled(), "cudnn_sdp": cuda.cudnn_sdp_enabled(),
            "math_sdp_reduced_precision": cuda.fp16_bf16_reduction_math_sdp_allowed(),
            "matmul_allow_tf32": cuda.matmul.allow_tf32}


def apply_policy(name):
    options = policy(name)
    if name == "fp32-math":
        import torch
        cuda = torch.backends.cuda
        cuda.matmul.allow_tf32 = False
        cuda.matmul.allow_fp16_reduced_precision_reduction = False
        cuda.enable_flash_sdp(False)
        cuda.enable_mem_efficient_sdp(False)
        cuda.enable_cudnn_sdp(False)
        cuda.enable_math_sdp(True)
        cuda.allow_fp16_bf16_reduction_math_sdp(False)
    return options


def validate_fp32_record(record):
    """Do not relabel an old promoted-weight diagnostic as native FP32 proof."""
    expected = {"flash_sdp": False, "mem_efficient_sdp": False, "cudnn_sdp": False,
                "math_sdp": True, "math_sdp_reduced_precision": False, "matmul_allow_tf32": False,
                "fp16_reduced_precision_reduction": False}
    if (record.get("dtype") not in {"float32", "torch.float32"} or
            record.get("attention_implementation") != "sdpa" or
            record.get("parameter_dtypes") != ["torch.float32"] or
            record.get("checkpoint_load_dtype") != "float32" or
            record.get("promote_fp16_weights_for_control", False) or
            any(record.get("numerical_settings", {}).get(key) != value for key, value in expected.items())):
        raise ValueError("FP32 serving/reference execution policy differs or is incomplete")

"""Load and inspect a causal language model without running inference."""

import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "distilgpt2"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or select CUDA when it is available."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_device = torch.device(device)

    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported.")

    return resolved_device


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | torch.device | None = None,
    *,
    dtype: str | torch.dtype | None = None,
    revision: str | None = None,
    attn_implementation: str | None = None,
) -> tuple[Any, Any]:
    """Load a causal language model and tokenizer onto the selected device."""
    resolved_device = resolve_device(device)
    options = {} if revision is None else {"revision": revision}
    model_options = dict(options)
    if dtype is not None:
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        resolved_dtype = dtypes.get(dtype) if isinstance(dtype, str) else dtype
        if resolved_dtype not in dtypes.values():
            raise ValueError("dtype must be float32, float16 or bfloat16.")
        if resolved_device.type == "cuda" and resolved_dtype == torch.bfloat16:
            with torch.cuda.device(resolved_device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError("Native CUDA BF16 is unavailable; select float16.")
        model_options["dtype"] = resolved_dtype
    if attn_implementation is not None:
        model_options["attn_implementation"] = attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_options)
    model.eval()
    model.to(resolved_device)
    return model, tokenizer


def get_model_metadata(model_name: str, model: Any, tokenizer: Any) -> dict[str, Any]:
    """Collect useful model and tokenizer facts without executing the model."""
    config = model.config
    model_device = next(model.parameters()).device

    return {
        "model identifier": model_name,
        "model class": type(model).__name__,
        "tokenizer class": type(tokenizer).__name__,
        "device": str(model_device),
        "evaluation mode": not model.training,
        "vocabulary size": tokenizer.vocab_size,
        "parameter count": sum(parameter.numel() for parameter in model.parameters()),
        "hidden size": getattr(config, "hidden_size", None),
        "transformer layers": getattr(config, "num_hidden_layers", None),
        "EOS token ID": tokenizer.eos_token_id,
        "BOS token ID": tokenizer.bos_token_id,
        "PAD token ID": tokenizer.pad_token_id,
    }


def main() -> None:
    """Load a model and tokenizer, then print configuration metadata."""
    parser = argparse.ArgumentParser(
        description="Load a causal language model and inspect its metadata."
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model identifier (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--revision", help="Optional immutable Hugging Face commit")
    parser.add_argument(
        "--device",
        help="Device to use, such as 'cpu', 'cuda', or 'cuda:0' (default: auto)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.device, dtype=args.dtype, revision=args.revision)
    metadata = get_model_metadata(args.model_name, model, tokenizer)

    for label, value in metadata.items():
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()

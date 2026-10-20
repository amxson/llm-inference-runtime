from unittest.mock import MagicMock

import pytest
import torch

from llm_runtime import model_loader


def test_load_model_and_tokenizer_uses_requested_model_and_cpu(monkeypatch) -> None:
    tokenizer = object()
    model = MagicMock()
    tokenizer_loader = MagicMock(return_value=tokenizer)
    model_loader_mock = MagicMock(return_value=model)

    monkeypatch.setattr(
        model_loader.AutoTokenizer, "from_pretrained", tokenizer_loader
    )
    monkeypatch.setattr(
        model_loader.AutoModelForCausalLM, "from_pretrained", model_loader_mock
    )

    loaded_model, loaded_tokenizer = model_loader.load_model_and_tokenizer(
        "example/model", device="cpu"
    )

    tokenizer_loader.assert_called_once_with("example/model")
    model_loader_mock.assert_called_once_with("example/model")
    model.eval.assert_called_once_with()
    model.to.assert_called_once_with(torch.device("cpu"))
    assert loaded_model is model
    assert loaded_tokenizer is tokenizer


def test_automatic_device_selection_follows_cuda_availability(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert model_loader.resolve_device() == torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert model_loader.resolve_device() == torch.device("cpu")


def test_explicit_cuda_request_fails_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        model_loader.load_model_and_tokenizer("example/model", device="cuda")


def test_loader_explicit_dtype_revision_and_attention(monkeypatch):
    tokenizer_loader, model_loader_mock = MagicMock(), MagicMock()
    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(model_loader.AutoModelForCausalLM, "from_pretrained", model_loader_mock)
    model_loader.load_model_and_tokenizer("Qwen/Qwen2.5-0.5B", "cpu", dtype="float16",
                                         revision="abc", attn_implementation="eager")
    tokenizer_loader.assert_called_once_with("Qwen/Qwen2.5-0.5B", revision="abc")
    model_loader_mock.assert_called_once_with("Qwen/Qwen2.5-0.5B", revision="abc",
                                             dtype=torch.float16, attn_implementation="eager")


def test_invalid_dtype_fails_before_download(monkeypatch):
    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("download"))
    with pytest.raises(ValueError, match="dtype"):
        model_loader.load_model_and_tokenizer(device="cpu", dtype="int8")

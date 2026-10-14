import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from llm_runtime.benchmark_quantization import convert_conv1d, prepare_quantized_model, run_experiment, weight_payload_bytes
from runtime_fakes import FakeTokenizer


def small_model():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=32, n_embd=8, n_layer=1, n_head=2)).eval()


def test_conv1d_adapter_preserves_affine_operation():
    model = torch.nn.Sequential(Conv1D(5, 3))
    x = torch.randn(2, 4, 3)
    expected = model(x)
    assert convert_conv1d(model) == ["0"]
    torch.testing.assert_close(model(x), expected)


def test_quantized_projections_execute_without_mutating_baseline_or_head():
    model, tokenizer = small_model(), FakeTokenizer()
    quantized, details = prepare_quantized_model(model, tokenizer, "a")
    assert details["quantized_module_count"] == 4
    assert isinstance(model.transformer.h[0].attn.c_attn, Conv1D)
    assert isinstance(quantized.lm_head, torch.nn.Linear)
    assert quantized.lm_head.weight is quantized.transformer.wte.weight
    assert weight_payload_bytes(quantized) < weight_payload_bytes(model)
    with torch.inference_mode():
        logits = quantized(**tokenizer("a", "pt"), use_cache=False).logits
    assert torch.isfinite(logits).all() and logits.dtype == torch.float32


def test_storage_counts_tied_parameters_once():
    model = torch.nn.Module()
    model.first = torch.nn.Linear(3, 4, bias=False)
    model.second = torch.nn.Linear(3, 4, bias=False)
    model.second.weight = model.first.weight
    assert weight_payload_bytes(model) == 3 * 4 * 4


def test_benchmark_records_real_quantization_and_alternated_pairs():
    result = run_experiment(small_model(), FakeTokenizer(), "a", 2, 0, 2)
    assert result["config"]["measured_execution_order"] == [["fp32", "int8"], ["int8", "fp32"]]
    assert len(result["int8"]["runs"]) == 2
    assert result["setup"]["quantized_module_count"] == 4
    assert result["comparison"]["weight_payload_reduction_bytes"] > 0
    assert isinstance(result["comparison"]["exact_token_id_parity"], bool)


def test_no_projection_is_not_fake_quantization():
    with pytest.raises(ValueError, match="Conv1D"):
        prepare_quantized_model(torch.nn.Linear(3, 4), FakeTokenizer(), "a")

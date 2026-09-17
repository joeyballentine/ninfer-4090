from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.formats import get_format
from tools.convert.calibration import HessianStore
from tools.convert.quantization.gptq import gptq_quantize_matrix
from tools.convert.quantization.groupwise import quantize_matrix


FORMATS = ["q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16"]


def _dequantize(quantized, columns):
    rows, groups, group_size = quantized.codes.shape
    values = quantized.codes.to(torch.float32) * quantized.scales.to(
        torch.float32
    ).unsqueeze(-1)
    return values.reshape(rows, groups * group_size)[:, :columns]


def _objective(weight, quantized, hessian):
    residual = weight.to(torch.float32) - _dequantize(quantized, weight.shape[1])
    return float(((residual @ hessian) * residual).sum())


def _spd(columns, generator, *, correlation=4):
    factor = torch.randn(correlation * columns, columns, generator=generator)
    mixing = torch.randn(columns, columns, generator=generator) / columns**0.5
    return ((factor @ mixing).transpose(0, 1) @ (factor @ mixing)) / columns + 0.05 * (
        torch.eye(columns)
    )


@pytest.mark.parametrize("format", FORMATS)
def test_identity_hessian_reduces_to_round_to_nearest(format) -> None:
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn(24, 256, generator=generator)
    reference = quantize_matrix(weight, format, device="cpu")
    quantized = gptq_quantize_matrix(
        weight, format, torch.eye(256), device="cpu", block_size=128
    )
    assert torch.equal(quantized.codes, reference.codes)
    assert torch.equal(
        quantized.scales.view(torch.int16), reference.scales.view(torch.int16)
    )
    assert quantized.codes.dtype == torch.int8 and quantized.scales.dtype == torch.float16


@pytest.mark.parametrize("format", FORMATS)
def test_correlated_hessian_beats_round_to_nearest(format) -> None:
    generator = torch.Generator().manual_seed(4321)
    hessian = _spd(256, generator)
    weight = torch.randn(64, 256, generator=generator)
    reference = quantize_matrix(weight, format, device="cpu")
    quantized = gptq_quantize_matrix(weight, format, hessian, device="cpu")
    searched = gptq_quantize_matrix(
        weight, format, hessian, device="cpu", candidates=21
    )
    spec = get_format(format)
    assert int(quantized.codes.min()) >= spec.qmin
    assert int(quantized.codes.max()) <= spec.qmax
    words = quantized.scales.view(torch.int16).to(torch.int32) & 0xFFFF
    assert bool((words < 0x7C00).all())
    assert _objective(weight, quantized, hessian) < _objective(
        weight, reference, hessian
    )
    assert _objective(weight, searched, hessian) < _objective(
        weight, quantized, hessian
    )


def test_act_order_keeps_contiguous_groups_and_improves_on_rtn() -> None:
    generator = torch.Generator().manual_seed(99)
    hessian = _spd(256, generator)
    weight = torch.randn(32, 256, generator=generator)
    reference = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    ordered = gptq_quantize_matrix(
        weight, "q4_g64_fp16", hessian, device="cpu", act_order=True, candidates=21
    )
    assert ordered.codes.shape == reference.codes.shape
    assert ordered.scales.shape == reference.scales.shape
    assert _objective(weight, ordered, hessian) < _objective(
        weight, reference, hessian
    )


def test_padded_columns_stay_zero_and_shapes_match() -> None:
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(8, 192, generator=generator)
    reference = quantize_matrix(weight, "q5_g64_fp16", device="cpu")
    quantized = gptq_quantize_matrix(
        weight, "q5_g64_fp16", _spd(192, generator), device="cpu"
    )
    assert quantized.codes.shape == reference.codes.shape == (8, 4, 64)
    assert int(quantized.codes.reshape(8, 256)[:, 192:].abs().max()) == 0


def test_invalid_plan_inputs_are_rejected() -> None:
    weight = torch.randn(4, 128)
    with pytest.raises(ValueError):
        gptq_quantize_matrix(weight, "q4_g64_fp16", torch.eye(64), device="cpu")
    with pytest.raises(ValueError):
        gptq_quantize_matrix(
            weight, "q4_g64_fp16", torch.eye(128), device="cpu", block_size=96
        )
    with pytest.raises(ValueError):
        gptq_quantize_matrix(weight, "q4_g64_fp16", torch.zeros(128, 128), device="cpu")
    with pytest.raises(ValueError):
        gptq_quantize_matrix(weight, "bf16", torch.eye(128), device="cpu")


def test_hessian_store_reads_safetensors_numpy_and_aliases(tmp_path) -> None:
    hessian = torch.eye(64, dtype=torch.float32) * 3.0
    save_file({"text/layers/0/ffn_input": hessian}, str(tmp_path / "hessians.safetensors"))
    (tmp_path / "text" / "layers" / "1").mkdir(parents=True)
    np.save(tmp_path / "text" / "layers" / "1" / "ffn_input.npy", hessian.numpy())
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sites": {"text/layers/2/ffn_input": "text/layers/0/ffn_input"}}),
        encoding="utf-8",
    )
    store = HessianStore(tmp_path)
    for site in (
        "text/layers/0/ffn_input",
        "text/layers/1/ffn_input",
        "text/layers/2/ffn_input",
    ):
        assert torch.equal(store.hessian(site, 64), hessian)
    with pytest.raises(ValueError):
        store.hessian("text/layers/3/ffn_input", 64)
    with pytest.raises(ValueError):
        store.hessian("text/layers/0/ffn_input", 32)
    with pytest.raises(ValueError):
        HessianStore(tmp_path / "missing")

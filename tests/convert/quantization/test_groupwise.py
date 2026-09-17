from __future__ import annotations

import pytest
import torch

from tools.artifact.formats import get_format
from tools.artifact.codecs.row_split import (
    decode_row_split_codes,
    dequantize_row_split,
    encode_row_split,
)
from tools.convert.quantization.groupwise import (
    quantize_matrix,
    quantize_matrix_mse,
)


def test_quantization_uses_stored_fp16_scale_and_zero_padding() -> None:
    weight = torch.tensor(
        [[-7.0, -1.0, 0.0, 1.0, 7.0, 3.5, -3.5, 0.25] + [0.0] * 57],
        dtype=torch.bfloat16,
    )
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert quantized.codes.shape == (1, 2, 64)
    assert quantized.scales.shape == (1, 2)
    expected_codes = torch.zeros((1, 2, 64), dtype=torch.int8)
    expected_codes[0, 0, :8] = torch.tensor([-7, -1, 0, 1, 7, 4, -4, 0])
    assert torch.equal(quantized.codes, expected_codes)
    assert quantized.scales.tolist() == [[1.0, 0.0]]
    assert torch.count_nonzero(quantized.codes[0, 1]) == 0
    assert quantized.scales[0, 1] == 0

    payload = encode_row_split(
        quantized.codes, quantized.scales, "q4_g64_fp16", weight.shape
    )
    scales, codes = decode_row_split_codes(payload, "q4_g64_fp16", tuple(weight.shape))
    assert torch.equal(scales, quantized.scales)
    assert torch.equal(codes, quantized.codes)
    decoded = dequantize_row_split(
        payload, "q4_g64_fp16", tuple(weight.shape), dtype=torch.float32
    )
    expected = expected_codes.float().reshape(1, 128)[:, :65]
    assert torch.equal(decoded, expected)


def test_quantization_uses_canonical_scale_rounding_on_cuda() -> None:
    # BF16 word 0x3636 divided by Q4 qmax is a known CUDA division boundary:
    # approximate device division rounds the FP16 scale to word 7, while the
    # ordered RNE oracle requires word 6.
    value = torch.tensor([0x3636], dtype=torch.int16).view(torch.bfloat16)
    weight = value.repeat(64).reshape(1, 64)
    cpu = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(cpu.scales.view(torch.int16)[0, 0]) == 6
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "q4_g64_fp16", device="cuda")
        assert torch.equal(
            cuda.scales.cpu().view(torch.int16), cpu.scales.view(torch.int16)
        )
        assert torch.equal(cuda.codes.cpu(), cpu.codes)


def test_quantization_uses_reciprocal_multiply_and_ties_to_even() -> None:
    words = torch.tensor([0x41A0B334, 0x417C7BFF], dtype=torch.int32).view(
        torch.float32
    )
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, :2] = words
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 0x41BD
    assert quantized.codes[0, 0, :2].tolist() == [7, 6]
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "q4_g64_fp16", device="cuda")
        assert torch.equal(cuda.scales.cpu(), quantized.scales)
        assert torch.equal(cuda.codes.cpu(), quantized.codes)

    tie_weight = torch.zeros((1, 64), dtype=torch.float32)
    tie_weight[0, :7] = torch.tensor([7.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    ties = quantize_matrix(tie_weight, "q4_g64_fp16", device="cpu")
    assert ties.codes[0, 0, :7].tolist() == [7, 0, 2, 2, 0, -2, -2]


def test_nonzero_scale_underflow_uses_smallest_fp16_subnormal() -> None:
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, 0] = torch.finfo(torch.float32).tiny
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 1


def _group_errors(weight, quantized, format):
    spec = get_format(format)
    padded = torch.zeros(
        (weight.shape[0], quantized.codes.shape[1] * spec.group_size),
        dtype=torch.float32,
    )
    padded[:, : weight.shape[1]] = weight.to(torch.float32)
    grouped = padded.reshape(quantized.codes.shape)
    residual = grouped - quantized.codes.to(torch.float32) * quantized.scales.to(
        torch.float32
    ).unsqueeze(-1)
    return (residual * residual).sum(dim=2)


def _heavy_tailed(rows, columns, generator):
    normal = torch.randn(rows, columns, generator=generator)
    scale = torch.exp(2.5 * torch.randn(rows, columns, generator=generator))
    return normal * scale


@pytest.mark.parametrize("format", ["q4_g64_fp16", "q5_g64_fp16", "q8_g32_fp16"])
@pytest.mark.parametrize("kind", ["gaussian", "heavy_tailed"])
def test_mse_search_never_increases_group_error(format, kind) -> None:
    generator = torch.Generator().manual_seed(20260917)
    weight = (
        torch.randn(48, 512, generator=generator)
        if kind == "gaussian"
        else _heavy_tailed(48, 512, generator)
    )
    absmax = quantize_matrix(weight, format, device="cpu")
    searched = quantize_matrix_mse(weight, format, device="cpu")
    assert searched.codes.shape == absmax.codes.shape
    assert searched.codes.dtype == absmax.codes.dtype == torch.int8
    assert searched.scales.shape == absmax.scales.shape
    assert searched.scales.dtype == absmax.scales.dtype == torch.float16

    spec = get_format(format)
    assert int(searched.codes.min()) >= spec.qmin
    assert int(searched.codes.max()) <= spec.qmax
    words = searched.scales.view(torch.int16).to(torch.int32) & 0xFFFF
    assert bool((words < 0x7C00).all()) and bool((words >= 0).all())

    absmax_error = _group_errors(weight, absmax, format)
    searched_error = _group_errors(weight, searched, format)
    assert bool((searched_error <= absmax_error + 1e-6).all())
    assert float(searched_error.sum()) <= float(absmax_error.sum())
    if spec.bits <= 5:
        # Below six bits the clipping choice is material, not a rounding detail.
        assert float(searched_error.sum()) < 0.97 * float(absmax_error.sum())


def test_mse_search_reproduces_absmax_without_candidates() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(8, 256, generator=generator)
    absmax = quantize_matrix(weight, "q5_g64_fp16", device="cpu")
    single = quantize_matrix_mse(weight, "q5_g64_fp16", device="cpu", candidates=1)
    assert torch.equal(single.codes, absmax.codes)
    assert torch.equal(single.scales.view(torch.int16), absmax.scales.view(torch.int16))


def test_mse_search_is_deterministic_and_weights_importance() -> None:
    generator = torch.Generator().manual_seed(11)
    weight = _heavy_tailed(16, 128, generator)
    first = quantize_matrix_mse(weight, "q4_g64_fp16", device="cpu")
    second = quantize_matrix_mse(weight, "q4_g64_fp16", device="cpu")
    assert torch.equal(first.codes, second.codes)
    assert torch.equal(first.scales.view(torch.int16), second.scales.view(torch.int16))

    importance = torch.rand(128, generator=generator) * 10.0
    weighted = quantize_matrix_mse(
        weight, "q4_g64_fp16", device="cpu", importance=importance
    )
    absmax = quantize_matrix(weight, "q4_g64_fp16", device="cpu")

    def weighted_error(quantized):
        spec = get_format("q4_g64_fp16")
        grouped = weight.to(torch.float32).reshape(16, 2, 64)
        residual = grouped - quantized.codes.to(torch.float32) * quantized.scales.to(
            torch.float32
        ).unsqueeze(-1)
        return (residual * residual * importance.reshape(1, 2, 64)).sum()

    assert float(weighted_error(weighted)) <= float(weighted_error(absmax))

    with pytest.raises(ValueError):
        quantize_matrix_mse(weight, "q4_g64_fp16", device="cpu", importance=-importance)
    with pytest.raises(ValueError):
        quantize_matrix_mse(weight, "q4_g64_fp16", device="cpu", candidates=0)

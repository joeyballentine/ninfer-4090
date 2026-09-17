"""Grouped symmetric quantization used by NInfer artifact converters.

The persistent numeric format fixes the code range, group size, and binary16
scale.  Model-specific recipes decide which tensors use those formats; this
module only performs the registered numeric transform.  Two scale rules share
that transform: the group absmax, and a per-group clipping search minimizing
the group's reconstruction error over the same canonical binary16 scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.artifact.layouts import (
    row_split_geometry,
)
from tools.artifact.formats import QuantFormat, get_format

_FP16_MIN_SUBNORMAL = 2.0**-24

MSE_CANDIDATES = 21
MSE_LOWEST_FACTOR = 0.80


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Physical code groups and binary16 scales for one logical matrix."""

    codes: torch.Tensor
    scales: torch.Tensor


def _canonical_scale_words(
    max_abs: torch.Tensor,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical binary16 scales and binary32 reciprocals on the host.

    CUDA division is not correctly rounded at every binary16 scale boundary.
    The host oracle performs the specified division in binary64, explicitly
    rounds through binary32 and binary16, then computes the reciprocal in the
    same way.  A binary32 input divided by these small integer denominators has
    enough binary64 precision for the final binary32 rounding to be exact.
    """

    host_max = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (host_max.astype(np.float64) / float(qmax)).astype(np.float32)
        scale = raw_scale.astype(np.float16)
    underflow = (scale == 0) & (host_max > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((host_max > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("grouped quantization scale is not finite and positive")

    reciprocal = np.zeros(host_max.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def pick_device(preferred: str | torch.device = "cuda") -> torch.device:
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _grouped_values(
    weight: torch.Tensor,
    format: str | QuantFormat,
    device: str | torch.device | None,
) -> tuple[QuantFormat, torch.Tensor, torch.device]:
    """Return the format, the padded ``[N,groups,group_size]`` view and its device."""

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(
            f"grouped quantization requires rank 2, got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical
    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    return spec, grouped, target


def group_codes(
    grouped: torch.Tensor, reciprocal: torch.Tensor, spec: QuantFormat
) -> torch.Tensor:
    """Select codes from grouped values and the canonical binary32 reciprocals."""

    return torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    )


def clipping_factors(candidates: int) -> tuple[float, ...]:
    """Return descending clipping factors covering ``[MSE_LOWEST_FACTOR, 1]``.

    The first factor is exactly one, so a group whose absmax scale already
    minimizes the searched error keeps the ``grouped_absmax`` result.
    """

    if type(candidates) is not int or candidates < 1:
        raise ValueError("clipping search requires a positive candidate count")
    if candidates == 1:
        return (1.0,)
    step = (1.0 - MSE_LOWEST_FACTOR) / (candidates - 1)
    return tuple(1.0 - index * step for index in range(candidates))


def group_importance(
    importance: torch.Tensor, spec: QuantFormat, geometry, device: torch.device
) -> torch.Tensor:
    """Return a ``[1,groups,group_size]`` nonnegative per-input-channel weight."""

    values = importance.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if values.numel() != geometry.k:
        raise ValueError(
            f"importance has {values.numel()} entries, expected {geometry.k}"
        )
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("importance must be finite and nonnegative")
    padded = torch.zeros(geometry.k_pad, dtype=torch.float32, device=device)
    padded[: geometry.k].copy_(values)
    return padded.reshape(1, geometry.groups_per_row, spec.group_size)


def search_group_scales(
    grouped: torch.Tensor,
    spec: QuantFormat,
    candidates: int,
    *,
    importance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one clipping factor per group of a ``[...,group_size]`` tensor.

    Every candidate scale passes through the canonical binary16 word, so the
    search only reorders choices the stored format can already represent.  The
    factors descend from one and ties keep the earlier, larger factor, which
    makes a single candidate identical to the plain absmax scale.  ``importance``
    broadcasts against *grouped* and weights the squared error per channel.
    """

    factors = clipping_factors(candidates)
    device = grouped.device
    host_max = grouped.abs().amax(dim=-1).detach().cpu().to(torch.float32)
    if len(factors) == 1:
        words, reciprocal = _canonical_scale_words(host_max, spec.qmax)
        return words.to(device), reciprocal.to(device)

    shape = tuple(host_max.shape)
    count = len(factors)
    scale_words = torch.empty((*shape, count), dtype=torch.float16)
    reciprocals = torch.empty((*shape, count), dtype=torch.float32)
    errors = torch.empty((*shape, count), dtype=torch.float32, device=device)
    for index, factor in enumerate(factors):
        clipped = (host_max.to(torch.float64) * factor).to(torch.float32)
        words, reciprocal = _canonical_scale_words(clipped, spec.qmax)
        scale_words[..., index] = words
        reciprocals[..., index] = reciprocal
        codes = group_codes(grouped, reciprocal.to(device), spec)
        chosen = words.to(device=device, dtype=torch.float32).unsqueeze(-1)
        residual = grouped - codes * chosen
        residual = residual * residual
        if importance is not None:
            residual = residual * importance
        errors[..., index] = residual.sum(dim=-1)

    best = errors[..., 0].clone()
    choice = torch.zeros(shape, dtype=torch.int64, device=device)
    for index in range(1, count):
        improved = errors[..., index] < best
        best = torch.where(improved, errors[..., index], best)
        choice = torch.where(improved, index, choice)

    picked = choice.detach().cpu().unsqueeze(-1)
    scales = torch.gather(scale_words, -1, picked).squeeze(-1).to(device)
    reciprocal = torch.gather(reciprocals, -1, picked).squeeze(-1).to(device)
    return scales, reciprocal


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values, including registered K padding.

    Scales are rounded to binary16 before codes are selected because those are
    the exact scales consumed after loading.  Padding values are zero and do
    not affect a partially populated final group.
    """

    spec, grouped, target = _grouped_values(weight, format, device)
    max_abs = grouped.abs().amax(dim=2)
    host_scales, host_reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    scales = host_scales.to(target)
    reciprocal = host_reciprocal.to(target)
    codes = group_codes(grouped, reciprocal, spec).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales)


def quantize_matrix_mse(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
    candidates: int = MSE_CANDIDATES,
    importance: torch.Tensor | None = None,
) -> QuantizedMatrix:
    """Quantize ``[N,K]`` values with a per-group clipping-scale search.

    Each group evaluates ``absmax * factor / qmax`` for every clipping factor,
    rounds it through the same canonical binary16 word as ``quantize_matrix``,
    and keeps the factor with the smallest reconstruction error.  ``importance``
    optionally weights that error per logical input channel.  Equal errors keep
    the largest factor, so the search never replaces an already optimal absmax
    scale and never selects a higher group error than ``quantize_matrix``.
    """

    spec, grouped, target = _grouped_values(weight, format, device)
    geometry = row_split_geometry(spec, weight.shape)
    channel_weight = (
        None
        if importance is None
        else group_importance(importance, spec, geometry, target)
    )
    scales, reciprocal = search_group_scales(
        grouped, spec, candidates, importance=channel_weight
    )
    codes = group_codes(grouped, reciprocal, spec).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales)

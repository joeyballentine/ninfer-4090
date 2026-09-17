"""GPTQ code selection for the grouped symmetric integer formats.

The algorithm is the one of Frantar et al.: optimal brain quantization with
lazy batch updates over a damped inverse Hessian.  It only changes the order
and the error compensation used when codes are selected; the stored words stay
those of :mod:`tools.convert.quantization.groupwise`, so the consumer kernels
see the same codes-and-scales representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.artifact.formats import QuantFormat, get_format
from tools.artifact.layouts import row_split_geometry

from .groupwise import (
    QuantizedMatrix,
    group_codes,
    pick_device,
    search_group_scales,
)

GPTQ_BLOCK_SIZE = 128
GPTQ_DAMPING = 0.01
_CHOLESKY_ATTEMPTS = 6


def _damped_inverse_cholesky(
    hessian: torch.Tensor, damping: float
) -> tuple[torch.Tensor, float]:
    """Return the upper Cholesky factor of the damped inverse Hessian."""

    mean = float(torch.diagonal(hessian).mean())
    if not mean > 0.0:
        raise ValueError("calibration Hessian has a nonpositive mean diagonal")
    factor = damping
    for _ in range(_CHOLESKY_ATTEMPTS):
        damped = hessian.clone()
        damped.diagonal().add_(factor * mean)
        try:
            lower = torch.linalg.cholesky(damped)
            inverse = torch.cholesky_inverse(lower)
            return torch.linalg.cholesky(inverse, upper=True), factor
        except RuntimeError:
            factor *= 10.0
    raise ValueError("damped calibration Hessian is not positive definite")


@dataclass(frozen=True, slots=True)
class GptqPlan:
    """Column order, inverse-Hessian factor and scale policy for one input."""

    spec: QuantFormat
    columns: int
    logical_columns: int
    inverse: torch.Tensor
    order: torch.Tensor | None
    group_of_column: torch.Tensor
    importance: torch.Tensor
    candidates: int
    block_size: int
    damping: float


def gptq_plan(
    format: str | QuantFormat,
    logical_columns: int,
    hessian: torch.Tensor,
    *,
    device: str | torch.device | None = None,
    block_size: int = GPTQ_BLOCK_SIZE,
    damping: float = GPTQ_DAMPING,
    act_order: bool = False,
    candidates: int = 1,
) -> GptqPlan:
    """Factor one calibration Hessian into the reusable per-input GPTQ plan.

    The Hessian covers the logical input channels; registered K padding is
    appended as an independent identity block, so padded columns quantize to
    zero without disturbing the logical ones.  A channel the corpus never
    activated is decoupled the same way and keeps its own round-to-nearest
    code rather than being discarded.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("GPTQ requires a quantized numeric format")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("GPTQ block size must be positive")
    if block_size % spec.group_size:
        raise ValueError("GPTQ block size must be a multiple of the format group size")
    if not (isinstance(damping, float) and damping > 0.0):
        raise ValueError("GPTQ damping must be a positive fraction")
    geometry = row_split_geometry(spec, (1, logical_columns))
    if geometry.k_pad % block_size:
        raise ValueError("GPTQ block size must divide the padded column count")
    if hessian.dim() != 2 or hessian.shape != (logical_columns, logical_columns):
        raise ValueError(
            f"calibration Hessian must be [{logical_columns},{logical_columns}], "
            f"got {tuple(hessian.shape)}"
        )
    target = pick_device() if device is None else pick_device(device)
    columns = geometry.k_pad
    square = torch.eye(columns, dtype=torch.float32, device=target)
    logical = hessian.detach().to(device=target, dtype=torch.float32)
    if not bool(torch.isfinite(logical).all()):
        raise ValueError("calibration Hessian contains NaN or infinity")
    symmetric = 0.5 * (logical + logical.transpose(0, 1))
    square[:logical_columns, :logical_columns] = symmetric
    if not bool((torch.diagonal(symmetric) > 0.0).any()):
        raise ValueError("calibration Hessian has no positive diagonal entry")
    dead = torch.diagonal(square) <= 0.0
    if bool(dead.any()):
        square[dead, :] = 0.0
        square[:, dead] = 0.0
        square.diagonal()[dead] = 1.0
    importance = torch.diagonal(square).clone()
    group_of_column = torch.arange(columns, device=target) // spec.group_size
    order = None
    if act_order:
        order = torch.argsort(importance, descending=True, stable=True)
        group_of_column = group_of_column.index_select(0, order)
        square = square.index_select(0, order).index_select(1, order)
    inverse, used = _damped_inverse_cholesky(square, damping)
    return GptqPlan(
        spec=spec,
        columns=columns,
        logical_columns=logical_columns,
        inverse=inverse,
        order=order,
        group_of_column=group_of_column,
        importance=importance,
        candidates=candidates,
        block_size=block_size,
        damping=used,
    )


def gptq_quantize_rows(plan: GptqPlan, weight: torch.Tensor) -> QuantizedMatrix:
    """Quantize ``[N,K]`` rows against a prepared plan, sharing its Hessian."""

    spec = plan.spec
    if weight.dim() != 2 or weight.shape[1] != plan.logical_columns:
        raise ValueError(
            f"GPTQ rows must be [N,{plan.logical_columns}], got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    target = plan.inverse.device
    rows, group_size = weight.shape[0], spec.group_size
    groups = plan.columns // group_size
    values = torch.zeros((rows, plan.columns), dtype=torch.float32, device=target)
    values[:, : plan.logical_columns] = weight.detach().to(
        device=target, dtype=torch.float32
    )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("GPTQ source contains NaN or infinity")
    importance = (
        plan.importance.reshape(1, groups, group_size) if plan.candidates > 1 else None
    )

    scales = torch.zeros((rows, groups), dtype=torch.float16, device=target)
    reciprocals = torch.zeros((rows, groups), dtype=torch.float32, device=target)
    if plan.order is None:
        ordered = values
    else:
        # Act-order keeps static groups: the stored layout needs contiguous
        # original groups, so their scales come from the unpermuted weights.
        scales, reciprocals = search_group_scales(
            values.reshape(rows, groups, group_size),
            spec,
            plan.candidates,
            importance=importance,
        )
        ordered = values.index_select(1, plan.order)

    codes = torch.zeros((rows, plan.columns), dtype=torch.float32, device=target)
    for start in range(0, plan.columns, plan.block_size):
        stop = start + plan.block_size
        block = ordered[:, start:stop].clone()
        residuals = torch.zeros_like(block)
        local = plan.inverse[start:stop, start:stop]
        for offset in range(stop - start):
            column = start + offset
            group = int(plan.group_of_column[column])
            if plan.order is None and column % group_size == 0:
                words, reciprocal = search_group_scales(
                    block[:, offset : offset + group_size],
                    spec,
                    plan.candidates,
                    importance=(
                        None
                        if importance is None
                        else importance[:, group, :].reshape(1, group_size)
                    ),
                )
                scales[:, group] = words
                reciprocals[:, group] = reciprocal
            current = block[:, offset]
            code = torch.clamp(
                torch.round(current * reciprocals[:, group]), spec.qmin, spec.qmax
            )
            codes[:, column] = code
            error = (current - code * scales[:, group].to(torch.float32)) / local[
                offset, offset
            ]
            update = error.unsqueeze(1) * local[offset, offset:].unsqueeze(0)
            block[:, offset:] -= update
            residuals[:, offset] = error
        if stop < plan.columns:
            ordered[:, stop:] -= residuals @ plan.inverse[start:stop, stop:]

    if plan.order is not None:
        codes = codes.index_select(1, torch.argsort(plan.order))
    return QuantizedMatrix(
        codes=codes.reshape(rows, groups, group_size).to(torch.int8), scales=scales
    )


def gptq_quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    hessian: torch.Tensor,
    *,
    device: str | torch.device | None = None,
    block_size: int = GPTQ_BLOCK_SIZE,
    damping: float = GPTQ_DAMPING,
    act_order: bool = False,
    candidates: int = 1,
) -> QuantizedMatrix:
    """Quantize one ``[N,K]`` matrix against one calibration Hessian."""

    plan = gptq_plan(
        format,
        weight.shape[1],
        hessian,
        device=device,
        block_size=block_size,
        damping=damping,
        act_order=act_order,
        candidates=candidates,
    )
    return gptq_quantize_rows(plan, weight)

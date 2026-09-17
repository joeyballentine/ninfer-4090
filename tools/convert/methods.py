"""Method requests/results and the built-in adapters to quantization or encoded import.

Methods own input traversal, chunking, and auxiliary values. Numerical algorithms
return codes/scales; artifact.tensor_output owns their physical byte placement.
User methods accept the same PrepareRequest and return a PreparedMethod.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import prod
import struct
from typing import Callable, Mapping

import torch

from tools.artifact.formats import (
    DirectFormat,
    QuantFormat,
    get_format,
    valid_positive_fp32_word,
)
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput

from .calibration import HessianStore
from .quantization.fp8_row import quantize_bf16_rows
from .quantization.gptq import (
    GPTQ_BLOCK_SIZE,
    GPTQ_DAMPING,
    gptq_plan,
    gptq_quantize_rows,
)
from .quantization.groupwise import MSE_CANDIDATES, quantize_matrix, quantize_matrix_mse
from .sources.logical import EncodedRows, LogicalSource

UseKey = tuple[str, str]
AuxiliaryKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class MethodInput:
    parameter: str
    source: LogicalSource
    uses: tuple[UseKey, ...]


@dataclass(frozen=True, slots=True)
class AuxiliaryValue:
    format: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def activation_divisor(cls, value: bytes | float) -> AuxiliaryValue:
        raw = value if isinstance(value, bytes) else struct.pack("<f", value)
        if len(raw) != 4 or not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
            raise ValueError("activation input divisor must be positive finite FP32")
        return cls("fp32", (), raw)


@dataclass(frozen=True, slots=True)
class PreparedMethod:
    produce: Callable[[TensorOutput], None]
    auxiliaries: Mapping[AuxiliaryKey, AuxiliaryValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    target: TensorSpec
    inputs: tuple[MethodInput, ...]
    policies: Mapping[UseKey, str]
    parameters: Mapping[str, object]
    device: str = "cuda"
    rows_per_chunk: int = 512
    auxiliary_overrides: Mapping[AuxiliaryKey, AuxiliaryValue] = field(
        default_factory=dict
    )
    source_offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        values = [0]
        for item in self.inputs:
            values.append(values[-1] + prod(item.source.shape))
        object.__setattr__(self, "source_offsets", tuple(values))

    @property
    def source(self) -> LogicalSource:
        if len(self.inputs) != 1:
            raise ValueError("this method has multiple inputs; use request.inputs")
        return self.inputs[0].source

    def require_input_shape(self, shape: tuple[int, ...]) -> None:
        if self.source.shape != tuple(shape):
            raise ValueError(f"source shape {self.source.shape} differs from {shape}")

    def job(self, *, produce, auxiliaries=None) -> PreparedMethod:
        return PreparedMethod(produce, {} if auxiliaries is None else dict(auxiliaries))

    def values(self, begin: int, end: int) -> torch.Tensor:
        """Read the ordered logical inputs as one C-order parent element sequence."""
        pieces = []
        if not 0 <= begin <= end <= self.source_offsets[-1]:
            raise ValueError(f"{self.target.id}: requested values exceed method inputs")
        index = bisect_right(self.source_offsets, begin) - 1
        while begin < end:
            high = min(end, self.source_offsets[index + 1])
            pieces.append(
                self.inputs[index].source.values(
                    begin - self.source_offsets[index],
                    high - self.source_offsets[index],
                )
            )
            begin = high
            index += 1
        if not pieces:
            return torch.empty(0)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def encoded_rows(self, begin: int, end: int) -> EncodedRows:
        pieces = []
        cursor = 0
        for item in self.inputs:
            source = item.source
            if len(source.shape) != 2 or source.shape[1] != self.target.shape[1]:
                raise ValueError(
                    f"{item.parameter}: encoded grouping requires complete rows"
                )
            low, high = max(begin, cursor), min(end, cursor + source.shape[0])
            if low < high:
                if source.read_encoded is None:
                    raise ValueError(f"{item.parameter}: encoded rows are unavailable")
                pieces.append(source.read_encoded(low - cursor, high - cursor))
            cursor += source.shape[0]
        if not pieces or end > cursor:
            raise ValueError("encoded method input range is invalid")
        first = pieces[0]
        if any(
            (p.format, p.weight_divisor) != (first.format, first.weight_divisor)
            for p in pieces
        ):
            raise ValueError("encoded inputs cannot share one parent format/divisor")
        if len(pieces) == 1:
            return first
        return EncodedRows(
            first.format,
            torch.cat([p.codes for p in pieces]),
            torch.cat([p.scales for p in pieces]),
            first.weight_divisor,
        )


Method = Callable[[PrepareRequest], PreparedMethod]
_DIRECT_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}


def _preflight(
    request: PrepareRequest,
    *,
    values: bool = True,
    parameters: tuple[str, ...] = (),
) -> None:
    unknown = sorted(set(request.parameters) - set(parameters))
    if unknown:
        accepted = ", ".join(parameters) if parameters else "none"
        raise ValueError(
            f"{request.target.id}: unknown numerical parameters {unknown}; "
            f"this method accepts {accepted}"
        )
    if prod(request.target.shape) != sum(
        prod(item.source.shape) for item in request.inputs
    ):
        raise ValueError(
            f"{request.target.id}: parent and logical source counts differ"
        )
    if type(request.rows_per_chunk) is not int or request.rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    if values:
        for item in request.inputs:
            item.source.values(0, 1)


def cast_direct(request: PrepareRequest) -> PreparedMethod:
    """Convert values at the explicit target BF16/FP32/INT32 boundary."""
    if not isinstance(get_format(request.target.format), DirectFormat):
        raise ValueError("cast_direct requires a direct target format")
    _preflight(request)
    dtype = _DIRECT_DTYPES[request.target.format]
    chunk = request.rows_per_chunk * (
        request.target.shape[-1] if len(request.target.shape) > 1 else 1
    )
    elements = prod(request.target.shape)

    def produce(output):
        for begin in range(0, elements, chunk):
            values = request.values(begin, min(elements, begin + chunk))
            if dtype == torch.int32 and (
                bool((values < -(1 << 31)).any())
                or bool((values > (1 << 31) - 1).any())
            ):
                raise ValueError(
                    "int32 conversion source is outside the representable range"
                )
            values = values.to(dtype=dtype)
            output.write_values(begin, values)

    return request.job(produce=produce)


def grouped_absmax(request: PrepareRequest) -> PreparedMethod:
    """Use the existing grouped max-abs, FP16-scale and code-rounding algorithm."""
    n, k = _integer_matrix(request, "grouped_absmax")
    _preflight(request)

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not values.dtype.is_floating_point:
                raise TypeError(
                    "grouped_absmax source must provide floating-point values"
                )
            encoded = quantize_matrix(
                values, request.target.format, device=request.device
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def _integer_matrix(request: PrepareRequest, method: str) -> tuple[int, int]:
    if (
        not isinstance(get_format(request.target.format), QuantFormat)
        or len(request.target.shape) != 2
    ):
        raise ValueError(f"{method} requires a grouped-integer matrix target")
    return request.target.shape


def _integer_parameter(request: PrepareRequest, name: str, default: int) -> int:
    value = request.parameters.get(name, default)
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def grouped_mse(request: PrepareRequest) -> PreparedMethod:
    """Search each group's clipping scale for the smallest squared error."""
    n, k = _integer_matrix(request, "grouped_mse")
    _preflight(request, parameters=("candidates",))
    candidates = _integer_parameter(request, "candidates", MSE_CANDIDATES)

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not values.dtype.is_floating_point:
                raise TypeError("grouped_mse source must provide floating-point values")
            encoded = quantize_matrix_mse(
                values,
                request.target.format,
                device=request.device,
                candidates=candidates,
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def grouped_gptq(request: PrepareRequest) -> PreparedMethod:
    """Select codes with GPTQ error compensation against a calibration Hessian."""
    n, k = _integer_matrix(request, "grouped_gptq")
    _preflight(
        request,
        parameters=(
            "calibration",
            "block_size",
            "damping",
            "act_order",
            "mse",
            "candidates",
        ),
    )
    directory = request.parameters.get("calibration")
    if not isinstance(directory, str) or not directory:
        raise ValueError("grouped_gptq requires a calibration directory")
    block_size = _integer_parameter(request, "block_size", GPTQ_BLOCK_SIZE)
    damping = request.parameters.get("damping", GPTQ_DAMPING)
    if type(damping) is not float or not 0.0 < damping < 1.0:
        raise ValueError("damping must be a fraction in (0, 1)")
    act_order = request.parameters.get("act_order", False)
    mse = request.parameters.get("mse", False)
    if type(act_order) is not bool or type(mse) is not bool:
        raise ValueError("act_order and mse must be boolean")
    candidates = _integer_parameter(request, "candidates", MSE_CANDIDATES) if mse else 1
    store = HessianStore(directory)
    sites = []
    for item in request.inputs:
        if len(item.source.shape) != 2 or item.source.shape[1] != k:
            raise ValueError(f"{item.parameter}: GPTQ requires complete [N,K] rows")
        if len(item.uses) != 1:
            raise ValueError(
                f"{item.parameter}: GPTQ needs exactly one mathematical input,"
                f" got {len(item.uses)}"
            )
        sites.append(item.uses[0][1])

    def produce(output):
        row = 0
        plans = {}
        for item, site in zip(request.inputs, sites):
            # Projections packed into one parent usually read one input and
            # then share its factorization.
            if site not in plans:
                plans[site] = gptq_plan(
                    request.target.format,
                    k,
                    store.hessian(site, k),
                    device=request.device,
                    block_size=block_size,
                    damping=damping,
                    act_order=act_order,
                    candidates=candidates,
                )
            plan = plans[site]
            rows = item.source.shape[0]
            for begin in range(0, rows, request.rows_per_chunk):
                end = min(rows, begin + request.rows_per_chunk)
                values = item.source.rows(begin, end)
                if not values.dtype.is_floating_point:
                    raise TypeError(
                        "grouped_gptq source must provide floating-point values"
                    )
                encoded = gptq_quantize_rows(plan, values)
                output.write_codes(row + begin, encoded.codes, encoded.scales)
            row += rows

    return request.job(produce=produce)


def fp8_row_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Round inputs to BF16, then quantize to FP8 codes with BF16 row scales."""
    if request.target.format != "fp8_e4m3fn_row_bf16" or len(request.target.shape) != 2:
        raise ValueError("fp8_row_maxabs requires the row-scaled FP8 matrix format")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = (
                request.values(begin * k, end * k)
                .reshape(end - begin, k)
                .to(torch.bfloat16)
            )
            encoded = quantize_bf16_rows(values)
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def import_encoded(request: PrepareRequest) -> PreparedMethod:
    """Preserve the current FP8/NVFP4 source codes, scales and weight divisor."""
    if (
        request.target.format not in ("nvfp4", "fp8_e4m3fn_row_bf16")
        or len(request.target.shape) != 2
    ):
        raise ValueError("import_encoded requires a known encoded matrix target")
    _preflight(request, values=False)
    auxiliaries = {}
    weight_divisor = None
    for item in request.inputs:
        source = item.source
        if source.read_encoded is None:
            raise ValueError(f"{item.parameter}: encoded rows are unavailable")
        first = source.read_encoded(0, 1)
        if first.format != request.target.format:
            raise ValueError(
                f"{item.parameter}: source {first.format} differs from target {request.target.format}"
            )
        if first.format == "nvfp4":
            if weight_divisor is None:
                weight_divisor = first.weight_divisor
            elif first.weight_divisor != weight_divisor:
                raise ValueError(
                    "NVFP4 weight divisors differ; choose separate parents or a conversion method"
                )
            for parameter, input_name in item.uses:
                key = (parameter, input_name, "activation_input_divisor")
                if key in request.auxiliary_overrides:
                    auxiliaries[key] = request.auxiliary_overrides[key]
                elif request.policies[(parameter, input_name)] == "AllowA4":
                    if source.input_divisor is None:
                        raise ValueError(
                            f"{parameter}: supply an activation divisor for AllowA4"
                        )
                    auxiliaries[key] = AuxiliaryValue.activation_divisor(
                        source.input_divisor()
                    )
    n = request.target.shape[0]
    chunk = (
        max(128, request.rows_per_chunk // 128 * 128)
        if request.target.format == "nvfp4"
        else request.rows_per_chunk
    )

    def produce(output):
        for begin in range(0, n, chunk):
            words = request.encoded_rows(begin, min(n, begin + chunk))
            output.write_codes(begin, words.codes, words.scales, words.weight_divisor)

    return request.job(produce=produce, auxiliaries=auxiliaries)


METHODS: dict[str, Method] = {
    "cast_direct": cast_direct,
    "grouped_absmax": grouped_absmax,
    "grouped_mse": grouped_mse,
    "grouped_gptq": grouped_gptq,
    "fp8_row_maxabs": fp8_row_maxabs,
    "import_encoded": import_encoded,
}

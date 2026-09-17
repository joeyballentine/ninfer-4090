from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.row_split import decode_row_split_codes
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

FORMAT = "q4_g64_fp16"


def _model(rows=8, columns=128):
    generator = torch.Generator().manual_seed(2026)
    model = Model({"text": {"config": {}}})
    for name in ("query", "key"):
        values = torch.randn(rows, columns, generator=generator).to(torch.bfloat16)
        model.add(
            Parameter(
                name, (rows, columns), array_source(values, name), inputs=("site",)
            )
        )
    model.packing_groups = [("query", "key")]
    return model


def _convert(path, model, recipe):
    prepared = recipe.prepare(device="cpu", rows_per_chunk=4)
    with ArtifactWriter(
        path,
        [job.spec for job in prepared.weights],
        components=model.components,
        bindings=prepared.bindings,
        uses=prepared.uses,
    ) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
    return prepared


def _codes(path, model, name):
    artifact = Artifact(path)
    (object_id, begin, end), = binding_parts(
        artifact.directory.bindings[name], artifact.by_id
    )
    obj = artifact.object(object_id)
    scales, codes = decode_row_split_codes(
        artifact.read_object(object_id), obj.format, obj.shape
    )
    width = model.parameters[name].shape[1] // 64
    return scales[begin // 128 : end // 128], codes[begin // 128 : end // 128], width


def _identity_calibration(directory, columns):
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {"site": torch.eye(columns, dtype=torch.float32)},
        str(directory / "hessians.safetensors"),
    )
    return str(directory)


def test_grouped_mse_and_gptq_share_the_absmax_output_format(tmp_path) -> None:
    model = _model()
    calibration = _identity_calibration(tmp_path / "calibration", 128)
    results = {}
    for label, method, parameters in (
        ("absmax", "grouped_absmax", None),
        ("mse", "grouped_mse", None),
        ("gptq", "grouped_gptq", {"calibration": calibration}),
    ):
        recipe = Recipe(model)
        recipe.assign(
            ("query", "key"), format=FORMAT, method=method, parameters=parameters
        )
        prepared = _convert(tmp_path / f"{label}.ninfer", model, recipe)
        # Both new methods stay eligible for the automatic packing group.
        assert len(prepared.weights) == 1
        assert prepared.weights[0].spec.shape == (16, 128)
        results[label] = _codes(tmp_path / f"{label}.ninfer", model, "query")

    for label in ("mse", "gptq"):
        scales, codes, _ = results[label]
        assert codes.shape == results["absmax"][1].shape
        assert codes.dtype == torch.int8 and scales.dtype == torch.float16
        assert int(codes.min()) >= -8 and int(codes.max()) <= 7
    # Identity calibration makes GPTQ the plain round-to-nearest result.
    assert torch.equal(results["gptq"][1], results["absmax"][1])
    assert torch.equal(
        results["gptq"][0].view(torch.int16), results["absmax"][0].view(torch.int16)
    )
    assert not torch.equal(results["mse"][0], results["absmax"][0])


def test_method_parameters_are_validated(tmp_path) -> None:
    model = _model()
    recipe = Recipe(model)
    recipe.assign("query", format=FORMAT, method="grouped_mse", parameters={"clip": 1})
    with pytest.raises(ValueError, match="unknown numerical parameters"):
        recipe.prepare(device="cpu")

    recipe = Recipe(model)
    recipe.assign("query", format=FORMAT, method="grouped_gptq")
    with pytest.raises(ValueError, match="calibration directory"):
        recipe.prepare(device="cpu")

    recipe = Recipe(model)
    recipe.assign(
        "query",
        format=FORMAT,
        method="grouped_gptq",
        parameters={"calibration": str(tmp_path), "damping": 2.0},
    )
    with pytest.raises(ValueError, match="damping"):
        recipe.prepare(device="cpu")

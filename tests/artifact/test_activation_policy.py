from __future__ import annotations

import hashlib

import pytest
import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import parse_directory
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source
from tools.set_activation_policy import main, set_activation_policy

# name, format, mathematical input, initial activation policy
PARAMETERS = (
    ("text/layers/0/attention/query", "q4_g64_fp16", "text/layers/0/mixer_input", None),
    ("text/layers/0/attention/key", "q4_g64_fp16", "text/layers/0/mixer_input", None),
    ("text/layers/0/attention/value", "q5_g64_fp16", "text/layers/0/mixer_input", None),
    ("text/layers/0/gdn/a_projection", "bf16", "text/layers/0/mixer_input", None),
    ("text/layers/0/mlp/down", "q5_g64_fp16", "text/layers/0/mlp/product", None),
    ("text/output_head", "q8_g32_fp16", "text/final_hidden", None),
    ("mtp/layers/0/mlp/down", "q5_g64_fp16", "mtp/layers/0/mlp/product", "AllowA4"),
)


def _artifact(path, *, max_file_bytes=None):
    generator = torch.Generator().manual_seed(2026)
    model = Model({"text": {"config": {}}, "mtp": {"config": {}}})
    recipe_formats = {}
    for name, format, input_name, policy in PARAMETERS:
        values = torch.randn(8, 128, generator=generator).to(torch.bfloat16)
        model.add(
            Parameter(
                name,
                (8, 128),
                array_source(values, name),
                inputs=(input_name,),
                residency="text" if name.startswith("text/") else "mtp",
            )
        )
        recipe_formats[name] = (format, policy)
    recipe = Recipe(model)
    for name, (format, policy) in recipe_formats.items():
        if format != "bf16":
            recipe.assign(name, format=format, method="grouped_absmax")
        if policy is not None:
            recipe.assign(name, activation_policy=policy)
    prepared = recipe.prepare(device="cpu", rows_per_chunk=4)
    limit = {} if max_file_bytes is None else {"max_file_bytes": max_file_bytes}
    with ArtifactWriter(
        path,
        [job.spec for job in prepared.weights],
        components=model.components,
        bindings=prepared.bindings,
        uses=prepared.uses,
        metadata={"name": "policy-fixture"},
        **limit,
    ) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
    return path


def _policies(path):
    with Artifact(path) as artifact:
        return {
            (use["parameter"], use["input"]): use.get("activation_policy")
            for use in artifact.directory.uses
        }


def _payload_digest(path):
    with Artifact(path) as artifact:
        digest = hashlib.sha256()
        for chunk in artifact.iter_range(0, artifact.payload_bytes):
            digest.update(chunk)
        return digest.hexdigest()


def test_selected_records_flip_and_the_payload_is_copied_unchanged(tmp_path):
    source = _artifact(tmp_path / "in.ninfer")
    output = tmp_path / "out.ninfer"
    summary = set_activation_policy(
        source,
        output,
        policy="AllowA8",
        patterns=("text/layers/*/attention/query", "text/layers/*/attention/value"),
    )
    assert summary["selected"] == 2 and summary["changed"] == 2
    assert summary["previous"] == {"A16Only": 2}
    assert summary["files"] == [str(output)]

    before, after = _policies(source), _policies(output)
    flipped = {
        ("text/layers/0/attention/query", "text/layers/0/mixer_input"),
        ("text/layers/0/attention/value", "text/layers/0/mixer_input"),
    }
    assert set(after) == set(before)
    for key, policy in after.items():
        assert policy == ("AllowA8" if key in flipped else before[key])
    assert before[("mtp/layers/0/mlp/down", "mtp/layers/0/mlp/product")] == "AllowA4"

    assert _payload_digest(output) == _payload_digest(source)
    with Artifact(source) as original, Artifact(output) as rewritten:
        assert rewritten.directory.objects == original.directory.objects
        assert rewritten.directory.bindings == original.directory.bindings
        assert rewritten.directory.metadata == original.directory.metadata
        assert rewritten.artifact_id != original.artifact_id
        parse_directory(rewritten.directory.to_json(), entry_name=output.name)
        for obj in rewritten.objects:
            assert rewritten.read_object(obj.id) == original.read_object(obj.id)


def test_text_projections_selects_the_q4_and_q5_layer_projections(tmp_path):
    source = _artifact(tmp_path / "in.ninfer")
    output = tmp_path / "out.ninfer"
    summary = set_activation_policy(
        source, output, policy="AllowA8", text_projections=True
    )
    assert summary["selected"] == 4 and summary["changed"] == 4
    permitted = {
        name for (name, _), policy in _policies(output).items() if policy == "AllowA8"
    }
    assert permitted == {
        "text/layers/0/attention/query",
        "text/layers/0/attention/key",
        "text/layers/0/attention/value",
        "text/layers/0/mlp/down",
    }


def test_formats_restrict_the_selection(tmp_path):
    source = _artifact(tmp_path / "in.ninfer")
    output = tmp_path / "out.ninfer"
    summary = set_activation_policy(
        source,
        output,
        policy="AllowA8",
        patterns=("text/*",),
        formats=("q4_g64_fp16",),
    )
    assert summary["selected"] == 2
    permitted = {
        name for (name, _), policy in _policies(output).items() if policy == "AllowA8"
    }
    assert permitted == {
        "text/layers/0/attention/query",
        "text/layers/0/attention/key",
    }


def test_the_source_file_segmentation_is_reproduced(tmp_path):
    source = _artifact(tmp_path / "in.ninfer", max_file_bytes=8192)
    output = tmp_path / "out.ninfer"
    summary = set_activation_policy(
        source, output, policy="AllowA8", text_projections=True
    )
    with Artifact(source) as original, Artifact(output) as rewritten:
        original_files = original.directory.files
        assert len(original_files) > 1
        assert [file.payload_bytes for file in rewritten.directory.files] == [
            file.payload_bytes for file in original_files
        ]
        assert [file.path for file in rewritten.directory.files[1:]] == [
            f"{output.name}.part-{index:04d}" for index in range(1, len(original_files))
        ]
    assert len(summary["files"]) == len(original_files)
    assert _payload_digest(output) == _payload_digest(source)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"policy": "AllowA2", "patterns": ("text/*",)}, "unknown activation policy"),
        ({"policy": "AllowA8", "patterns": ("vision/*",)}, "matched no Use record"),
        ({"policy": "AllowA8"}, "select records with"),
        (
            {"policy": "AllowA8", "patterns": ("text/layers/*",)},
            "not stored in an integer groupwise format",
        ),
        (
            {"policy": "AllowA8", "patterns": ("text/*",), "formats": ("bf16",)},
            "--formats accepts integer groupwise formats",
        ),
    ],
)
def test_invalid_selections_and_policies_are_rejected(tmp_path, arguments, message):
    source = _artifact(tmp_path / "in.ninfer")
    output = tmp_path / "out.ninfer"
    with pytest.raises(ValueError, match=message):
        set_activation_policy(source, output, **arguments)
    assert not output.exists()


def test_lowering_a_permission_requires_force(tmp_path):
    source = _artifact(tmp_path / "in.ninfer")
    output = tmp_path / "out.ninfer"
    selection = {"policy": "AllowA8", "patterns": ("mtp/layers/*/mlp/down",)}
    with pytest.raises(ValueError, match="would lose permission"):
        set_activation_policy(source, output, **selection)
    assert not output.exists()
    summary = set_activation_policy(source, output, force=True, **selection)
    assert summary["previous"] == {"AllowA4": 1} and summary["changed"] == 1
    assert (
        _policies(output)[("mtp/layers/0/mlp/down", "mtp/layers/0/mlp/product")]
        == "AllowA8"
    )
    assert _payload_digest(output) == _payload_digest(source)


def test_the_output_must_be_a_new_path(tmp_path):
    source = _artifact(tmp_path / "in.ninfer")
    output = tmp_path / "out.ninfer"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        set_activation_policy(source, output, policy="AllowA8", text_projections=True)
    assert output.read_bytes() == b"existing"
    with pytest.raises(FileExistsError):
        set_activation_policy(source, source, policy="AllowA8", text_projections=True)


def test_command_line_reports_the_change(tmp_path, capsys):
    source = _artifact(tmp_path / "in.ninfer")
    output = tmp_path / "out.ninfer"
    main(
        [
            str(source),
            str(output),
            "--policy",
            "AllowA8",
            "--select",
            "text/layers/*/mlp/down",
            "--formats",
            "q4_g64_fp16,q5_g64_fp16",
        ]
    )
    printed = capsys.readouterr().out
    assert "1 of 7 Use records selected, 1 changed" in printed
    assert "A16Only -> AllowA8: 1 (changed)" in printed
    with Artifact(output) as rewritten:
        assert rewritten.directory.uses[4]["activation_policy"] == "AllowA8"


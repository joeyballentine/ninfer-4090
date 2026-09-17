#!/usr/bin/env python3
"""Rewrite the activation permissions of an existing NInfer v3 artifact.

Run: python3 tools/set_activation_policy.py IN.ninfer OUT.ninfer --policy P --select PATTERN

A Use record's ``activation_policy`` states which activation precisions the
runtime may use for one mathematical input of one parameter.  It permits a
route; it never selects one.  The sm_89 FP8 prefill route in particular is
admitted only for a Q4/Q5 projection input whose Use carries ``AllowA8`` (or
``AllowA4``), and it additionally requires ``--prefill-a8 fp8`` at startup.

Recipes set this permission at conversion time.  This tool sets it on an
artifact that already exists: the directory is re-serialized with the new Use
records and the logical payload is copied byte for byte, so stored weights,
object layout and bindings are unchanged.  The output must be a new path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fnmatch import fnmatchcase
import os
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

# Running this file as a script puts tools/ on sys.path, not the repository root.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.artifact.file_io import IO_CHUNK_BYTES, Writeback
from tools.artifact.formats import QUANT_FORMATS
from tools.artifact.framing import HEADER, MAGIC, PART_MAGIC, PAYLOAD_ALIGNMENT
from tools.artifact.reader import Artifact
from tools.artifact.schema import (
    ACTIVATION_POLICIES,
    ArtifactError,
    binding_parts,
    integer,
)
from tools.artifact.writer import DEFAULT_MAX_FILE_BYTES, layout_directory

DEFAULT_POLICY = "A16Only"
POLICY_RANK = {"A16Only": 0, "AllowA8": 1, "AllowA4": 2}
TEXT_PROJECTION_PATTERN = "text/layers/*"
TEXT_PROJECTION_FORMATS = ("q4_g64_fp16", "q5_g64_fp16")


def parameter_formats(directory, objects) -> dict[str, tuple[str, ...]]:
    """Return the distinct stored formats behind every logical parameter."""

    result = {}
    for name, binding in directory.bindings.items():
        formats = {
            objects[object_id].format
            for object_id, _, _ in binding_parts(binding, objects, name)
        }
        result[name] = tuple(sorted(formats))
    return result


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(name == pattern or fnmatchcase(name, pattern) for pattern in patterns)


def select_uses(directory, formats, patterns, selected_formats) -> tuple[int, ...]:
    """Resolve the command-line selection to indices into the Use records."""

    if not patterns and not selected_formats:
        raise ValueError(
            "select records with --select, --formats or --text-projections"
        )
    patterns = patterns or ("*",)
    unknown = [item for item in selected_formats if item not in QUANT_FORMATS]
    if unknown:
        raise ValueError(
            f"--formats accepts integer groupwise formats "
            f"{', '.join(sorted(QUANT_FORMATS))}; got {', '.join(unknown)}"
        )
    matched = []
    rejected = []
    for index, use in enumerate(directory.uses):
        name = use["parameter"]
        if not _matches(name, patterns):
            continue
        stored = formats[name]
        if selected_formats and not set(stored) <= set(selected_formats):
            continue
        if not set(stored) <= set(QUANT_FORMATS):
            rejected.append(f"{name} ({'/'.join(stored)})")
            continue
        matched.append(index)
    if rejected:
        raise ValueError(
            "selection contains parameters that are not stored in an integer "
            f"groupwise format: {', '.join(sorted(set(rejected))[:8])}; restrict it "
            "with --formats or a narrower --select"
        )
    if not matched:
        raise ValueError("selection matched no Use record")
    return tuple(matched)


def rewritten_uses(
    directory, selection, policy, *, force
) -> tuple[list[dict], Counter]:
    """Return the new Use records and the old policy counts of the selection."""

    uses = [dict(use) for use in directory.uses]
    previous = Counter()
    lowered = []
    for index in selection:
        use = uses[index]
        old = use.get("activation_policy", DEFAULT_POLICY)
        previous[old] += 1
        if POLICY_RANK[policy] < POLICY_RANK[old]:
            lowered.append(f"{use['parameter']}@{use['input']} ({old})")
        use["activation_policy"] = policy
    if lowered and not force:
        raise ValueError(
            f"{len(lowered)} selected records would lose permission, for example "
            f"{', '.join(sorted(lowered)[:4])}; a stored NVFP4 weight needs its A4 "
            "permission to run its private activation route. Pass --force to lower them"
        )
    return uses, previous


def segmentation_limit(source: Artifact, override: int | None) -> int:
    """Reproduce the source's file segmentation unless a limit is given explicitly."""

    if override is not None:
        return integer(override, "maximum file bytes", positive=True)
    if len(source.directory.files) == 1:
        return DEFAULT_MAX_FILE_BYTES
    return source.payload_offset + source.directory.files[0].payload_bytes


def _write(fd: int, offset: int, data: bytes, writeback: Writeback) -> None:
    view = memoryview(data).cast("B")
    while view:
        count = os.pwrite(fd, view[:IO_CHUNK_BYTES], offset)
        if count <= 0:
            raise OSError(f"short write at file offset {offset}")
        view = view[count:]
        offset += count
        writeback.written(fd, count)


def copy_with_uses(
    source: Artifact,
    uses: list[dict],
    output: Path,
    max_file_bytes: int,
    *,
    segments: tuple[int, ...] | None = None,
) -> list[Path]:
    """Re-serialize the directory and stream the unchanged payload into new files."""

    description = source.directory.to_json()
    description["uses"] = uses
    description.pop("files")
    directory, encoded, entry_start = layout_directory(
        output.name,
        description,
        source.payload_bytes,
        max_file_bytes=max_file_bytes,
    )
    if segments is not None and segments != tuple(
        file.payload_bytes for file in directory.files
    ):
        raise ArtifactError(
            "the rewritten directory does not reproduce the source file "
            "segmentation; pass --max-file-bytes explicitly"
        )
    targets = [output] + [output.parent / file.path for file in directory.files[1:]]
    for target in targets:
        if target.exists():
            raise FileExistsError(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_id = uuid4().bytes
    writeback = Writeback()
    fds: list[int] = []
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        cursor = 0
        for index, file in enumerate(directory.files):
            fd, name = tempfile.mkstemp(
                prefix=f".{targets[index].name}.",
                suffix=".tmp",
                dir=targets[index].parent,
            )
            fds.append(fd)
            temporary.append(Path(name))
            start = entry_start if index == 0 else PAYLOAD_ALIGNMENT
            os.ftruncate(fd, integer(start + file.payload_bytes, "file bytes"))
            _write(
                fd,
                0,
                HEADER.pack(
                    MAGIC if index == 0 else PART_MAGIC,
                    len(encoded) if index == 0 else index,
                    artifact_id,
                ),
                writeback,
            )
            if index == 0:
                _write(fd, HEADER.size, encoded, writeback)
            offset = start
            for chunk in source.iter_range(cursor, file.payload_bytes):
                _write(fd, offset, chunk, writeback)
                offset += len(chunk)
            cursor += file.payload_bytes
        writeback.flush()
        while fds:
            os.close(fds.pop())
        for index in [*range(1, len(targets)), 0]:
            os.link(temporary[index], targets[index])
            published.append(targets[index])
    except BaseException:
        while fds:
            os.close(fds.pop())
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return targets


def verify(output: Path, uses: list[dict], source_directory) -> None:
    """Re-read the published entry through the Python schema validator."""

    with Artifact(output) as result:
        directory = result.directory
        if list(directory.uses) != uses:
            raise ArtifactError(f"{output}: Use records were not written as requested")
        if directory.objects != source_directory.objects:
            raise ArtifactError(f"{output}: object records changed")
        if (
            directory.bindings != source_directory.bindings
            or directory.components != source_directory.components
            or directory.metadata != source_directory.metadata
            or directory.provenance != source_directory.provenance
        ):
            raise ArtifactError(f"{output}: directory content other than Uses changed")
        for obj in directory.objects:
            result.object(obj.id)


def set_activation_policy(
    input_path: Path,
    output_path: Path,
    *,
    policy: str,
    patterns: tuple[str, ...] = (),
    formats: tuple[str, ...] = (),
    text_projections: bool = False,
    force: bool = False,
    max_file_bytes: int | None = None,
) -> dict:
    """Copy *input_path* to *output_path* with the selected Use records rewritten."""

    input_path, output_path = Path(input_path), Path(output_path)
    if policy not in ACTIVATION_POLICIES:
        raise ValueError(
            f"unknown activation policy {policy!r}; expected one of "
            f"{', '.join(sorted(ACTIVATION_POLICIES))}"
        )
    if output_path.exists() or input_path.resolve() == output_path.resolve():
        raise FileExistsError(f"the output must be a new path: {output_path}")
    patterns = tuple(patterns)
    formats = tuple(formats)
    if text_projections:
        patterns += (TEXT_PROJECTION_PATTERN,)
        formats = formats or TEXT_PROJECTION_FORMATS
    with Artifact(input_path) as source:
        stored = parameter_formats(source.directory, source.by_id)
        selection = select_uses(source.directory, stored, patterns, formats)
        uses, previous = rewritten_uses(
            source.directory, selection, policy, force=force
        )
        files = copy_with_uses(
            source,
            uses,
            output_path,
            segmentation_limit(source, max_file_bytes),
            segments=(
                None
                if max_file_bytes is not None
                else tuple(file.payload_bytes for file in source.directory.files)
            ),
        )
        try:
            verify(output_path, uses, source.directory)
        except BaseException:
            for path in files:
                path.unlink(missing_ok=True)
            raise
        summary = {
            "input": str(input_path),
            "output": str(output_path),
            "uses": len(source.directory.uses),
            "selected": len(selection),
            "changed": sum(
                count for name, count in previous.items() if name != policy
            ),
            "policy": policy,
            "previous": dict(sorted(previous.items())),
            "files": [str(path) for path in files],
            "payload_bytes": source.payload_bytes,
        }
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="existing v3 artifact entry")
    parser.add_argument("output", type=Path, help="new entry path to write")
    parser.add_argument(
        "--policy",
        required=True,
        help="activation permission to store: "
        + ", ".join(sorted(ACTIVATION_POLICIES)),
    )
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="PATTERN",
        help="parameter name or shell-style pattern such as"
        " 'text/layers/*/attention/query'; repeatable",
    )
    parser.add_argument(
        "--formats",
        default="",
        metavar="LIST",
        help="comma-separated stored formats to restrict the selection to, such as"
        " q4_g64_fp16,q5_g64_fp16",
    )
    parser.add_argument(
        "--text-projections",
        action="store_true",
        help=f"shorthand for --select {TEXT_PROJECTION_PATTERN} --formats"
        f" {','.join(TEXT_PROJECTION_FORMATS)}, the 24 GB recipe's set",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow records to lose permission, such as AllowA4 to AllowA8",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        help="file size limit for the output;"
        " the source segmentation is kept by default",
    )
    args = parser.parse_args(argv)
    summary = set_activation_policy(
        args.input,
        args.output,
        policy=args.policy,
        patterns=tuple(args.select),
        formats=tuple(item for item in args.formats.split(",") if item),
        text_projections=args.text_projections,
        force=args.force,
        max_file_bytes=args.max_file_bytes,
    )
    print(f"{summary['input']} -> {summary['output']}")
    print(
        f"  {summary['selected']} of {summary['uses']} Use records selected,"
        f" {summary['changed']} changed"
    )
    for name, count in summary["previous"].items():
        mark = "unchanged" if name == summary["policy"] else "changed"
        print(f"  {name} -> {summary['policy']}: {count} ({mark})")
    print(
        f"  payload {summary['payload_bytes']} bytes copied unchanged,"
        f" {len(summary['files'])} output files"
    )


if __name__ == "__main__":
    main()

"""Precomputed calibration Hessians for the GPTQ conversion method.

NInfer has no Python model-inference route, so the activation statistics are
produced outside the converter by ``tools/calibrate_hessians.py`` and read back
here.  One Hessian ``H = sum x x^T`` belongs to one mathematical input of the
logical model, not to one parameter: projections that consume the same input,
such as attention query and key, share it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HESSIAN_FILE = "hessians.safetensors"
MANIFEST_FILE = "manifest.json"


class HessianStore:
    """Read per-input Hessians from a calibration directory.

    The directory holds ``hessians.safetensors`` keyed by input name, or one
    ``<input name>.npy`` per input below it, or both.  An optional
    ``manifest.json`` with a ``{"sites": {input: key}}`` mapping lets several
    inputs share one stored matrix.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise ValueError(f"calibration directory does not exist: {self.path}")
        self.sites: dict[str, str] = {}
        manifest = self.path / MANIFEST_FILE
        if manifest.exists():
            record = json.loads(manifest.read_text(encoding="utf-8"))
            sites = record.get("sites", {})
            if not isinstance(sites, dict):
                raise ValueError(f"{manifest}: sites must map input names to keys")
            self.sites = {str(key): str(value) for key, value in sites.items()}
        self.file = self.path / HESSIAN_FILE
        self.keys: frozenset[str] = frozenset()
        if self.file.exists():
            from safetensors import safe_open

            with safe_open(self.file, framework="pt") as stream:
                self.keys = frozenset(stream.keys())

    def key(self, site: str) -> str:
        return self.sites.get(site, site)

    def hessian(self, site: str, columns: int) -> torch.Tensor:
        """Return the ``[columns,columns]`` float32 Hessian recorded for *site*."""

        key = self.key(site)
        if key in self.keys:
            from safetensors import safe_open

            with safe_open(self.file, framework="pt") as stream:
                values = stream.get_tensor(key)
        else:
            array = self.path / (key + ".npy")
            if not array.exists():
                raise ValueError(
                    f"{site}: no calibration Hessian in {self.path}"
                    f" (looked for key {key!r} and {array.name})"
                )
            values = torch.from_numpy(np.load(array).copy())
        values = values.to(torch.float32)
        if values.dim() != 2 or values.shape != (columns, columns):
            raise ValueError(
                f"{site}: calibration Hessian is {tuple(values.shape)},"
                f" expected [{columns},{columns}]"
            )
        return values

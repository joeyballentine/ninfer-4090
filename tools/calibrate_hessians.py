#!/usr/bin/env python3
"""Collect per-input calibration Hessians for the converter's GPTQ method.

Run: python3 tools/calibrate_hessians.py --model DIR --corpus FILE --out DIR

The engine has no Python inference route, so this script runs the Hugging Face
Qwen3.5 reference implementation once over a calibration corpus, accumulates
``H = sum x x^T`` for every linear input, and writes them under the logical
input names the converter uses.  ``tools/convert`` then reads that directory
with ``--calibration DIR``.  ``transformers`` and ``safetensors`` are required
here only; conversion itself does not need them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# HF module suffix -> logical mathematical input of the NInfer model, matching
# tools/convert/qwen3_5.py.  Projections sharing an input share one Hessian.
LAYER_SITES = {
    "self_attn.q_proj": "mixer_input",
    "self_attn.k_proj": "mixer_input",
    "self_attn.v_proj": "mixer_input",
    "self_attn.o_proj": "attention/gated_output",
    "linear_attn.in_proj_a": "mixer_input",
    "linear_attn.in_proj_b": "mixer_input",
    "linear_attn.in_proj_qkv": "mixer_input",
    "linear_attn.in_proj_z": "mixer_input",
    "linear_attn.out_proj": "gdn/gated_output",
    "mlp.gate_proj": "ffn_input",
    "mlp.up_proj": "ffn_input",
    "mlp.down_proj": "mlp/product",
}


def logical_site(name: str) -> str | None:
    """Map one Hugging Face linear module name to its logical input name."""

    parts = name.split(".")
    if "layers" not in parts:
        return None
    index = parts.index("layers")
    if index + 1 >= len(parts) or not parts[index + 1].isdigit():
        return None
    suffix = ".".join(parts[index + 2 :])
    site = LAYER_SITES.get(suffix)
    if site is None:
        return None
    component = "mtp" if parts[0] == "mtp" else "text"
    return f"{component}/layers/{parts[index + 1]}/{site}"


class Accumulator:
    """Running ``sum x x^T`` in float64 for one mathematical input."""

    def __init__(self, columns: int, device: torch.device):
        self.total = torch.zeros((columns, columns), dtype=torch.float64, device=device)
        self.samples = 0

    def add(self, values: torch.Tensor) -> None:
        flat = values.reshape(-1, values.shape[-1]).to(torch.float64)
        self.total += flat.transpose(0, 1) @ flat
        self.samples += flat.shape[0]


def read_corpus(path: Path, tokenizer, sequence: int, sequences: int):
    text = path.read_text(encoding="utf-8")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    usable = min(sequences, ids.numel() // sequence)
    if usable < 1:
        raise ValueError("calibration corpus is shorter than one sequence")
    return ids[: usable * sequence].reshape(usable, sequence)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="HF checkpoint dir")
    parser.add_argument("--corpus", type=Path, required=True, help="UTF-8 text file")
    parser.add_argument("--out", type=Path, required=True, help="calibration dir")
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float16"),
        help="stored Hessian element type",
    )
    args = parser.parse_args(argv)

    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    device = torch.device(args.device)
    accumulators: dict[str, Accumulator] = {}
    handles = []

    def hook(site: str):
        def record(module, inputs, output):
            values = inputs[0].detach()
            store = accumulators.get(site)
            if store is None:
                store = Accumulator(values.shape[-1], device)
                accumulators[site] = store
            store.add(values)

        return record

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        site = logical_site(name)
        if site is not None:
            handles.append(module.register_forward_hook(hook(site)))
    if not handles:
        raise ValueError("no known linear modules found; check the checkpoint")

    batches = read_corpus(args.corpus, tokenizer, args.sequence, args.sequences)
    with torch.inference_mode():
        for index in range(batches.shape[0]):
            model(batches[index : index + 1].to(device))
            print(f"[{index + 1}/{batches.shape[0]}] calibration sequence", flush=True)
    for handle in handles:
        handle.remove()

    stored = torch.float16 if args.dtype == "float16" else torch.float32
    tensors = {}
    report = {}
    for site, store in accumulators.items():
        mean = store.total / max(store.samples, 1)
        tensors[site] = mean.to(device="cpu", dtype=stored).contiguous()
        report[site] = {"columns": mean.shape[0], "tokens": store.samples}
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "hessians.safetensors"))
    manifest = {
        "model": str(args.model),
        "corpus": str(args.corpus),
        "sequence": args.sequence,
        "sequences": int(batches.shape[0]),
        "dtype": args.dtype,
        "inputs": report,
        "sites": {},
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(tensors)} Hessians to {args.out}", flush=True)


if __name__ == "__main__":
    main()

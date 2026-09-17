# Weight conversion

NInfer's converter creates `.ninfer` artifacts from local weights and a Python recipe. A recipe
can reuse an official conversion, change selected layers or projections, combine sources, or call
your own conversion method. The artifact contains the resulting configuration, encoded weights,
logical bindings and frontend resources.

Run the commands below from the repository root.

## Upgrade an existing v2 artifact

The offline upgrade tool supports the official Qwen3.6/3.8-27B groupwise-int and NVFP4 artifacts,
and Qwen3.6-35B-A3B groupwise-int. Update your checkout to the current `master` and
[rebuild NInfer](../README.md#quick-start), then run with Python 3.11:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
  models/qwen3_8_27b_nvfp4.ninfer \
  models/qwen3_8_27b_nvfp4.v3.ninfer
```

The output must use a new path. After upgrading, use it directly or rename it to replace the
original file. Stored weight values and formats are preserved. The upgrade also installs the
matching template from `tools/chat_templates/`. Published SHA-256 checksums apply only to
downloaded files.

## Start with an official recipe

Source-weight conversion requires a Python 3.11 environment with PyTorch and NumPy. It uses CUDA
by default; `--device cpu` selects CPU conversion. The input paths below are placeholders for your
local checkpoint directories.

For Qwen3.6-27B floating-point source weights:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --proposal \
  --name qwen3.6-27b \
  --out models/qwen3_6_27b.ninfer
```

`--components` defaults to `text`. Include only the optional components you want to distribute.
`--proposal` adds the indexed proposal head used by speculative decoding; it uses the repository's
token ranking and defaults to 131,072 rows. The ordinary full-vocabulary output head is retained.

The built-in recipes are ordinary Python functions in
[`official_recipes.py`](../tools/convert/official_recipes.py):

| Recipe | Main representation choices | Additional source |
|---|---|---|
| `qwen3_6_27b` | Q4/Q5 projections, Q6 vocabulary weights | None |
| `qwen3_8_27b` | Q4/Q5 projections, Q8 vocabulary weights | None |
| `qwen3_6_35b_a3b` | Q4 experts, Q5/Q6 expert down, Q8 shared/projection weights | None |
| `qwen3_6_27b_nvfp4` | Imported NVFP4, selected BF16 projections, Q8 vocabulary weights | `quantized` |
| `qwen3_8_27b_nvfp4` | Imported NVFP4/FP8, FP8 embedding generated from BF16 | `quantized` |

These names select conversion choices. Runtime execution is selected from the architecture,
configuration and actual bindings stored in the artifact. `--name` sets the public model name;
it does not select kernels.

## Maintained deployment recipes

`official_recipes.py` holds the conversions behind the published model cards and their checksums.
[`tools/convert/recipes/`](../tools/convert/recipes/) holds maintained recipes that target a
deployment instead, may depend on calibration data you produce, and are selected by file path:

| Recipe file | Target |
|---|---|
| [`qwen3_8_27b_24gb.py`](../tools/convert/recipes/qwen3_8_27b_24gb.py) | Qwen3.8-27B on one 24 GB card, tuned for coding accuracy |

`qwen3_8_27b_24gb` starts from `qwen3_8_27b` and changes only the Text representation and
its activation permissions:

- the 248,320 x 5,120 token embedding moves from Q8 to Q6. The embedding is a gather, not a matmul,
  so the narrower codes cost no arithmetic, and the artifact loses 357,780,480 bytes: 1.258 GiB at
  Q8 against 0.925 GiB at Q6. On a card where the official 19.03 GiB artifact leaves little room,
  that third of a gibibyte goes to KV pages and workspace. The official `qwen3_6_27b` conversion
  already stores both vocabulary matrices at Q6, and the runtime carries the Q6 embedding gather
  and the Q6 `n248320_k5120` linear shape;
- the output head stays at Q8, where full-vocabulary logit margins are worth the bytes;
- every Q4 and Q5 Text-layer projection selects its codes with `grouped_mse`, or with
  `grouped_gptq` when `--calibration` supplies Hessians;
- those same projections carry `activation_policy="AllowA8"`, which permits the sm_89 FP8 prefill
  route on an RTX 4090. It permits it only: A16 stays the default compute, and the runtime still
  needs [`--prefill-a8 fp8`](cli.md#common-options). Stored weights are identical either way.

Vision, MTP and DFlash2 keep the official assignments, and the vocabulary matrices keep the default
`A16Only`.

```bash
python3 tools/calibrate_hessians.py \
  --model /path/to/Qwen3.8-27B \
  --corpus /path/to/calibration.txt \
  --out out/qwen3_8_27b-hessians

python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe tools/convert/recipes/qwen3_8_27b_24gb.py \
  --calibration out/qwen3_8_27b-hessians \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_24gb.ninfer
```

Leave out `--calibration` to use `grouped_mse` everywhere, which needs no corpus and no
`transformers`. Conversion itself is unchanged otherwise; see
[Evaluate a quantization choice](#evaluate-a-quantization-choice) for comparing the results.

For a Qwen3.8-27B NVFP4/FP8 artifact with DFlash2:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_nvfp4.ninfer
```

MTP and Vision use the main source. DFlash and DFlash2 use the corresponding named source, supplied
as `--source dflash=PATH` or `--source dflash2=PATH`. An artifact may contain several optional
components; the Engine loads only the ones selected at startup, including at most one speculative
backend. Component availability and startup selection are independent.

## Change part of a recipe

Save the following as `my_recipe.py`:

```python
from tools.convert.official_recipes import qwen3_6_27b


def configure(model, recipe, sources):
    qwen3_6_27b(model, recipe, sources)
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method="grouped_absmax",
    )
```

Run it with the same source and component selection:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe my_recipe.py \
  --components text,vision,mtp \
  --proposal \
  --out models/my_qwen.ninfer
```

The default entry function is `configure`; `--recipe my_recipe.py:customize` selects another
function. Alternatively, use an official `--recipe` and put only the changes in an `--override`
file. Overrides run after the base recipe and optional proposal-head setup.

`model.parameters` maps logical names to their shape, source and mathematical inputs. To see the
available names for your selected components, a recipe can print them:

```python
for name, parameter in model.parameters.items():
    print(name, parameter.shape, parameter.inputs)
```

`recipe.assign` accepts one name, a list of names, or shell-style patterns such as
`text/layers/*/mlp/down`. A selector that matches nothing fails. Assignments run in Python order;
later assignments replace only the explicitly supplied choices. Changing `format` does not
automatically change `method`. Use `layout="auto"` to select the registered layout for the format
when overriding an earlier explicit layout choice.

The principal choices are:

| Argument | Meaning |
|---|---|
| `format` | Persistent numeric format |
| `layout` | Physical encoding; inferred from format unless explicitly set |
| `method` | Built-in method name or Python callable |
| `source` | Logical values or encoded rows from the selected source |
| `parameters` | JSON-serializable numerical parameters passed to the method |
| `rows=(begin, end)` | Override complete leading-axis rows in a half-open range |
| `activation_policy` | Permission for activation precision at the parameter's mathematical inputs |

For example, `rows=(0, 128)` can give the first 128 rows a different format. This creates multiple
physical parts when necessary. The container can represent that result; the intended Op must also
support consuming those parts. Current native projections generally require a contiguous parent
region, so arbitrary splits of one projection are not automatically executable.

## Formats, methods and activation precision

The converter currently writes these formats:

| Format | Built-in method for floating-point input | Import of already encoded input |
|---|---|---|
| `bf16`, `fp32`, `int32` | `cast_direct` | Direct words through the source reader |
| `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | `grouped_absmax`, `grouped_mse`, `grouped_gptq` | Supply a custom method/source if needed |
| `fp8_e4m3fn_row_bf16` | `fp8_row_maxabs` | `import_encoded` |
| `nvfp4` | Supply a custom quantizer | `import_encoded` |

`grouped_absmax` stores one FP16 scale per group and signed integer codes. `fp8_row_maxabs` first
rounds input values to BF16, then produces E4M3FN codes and one BF16 multiplier per row.
`import_encoded` preserves compatible code and scale words, including NVFP4's matrix weight divisor.
It does not dequantize and requantize them.

`grouped_mse` and `grouped_gptq` write exactly the words `grouped_absmax` writes: int8 codes in the
format's range and one canonical binary16 scale per group. Only the choice of those words changes,
so an artifact using them loads and runs through the same kernels.

### Choose integer codes

| Method | Scale per group | Code selection | Extra input |
|---|---|---|---|
| `grouped_absmax` | `absmax / qmax` | Independent round to nearest | None |
| `grouped_mse` | Search of `absmax * f / qmax` | Independent round to nearest | None |
| `grouped_gptq` | Absmax or the same search, on error-compensated weights | GPTQ order with error compensation | Calibration Hessians |

`grouped_mse` evaluates 21 clipping factors `f` descending from 1.00 to 0.80, rounds every candidate
scale through the same binary16 word `grouped_absmax` would store, and keeps the factor with the
smallest squared reconstruction error in that group. The factors descend from one and ties keep the
larger factor, so a group whose absmax scale is already optimal reproduces the `grouped_absmax`
result and no group ends up with a larger error. The search costs one extra pass per candidate and
needs no calibration data. On random Gaussian weights it removes about 22% of the Q4 squared error
and 14% of the Q5 squared error; on heavy-tailed weights about 9% and 6%. At Q8 the gain is
negligible, because 255 levels leave almost nothing to clip.

| Parameter | Default | Meaning |
|---|---|---|
| `candidates` | 21 | Clipping factors searched over `[0.80, 1.00]`; `1` is plain absmax |

`grouped_gptq` implements GPTQ (Frantar et al.): it factors the damped inverse Hessian of the
calibration activations, sweeps the input columns in blocks of 128 with lazy batch updates, and
pushes each column's rounding error onto the columns it has not quantized yet. Group scales come
from the error-compensated weights, with the absmax rule or with the `grouped_mse` search weighted
by `diag(H)`. With `H = I` the method reduces exactly to `grouped_absmax`, codes and scales
included; that identity is a test.

| Parameter | Default | Meaning |
|---|---|---|
| `calibration` | required | Calibration directory, normally the `--calibration` path |
| `mse` | `false` | Use the clipping search for the group scales |
| `candidates` | 21 | Clipping factors when `mse` is set |
| `block_size` | 128 | Columns per lazy-update block; must divide the padded K |
| `damping` | 0.01 | Added to `diag(H)` as a fraction of its mean |
| `act_order` | `false` | Quantize columns by descending `diag(H)` |

`act_order` reorders the sweep but not the storage: the row-split layout needs contiguous groups of
64 (or 32) original input channels, so act-order runs with static groups, computing the group scales
from the unpermuted weights and permuting the codes back. It usually helps least where groups are
already small, so it stays off by default.

A calibration Hessian belongs to one mathematical input, not to one parameter. Attention query and
key read the same input and share one matrix. A projection with several mathematical inputs, such as
`text/output_head`, has no single calibration site and is rejected by `grouped_gptq`.

## Calibrate for GPTQ

NInfer has no Python model-inference route, so activation statistics are collected outside the
converter. `tools/calibrate_hessians.py` runs the Hugging Face reference implementation of the
checkpoint over a calibration corpus, accumulates `H = sum x x^T` per linear input in FP64, and
writes them under the logical input names the converter uses. It needs `transformers` and
`safetensors`; conversion itself does not.

```bash
python3 tools/calibrate_hessians.py \
  --model /path/to/Qwen3.8-27B \
  --corpus /path/to/calibration.txt \
  --sequence 2048 --sequences 128 \
  --out out/qwen3_8_27b-hessians
```

Use a calibration corpus that matches the intended use. For a coding artifact, concatenate source
files in the languages you work in; 128 sequences of 2,048 tokens is a reasonable starting point.
The output directory contains `hessians.safetensors`, keyed by input name such as
`text/layers/17/ffn_input`, and a `manifest.json` recording the model, corpus, token counts per
input and an optional `sites` map that lets several inputs share one stored matrix. A directory of
`<input name>.npy` files is read as well, so statistics from another tool can be supplied without
using the script. `--dtype float16` halves the directory size; the converter upcasts to FP32.

Pass the directory to conversion with `--calibration DIR`. Recipes that declare a `calibration`
keyword parameter receive it; supplying it to a recipe that does not accept it is an error rather
than a silent omission.


The exact numeric and packing rules are in [numeric formats](maintainer/tensor-formats.md) and
[storage layouts](maintainer/storage-layouts.md). Source format names alone do not establish
compatibility: scale direction, granularity, code meaning and axis order must also match.

Activation permissions are independent of the stored weight format:

| Policy | Permitted activation paths |
|---|---|
| `A16Only` | A16 |
| `AllowA8` | A16, A8 |
| `AllowA4` | A16, A8, A4 |

They permit choices; they do not force a kernel to use the lowest precision. A fused operation that
shares one activation across several projections must respect the intersection of their
permissions. `recipe.use(parameter, input_name, ...)` can set one mathematical input independently;
the names are available in `parameter.inputs`.

An NVFP4 A4 input requires a positive finite activation divisor. `import_encoded` obtains it from
the selected source, or a recipe supplies it through
`recipe.use(..., auxiliaries={"activation_input_divisor": value})`. Shared weights retain separate
Use records; sharing weights does not share calibration implicitly.

## Permit 8-bit activations on an existing artifact

Activation permissions live in the artifact's Use records, so an artifact converted before a route
existed keeps `A16Only` until those records are rewritten.
`tools/set_activation_policy.py` rewrites them instead of reconverting: it re-serializes the
directory, copies the payload byte for byte into a new file, and leaves stored weights, object
records and bindings unchanged.

The official `qwen3_8_27b` artifact stores its Text-layer projections at Q4/Q5 under `A16Only`,
which is the one condition of the [Ada FP8 prefill route](maintainer/ada-fp8-prefill.md) that no
startup flag can satisfy:

```bash
python3 tools/set_activation_policy.py \
  models/qwen3_8_27b.ninfer \
  models/qwen3_8_27b.a8.ninfer \
  --policy AllowA8 \
  --text-projections
```

`--text-projections` is the shorthand for `--select 'text/layers/*' --formats
q4_g64_fp16,q5_g64_fp16`, the set `qwen3_8_27b_24gb` permits at conversion time. Select another set
with repeated `--select PATTERN`, with `--formats LIST`, or with both; every selected record must
belong to a parameter stored in an integer groupwise format. The tool prints how many records it
changed and which policies they carried.

The output must be a new path and keeps the source's file segmentation. Lowering a permission
requires `--force`, because a stored NVFP4 weight needs its `AllowA4` record to run its private
activation route. The conversion report next to the source describes the original conversion and is
not copied, and published SHA-256 checksums apply to the downloaded file only.

## Fused parents and logical projections

The Qwen adapter exposes Q, K, gate and V separately, even when they came from fused source tensors.
It also supplies finite packing groups for attention, GDN, MLP and MoE. Compatible selections are
packed into a shared parent automatically by the built-in methods.

For the Dense groupwise recipe, attention Q/K form one Q4 parent and gate/V form one Q5 parent.
The native fused Op receives two weights. To use the supported single-parent FP8 form, an override
can assign all four projections together:

```python
def configure(model, recipe, sources):
    for layer, kind in enumerate(model.config["layer_types"]):
        if kind != "full_attention":
            continue
        prefix = f"text/layers/{layer}/attention/"
        recipe.assign(
            [prefix + role for role in ("query", "key", "gate", "value")],
            format="fp8_e4m3fn_row_bf16",
            layout="auto",
            method="fp8_row_maxabs",
            activation_policy="AllowA8",
        )
```

Use this file as `--override` after `qwen3_6_27b` or `qwen3_8_27b`. It leaves the other parameters
under that recipe. The adapter handles source Q/gate row order; the override works with logical
projections. Changing their representation can change numerical results and the physical kernels.

For explicit organization, `recipe.group([names...])` concatenates compatible selections in the
given order, `recipe.separate(names)` disables automatic grouping for those parameters, and
`recipe.share(parameter, target)` binds equal-shaped parameters to the same physical data.
Explicit groups must be disjoint and use unsplit selections with matching format, layout, method
and method parameters. NVFP4 parents also require a common weight divisor. Automatic grouping is
limited to built-in methods; custom methods can request explicit groups.

Grouping chooses storage. Model execution code chooses the supported fused implementation. The
loader uploads the stored representation, without repacking an inconvenient arrangement.

## Read another source

`--model` supplies the main config, default resources and the source named `base`. Add other
Safetensors sources with repeated `--source NAME=PATH`; they are opened when used. Single-file
Safetensors and indexed shards are supported. Additional tensor-only sources can omit model config;
sources carrying config are checked against the relevant model geometry.

To replace a logical parameter from another compatible checkpoint in a recipe:

```python
name = "text/layers/0/mlp/down"
recipe.assign(
    name,
    source=model.source(name, sources["alternate"]),
    format="q6_g64_fp16",
    method="grouped_absmax",
)
```

Supply `--source alternate=/path/to/alternate-checkpoint`. `model.source` applies the architecture's
source-name and axis mapping, including Q/gate extraction. The built-in compressed-tensors reader
understands the implemented per-row FP8 and NVFP4 code/scale conventions. It can expose decoded
values for another quantizer or encoded rows for exact import.

For another file format or quantization convention, provide a `LogicalSource`. Its value reader
accepts flat C-order element bounds and returns exactly that range. For example, a recipe can read
a logical matrix from a NumPy file stored beside the recipe:

```python
from pathlib import Path
import numpy as np
import torch
from tools.convert.sources.logical import LogicalSource


def configure(model, recipe, sources):
    name = "text/layers/0/mlp/down"
    path = Path(__file__).with_name("mlp-down.npy")
    data = np.load(path, mmap_mode="r")
    if tuple(data.shape) != model.parameters[name].shape or not data.flags.c_contiguous:
        raise ValueError("mlp-down.npy must have the logical shape and C-order storage")

    def read_values(begin, end):
        values = data.reshape(-1)[begin:end].astype(np.float32, copy=True)
        return torch.from_numpy(values)

    source = LogicalSource(tuple(data.shape), str(path), read_values)
    recipe.assign(name, source=source, format="q6_g64_fp16", method="grouped_absmax")
```

Use this as an override. The NumPy file must already follow the logical row/column order. For an
unfamiliar quantized source, its reader performs the corresponding decoding before returning
values. To preserve existing compatible encoded words, also provide `read_encoded` returning
`EncodedRows`, and the format's required divisor accessors. Their definitions are in
[`sources/logical.py`](../tools/convert/sources/logical.py).

## Write a conversion method

A method receives a `PrepareRequest` and returns `request.job(produce=...)`. Preparation validates
the target and determines auxiliary values. The `produce` function reads bounded source regions
and writes values or codes/scales through `TensorOutput`; the writer owns placement and file I/O.

This example adds explicit clipping before the existing grouped quantizer. It demonstrates the
method interface; the clipping threshold is a numerical choice made by the recipe author.

```python
import math
import torch
from tools.artifact.formats import QuantFormat, get_format
from tools.convert.quantization.groupwise import quantize_matrix


def clipped_grouped(request):
    if len(request.target.shape) != 2 or not isinstance(
        get_format(request.target.format), QuantFormat
    ):
        raise ValueError("clipped_grouped requires a grouped-integer matrix")
    limit = float(request.parameters["clip"])
    if not math.isfinite(limit) or limit <= 0 or request.rows_per_chunk <= 0:
        raise ValueError("clip and rows_per_chunk must be positive")
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("source contains non-finite values")
            encoded = quantize_matrix(
                values.clamp(-limit, limit),
                request.target.format,
                device=request.device,
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def configure(model, recipe, sources):
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method=clipped_grouped,
        parameters={"clip": 1.0},
    )
```

Use this as an override. `request.values` traverses the prepared logical inputs in parent order,
including explicit groups. `output.write_codes` performs the registered packing and validates
codes/scales; the method should not duplicate that byte-layout logic. Direct output uses
`output.write_values`. Keep source blocks and temporary device tensors bounded to the method's
working set. `--rows-per-chunk` defaults to 512; custom methods own how they use it.

## Evaluate a quantization choice

Code selection changes numerical results, not the artifact's execution route, so compare artifacts
with [`ninfer-perplexity`](perplexity.md) on the fixed corpus. Build the alternatives from the same
checkpoint, with the same components and the same recipe except for the method being measured:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_absmax.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --kv-dtype int8 \
  --output profiles/perplexity/absmax
```

Repeat it for the `grouped_mse` and `grouped_gptq` artifacts, keeping the corpus selection, the
default 4,096-token context and 2,048-token stride, and `--kv-dtype int8` fixed, because
`int8` is the KV representation a 24 GB card actually has room for. Each run writes an unrounded
`report.json` with per-stream, per-domain and overall values.

The corpus reports four domains. For programming use, the number that decides the comparison is the
NInfer C++/CUDA code domain; the English and Chinese reference domains show whether a calibration
corpus has skewed the model away from general text. Expect the ordering
`grouped_absmax` > `grouped_mse` > `grouped_gptq` in perplexity, with the largest gaps at Q4, and
treat a code-domain regression against `grouped_absmax` as a calibration problem rather than a
method problem. The Q6 embedding is a separate variable: measure it on its own before combining it
with a code-selection change.

## Resources, files and inspection

Text includes `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and
`generation_config.json`. Vision adds its image and video processor configs. Resources come from
`--model`; `--resource ROLE=PATH` replaces a selected resource:

```text
--resource chat_template.jinja=/path/to/chat_template.jinja
```

The official conversion examples select these maintained templates:

| Model | Template | Defaults |
|---|---|---|
| Qwen3.6 Dense/MoE | [qwen3_6.jinja](../tools/chat_templates/qwen3_6.jinja) | thinking on; closed-turn reasoning omitted |
| Qwen3.8 | [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja) | thinking on; effort `xhigh`; closed-turn reasoning retained |

Use your own Jinja file to change the artifact's default template. A startup
[`--chat-template FILE`](cli.md#text-input) overrides the stored template.
`generation_config.json` is preserved; sampling presets remain determined by the architecture
and explicit application/request settings.

The default maximum file size is 32,000,000,000 bytes, including framing. Smaller artifacts remain
one file. Larger artifacts use an entry such as `models/my_qwen.ninfer` plus
`my_qwen.ninfer.part-0001`, `my_qwen.ninfer.part-0002`, and so on in the same directory. Pass only the
entry path to NInfer and keep all its recorded parts together. `--max-file-bytes` changes the limit.

Conversion writes `models/my_qwen.ninfer.conversion.json` alongside the artifact, recording sources,
methods, formats, component configs, files and timing. Existing output files are not overwritten.
The report is useful for reproducing a recipe; the Engine reads the artifact itself.

```bash
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
python3 -m tools.artifact.inspect models/my_qwen.ninfer --json
```

Inspection reads directory facts without running inference. Conversion rejects missing logical
coverage, invalid source geometry, unsupported encodings and invalid method output. Actual Op
support is checked by consumers during preparation, resource queries, warmup or execution. A
valid file may need additional Op support before its chosen combination can run. Exercise the
phases and optional components you intend to use through the normal [CLI](cli.md) or
[serving](serving.md) route.

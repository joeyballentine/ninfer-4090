# RTX 4090 hardware validation

Backlog of work on branch `claude/v3-sm89-port` that needs a real Ada device. Every sm_89 change on
this branch was compile-checked without a GPU. Nothing below has run. This file is the single list.
The per-area references keep the contracts.

Work the sections in order. Section 2 gates section 5, and section 4 produces the artifact that
sections 5 and 6 measure.

## 1. Environment

| Item | Required |
|---|---|
| Device | RTX 4090, 24 GB, sm_89 |
| CUDA | 13.3 or later |
| Python | 3.11 with PyTorch and NumPy; `transformers` and `safetensors` for calibration |
| Free disk | about 80 GB: 54 GB BF16 checkpoint, 19 GB artifact, the rest for Hessians and reports |

Build with the `dev` preset. It turns on `BUILD_TESTING` and `NINFER_BUILD_BENCHMARKS`, which the
`release` preset leaves off:

```bash
cmake --preset dev
cmake --build --preset dev --parallel
```

Pass `-DCMAKE_CUDA_ARCHITECTURES=89`. `CMakeLists.txt` defaults to `120a` and accepts only those two.

### Windows

The branch was written on Linux, so the MSVC path needed work. What it costs:

* Configure and build from a Developer Command Prompt, or call `vcvars64.bat` first. The Ninja
  generator records the compiler path but not `INCLUDE`, so `cl.exe` otherwise fails on `<memory>`.
* `curl` and FFmpeg come from vcpkg, which supplies `pkgconf` and the `.pc` files that
  `cmake/Dependencies.cmake` reads. A repo-local `ffmpeg/` directory takes precedence if present.
* `ninfer_artifact_materialization_test` loses its CUDA fault injection. The interception uses the
  GNU linker's `--wrap`, which `link.exe` has no equivalent for, so MSVC builds drop
  `materialization_cuda_errors.cpp` and define `NINFER_NO_LINK_WRAP`. The file-roundtrip modes the
  writer interop test drives are unaffected. Run that coverage on Linux.

## 2. Execute the GPU tests

First run on an RTX 4090, 2026-09-20, CUDA 13.3, driver 616.56, Windows/MSVC. Each entry names the
commit that added the code.

| Test target | Covers | Commit | Result |
|---|---|---|---|
| `ninfer_linear_q4_a8_prefill_test` | Q4 FP8 prefill route correctness | `e0c93f48`, `ae701a2d` | pass |
| `ninfer_linear_q5_a8_prefill_test` | Q5 FP8 prefill route correctness | `e0c93f48`, `ae701a2d` | pass |
| `ninfer_kv_cache_append_test` | rk8v4 and rk2v4-e8 storage numerics, D256 key rotation | `775147e6`, `1d02d180`, `5a09feca` | pass |
| `ninfer_sampling_test` | device token-id licensing bitmask | `9e8f978e` | pass |
| `ninfer_mtp_round_test` | MTP draft window at K=15 | `cc3f5b02` | pass |
| `ninfer_softmax_attention_test` | the same codecs on the read side | `775147e6`, `1d02d180`, `5a09feca` | crash fixed; 4 marginal failures, see below |

Both prefill tests returned 0, not the 77 skip code, so the FP8 prefill routes executed. The
`--nvfp4-only` and `--k8v4-only` variants skip on sm_89 as intended: those KV kernels are SM120.

### Fixed: the batch suites ran SM120 storages on sm_89

The binary exited `0xC0000409` (`__fastfail`) with no output. Relinking with a 64 MB stack did not
change it, so it was corruption, not stack exhaustion. `run_batch_cases` and `run_dflash2_cases`
each looped over all five storages including `Nvfp4Group16` and `Fp8KeyNvfp4Value`, whose kernels
are SM120 and link stubs on sm_89. Every other suite in the file guards them, and `--nvfp4-only`
and `--k8v4-only` skip on sm_89; the two batch loops did not. Both now take `kBatchStorages`,
which drops those two under `NINFER_SM89`.

The crash masked the whole file. With it fixed, `verify_workspace_capacity_contract`, both
geometries, `run_rotated_cases`, `run_fp8_cases`, the packed and context suites all pass.

### Open, low severity: INT8 gross bound on high-magnitude outputs

Four `causal batch d256-h24-kv4 int8-g64 phase=0` cases still exceed `kAttentionInt8Criterion`.
With `NINFER_OP_REPORT_STATS=1`:

| Case | `rel_l2` ratio | `gross_ratio` | `max_reference` |
|---|---|---|---|
| W=5 B=8 | 0.614 | 1.017 | 0.99994 |
| W=8 B=1 | 0.644 | 1.043 | 0.99994 |
| W=12 B=8 | 0.647 | 1.005 | 0.99994 |
| W=8 B=6 | 0.663 | 1.035 | 0.99994 |

Read those two columns together. `relative_l2` passes at 61 to 66 percent of its limit in every
case, so the aggregate output is sound. Only the single worst element exceeds, by 0.5 to 4.3
percent. `gross_error_limit` is `1.1e-3 + 3.0e-3 * max_reference`, which is 4.0998e-3 here, and
int8 dequantization alone contributes up to `amax/254`, 3.94e-3, before BF16 output rounding. The
bound sits essentially at the storage's own theoretical worst case.

Every failing case has `max_reference` at 0.99994. The `d256-h24-kv4 int8-g64` cases that pass
carry `max_reference` between 0.021 and 0.175 and `gross_ratio` at 0.08 to 0.40. The bound is only
reachable when an attention row concentrates on few keys and the output approaches 1.0, which the
DFlash2 batch cases construct and the A1/A3 cases do not.

This is not the D256 rotation. The test's own control comparison writes the same keys through the
standalone `kv_cache_append` and compares the cache byte for byte, and that check passes, so
`5a09feca`'s claim that the fused write matches the standalone append holds on hardware. Tracing
the factorization confirms it: `hadamard_d64_fragment_inplace` plus
`normalized_hadamard_d256_group_value_from_h64` reproduce `normalized_hadamard_d256_inplace`
operand for operand and stage for stage across all four groups, including the single `2^-4`.

Deciding this needs a number the sm_89 build cannot produce: the same `gross_ratio` on SM120,
where this criterion was set. If 120a also lands near 1.0 the bound is simply tight for int8 V at
unit output and should be widened with the derivation recorded. If 120a lands well below, the
sm_89 accumulation order is worth a look. Do not widen it on sm_89 evidence alone.

Both KV targets share the rotated-family oracle in `tests/ops/kv_rotated_codec.h`. Each also
registers `--nvfp4-only` and `--k8v4-only` variants as separate ctest entries; on sm_89 those link
stubs and skip.

The two prefill tests skip themselves off sm_89 builds, off non-Ada devices, and with no GPU. A
silent skip reads as a pass in the ctest summary, so confirm each one ran rather than trusting the
exit code.

`ninfer_linear_bench` needs a local `cuda_profiler_api.h` stub that this CUDA package does not ship
(`ae701a2d`). Fix that before section 5.

Not covered by any test: end-to-end generation under `--structured-output-language` (`69b70c28`).
Exercise it through the CLI.

## 3. Download the source weights

The reconversion needs the BF16 checkpoint, not the published artifact. Revisions are pinned in
[`model-cards/Qwen3.8-27B-NInfer/artifact-manifest.json`](../../model-cards/Qwen3.8-27B-NInfer/artifact-manifest.json).

| Source | Repository | Revision |
|---|---|---|
| Base | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |

## 4. Reconvert for 24 GB

[`tools/convert/recipes/qwen3_8_27b_24gb.py`](../../tools/convert/recipes/qwen3_8_27b_24gb.py)
starts from the official `qwen3_8_27b` recipe and changes two things. The token embedding drops from
Q8 to Q6, which returns 357,780,480 bytes to KV pages and workspace. Every Q4/Q5 Text projection
carries `activation_policy="AllowA8"`, the one precondition of the FP8 prefill route that no startup
flag can satisfy. Code selection uses `grouped_mse`, or `grouped_gptq` when `--calibration` supplies
Hessians.

Convert with `grouped_mse` first. Leaving out `--calibration` needs no corpus and no `transformers`,
and it produces the artifact every measurement in sections 2 and 5 depends on. GPTQ is the optional
upgrade and the third artifact of the section 6 comparison, so it comes after:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe tools/convert/recipes/qwen3_8_27b_24gb.py \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_24gb.ninfer
```

Full procedure, including how to change part of a recipe, is in
[weight conversion](../weight-conversion.md).

For the later GPTQ artifact, collect Hessians with `tools/calibrate_hessians.py` and pass
`--calibration DIR`. Two constraints on that run:

* The script loads the reference implementation with `device_map=args.device`, default `cuda`.
  A 27B BF16 checkpoint is about 55 GB and does not fit on a 24 GB card. Pass `--device auto` so
  accelerate shards it across GPU and host.
* Pick the corpus against section 6, not only against intended use. See the leakage note there.

## 5. Measure the FP8 prefill route

Op level, both directions on the same geometries, extents, inputs, cache and timing conditions.
The suite needs `--policy a8`; without it every point resolves to the A16 route and the two CSVs
come out identical:

```bash
./build/bench/ninfer_linear_bench \
  --suite ada_fp8_prefill --policy a8 --prefill-a8 off --csv-out profiles/bench/prefill_bf16.csv
./build/bench/ninfer_linear_bench \
  --suite ada_fp8_prefill --policy a8 --prefill-a8 fp8 --csv-out profiles/bench/prefill_a8.csv
```

### Measured, 2026-09-20

End to end on the section 4 artifact, 7,680 prompt tokens, `--kv-dtype int8`, `--max-new 1`:
2.03k tok/s at `--prefill-a8 off` against 2.17k at `fp8`, a gain of **1.07x**. The Op suite totals
1.064x, so the product route is not losing anything the kernels deliver. Neither reaches the 1.3x
bar.

Median microseconds per point, `off` / `fp8`:

| Geometry | N | T=128 | T=512 | T=2048 |
|---|---|---|---|---|
| `q4_attn_out` | 4096 | 0.83x | 1.48x | 1.45x |
| `q4_residual` | 5120 | 1.06x | 1.30x | 1.47x |
| `q4_qkv` | 6144 | 1.08x | 1.33x | 1.47x |
| `q4_gdn_in` | 7168 | 1.37x | 2.06x | 1.35x |
| `q4_gate_up` | 34816 | 1.08x | **0.86x** | **0.88x** |
| `q5_qkv` | 6144 | 0.96x | 1.01x | 1.14x |
| `q5_gdn_in` | 7168 | 0.97x | 1.11x | 1.14x |
| `q5_residual` | 5120 | 1.04x | 1.02x | 1.13x |
| `q5_down` | 17408 | 1.03x | 0.99x | 1.10x |

Two results decide where the remaining work is.

**`q4_gate_up` regresses.** At N=34816 the FP8 route is slower than BF16 at T=512 and T=2048,
5,583 us against 6,373 us at T=2048. It reaches 114.6 useful TFLOPs where `q4_gdn_in` reaches
187.0 on the same T. This is the largest geometry in the model and the one that dominates prefill
FLOPs, and the single tile is tuned for the others.

This corrects the plan in [Ada FP8 prefill](ada-fp8-prefill.md) section 8, which calls wiring
LinearSwiGLU "the larger remaining win". The gate/up bank is exactly the geometry that regresses,
so routing the fused Op to this family would make prefill slower, not faster. Tile exploration is
the prerequisite, not the follow-up. Reverse the order of those two items.

**Q5 gains little.** Every Q5 geometry lands between 1.10x and 1.14x at T=2048 where the healthy
Q4 ones reach 1.45x, and Q5 covers `q5_down` at K=17408, the second-largest bank. Worth a separate
look at the Q5 codec's decode path.

Quality, on the code domain, using the artifact from section 4:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_24gb.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --kv-dtype fp8 --prefill-a8 off --output profiles/ppl/prefill_off
./build/apps/ninfer-perplexity models/qwen3_8_27b_24gb.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --kv-dtype fp8 --prefill-a8 fp8 --output profiles/ppl/prefill_fp8
```

Repeat the `off` run to establish the evaluator's noise band, and report it alongside the delta.

Go / no-go, from [Ada FP8 prefill](ada-fp8-prefill.md) section 7. Change the
`EngineOptions::prefill_a8` default to `Fp8`, and the CLI default with it, only when both hold:

* the code-domain perplexity delta between `fp8` and `off` sits inside the noise band; and
* prefill throughput at pp2048 improves by at least 1.3x end to end through the product route, not
  only in the Op benchmark.

Below 1.3x with clean perplexity, the route stays opt-in and the gap is a tile or pipeline question.
Outside the noise band, revisit activation scale granularity first: the current scheme is one scale
per token row, and the weight side has no error to spend.

**Verdict: the route stays opt-in.** 1.07x fails the throughput condition, so `prefill_a8` keeps
its `Off` default and no code changes. The gap is a tile question, which is where the measured
table above points. Perplexity is still worth running for correctness confidence, but it cannot
change this decision.

## 6. Compare quantization methods

Section 4 changes two variables at once. Build the alternatives from the same checkpoint with the
same components, varying one thing at a time. Measure the Q6 embedding on its own before combining
it with a code-selection change.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_absmax.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --kv-dtype int8 \
  --output profiles/perplexity/absmax
```

Repeat for the `grouped_mse` and `grouped_gptq` artifacts. Hold the corpus selection, the default
4,096-token context, the 2,048-token stride and `--kv-dtype int8` fixed; `int8` is the KV
representation a 24 GB card has room for. Expect the ordering `grouped_absmax` > `grouped_mse` >
`grouped_gptq` in perplexity, with the largest gaps at Q4. A code-domain regression against
`grouped_absmax` is a calibration problem, not a method problem.

The English and Chinese reference domains show whether the calibration corpus skewed the model away
from general text.

A GPTQ calibration corpus must not overlap the `ninfer_code` streams.
`eval/corpora/perplexity-1m/provenance` pins them to `Neroued/ninfer` at revision `f8fc0f50`, roots
`include/ninfer/`, `src/` and `apps/`, about 960 KB over four streams. Calibrating on this
repository fits the Hessians on the text the comparison then scores, and `grouped_gptq` wins on
memorization. Near-duplicate kernel files make an exclusion list insufficient. Draw the C++/CUDA
part of the corpus from a codebase that is not in the streams.

Calibrating only on other languages has the opposite cost: the `ninfer_code` domain goes
out-of-distribution and a small regression against `grouped_mse` stops meaning anything. A corpus
weighted toward the languages the artifact serves, with a C++/CUDA remainder disjoint from the
streams, keeps both the fit and the metric usable.

## 7. Publish numbers

[README](../../README.md) currently states that nothing on the sm_89 layer beyond the original
fork's measurements has been timed on hardware, and that its RTX 4090 figures are the arithmetic of
the memory budget. Replace them once sections 5 and 6 produce measurements. Follow the publication
rules in [performance methodology](../performance/methodology.md).

## 8. Not scheduled

Work that a device unblocks but that this backlog does not cover:

* **Tile exploration, first.** One tile is instantiated, chosen to match the BF16 prefill route
  rather than measured. Section 5 shows it is wrong for N=34816 and weak for every Q5 geometry.
  Apply the candidate sweep in [op development](op-development.md) section 7.1: instantiate the
  small overlapping candidate set, decide, delete the losers.
* **LinearSwiGLU and LinearAdd on the FP8 family, after the tile.** The kernel body already carries
  the `Epilogue`, `Output`, `RowPolicy` and `PairRows` hooks, but neither fused Op's plan is wired
  to it, so both keep their A16 routes. Wiring LinearSwiGLU needs a `Q4LinearSwiGluScheduleId`
  entry, a workspace-capacity branch and a launcher shaped like
  `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu`. Do not wire it before the tile work: the
  gate/up bank is the geometry that currently regresses, so routing the fused Op to this family
  would cost prefill throughput.
* **DirectStorage restore.** `ContextDiskTransferPort` declares the seam under
  `_WIN32 && NINFER_DIRECTSTORAGE` and nothing implements it (`cf247e5a`). It needs a Windows SDK
  and a D3D12 device. An implementation must wait on the shared D3D12 fence from a CPU thread before
  releasing any staging resource; releasing COM resources while the DirectStorage queue still has
  requests in flight crashes inside the NVIDIA D3D12 driver.
* **Context cache restore timing.** [CLI](../cli.md) and [serving](../serving.md) quote the donor
  fork's NVMe figure, 367 ms to restore a 152k-token checkpoint against 162 s of cold prefill. This
  implementation's portable path has not been measured.

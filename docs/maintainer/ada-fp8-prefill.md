# Ada FP8 prefill for the groupwise row-split Linear formats

Family reference for the `q4_a8_prefill` / `q5_a8_prefill` routes. It specializes
[Op development rules](op-development.md) for one private implementation family; it does not change
the `linear()` contract, which stays BF16 in and BF16 out at every policy.

## 1. Why the route exists

On an RTX 4090 the 27B groupwise artifact decodes at the memory roofline, but prefill is tensor-core
bound: about 2,100 tok/s at pp2048, roughly 70% of Ada's BF16 dense rate with FP32 accumulate
(165 TFLOPS). Ada's E4M3 MMA with FP32 accumulate runs at twice the BF16 rate, so the contraction
has a factor of two of headroom that a BF16 operand cannot reach.

The weights are Q4/Q5 groupwise integers with one binary16 scale per 64 values along K
([tensor formats](tensor-formats.md), [storage layouts](storage-layouts.md), `row_split_k128_v1`).
The interesting property is that the *weight* side can move to FP8 without losing anything, so the
only new error is on the activation side, where the profile is already qualified upstream.

## 2. Weight side: the decode is exact

A Q4 lane is a 4-bit two's-complement integer in `[-8,7]`; a Q5 lane is a 5-bit one in `[-16,15]`.
E4M3 has a 4-bit exponent and 3 mantissa bits, so it represents every integer of magnitude up to 16
exactly, and 16 is far below its 448 finite maximum. The decode is therefore a recode of an integer
into another exact representation, not a quantization:

| `|v|` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E4M3 | `0x38` | `0x40` | `0x44` | `0x48` | `0x4A` | `0x4C` | `0x4E` | `0x50` | `0x51` | `0x52` | `0x53` | `0x54` | `0x55` | `0x56` | `0x57` | `0x58` |

with the sign in bit 7 and zero as `0x00`. `q4_a8_prefill_codec.cuh` and `q5_a8_prefill_codec.cuh`
hold exactly this table, packed into 64-bit words and indexed by the code, so the decode is a shift
and a mask rather than a float round-trip.

The group scale is deliberately *not* folded into the E4M3 byte: a scaled code is a general real
number and would be quantized. It is applied in FP32 after the MMA instead (section 4), so the
weight-side error of the route is one FP32 multiply, about `2^-24` relative — two orders of
magnitude below the BF16 output rounding the A16 routes already accept. The A8 route adds no weight
error relative to the A16 route.

## 3. Activation side: the error model

The activation is BF16 and cannot be recoded exactly. It is materialized once per call, per token
row `m`:

```text
s[m]    = absmax(x[m, :]) / 448
xq[m,k] = round_to_nearest_e4m3(x[m,k] / s[m])
```

This is the row-scaled FP8 family's quantizer verbatim
(`src/ops/linear/fp8/fp8_a8.cu`), reused rather than forked: the code plane is `[T,K]` E4M3 bytes
and the scale plane is one FP32 per token, which is the layout this GEMM wants for its `m16` MMA
operand. The workspace helpers are shared with it too (`A8PrefillWorkspace` is `Fp8A8Workspace`).

Error model, which is also the derivation the tests cite:

* E4M3 normals have a `2^-3` relative ulp, so round-to-nearest gives `|dx| <= 2^-4 |x| = 0.0625|x|`.
* The subnormal floor contributes at most `s * 2^-10`, i.e. about `absmax * 2^-19` in absolute
  terms, and is negligible next to the relative term.
* Treating the per-element relative errors as independent and uniform on `[-2^-4, 2^-4]`, their
  standard deviation is `2^-4 / sqrt(3) = 0.036`.
* In a dot product of length K, both the signal and the error accumulate as `sqrt(K)`, so the
  *relative* L2 error of an output column stays near 0.036 and does not grow with K.

That is why the suite-owned A8 criterion (relative-L2 allowance 0.04, gross pointwise cap 0.06 in
`tests/ops/linear/linear_test_common.cpp`) applies unchanged to this route. It is the same
arithmetic profile as the row-scaled FP8 A8 route, so it is not a second criterion.

The activation materialization is a separate launch that reads `T*K` BF16 and writes `T*K` bytes
plus `T` floats. At the prefill extents this is a small fraction of the weight traffic the GEMM
already moves, but it is real and belongs to the measurement (section 6), not to a contraction-only
timing.

## 4. Kernel

`src/ops/linear/a8_prefill/a8_prefill_mma.cuh`. One CTA owns a `(BlockRows x BlockTokens)` output
tile and walks K one quant group at a time.

Orientation. The MMA is `[token,K] x [K,row]`: activations are the `m16` operand, weights are the
`n8` operand, so the accumulator's contiguous axis is the public output-row axis and the epilogue
can emit aligned BF16 vectors. This is the same orientation as the row-scaled FP8 A8 kernel, whose
fragment plumbing, 16-byte segment swizzle and output policies this body follows.

Tile. `BlockRows = 64`, `BlockTokens = 128`, `BlockK = 64`, three pipeline stages, four warps. That
is the tile the BF16 prefill routes of these formats already settle on at T >= 128
(`Q4MmaR64C128Schedule`, 64 weight rows x 128 token columns, one 64-wide quant group per K step);
only the operand roles are swapped, so each warp still owns 16 accumulator fragments
(`kMmaTokens = 4`, `kMmaRows = 4`). Keeping the tile fixed means the A/B measurement in section 6
compares instruction sets and not tile choices.

Staging, per K tile:

* activations: `cp.async.cg` 16-byte segments into `Stages * 128 * 64` bytes;
* weights: the raw 32-byte code group per row (`cp.async.cg`), the 8-byte Q5 high-bit group
  (`cp.async.ca`, because `.cg` is 16-byte only), and the one binary16 group scale per row, which is
  strided by `groups_per_row` and is therefore gathered rather than copied asynchronously;
* decode: each thread turns four logical K values into one 32-bit store of E4M3 bytes into a single
  `64 x 64` byte plane, under the same segment swizzle, so `ldmatrix` reads it like any other
  operand tile.

Scales. The activation scale depends only on the token, so it is applied once in the epilogue. The
weight group scale changes every 64-wide slab, so it must be applied *before* the slab joins the
running total. Per `(mma_token, mma_row)` pair the kernel runs the slab's two `m16n8k32` MMA steps
into a four-register private partial and then does

```text
accumulator[j] += partial[j] * weight_scale[row(j)]
```

which costs four FMAs per fragment per 64 values of K and keeps the accumulator count unchanged.

Shared memory. Static, and the output tile aliases the staging planes after the contraction:
35,200 B for Q4 and 36,736 B for Q5, both inside the 48 KiB static block budget, so no
`cudaFuncSetAttribute` opt-in is needed. The schedule `static_assert`s this; a larger tile would
have to move to the dynamic allocation and Ada's 101,376-byte opt-in maximum.

ptxas, CUDA 13.4, `-Xptxas -v`, `--generate-code=arch=compute_89,code=[compute_89,sm_89]`:

| translation unit | registers | stack | spill stores | spill loads | static shared |
|---|---:|---:|---:|---:|---:|
| `q4_a8_prefill.cu` (both `FullTokens` instances) | 173 | 0 | 0 | 0 | 35,200 B |
| `q5_a8_prefill.cu` (both `FullTokens` instances) | 176 | 0 | 0 | 0 | 36,736 B |

ptxas accepts `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` for `sm_89`; the `kind::f8f6f4`
modifier that `ops/common/mma.cuh` selects at `__CUDA_ARCH__ >= 1200` is Blackwell-only and is not
used here. With `__launch_bounds__(128, 2)` the register count still admits two CTAs per SM, and two
CTAs of shared memory fit Ada's 100 KiB per-SM carveout.

## 5. Route admission

Every condition must hold, and the last one defaults to off:

1. the weight is `Q4_G64_FP16` or `Q5_G64_FP16` in `row_split_k128_v1` with unpadded K;
2. the `LinearPolicy` permits A8 (`AllowA8` or `AllowA4`);
3. the build defines `NINFER_SM89`; the two instantiating translation units emit nothing otherwise;
4. `T >= 128`, one token tile, below which the contraction is not tensor-core bound at these shapes
   and the extra activation pass is not repaid;
5. the `(N,K)` pair is registered by the shape-owned selector;
6. `ops::set_prefill_a8_routes_enabled(true)`, which `EngineOptions::prefill_a8` sets from
   `--prefill-a8 fp8`.

Registered geometries: Q4 `4096x5120`, `5120x6144`, `6144x5120`, `7168x5120`, `34816x5120`; Q5
`6144x5120`, `7168x5120`, `5120x6144`, `5120x17408`. K is restricted to the three extents for which
the shared activation quantizer compiles an instance (5120, 6144, 17408).

The gate participates in route selection, so it also decides what
`linear_workspace_capacity_bytes()` reserves for these formats. Set it once, before the first
capacity query; flipping it while a planned capacity is live is a programming error. `Engine`
does this in its constructor, ahead of model construction.

With the gate off, `select_q{4,5}_a8_prefill_launch` returns null, the capacity query returns 0 and
dispatch is byte-for-byte the previous A16 path.

## 6. How to measure

Op level, both directions on the same geometries, extents, inputs, cache and timing conditions:

```bash
cmake --build build --parallel --target ninfer_linear_bench
./build/bench/ninfer_linear_bench \
  --suite ada_fp8_prefill --prefill-a8 off --csv-out profiles/bench/prefill_bf16.csv
./build/bench/ninfer_linear_bench \
  --suite ada_fp8_prefill --prefill-a8 fp8 --csv-out profiles/bench/prefill_a8.csv
```

The suite runs the nine registered geometries at T = 128, 512 and 2048 under `--policy a8`; with
the gate off the same points resolve to the BF16 MMA routes, so the two CSVs differ only in the
selected implementation. The timed region is the public `linear()` call, so it includes the
activation materialization launch.

Quality, on the code domain:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --kv-dtype fp8 --prefill-a8 off --output profiles/ppl/prefill_off
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --kv-dtype fp8 --prefill-a8 fp8 --output profiles/ppl/prefill_fp8
```

`ninfer-ppl-1m-v1` carries a NInfer C++/CUDA code domain; compare that domain's per-stream
perplexity between the two runs. Use the same manifest, KV representation, context and stride for
both, and report the evaluator's noise band (repeat the `off` run) alongside the delta. Scoring is
prefill-shaped at the default 4,096-token context, so it exercises the route at the extents that
matter; a decode-only measurement does not.

## 7. Go / no-go

Enable the route by default (that is, change `EngineOptions::prefill_a8` to `Fp8` and the CLI
default with it) only when both hold on an RTX 4090:

* the code-domain perplexity delta between `--prefill-a8 fp8` and `--prefill-a8 off` is within the
  evaluator's measured noise band; and
* prefill throughput improves by at least 1.3x at pp2048, measured end to end through the product
  route, not only in the Op benchmark.

If the throughput gain lands below 1.3x while perplexity is clean, the route stays opt-in and the
gap is a tile/pipeline question, not a numerics one. If perplexity moves outside the noise band, the
activation scale granularity is the first thing to revisit: the current scheme is one scale per
token row, and the weight side has no error to spend.

## 8. Not covered yet

* **GPU validation.** The route has never been executed. It was developed and compile-checked
  without a CUDA device, so every claim in sections 4 and 5 about numbers is a ptxas or source-level
  claim, and every claim about correctness is unproven until
  `ninfer_linear_q4_a8_prefill_test` and `ninfer_linear_q5_a8_prefill_test` run on Ada.
* **LinearSwiGLU and LinearAdd.** The kernel body already carries the `Epilogue`, `Output`,
  `RowPolicy` and `PairRows` hooks the row-scaled FP8 family uses to serve the fused gate/up fold
  and the residual add from one contraction, but neither fused Op's plan is wired to this family
  yet, so both keep their A16 routes. Wiring LinearSwiGLU is the larger remaining win, because the
  `34816x5120` gate/up bank dominates prefill FLOPs; it needs a `Q4LinearSwiGluScheduleId` entry, a
  workspace-capacity branch and a launcher of the shape of
  `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu`.
* **Tile exploration.** One tile is instantiated, chosen to match the BF16 prefill route rather than
  measured. Once a device is available, the candidate sweep in
  [op development](op-development.md) section 7.1 applies: instantiate the small overlapping
  candidate set, decide, and delete the losers.

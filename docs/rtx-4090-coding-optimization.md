# RTX 4090 coding workload: where the time goes and what to change

This note is the working plan for making NInfer-4090 as fast and as accurate as a single 24 GB
RTX 4090 (`sm_89`) allows for programming workloads: coding agents (Qwen Code, OpenAI- and
Anthropic-protocol tools), long repository contexts, and structured tool-call output. It records
the upstream sync state, the measured limits of the card, the configuration to run today, and the
engineering candidates ranked by payoff.

## Upstream sync state

| Source | Relationship | State on this branch |
|---|---|---|
| `UDPSendToFailed/ninfer-4090` `feat/rtx-4090-sm89-native` | direct parent fork | merged through v1.2.0 (clean merge) |
| `feat/qwen-code-compat` | this repository | fast-forwarded (v1.0.0 merge plus `enable_thinking` in `chat_template_kwargs`) |
| `Don-Chad/ninfer-3090` `master` | grandparent fork | not merged as a whole; ten commits cherry-picked, one ported by hand |
| `Neroued/ninfer` `master` | root upstream | not merged; the tree moved to `src/models`, v3 artifacts, and `sm_120a` only |

Don-Chad `master` is 168 commits ahead of the 4090 lineage, but 155 of them are not a patch-level
match for anything here and only 27 apply without conflicts. The remainder is the Neroued
`resource_manager` / protocol-adapter / softmax-attention restructuring, which deletes the files
that hold this fork's E8 lattice KV cache, GQA prefill, and OpenAI schema code. A whole-branch merge
would re-port those features. The commits taken instead:

- `fix(gdn)`: pairwise K reduction in the unsplit gating projection (numerical accuracy);
- `fix(runtime)`: startup no longer aborts on a device-wide memory reading;
- `fix(ops)`, `fix(sparse-moe)`: gather index lifetimes and prefill scan synchronization;
- `fix(portability)`: `<cstdlib>` before nvtx3 in both NVTX headers;
- `perf(serve)`: incremental response-context accounting, linear tool-call whitespace streaming,
  no duplicate stream readiness probes;
- `fix(docker)`: forward-compat libraries removed so GeForce cards run the image;
- `fix(engine)` (hand port): the executor worker thread and Engine teardown bind the selected CUDA
  device, so `--device N` is honored on every thread.

Neroued `master` now requires v3 artifacts, runs the Qwen3.5 family from `src/models/qwen3_5`,
executes the artifact's Jinja chat template, and ships DFlash2 for Qwen3.8-27B. Sibling forks show
that an `sm_89` port on top of that tree is feasible (`natpate/ninfer-windows` `dev` tracks it on
`sm_120a`; `Ambolio/ninfer-4090-windows` ports v1.0.7 to `sm_89` with `rk4v4-e8` and DFlash2). The
right way to take those features is a re-port of this fork's kernels onto the upstream tree, not a
merge; see the last section.

## What the card can do

Decode is memory-bound. The groupwise artifact is 16.96 GiB and the RTX 4090 moves about 1 TB/s,
so one forward pass over the weights costs about 18 ms and the roofline is roughly 55 tok/s for a
single sequence. Measured MTP0 decode is 51.9 tok/s, about 94% of that ceiling. Nothing in the
decode kernels will change single-token decode meaningfully; only producing more accepted tokens
per weight pass does (speculation, batching).

Prefill is tensor-core bound. At 2,146 tok/s (pp2048) the engine sustains about 116 TFLOPS of
effective dense work against Ada's 165 TFLOPS BF16 rate with FP32 accumulate. Ada also has INT8 and
FP8 tensor cores at two to four times that rate, and none of the GEMM routes here use them.

Memory is the third limit. Weights plus the MTP proposal head leave about 7 GiB for KV cache and
workspace, and Windows DWM plus background applications take 1 to 3 GiB more unless the GPU is
dedicated. Every byte per token of KV precision trades directly against context length.

## Run this today for coding

```powershell
ninfer-serve.exe qwen3_8_27b.ninfer ^
  --kv-dtype rk8v4 --max-context 160000 ^
  --spec mtp --draft-tokens 5 --lm-head-draft ^
  --prefill-chunk 2048 ^
  --prompt-cache --preserve-thinking ^
  --max-concurrency 1
```

- `rk8v4` keeps keys at 8 bits (99.4% cosine similarity to FP32) and fits about 210k tokens
  with MTP; use `int8` (99.8%) below about 160k tokens when the workload is precision-critical, and
  `rk4v4-e8` (98.7%) only when the repository context needs 300k or more. `rk2v4-e8` is a retrieval
  mode, not a coding mode.
- MTP with the LM-head draft is where code decode speed comes from. Structured code accepts 75 to
  90% of draft tokens, so decode runs at 100 to 150 tok/s against a 52 tok/s base. `--draft-tokens`
  4 to 7 is the useful band; measure acceptance in the request log for your prompts and lower K if
  acceptance drops below about 60%. Prompt-lookup drafting is automatic under MTP and copies
  repeated spans from the context, which is why edits of existing files decode faster than fresh
  code.
- `--prefill-chunk 2048` or `4096` raises deep-prefill throughput (2,637 tok/s at pp4096 versus
  2,146 at chunk 1024) at the cost of workspace memory; file dumps and repository maps are long
  prompts.
- `--prompt-cache` (DirectStorage on Windows) turns a cold 77k-token restore into 150 ms; every
  agent turn re-sends the whole history, so this is the largest time-to-first-token lever.
- `--preserve-thinking` keeps the rendered history append-only, so the prefix cache is reused
  across turns. The official template drops closed-turn reasoning; if you want that behavior,
  expect a prefix rebuild per turn.
- Sampling defaults are the Qwen recommendations per thinking mode (1.0/0.95/20 thinking,
  0.7/0.80/20 non-thinking). Coding agents that pass `temperature` explicitly override them.
- On a GPU that does not drive the desktop, add `--wddm-evictable-budget` and leave the
  `--max-context` about 20k to 30k tokens under the published ceiling. Moving the desktop to an
  iGPU or a second card frees 1 to 3 GiB, which is 30k to 60k tokens of `rk8v4` context.

## Upgrade candidates, ranked

### 1. Measure accuracy before changing precision

Neroued added a causal scoring evaluator (`apps/perplexity`, `feat(perplexity): add causal scoring
evaluator` plus `feat(ops): add target logprob reduction`, about 2,100 lines). It is the only way
to decide KV mode, draft depth, or a weight re-quantization on evidence rather than cosine
similarity. Port it and add a code corpus fixture (repository files, diffs, JSON tool arguments).
Every item below that touches numerics should report perplexity on that fixture. Effort: medium;
the port conflicts in `engine.cpp`, `program_impl.h`, and `layouts_impl.h`, all mechanical.

### 2. INT8 or FP8 tensor-core prefill

The weights are Q4/Q5/Q6 groupwise integers; Q4 and Q5 codes are exactly representable in INT8 and
in FP8 `e4m3`, so dequantizing to an 8-bit MMA operand loses nothing on the weight side. Ada's
INT8 MMA runs at four times, and FP8 at two times, the BF16 FP32-accumulate rate. Per-token INT8 or
FP8 activation quantization in the prefill GEMMs (with per-group scales applied per 64-wide K
slab in the epilogue) is the one accuracy risk and is exactly what item 1 measures. Expected
effect: 1.4x to 1.8x prefill on long prompts, which for an agent is time-to-first-token on every
turn that misses the prefix cache. Effort: high (new GEMM route family under `src/ops/linear`,
`linear_swiglu`, `linear_add`, and the attention/GDN input projections; the route-selection
machinery already exists). Blackwell NVFP4 is not an option on Ada.

### 3. Better drafting for code

Speculation is the only decode lever, so its acceptance on code is the metric to optimize.

- Sweep `--draft-tokens` on real agent transcripts and publish the K-versus-acceptance curve for
  code, diffs, and JSON; the README numbers are corpus replays and overstate acceptance.
- Extend prompt lookup to prefer the longest match and to match across the tool-result boundary,
  since agent edits quote the file they are about to rewrite.
- Evaluate DFlash2 for Qwen3.8-27B. Upstream ships companion weights and measures, on an RTX 5090,
  191 tok/s at 53.5% acceptance on code with DFlash2 K=7 against 200 tok/s at 76.3% with MTP3, and
  267 versus 224 tok/s on structured output. It is a win for JSON and tool arguments, parity for
  code, and it costs about 1.7B draft parameters of VRAM (roughly 1 GiB groupwise, 3.3 GiB BF16)
  plus a 40 MiB ring. On 24 GB that is context you give up, so it belongs behind item 1's
  measurement, not on by default.

### 4. Exact chat template execution

Upstream now executes the artifact's Jinja template with the llama.cpp Jinja engine
(`feat(frontend): execute custom jinja chat templates`, `build: import llama.cpp jinja source
base`). This fork renders recognized templates from C++ (`chat_template.cpp`). Tool schemas,
parallel tool calls, and thinking toggles are where hand-rendered templates drift from the trained
format, and drift shows up as malformed tool calls in coding agents. Effort: medium-high; it also
brings the `chat_template_kwargs` surface into line with upstream.

### 5. Tolerant tool-call parsing

`tool_call_parser.cpp` falls back to plain text when a `<tool_call>` block is malformed or cut by
the stop token, so the agent receives prose instead of a call. Add repair for the common failures
(missing closing tag at end of sequence, trailing commas, JSON wrapped in a code fence, unescaped
newlines in string arguments) behind a flag, and log every repair. Effort: low. Pair it with the
existing structured-JSON grammar for `response_format` requests.

### 6. KV precision per layer

The KV modes are uniform across layers. Attention layers early in the stack propagate key error
into every later layer, while late layers tolerate 4-bit keys. A mixed schedule (INT8 or `rk8v4`
for the first N attention layers, `rk4v4-e8` after) could hold `rk8v4` accuracy at close to
`rk4v4-e8` capacity. The paged KV container already stores per-layer layouts, so the change is in
the layout tables and the attention route selection. Effort: medium; only worth doing with item 1.

### 7. Weight re-quantization is the last resort

The current Q4/Q5 mix is already at the bandwidth roofline. Converting the remaining Q5 tensors to
Q4 would save roughly 1.5 GiB (about 8% decode speed and 30k tokens of context) and is the one
change on this list that trades away model accuracy on code. Do not do it without item 1's
numbers, and keep the attention projections and the LM head at their current widths.

### 8. Concurrency for agent fan-out

Agents that run subtasks in parallel benefit from `--max-concurrency 2` to `4`: aggregate decode
scaled from 102 tok/s at C1 to 193 tok/s at C4 on the early sm_89 build, and the fixed-lane
executor forms a compact batch every round. The KV budget is shared across lanes, so set
`--max-context` per lane accordingly.

## Longer term: re-port onto Neroued v3

The root upstream's v3 tree is where DFlash2, the Jinja frontend, the perplexity evaluator, the
resource manager, and the current linear tuning live, and its build rejects every architecture
but `sm_120a`. The 4090 features that must survive a re-port are well delimited:

- `rk8v4`, `rk4v4`, `rk4v4-e8`, and `rk2v4-e8` KV codecs and their attention prefill/decode routes
  (`src/ops/kernel/e8_*`, `gqa_attention_*`, `kv_cache_append_*`);
- Ada small-T MMA routes and SM-count-sized launch policies (`kTargetSmCount`, the Q4/Q5/W8
  `small_t` files);
- WDDM residency locking, D3D12 fence handling, and DirectStorage prompt cache (`src/core/arena.cu`,
  `disk_state_cache`);
- the embedded WebUI, `/metrics`, `/slots`, structured JSON, and the Qwen Code compatibility fixes
  in `src/serve`.

`Ambolio/ninfer-4090-windows` (v1.0.7 base) and `natpate/ninfer-windows` `dev` (current master)
are the closest reference trees for that work.

# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense and MoE architectures on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

Five official artifacts are available. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Quick start

NInfer requires 64-bit Linux or Windows 11, an NVIDIA GeForce RTX 5090 (`sm_120a`) or RTX 4090
(`sm_89`), a CUDA toolkit supporting that architecture, CMake 3.28 or newer, a C++20 host
compiler, Ninja, `pkg-config`, FFmpeg development libraries (`libavformat`, `libavcodec`,
`libavutil`, and `libswscale`), and `libcurl >= 7.85`. CUDA 13.1 is the validated development
toolkit on the RTX 5090 and CUDA 13.3/13.4 on the RTX 4090; CMake does not impose a CUDA version
floor. The build rejects CUDA architectures other than `sm_120a` and `sm_89`.

### RTX 4090 (`sm_89`) build

This tree carries the Ada Lovelace port layer described in [NOTICE](NOTICE). Configure with the
architecture set explicitly:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

On Windows, run both from a Developer Command Prompt or after `vcvars64.bat`, with vcpkg's
toolchain file; `vcpkg.json` supplies curl, FFmpeg and pkgconf.

Use the `groupwise-int` artifacts on Ada. The `nvfp4` artifacts need Blackwell FP4 tensor cores;
on `sm_89` the NVFP4 W4A4 routes and the `nvfp4` / `k8v4` KV cache modes are compiled as stubs
that fail with a clear error at startup. A v2 `.ninfer` download from before the v3 format is
upgraded once, in place of a re-download; weights are copied unchanged and the maintained chat
template is installed:

```bash
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir models
python3 tools/upgrade_ninfer_v2_to_v3.py models/qwen3_8_27b.ninfer models/qwen3_8_27b.v3.ninfer
```

#### Coding agent, one session, 128k context

`rk8v4` stores 8-bit rotated keys and 4-bit values. It costs 0.09% code-domain perplexity against
BF16 KV and is 1 GiB smaller than `int8` at 128k, which is what lets MTP fit beside it. `int8` with
MTP at 128k needs 5.1 GiB of runtime memory after weights and does not start on a card that also
drives the desktop. MTP with the LM-head draft at K=5 is the fastest draft window measured on the
4090; prompt lookup drafts from the sequence's own history whenever the head has no proposal.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_24gb.ninfer \
  --kv-dtype rk8v4 --max-context 131072 --kv-capacity 131072 \
  --spec mtp --draft-tokens 5 --lm-head-draft
```

Measured with 2.8 GiB of the card held by the Windows desktop: 3.86 GiB runtime, 1.35 GiB free,
1.49k tok/s prefill, 137 tok/s decode, and a needle retrieved from 130,048 tokens.

#### Long context, 256k

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_24gb.ninfer \
  --kv-dtype rk2v4-e8 --max-context 262144 --kv-capacity 262144 \
  --spec mtp --draft-tokens 5 --lm-head-draft \
  --device-state-slots 0 --prefill-chunk 1024
```

`rk2v4-e8` stores 2-bit E8 keys. It costs 3.1% code-domain perplexity, ten times what `rk4v4-e8`
costs, and it is the only mode measured that holds the model's full 262,144-token window with MTP
on 24 GB; `rk4v4-e8` at 256k needs 5.2 GiB of runtime and does not start. `--device-state-slots 0`
drops the cached device checkpoint, 150 MiB, and `--prefill-chunk 1024` shrinks the prefill
workspace. Together they leave 1.12 GiB free. Measured: 915 tok/s prefill, so 4 min 45 s for a cold
260,096-token prompt, 108 tok/s decode, needle retrieved. `max_context` above 262,144 is rejected
at startup.

#### Memory, reuse and the desktop

Keep the `free` figure on the `capacity` startup log line at about 1 GiB or more. Below that,
decode falls sharply as WDDM evicts to host memory: DFlash2 ran 189 tok/s at 32k with 1.60 GiB
free and 41 tok/s at 64k with 227 MiB, on the same prompt. `--kv-capacity auto` reserves 1 GiB of
headroom on top of this; pass the capacity explicitly when sizing near the limit. The Windows
desktop's share of the card moves with open applications, 1.7 to 3.5 GiB in these measurements,
so recheck `free` after changing what is open.

Follow-up turns reuse the computed prefix in process: a second turn over 130k tokens restored
130,041 of them and answered in 262 ms, including with `--device-state-slots 0`. `--prompt-cache`
persists prefixes to disk across restarts, but it is not used with a speculative backend, so a
restart pays the full prefill again.

The embedded web UI is served at `http://127.0.0.1:8080/`; OpenAI clients use
`/v1/chat/completions` and `/v1/responses`, Anthropic clients `/v1/messages`.

#### Agent fan-out, four lanes

Up to eight requests decode in one batch per round. The KV pool is shared across lanes, so a
wider pool mode (`rk8v4`, 8-bit keys) leaves room for one long parent plus several subagents.
Set `--default-max-tokens` explicitly: an omitted `max_tokens` otherwise reserves the whole
remaining context per request and serializes the lanes. Raise the pending timeout above its
30 second default so queued subagents wait instead of failing.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.v3.ninfer \
  --kv-dtype rk8v4 --max-context 131072 \
  --max-concurrency 4 --max-pending-requests 32 --pending-timeout-ms 600000 \
  --default-max-tokens 8192 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --prompt-cache
```

#### Measure before changing precision

`ninfer-perplexity` scores the fixed corpus; the NInfer C++/CUDA code domain is the number that
matters for programming use. Compare KV modes with the same artifact, and artifacts with the same
KV mode:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_24gb.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --kv-dtype int8
./build/apps/ninfer-perplexity models/qwen3_8_27b_24gb.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --kv-dtype rk4v4-e8
```

Measured on the `qwen3_8_27b_24gb` artifact with `--quick` (261,167 tokens), KV sizes at 128k:

| `--kv-dtype` | Code ppl | vs BF16 KV | Overall ppl | KV at 128k |
|---|---|---|---|---|
| `bf16` | 1.6552 | — | 4.3073 | |
| `int8` | 1.6550 | −0.01% | 4.3070 | 4.12 GiB |
| `rk8v4` | 1.6567 | +0.09% | 4.3114 | 3.12 GiB |
| `rk4v4-e8` | 1.6605 | +0.32% | 4.3221 | 2.12 GiB |
| `int8:4,rk4v4-e8` | 1.6603 | +0.31% | 4.3216 | |
| `int8:4,rk2v4-e8` | 1.7032 | +2.90% | 4.4429 | |
| `rk2v4-e8` | 1.7062 | +3.08% | 4.4515 | 1.62 GiB |

Two-tier `X:N,Y` schedules store the first N full-attention layers as X. On this model the
placement buys little: four `int8` layers recover 0.18 of the 3.08 points `rk2v4-e8` costs and
nothing measurable over `rk4v4-e8`. Choose the single mode by memory budget.

#### FP8 tensor-core prefill (off by default)

Ada's FP8 MMA has twice the BF16 tensor rate; the route keeps weights exact and quantizes
activations per token. Measured end to end it adds 7%, 2.03k to 2.17k tok/s at 7,680 prompt
tokens, because the 34816-row gate/up GEMM runs slower than BF16 on the current tile. It stays off
by default.

The route is admitted only when the artifact permits 8-bit activations. `qwen3_8_27b_24gb` does;
the official conversion does not. Grant it without touching the weights, then enable the switch:

```bash
python3 tools/set_activation_policy.py models/qwen3_8_27b.v3.ninfer \
  models/qwen3_8_27b.a8.ninfer --policy AllowA8 --text-projections
./build/apps/ninfer-serve models/qwen3_8_27b.a8.ninfer --kv-dtype rk8v4 --max-context 131072 \
  --kv-capacity 131072 --spec mtp --draft-tokens 5 --lm-head-draft --prefill-a8 fp8
```

Go/no-go criteria are in [Ada FP8 prefill](docs/maintainer/ada-fp8-prefill.md); per-GEMM results
are in [hardware validation](docs/maintainer/hardware-validation.md).

#### A better-quantized artifact for 24 GB

`tools/convert/recipes/qwen3_8_27b_24gb.py` keeps the official formats and kernels but selects
every Q4/Q5 code by a clip-optimal scale search (GPTQ when calibration Hessians are supplied),
stores the token embedding at Q6 (0.33 GiB smaller), and permits 8-bit activations. It needs the
BF16 checkpoint and a GPU; see [weight conversion](docs/weight-conversion.md), including
`tools/calibrate_hessians.py` and the evaluation protocol. Without calibration it converts in about
15 minutes on a 4090 to a 20,079,940,864-byte artifact. The recipes above were measured on it.

#### Other `sm_89` notes

- `--kv-dtype` gains `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8` and the two-tier `X:N,Y` form; `int8`
  stores keys under the same D256 rotation contract as the RTX 5090 build.
- `--spec mtp --draft-tokens K` accepts K up to 15, but with CUDA graphs on, MTP above K=7 aborts
  at startup with `cudaGraphExecUpdateErrorTopologyChanged`; `--no-cuda-graph` runs it. K=5 is
  the fastest window measured for both MTP and DFlash2. DFlash2 (`--spec dflash2`) works with
  artifacts that carry the companion weights, and its draft weights cost about 1.5 GiB more than
  MTP, so prefer MTP at long context.
- `response_format` (`json_object`, `json_schema`) and Responses `text.format` are enforced by
  grammar-constrained sampling (xgrammar, `NINFER_ENABLE_STRUCTURED_OUTPUT=ON`) with speculation
  kept on. `GET /metrics` exposes Prometheus counters, `GET /slots` the executor lanes, `GET /props`
  the web UI's model description. `NINFER_EMBED_WEBUI=OFF` drops the UI and its configure-time
  download.
- `--tolerant-tool-calls` accepts complete Qwen tool calls followed by trailing text.
  `--vision-max-tokens` caps the Vision workspace (default 8192 tokens). On Windows,
  `--wddm-evictable-budget` lets a GPU that does not drive the desktop budget against total VRAM
  minus a 512 MiB display floor.

Published performance below is for the RTX 5090. Measured on an RTX 4090 with the
`qwen3_8_27b_24gb` artifact, Windows 11 and CUDA 13.3, greedy, on a coding prompt: 47 tok/s
single-token decode, 139 tok/s with MTP at K=5 and 141 with DFlash2 at K=5 at 8k context, 189 tok/s
with DFlash2 at 32k, and 2.03k tok/s prefill at 7,680 tokens. Test results and open issues are in
[hardware validation](docs/maintainer/hardware-validation.md).

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% | 2.15× |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.

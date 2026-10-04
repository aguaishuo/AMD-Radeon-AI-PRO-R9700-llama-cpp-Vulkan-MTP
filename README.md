# AMD Radeon AI PRO R9700 / RX 9700 — llama.cpp Vulkan Inference Optimization Guide

> RDNA4 (gfx1201) Vulkan 推理优化指南 — 覆盖 Dense 与 MoE 两种模型架构
>
> RDNA4 (gfx1201) Vulkan inference optimization guide — covering Dense and MoE architectures
>
> **Latest data: 2026-10-04** (draft width 4 / DFlash2 drafter) · 2026-10-03 (long-context scaling) ·
> 2026-09-18 (per-model benchmarks) — llama.cpp
> build 10820, one model per role, all measured on the same box: MoE text (35B-A3B), vision
> (VL-30B-A3B), dense reasoning (27B + MTP). Only one runs at a time — 32 GB of VRAM does not hold
> two of these.

[![GPU](https://img.shields.io/badge/GPU-AMD%20Radeon%20AI%20PRO%20R9700-red)](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/r9700.html)
[![Backend](https://img.shields.io/badge/Backend-Vulkan-blue)](https://github.com/ggml-org/llama.cpp)
[![Discussion](https://img.shields.io/badge/Discussion-%2319890-blue)](https://github.com/ggml-org/llama.cpp/discussions/19890)

---

## Hardware / 硬件配置

| Component | Spec |
|---|---|
| GPU | AMD Radeon AI PRO R9700 / RX 9700 — 32GB GDDR6, 256-bit bus (~576 GB/s) |
| CPU | AMD Ryzen 7 5700X (8 cores) |
| OS | Ubuntu 24.04 |
| Kernel | Linux 7.0.0-31-generic |
| Driver | RADV Mesa 25.2.8 (Vulkan, `gfx1201`, `KHR_cooperative_matrix`), Vulkan loader 1.3.275 |
| llama.cpp | **build 10820** (`74a7c897f`, Vulkan backend) — retested 2026-09-18 |
| Key env | `RADV_DEBUG=nocompute` — essential for RDNA4 Vulkan perf |

---

## Models Tested / 已测试模型

**Current generation (2026-09-18 retest, llama.cpp build 10820):**

| Model | Architecture | Size | Quant | File Size | Effective params/token |
|---|---|---|---|---|---|
| Huihui-Qwen3.6-35B-A3B-abliterated | MoE | 35.5B total | Q4_K | ~21.7 GiB | ~3B |
| Qwen3.8-27B-ABLITERATED | Dense | 27B | Q4_K_M | ~16 GiB | 27B (full) |
| Qwen3-VL-30B-A3B-Instruct | MoE (VLM) | 30.5B total | Q4_K_M | ~18 GiB | ~3B |
| Qwen3.8-27B (base) | Dense | 27B | Q4_K_M | ~17 GiB | retired 2026-09-18 — duplicated the abliterated slot |

**Previous generation (historical data, kept for the optimization timeline):**

| Model | Architecture | Size | Quant | File Size | Effective params/token |
|---|---|---|---|---|---|
| Qwen3.6-27B | Dense | 27B | Q4_K_M | ~16.3 GiB | 27B (full) |
| Qwen3-30B-A3B | MoE | 30.53B total | Q4_K_M | ~17.3 GiB | ~5.9B |
| Qwen3.6-35B-A3B | MoE | 35.51B total | Q4_K | ~20.2 GiB | ~6.5B |

---

## Benchmark Results / 测试结果

### 🆕 2026-09-18 Retest — Qwen3.8 / Qwen3-VL generation (llama.cpp build 10820)

> Measured through the live `llama-server` OpenAI-compatible endpoint, `temperature=0`, thinking disabled
> (`chat_template_kwargs.enable_thinking=false`). Numbers are llama.cpp's own `timings`
> (`prompt_per_second` / `predicted_per_second`), not wall-clock estimates. Every prompt carries a unique
> nonce so no run reuses the KV cache.
>
> 通过运行中的 llama-server 实测，取 llama.cpp 自带 `timings` 字段；每次请求加随机前缀，避免命中 KV 缓存。

#### Huihui-Qwen3.6-35B-A3B-abliterated Q4_K — MoE (no MTP, `-b 4096 -ub 512`, 32768 ctx)

| Context | Metrics | pp (t/s) | tg (t/s) |
|---|---|---|---|
| **Short** (pp=26) | gen=16 | 228 | 114 |
| **Medium** (pp=186) | gen=16 | 553 | 113 |
| **Long** (pp=492) | gen=16 | **1805** | 114 |
| **Sustained** (pp=47) | gen=256 | 243 | 129 |
| | gen=512 | 389 | **145** |

**Real workload** — summarizing a 64 KB stage-1 JSON record (video frame analysis) into a structured report:

| Prompt tokens | pp (t/s) | tg (t/s) | Total wall |
|---|---|---|---|
| 16,813 | **2419** | **138** | 33.3 s |

> ⚠️ These numbers run **lower** than the 2026-07 figures recorded further down for the vanilla
> Qwen3.6-35B-A3B on build 9870 (tg 160 @ gen=512, pp 2273 @ 490 ctx). Two variables changed at once —
> abliterated weights (21.7 GiB vs 20.2 GiB) **and** the build — so this is not a controlled comparison and
> should not be read as a build regression. The real-workload prefill (2419 t/s at 16.8k tokens) is the
> highest sustained pp measured on this box.
>
> VRAM after load: 21.8 GB of 34.2 GB, so `--ctx-size` has room to grow well beyond 32768.

#### Qwen3.8-27B Q4_K_M — Dense + MTP (`--spec-draft-n-max 2`)

Two independent runs; ranges show run-to-run spread.

| Context | Metrics | pp (t/s) | tg (t/s) |
|---|---|---|---|
| **Short** (pp=26) | gen=16 | 89–91 | 61 |
| **Medium** (pp≈185) | gen=16 | 301–313 | 60–61 |
| **Long** (pp≈493) | gen=16 | **567–576** | 52–61 |
| **Sustained** (pp≈47) | gen=256 | 141–147 | 50–53 |
| | gen=512 | 143–145 | 52–54 |

> MTP draft acceptance on sustained generation: **67–69%**, mean accepted draft length 2.34–2.38
> (with `--spec-draft-n-max 2`). Short bursts hit 82–100% acceptance but are too small to be representative.
>
> ⚠️ Short-burst tg (gen=16) is noisy on a dense + MTP setup — the same case measured 52 and 61 t/s on two
> consecutive runs. Only the sustained 256/512-token figures are stable enough to compare configurations.

#### Qwen3.8-27B-ABLITERATED Q4_K_M — Dense + MTP (same flags)

| Context | Metrics | pp (t/s) | tg (t/s) |
|---|---|---|---|
| **Short** (pp=27) | gen=16 | 66 | 54 |
| **Medium** (pp=186) | gen=16 | 259 | 52 |
| **Long** (pp=492) | gen=16 | **510** | 52 |
| **Sustained** (pp=47) | gen=256 | 101 | 55 |
| | gen=512 | 114 | **56** |

> Draft acceptance 68–69%, mean length 2.36–2.38 — the abliterated weights behave the same as the base
> model here. tg sits ~3–8% below the base model; pp is ~10% lower.

#### Qwen3-VL-30B-A3B-Instruct Q4_K_M — MoE (no MTP, `-b 4096 -ub 512`)

| Context | Metrics | pp (t/s) | tg (t/s) |
|---|---|---|---|
| **Short** (pp=19) | gen=16 | 276 | 140 |
| **Medium** (pp=179) | gen=16 | 1201 | 141 |
| **Long** (pp=486) | gen=16 | **1986** | 139 |
| **Sustained** (pp=40) | gen=256 | 400 | 165 |
| | gen=512 | 652 | **179** |

> ⚠️ This container runs a locally built image (`llama-cpp-mtp:vulkan`, build `2d97363`) and sets
> `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` **without** `RADV_DEBUG=nocompute` —
> yet still reaches 179 t/s, the fastest sustained decode measured on this box. The two dense models
> run the official `ghcr.io/ggml-org/llama.cpp:server-vulkan` image with `RADV_DEBUG=nocompute`.

#### Generation-over-generation summary

| Model | Arch | tg @ sustained | pp @ ~490 ctx |
|---|---|---|---|
| Qwen3.6-35B-A3B (2026-07 data) | MoE | ~155–160 | ~243 |
| Qwen3.6-35B-A3B-abliterated (2026-09) | MoE | 129–145 | **1805** |
| Qwen3-VL-30B-A3B (2026-09) | MoE | **165–179** | **1986** |
| Qwen3.6-27B (2026-07 data) | Dense + MTP n-max 3 | ~59–60 | ~489 |
| Qwen3.8-27B (2026-09) | Dense + MTP n-max 2 | 50–54 | **567–576** |

> The dense 27B lost a few t/s moving from `--spec-draft-n-max 3` to `2`, but gained prefill throughput.
> On this box `n-max 2` was kept because it measured a higher acceptance rate in mixed production traffic;
> if you run mostly long single-turn generations, `n-max 3` is worth re-testing.

---

### 1️⃣ Qwen3.6-27B Q4_K_M — Dense Model (2026-07 data, build 9870)

> **Optimization path: MTP speculative decoding** — the R9700's 256-bit bus is the bottleneck for dense 27B. MTP boosts effective throughput by accepting ~2 tokens per step. This was the original optimization explored in the project.

#### Before optimization (old llama.cpp, no MTP, old Mesa)

| Test | t/s | Notes |
|---|---|---|
| tg (no MTP) | ~20 | Memory-bandwidth bound dense model |
| pp (1226 tokens) | ~525 | |

#### After optimization (latest llama.cpp + RADV_DEBUG=nocompute, MTP `-b 16384 -ub 2048`, `--spec-draft-n-max 3`)

| Context | Metrics | pp (t/s) | tg (t/s) |
|---|---|---|---|
| **Short** (pp=11) | gen=16 | 52 | 45 |
| | gen=197 | 53-71 | 60-63 |
| **Medium** (pp=35-170) | gen=16 | 110-117 | 57 |
| | gen=256-512 | 53 | 55-61 |
| **Long** (pp=490) | gen=16 | **489** | 55 |
| | gen=256-512 | 52-69 | 59-60 |

> **~3× speedup** vs no-MTP baseline (~20 t/s). MTP acceptance rate 95–97%.

#### Optimal llama-server flags (27B)

```bash
llama-server \
  --model Qwen3.6-27B-Q4_K_M.gguf \
  --n-gpu-layers 999 \
  --ctx-size 204800 \
  --parallel 1 \             # MTP requires single sequence
  --flash-attn on \          # required with large -b
  -b 16384 \                 # flash-attn REQUIRES large batch
  -ub 2048 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --spec-type draft-mtp \    # built-in MTP heads
  --spec-draft-n-max 3       # draft 3 tokens per step
```

> ⚠️ **Critical pairing**: `--flash-attn on` MUST be used with `-b 16384`. Without the large batch, flash-attn drops pp from 525 → 147. Without flash-attn but with large batch, pp also drops to ~133. Only the combination works.

---

### 2️⃣ Qwen3.6-35B-A3B Q4_K — MoE Model (2026-07 data, build 9870)

> **Optimization path: RADV_DEBUG=nocompute + latest llama.cpp** — MoE's sparse activation (~6.5B/token) means MTP is unnecessary. The bottleneck is kernel dispatch overhead on RDNA4, solved by `RADV_DEBUG=nocompute`.

`RADV_DEBUG=nocompute` | `KHR_cooperative_matrix` | `flash-attn on` | No MTP

#### llama-bench results (build 9870)

| Test | t/s | Notes |
|---|---|---|
| **tg128** | **164.6 ± 3.1** | **Fastest recorded R9700 35B result** |
| tg512 | 164.6 ± 3.1 | Stable across batch sizes |
| tg2048 | 165.0 ± 0.2 | Decode speed virtually unchanged |
| pp32 | 521.6 ± 38.7 | Short prefill |
| pp512 | 2901.2 ± 97.8 | Medium prefill |

#### Full server benchmark — llama.cpp build 9870, no MTP

| Context | Metrics | pp (t/s) | tg (t/s) |
|---|---|---|---|
| **Short** (pp=11) | gen=16 | 145 | 106 |
| | gen=161-168 | 197 | 137-145 |
| **Medium** (pp=170) | gen=16 | 500 | 133 |
| | gen=256 | 196 | 143 |
| | gen=512 | 216 | 157 |
| **Long** (pp=490) | gen=16 | **2273** | 150 |
| | gen=256 | 219 | 155 |
| | gen=512 | 243 | **160** |

#### Optimization Timeline (35B MoE)

| Stage | tg (t/s) | What changed |
|---|---|---|
| Initial (first report) | ~93 | Default llama.cpp settings |
| Master branch update | ~105 | Updated to latest llama.cpp |
| `RADV_DEBUG=nocompute` | ~127 | Disabled compute queue for RDNA4 |
| `KHR_cooperative_matrix` support | **~138** | Official deployment speed (API) |
| llama-bench peak | **~165** | Benchmark peak with latest build |

> From ~93 to ~165 t/s — **~77% improvement** through software optimizations alone.

#### Optimal llama-server flags (35B MoE)

```bash
llama-server \
  --model Qwen3.6-35B-A3B-Q4_K.gguf \
  --n-gpu-layers 999 \
  --ctx-size 32768 \
  --flash-attn on \
  -b 4096 \
  -ub 512 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0
```

> For 35B MoE, `-b 4096 -ub 512` is optimal. MTP (`--spec-type draft-mtp`) regresses performance to ~105 t/s and is **not recommended** for MoE.

---

### 3️⃣ Qwen3-30B-A3B Q4_K_M — MoE Model (Community Data)

From [llama.cpp discussion #19890](https://github.com/ggml-org/llama.cpp/discussions/19890) benchmarks on the same hardware:

| Test | t/s | Notes |
|---|---|---|
| **tg128** | **183.5 ± 1.0** | ~86% bandwidth utilization |
| pp512 | 3032.6 ± 23.5 | |
| pp1024 | 3009.0 ± 25.2 | |

---

### 🆕 Long-Context Scaling — Qwen3.8-27B Dense + MTP (2026-10-03)

> Depth series on the dense slot. Every depth measured over **≥256 generated tokens** with a unique
> per-request prefix; figures are llama.cpp's own `print_timing` (`prompt eval time` / `eval time`),
> never wall clock. Full method, raw log excerpts and cross-source comparison:
> **[`docs/long-context-scaling.md`](docs/long-context-scaling.md)**.

| Context | Prefill (t/s) | **Decode (t/s)** | MTP acceptance |
|---:|---:|---:|---:|
| 18K | 826 | **45.07** | 1.00 |
| 48K | 666 | **37.41** | 1.00 |
| 93K | 510 | **30.34** | 1.00 |
| 138K | 411 | **25.54** | 1.00 |
| **247K** | 264 | **18.07** | 1.00 |

**Findings**

- **No cliff — decode decays ~−15% per context doubling** (45.1 → 18.1 t/s across 18K → 247K); 247K
  retains 41% of the short-context rate. A **247,001-token** request completed with `truncated = 0`
  (947s end-to-end, full 262144 window).
- **MTP is *stronger* at depth, not weaker.** Acceptance stays at **1.00** (`170 accepted / 170
  generated`, mean accepted length 3.00) at every depth — versus 67–69% on short production traffic.
  The recurrent-state snapshot cost that motivates the usual "MTP hurts at long context" concern does
  not show up here at `--spec-draft-n-max 2`.
- **Prefill is the long-context cost, not decode.** 247K tokens take **15.6 minutes** to ingest before
  the first output token. That is the Dense-vs-MoE architectural gap, not a tuning failure — the MoE
  slot ingests a comparable depth in ~5 min. Route very long documents to the MoE slot; keep the dense
  27B for ≤100K deep-reasoning work.
- ⚠️ **Two benchmark traps this series flushed out** (both have produced false public reports for this
  model family): decode samples under ~100 tokens understate throughput by **2.3×** — at the same 18K
  depth an 8-token sample read **19.9 t/s** while a 256-token sample read **45.1 t/s** — and a multi-minute
  prefill can fail on a Python client when the host runs a **system proxy**: httpx (`trust_env=True` by
  default) and urllib read the macOS system proxy while `curl` does not, so a LAN endpoint gets dialed
  through the proxy and returns `502`/reset. Pass `trust_env=False`, or use `curl`.
- Reproduce with **[`scripts/bench-longctx.py`](scripts/bench-longctx.py)**.

**Cross-check vs community numbers**

| Source | Setup | Decode |
|---|---|---|
| **This box** | R9700 Vulkan, MTP n=2, Q4_K_M, empty ctx (sustained) | **55–56 t/s** |
| **This box** | same, at **18K ctx** | **45.1 t/s** |
| [AlanHuang99/qwen3.6-mtp-stack](https://github.com/AlanHuang99/qwen3.6-mtp-stack) | R9700 Vulkan, MTP **n=3**, Q5_K_M / Q6_K | 53.5 / 52.8 t/s (accept ~72%) |
| [discussion #21043](https://github.com/ggml-org/llama.cpp/discussions/21043) | R9700 RADV, Q4_K_M, **no MTP** | 29.1 t/s (32.5 with ASPM) |
| [club-3090 #94](https://github.com/noonghunna/club-3090/issues/94) | **RTX 3090** CUDA, MTP, **91K prompt** | 47.8 t/s |

> The published R9700 MTP figure (52.8–53.5 t/s at `n=3`, Q5/Q6) sits just below this box at `n=2` with
> the smaller Q4_K_M quant — consistent, since Q4_K_M reads less bandwidth. The RTX 3090's 47.8 t/s at
> 91K prompt scales to ~33 t/s at the R9700's 1.46× lower memory bandwidth, and **30.3 t/s measured at
> 93K lands at ~92% of that parity** — the denser card is not underperforming, the bus is the bus.

---

### 🆕 Draft Width 4 and the DFlash2 Drafter — Qwen3.8-27B Dense (2026-10-04)

> Two claims from the public R9700 HIP reports ([`sklarsa/inference-notes`](https://github.com/sklarsa/inference-notes))
> tested on this box's Vulkan path: **MTP draft width 4** (`--spec-draft-n-max 4 --spec-draft-p-min 0.10`)
> and the **DFlash2 block-diffusion drafter** ([`z-lab/Qwen3.8-27B-DFlash2-GGUF`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF),
> 1.1 GB — build 10820 supports it natively via `--spec-type draft-dflash`). Same protocol as the series
> above: unique prefix, ≥256 generated tokens, server-side `print_timing`. Full method and traps:
> **[`docs/dflash2-vs-mtp-draft-width.md`](docs/dflash2-vs-mtp-draft-width.md)**.

| Prompt | prod MTP n-max 2 | MTP n-max 4 | **DFlash2 n-max 7** |
|---:|---:|---:|---:|
| 1.7K | 48.82 | 63.07 (+29%) | 63.32 (**+30%**) |
| 16K | 44.84 | 55.19 (+23%) | **69.58** (**+55%**) |
| 52K | 36.82 | 43.55 (+18%) | **44.11** (**+20%**) |
| 104K | 29.07 | 32.35 (+11%) | **47.30** (**+63%**) |
| mean accepted length | 3.00 | 4.90 | **7.5** |
| prefill @104K | 465 | 463 | 458 |

- **`--spec-draft-n-max 2` → `4` is +18–29% decode for one flag and zero VRAM.** Acceptance moves 1.00 →
  0.98 while the mean accepted length goes 3.0 → 4.9 — the acceptance rate alone hides the whole gain.
- **DFlash2 at n-max 7 is the long-context winner:** **47.3 t/s at a 104K prompt vs 29.1 on the current
  production config**. It costs +1.85 GB VRAM and ≤2% of prefill. At n-max 3 (what the HIP report used)
  it loses to MTP — block drafting needs its width.
- ⚠️ **Speculative routes are not byte-identical at temperature 0.** The same coding prompt produced
  semantically identical code under MTP and DFlash2 but not byte-identical text (one `import` line moved).
  Gate correctness on semantics, not on a token-hash comparison.
- Reproduce with **[`scripts/bench-depths.py`](scripts/bench-depths.py)**.

---

## Benchmark Reproduction / 本地重现

### llama-bench

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DGGML_CPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-bench

# 35B MoE benchmark
RADV_DEBUG=nocompute GGML_VK_VISIBLE_DEVICES=0 \
  ./build/bin/llama-bench \
    -m /path/to/35B-moe.gguf \
    -n 512 -p 32 -ngl 999 -fa 1 -b 4096 -ub 512 -r 3

# 27B dense benchmark
RADV_DEBUG=nocompute GGML_VK_VISIBLE_DEVICES=0 \
  ./build/bin/llama-bench \
    -m /path/to/27B.gguf \
    -n 512 -p 32 -ngl 999 -fa 0 -b 16384 -ub 2048 -r 3
```

### Server benchmark (the 2026-09-18 numbers above)

[`scripts/bench-server.py`](scripts/bench-server.py) drives a running `llama-server` and reports its own
`timings` across short/medium/long prefill and sustained 256/512-token generation:

```bash
LLAMA_URL=http://localhost:8080 LLAMA_API_KEY=your-key \
  python3 scripts/bench-server.py Qwen3.8-27B-Q4K_M
```

Each request gets a random nonce prefix so the KV cache is never reused, and the long-generation cases use
a prompt that forces the model to keep producing tokens — otherwise the model stops early and the reported
tg is measured over a handful of tokens rather than the requested 256/512.

### Single curl check

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-alias","messages":[{"role":"user","content":"test"}],
       "max_tokens":128,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

Read `timings.prompt_per_second` / `timings.predicted_per_second` in the response.

> ⚠️ With thinking-capable Qwen3.x models, leave `enable_thinking` on and you are benchmarking reasoning
> tokens, not answer tokens — and a small `max_tokens` can return an empty `content` entirely.

---

## ✅ Optimizations That Work / 有效的优化项

### Must-have for all models: `RADV_DEBUG=nocompute` & latest llama.cpp

**Single most impactful setting for RDNA4 gfx1201 Vulkan.** Disables compute queue — routes all work through graphics queue, avoiding a performance regression on RADV.

```yaml
environment:
  - GGML_VK_VISIBLE_DEVICES=0
  - RADV_DEBUG=nocompute
```

Without this: ~120 t/s → With this: **~138 t/s** (+15%). Also pair with the **latest llama.cpp master** for `KHR_cooperative_matrix` support — older builds without it benchmark ~117-120 t/s vs **~165 t/s**.

### ✅ For 27B Dense: MTP Speculative Decoding

| Setting | tg Speedup |
|---|---|
| No MTP (baseline) | ~20 t/s |
| MTP (`--spec-type draft-mtp`, `--spec-draft-n-max 3`) | ~42 t/s (2×) |
| **MTP `--spec-draft-n-max 4 --spec-draft-p-min 0.10`** | **+18–29% over n-max 2 at every depth** |
| **DFlash2 (`-md` drafter, `--spec-type draft-dflash --spec-draft-n-max 7`)** | **+30% short → +63% at 104K** |
| Required pairing | `-b 16384 -ub 2048`, `--parallel 1`, `--flash-attn on` |
| Extra VRAM for DFlash2 | +1.85 GB (1.1 GB drafter + f16 draft KV) |

### ✅ For 35B MoE: Latest llama.cpp + RADV_DEBUG=nocompute

| Setting | tg Speedup |
|---|---|
| Old build, no RADV_DEBUG (baseline) | ~117-120 t/s |
| Latest build + RADV_DEBUG | **~138 t/s (+15-20%)** |
| llama-bench peak | **~165 t/s** |
| MTP | **Regresses** (-24%) — do not use |

### System-level — measured, and **not worth it for long-context work**

Both tweaks were benchmarked with and without, at four depths (256-token decode samples,
server-side timings, same box and build):

| Config | 18K dec | 93K dec | 138K dec | 247K dec | 18K pp | 93K pp | 247K pp |
|---|---|---|---|---|---|---|---|
| Factory (neither) | 45.08 | 30.34 | 25.54 | 18.07 | 818.6 | 510 | 264 |
| ASPM + GPU `high` | **50.94** | **35.59** | **30.13** | **21.76** | 747 | 435 | 229 |
| **ASPM only** | 46.36 | 29.76 | — | — | 795 | 491 | — |

- **The decode gain comes entirely from `power_dpm_force_performance_level=high`**, not
  from ASPM. ASPM by itself is within noise (+2.8% / -1.9%).
- **That same switch costs 9-16% prefill.** Peak prefill drops 264 -> 229 t/s at 247K,
  i.e. **15.6 -> 18.0 minutes** to ingest a full window.
- Net for long-context work: **wait ~2.4 extra minutes at 247K to save ~9 seconds per 1k
  generated tokens.** Not worth it - reverted to factory, and the post-revert run
  (44.58 t/s / 819 t/s) matches the factory baseline.

The `+10%` circulated in [discussion #21043](https://github.com/ggml-org/llama.cpp/discussions/21043)
is likely the GPU-clock effect attributed to ASPM, or a different driver/workload
combination - it did **not** reproduce here.

```bash
# If you still want the decode boost and rarely ingest long prompts:
echo high | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
# Revert:
echo auto | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
```

Resets on reboot. See [`scripts/system-optimize.sh`](scripts/system-optimize.sh) for a persistent setup.

---

## ❌ Optimizations That Don't Work / 无效或负优化项

| Optimization | 27B (dense) | 35B (MoE) | Reason |
|---|---|---|---|
| **MTP for MoE** | N/A | tg **-24%** (138→105) | MoE MTP head adds overhead; acceptance rate too low |
| AMDVLK driver | pp degrades | pp degrades | Prefill drops significantly |
| ROCm backend (HIP) | Slower | Slower | RDNA4 RADV Vulkan ~20% faster than ROCm HIP |
| Turboquant (turbo2/3/4) | Slower | Slower | CPU decompression bottleneck, GPU util ~30% |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | No effect | No effect | Already covered by `RADV_DEBUG=nocompute` |
| Dual GPU | tg **-25%** | tg **-25%** | PCIe split hurts decode more than it helps |

---

## Docker Setup / Docker 部署

### Build the image

```bash
# Clone latest llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Configure with Vulkan + CPU
cmake -B build -DGGML_VULKAN=ON -DGGML_CPU=ON -DCMAKE_BUILD_TYPE=Release

# Build the server binary
cmake --build build -j$(nproc) --target llama-server

# Build minimal Docker image
cat > Dockerfile << 'EOF'
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libc6 libstdc++6 libvulkan1 && rm -rf /var/lib/apt/lists/*
COPY build/bin/llama-server /app/llama-server
COPY build/bin/libggml-*.so* /app/
COPY build/bin/libllama*.so* /app/
ENV GGML_VK_VISIBLE_DEVICES=0
ENV RADV_DEBUG=nocompute
WORKDIR /app
ENTRYPOINT ["/app/llama-server"]
EOF

docker build -t llama-cpp-vulkan .
```

### Docker Compose

See [`docker-compose.yml`](docker-compose.yml) for two configs:
- `llama-35b` — MoE model, no MTP, `-b 4096 -ub 512`
- `llama-27b` — Dense model, MTP enabled, `-b 16384 -ub 2048`

---

## Optimization Impact Summary / 优化效果汇总

### 27B Dense Model

| Optimization | tg | pp | Notes |
|---|---|---|---|
| **MTP spec decoding** | **+100%** | — | Biggest single gain for dense models |
| **MTP draft width 4** (`--spec-draft-n-max 4 --spec-draft-p-min 0.10`) | **+18–29%** | ~0 | Free: one flag, no VRAM, mean accepted length 3.0 → 4.9 |
| **DFlash2 drafter, `--spec-draft-n-max 7`** | **+30% → +63% at 104K** | −≤2% | 1.1 GB drafter, +1.85 GB VRAM; the long-context winner |
| PCIe ASPM performance | ~0 | −3% | **No gain on this box** — see System-level |
| GPU power high | **+13–20%** | **−9–16%** | Trades prefill for decode — skip for long context |
| `--flash-attn on` + `-b 16384` | — | significant | Must use together |

### 35B MoE Model

| Optimization | tg | pp | Notes |
|---|---|---|---|
| **Latest llama.cpp master** | **+40%** | +15% | `KHR_cooperative_matrix` support |
| **`RADV_DEBUG=nocompute`** | **+15%** | Minor | RDNA4 gfx1201 must-have |
| PCIe ASPM performance | ~0 | −3% | **No gain on this box** — see System-level |
| GPU power high | **+13–20%** | **−9–16%** | Trades prefill for decode — skip for long context |
| `--flash-attn on` | negligible | significant | Standard recommendation |
| MTP | **-24%** | N/A | Only for dense models |

---

## References / 参考

- [llama.cpp Discussion #19890 — R9700 Performance Study](https://github.com/ggml-org/llama.cpp/discussions/19890)
- [llama.cpp RDNA4 Experiments Discussion #21043](https://github.com/ggml-org/llama.cpp/discussions/21043)
- [llama.cpp PR #27342 — DFlash2 speculative decoding](https://github.com/ggml-org/llama.cpp/pull/27342)
- [sklarsa/inference-notes — single-GPU Qwen3.8-27B on the R9700 (HIP)](https://github.com/sklarsa/inference-notes/blob/main/reports/2026-08-25-qwen3.8-27b-r9700.md)
- [z-lab/Qwen3.8-27B-DFlash2-GGUF — block-diffusion drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)
- [llama.cpp Vulkan Backend](https://github.com/ggml-org/llama.cpp)
- [RADV Mesa Driver](https://docs.mesa3d.org/drivers/radv.html)
- [AMD Radeon AI PRO R9700 Specs](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/r9700.html)

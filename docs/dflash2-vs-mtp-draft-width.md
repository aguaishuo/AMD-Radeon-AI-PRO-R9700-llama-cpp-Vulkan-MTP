# Speculative draft width and the DFlash2 drafter — Qwen3.8-27B on the R9700

**Measured 2026-10-04** · llama.cpp **build 10820** (`74a7c897f`, Vulkan) · RDNA4 `gfx1201` ·
`RADV_DEBUG=nocompute` · `--ctx-size 262144` · `-b 16384 -ub 2048` · KV `q4_0/q4_0` ·
Qwen3.8-27B-ABLITERATED Q4_K_M · one sequence, one GPU.

Every depth was measured over **≥256 generated tokens** with a unique per-request nonce; the figures are
llama.cpp's own `print_timing` lines (`prompt eval time` / `eval time`), never wall clock. A point is
printed only when `n_tokens` matches the prompt actually sent and `truncated = 0`.

> **⚠️ Read the [Boundary section](#-boundary-which-workload-the-drafter-actually-wins) before quoting any
> number below.** Every table on this page was measured with "copy the document back verbatim" prompts,
> the friendliest possible workload for a drafter. On realistic prompts — Chinese free-form writing and
> long-document summarisation — the DFlash2 drafter is **26–39% slower** than plain MTP n-max 2, and
> **production was reverted to MTP n-max 2** the same evening.

## Why this measurement exists

Two public R9700 reports (all on the HIP/ROCm backend) claimed:

1. **MTP draft width 4 beats width 2** at short context
   ([`sklarsa/inference-notes` 2026-08-25](https://github.com/sklarsa/inference-notes/blob/main/reports/2026-08-25-qwen3.8-27b-r9700.md)
   — 68.4 t/s at 8K with `--spec-draft-n-max 4 --spec-draft-p-min 0.10`);
2. **DFlash2 wins past ~131K**, and MTP wins at 8K.

Our production dense slot ran `--spec-draft-n-max 2`. Both claims were testable on this box with the
Vulkan path and no new dependencies for (1). The DFlash2 drafter is a 1.1 GB GGUF
([`z-lab/Qwen3.8-27B-DFlash2-GGUF`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)) that
**build 10820 supports natively** (`--spec-type draft-dflash`).

## TL;DR / 结论

| Prompt tokens | **prod** MTP n-max 2 | MTP n-max 4 (`p-min 0.10`) | **DFlash2** n-max 7 |
|---:|---:|---:|---:|
| 1.7K | 48.82 | 63.07 (**+29%**) | 63.32 (**+30%**) |
| 16K | 44.84 | 55.19 (**+23%**) | **69.58** (**+55%**) |
| 52K | 36.82 | 43.55 (**+18%**) | **44.11** (**+20%**) |
| 104K | 29.07 | 32.35 (**+11%**) | **47.30** (**+63%**) |

Prefill is untouched: 632 / 623 / 604 t/s at 52K and 465 / 463 / 458 t/s at 104K — the drafter costs
**≤2%** of prefill, and DFlash2's own draft context uses f16 KV, not the target's q4_0.

| Config | Mean accepted length | Acceptance | Extra VRAM |
|---|---|---|---|
| MTP n-max 2 (prod) | 3.00 (all depths) | 0.99–1.00 | — |
| MTP n-max 4 + `p-min 0.10` | 4.90 (all depths) | 0.98 | — |
| DFlash2 n-max 7 | 7.5–7.9 at ≤52K, 7.5 at 104K | 0.92–0.99 | **+1.85 GB** |
| DFlash2 n-max 3 (`p-min 0.30`) | 3.98 | 0.99 | +1.85 GB |

**Findings**

- **Draft width is the cheapest win on this box: n-max 2 → 4 is +18–29% decode for one flag and zero
  VRAM.** Acceptance barely moves (1.00 → 0.98) but the mean accepted length jumps 3.0 → 4.9, which is
  exactly where the speed comes from — the acceptance rate alone completely hides this
  (`1.00 × n=2` is *slower* than `0.98 × n=4`).
- **DFlash2 is the long-context path.** It holds a mean accepted length of **7.5** at 104K where MTP's
  width-4 gain has decayed to +11%: 47.3 t/s vs 29.1 t/s on the production config = **+63%**.
  The drafter loads on Vulkan in ~3 s and adds 1.85 GB (drafter + its f16 KV).
- **The width matters more for DFlash2 than the 2026-08 report assumed.** That report used
  `--spec-draft-n-max 3` and concluded "llama.cpp MTP wins at 8K"; at n-max 3 we reproduce the same
  result (51.72 t/s at 16K, *below* MTP n-max 4). At upstream's recommended **n-max 7** DFlash2 wins
  from 16K upward and never looks back. Block drafting (`block_size=8, n_extract=5`) is what the
  drafter was built for; capping it at 3 throws the mechanism away.
- **MTP's long-context behaviour is width-dependent, not depth-dependent.** At n-max 2 acceptance is
  1.00 at every depth and the mean length is pinned at 3.00 — the same shape we published in
  [`long-context-scaling.md`](long-context-scaling.md). Widening to 4 preserves that shape (4.90 at
  104K). There is no collapse to explain away.
- ⚠️ **Correctness gate — speculative paths are not bit-identical.** The same temperature-0 coding
  prompt produced semantically identical code under MTP and DFlash2, but not byte-identical output: a
  single `from collections import OrderedDict` line appeared one position earlier under MTP. Draft
  verification changes the target model's batch composition (n_draft+1 tokens per step), and a
  different reduction order can flip a near-tie token. **Do not claim byte-level determinism for
  cross-route greedy output**; gate quality on the semantics you can check.

## Reproduce

```bash
# MTP width 4 (no new files needed)
llama-server -m Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf \
  --n-gpu-layers 999 --ctx-size 262144 --parallel 1 --flash-attn on \
  -b 16384 -ub 2048 --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.10

# DFlash2 (add the drafter)
llama-server -m Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf \
  -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ngld 999 \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --n-gpu-layers 999 --ctx-size 262144 --parallel 1 --flash-attn on \
  -b 16384 -ub 2048 --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-dflash --spec-draft-n-max 7
```

Deep points require three separate llama-server lifecycles (one per config); each 104K point is ~4 min
of prefill plus ~9 s of generation. Use [`scripts/bench-depths.py`](../scripts/bench-depths.py).

## Measurement traps this sweep hit

- **A 65K-token payload cannot be passed as a command-line argument.** `curl -d '<json>'` from a
  subprocess dies with `OSError: [Errno 7] Argument list too long` (E2BIG) — and because the failure is
  raised *before* the request goes out, it looks like a completed sweep with missing deep points. Write
  the body to a file and use `-d @file`.
- **Calibrating filler tokens against the server still drifts.** A fixed Chinese filler line measured
  42.5 tokens/line at 40 lines and ~34 tokens/line at 3,000 lines — the incrementing line counter alone
  costs one extra token per additional digit. Report the `n_tokens` you actually got, not the depth you
  aimed for: the "130K" runs above landed at 104K.
- **Read decode only from `eval time`, never from the new `tg_3s` rolling line.** This build also logs
  `n_gen = 318, tg = 51.82 t/s, tg_3s = 70.72 t/s` mid-request; the 3-second window reads 40% high.
  The authoritative per-request number is the `eval time` line, checked against its token count.

## ⚠️ Boundary: which workload the drafter actually wins (measured the same evening)

**Everything above was measured on "copy the last 40 lines of the document verbatim" prompts.** That is
the friendliest possible workload for any drafter and it flatters every speculative route. A second pass
with realistic prompts — same box, `temperature 0`, ≥256 generated tokens, server-side timings — reverses
the recommendation:

| Workload (generated tokens) | **MTP n-max 2** | MTP n-max 4 | DFlash2 n-max 7 |
|---|---:|---:|---:|
| 中文说明文 · Chinese explanatory prose (400) | **36.98** (acc 0.59) | 29.60 (0.31) | 24.73 (0.15) |
| 小红书种草文案 · social copy (400) | **29.69** (0.39) | — | 18.13 (0.08) |
| 客户报价邮件 · business email (400) | **38.00** (0.62) | — | 28.14 (0.21) |
| 长文档总结 24K prompt · summarise a real document | **32.43** (0.69) | — | 23.52 (0.32) |
| Python 代码 · code generation (400) | 49.22 (0.93) | 56.69 (0.81) | **68.55** (0.80) |
| 数学分步推导 · step-by-step math (400) | 48.45 (0.90) | 55.19 (0.78) | **58.51** (0.64) |
| 逐行抄写 4.4K prompt · copy task (256) | 49.64 (1.00) | 60.13 (0.99) | **73.32** (0.92) |

**Read this before copying any number from this page:**

- **On Chinese free-form writing and on real long-document summarisation, DFlash2 is *slower* than the
  config it was meant to replace** — **12–27% slower at the report's own `n-max 3`**, and 26–39% slower at
  the `n-max 7` this page used (see the width A/B below; the first draft of this section overstated the
  gap by quoting only the n-max 7 runs). Drafter acceptance is 0.18–0.54 vs 0.39–0.74 for MTP. It retires the "+63% at 104K" headline in the TL;DR: that figure came from a
  104K *copy* prompt, and a 24K *summarise* prompt — the slot's actual job — puts DFlash2 at 23.5 t/s
  against MTP n-max 2's **32.4 t/s**.
- **Why the collapse:** the DFlash2 drafter is trained against the official `Qwen/Qwen3.8-27B`, while the
  slot serves an **abliterated** fine-tune. On low-entropy, highly predictable text (code, reformatting,
  verbatim copy) the two agree and acceptance stays high; on free-form prose the fine-tune's distribution
  drifts away from the drafter and acceptance falls off a cliff. **MTP heads ship inside the target
  model, so they do not suffer this.**
- **MTP n-max 4 is also workload-dependent**: +15–25% on code/math/copy, but **−20% on prose** (36.98 →
  29.60) since a wider block wastes drafting+verification on tokens that will not be accepted. The
  acceptance rate still looks respectable (0.31) while the mean accepted length stays at 2.23 — the
  block is simply thrown away. `--spec-draft-n-max 2` remains the best all-round value on this box.
- **This also corrects our own depth series.** `long-context-scaling.md` measured 45.07 t/s at an 18K
  *copy* prompt; the same slot on an 18–24K *summarise* prompt delivers **32.4 t/s**. The series is
  internally consistent for A/B-ing configurations but is **not** a forecast for real work.
- **Production decision (2026-10-04):** the slot was switched to DFlash2, validated end-to-end
  (mmproj + drafter booted, 28.0/32.6 GB VRAM, vision path intact), and then **reverted to MTP n-max 2**
  once the prose numbers landed. The drafter stays on disk for a future code-only route.

### Width matters on prose too — same-session A/B (2026-10-05)

Same box, same session, same throwaway container shape, identical prompts, `temperature 0`, server timings.
This is what makes the previous section's headline honest: **the report's `n-max 3` is 12–21% faster than
the `n-max 7` used above on prose**, so part of the "26–39%" gap was my own mis-set width.

| Workload | **DFlash2 n-max 3** (the report's value) | **MTP n-max 2** | DFlash2 n-max 7 |
|---|---:|---:|---:|
| 中文说明文 · Chinese explanatory prose | 27.55 (acc 0.339, mean 2.02) | **36.71** (0.590, 2.18) | 22.77 (0.150, 2.05) |
| 小红书文案 · social copy | 22.69 (0.184, 1.55) | **31.13** (0.392, 1.78) | 19.85 (0.076, 1.53) |
| 24K 文档总结 · real long-doc summarise | 32.16 (0.535, 2.61) | **36.40** (0.736, 2.47) | 28.84 (0.277, 2.94) |
| prefill @24K | 722 | **759** | 716 |
| prefill @326-token prompt | 119 | **194** | 114 |
| boot VRAM (GB) | 26.2 | **24.8** | 26.8 |

**What actually costs the time — measured, not inferred:**

- **The target model takes the same number of forward passes in all three configs.** 400 generated tokens
  ≈ 99 target steps under both MTP n-max 2 and DFlash2 n-max 7. The deficit is therefore **not** "the
  drafter cannot predict this model's prose" (accepted tokens per step are comparable: 2.18 vs 2.05 prose,
  2.47 vs 2.94 summarise) — it is **the drafter's own extra forward pass every step**. MTP's heads live
  inside the target model and cost almost nothing; a block-diffusion drafter is a second network that must
  run for every verification round.
- **That trade pays off only when acceptance is high.** On prose the drafter buys 0.2–0.6 extra tokens per
  step (not enough to pay for its own forward); on code and verbatim copy it buys 3–5 (mean accepted
  length 7.5–7.9) and wins big.
- **DFlash2 also adds a fixed per-request draft-prefill cost, which short prompts feel most:** prefill drops
  194 → 119 t/s on a 326-token prompt (−39%) versus only 759 → 722 at 24K (−5%). Short-chat traffic pays
  this on every request.
- **Correction to the attribution in the first draft:** the "the drafter was trained on the non-abliterated
  target" explanation was an **inference**, never tested — only the abliterated weights exist on this box.
  It is not needed to explain the measurements above. Testing it would require pulling the original
  `Qwen/Qwen3.8-27B` (~17 GB) and re-running this table.

**The reusable rule: A/B speculative decoders on your own prompt mix, never on a copy task.** If you
only take one number from this page, take the acceptance rate measured on prose.

## Cross-check vs the HIP reports

| | HIP (2026-08, `sklarsa`) | This box (Vulkan, 2026-10) |
|---|---|---|
| 8K / 16K MTP | 68.4 t/s @ n-max 4 | 55.2 t/s @ n-max 4, 69.6 t/s DFlash2 n7 |
| 131K MTP | 37.4 t/s @ n-max 3 | ~32 t/s @ 104K (n-max 4) |
| 131K DFlash2 | 41.0 t/s @ n-max 3 | **47.3 t/s @ 104K (n-max 7)** |
| Prefill @ 131K | 299–305 t/s | 458–465 t/s @ 104K |

The HIP build is ~20% faster at short context on MTP (Q4_1 target, and a local RDNA4 batched-MMVQ patch
that the Vulkan path does not have), while the Vulkan path prefills substantially faster. Their
DFlash2-long-context conclusion reproduced and strengthened here once the draft width was raised.

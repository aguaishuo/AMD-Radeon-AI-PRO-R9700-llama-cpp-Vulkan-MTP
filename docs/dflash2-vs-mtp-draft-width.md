# Speculative draft width and the DFlash2 drafter — Qwen3.8-27B on the R9700

**Measured 2026-10-04** · llama.cpp **build 10820** (`74a7c897f`, Vulkan) · RDNA4 `gfx1201` ·
`RADV_DEBUG=nocompute` · `--ctx-size 262144` · `-b 16384 -ub 2048` · KV `q4_0/q4_0` ·
Qwen3.8-27B-ABLITERATED Q4_K_M · one sequence, one GPU.

Every depth was measured over **≥256 generated tokens** with a unique per-request nonce; the figures are
llama.cpp's own `print_timing` lines (`prompt eval time` / `eval time`), never wall clock. A point is
printed only when `n_tokens` matches the prompt actually sent and `truncated = 0`.

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

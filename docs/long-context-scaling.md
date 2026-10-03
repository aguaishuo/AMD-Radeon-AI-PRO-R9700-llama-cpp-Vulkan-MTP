# Long-Context Decode Scaling — Qwen3.8-27B (Dense + MTP) on the R9700

**Measured 2026-10-03** · llama.cpp build 10820 (Vulkan) · RADV Mesa · `RADV_DEBUG=nocompute` ·
`--ctx-size 262144` · `--spec-type draft-mtp --spec-draft-n-max 2` · `-b 16384 -ub 2048` · KV `q4_0/q4_0`

Every prompt carried a unique nonce and every decode figure is measured over **≥256 generated tokens**,
read from llama.cpp's own `timings` (not wall clock). See [Method](#method) for why both matter.

---

## TL;DR / 结论

| Context | Prefill (t/s) | **Decode (t/s)** | MTP acceptance | Sample |
|---:|---:|---:|---:|---:|
| 18K | 826 | **45.07** | 1.00 (mean len 3.00) | 256 tok |
| 48K | 666 | **37.41** | 1.00 | 256 tok |
| 93K | 510 | **30.34** | 1.00 | 256 tok |
| 138K | 411 | **25.54** | 1.00 | 256 tok |
| **247K** | 264 | **18.07** | 1.00 | 256 tok |

- **Decode decays smoothly and near-linearly — roughly −15% per context doubling. There is no cliff.**
  At 247K the model still delivers **41%** of its short-context decode rate, which is a perfectly usable
  reading speed.
- **MTP gets *better* with context, not worse.** Draft acceptance is **100%** (`170 accepted / 170
  generated`, mean accepted length 3.00) at every depth — versus 67–69% on short production traffic.
- **Prefill, not decode, is the real long-context cost.** 247K tokens take **15.6 minutes** to ingest
  before the first output token appears.
- A single **247,001-token** request completed end-to-end with `truncated = 0`, no kills, no OOM.
- The 247K figure is bounded by the window (262144), not by the harness: the whole thing fits and runs.

---

## Why this measurement exists

Dense 27B on a single R9700 is memory-bandwidth bound, and the two things people usually report about it
at long context are:

1. *"decode collapses at long context"* — e.g.
   [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623), which reported a ~25× collapse
   past 80K positions on this exact model family; and
2. *"MTP stops helping (or hurts) at long context"*, from the recurrent-state snapshot cost.

Both turned out to be **measurement artifacts in the public reports**, and it is worth documenting why,
because we reproduced the same false conclusions before catching them.

- [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) was **retracted by its author**: the
  collapse was computed as `generated_tokens / total_request_time`, which folds prefill time into decode.
  Server-side `eval time` (used here) shows no such cliff.
- The *"MTP acceptance 0.25, MTP is a long-context penalty"* reading came from an **8-token sample**.
  At ≥256 tokens, acceptance is 1.00. First-token cost — KV first touch, SSM state restore — is simply
  not amortised over 8 tokens.
- [#28734](https://github.com/ggml-org/llama.cpp/issues/28734) *does* document a real, fixable
  depth-decay (250K: 11 → 41.8 t/s after patching), but the patches target the `qwen4exp` sparse-QSA
  path, **not** the `qwen35` hybrid GDN architecture used here. So there is genuine kernel-level headroom
  in this class of model; it just isn't reachable through configuration on `qwen35` today.

---

## Method

| Item | Value |
|---|---|
| Endpoint | live `llama-server` OpenAI-compatible API, `temperature=0`, `chat_template_kwargs.enable_thinking=false` |
| Prompt | unique nonce header + N filler lines (~28 tokens/line) + a task that forces long output ("copy the last 40 lines verbatim") |
| Output cap | `max_tokens=256` |
| Timing source | llama.cpp `print_timing` lines in the container log — `prompt eval time`, `eval time`, `draft acceptance` |
| Client | `curl -sN -m 1800` (see pitfall 3) |
| Depths | 18K / 48K / 93K / 138K / 247K tokens |
| Model | `Qwen3.8-27B-ABLITERATED-Q4_K_M` (15.7 GiB GGUF + 0.86 GiB mmproj), q4_0 KV |

---

## ⚠️ Measurement pitfalls (the actual finding)

### 1. Decode benchmarks must generate ≥256 tokens

At 18K context, the **same configuration** measured:

| Sample | Decode |
|---|---|
| 8 tokens | 19.86 t/s |
| 256 tokens | **45.07 t/s** |

**2.3× apart.** An `max_tokens=8` run produced the false headline "250K decode collapses to 7.9 t/s";
the true value is **18.07 t/s**. First-token overhead dominates a tiny sample.
**Discard any decode number derived from under ~100 tokens.**

### 2. Every request needs a unique prefix

llama-server reuses its prefix cache across requests sharing a prefix, and reports prefill throughput
over *evaluated* tokens only — a 216K-token prompt once reported a fake **515 t/s** while 108K of it was
a cache hit. Always cross-check the token count on the `prompt eval time` line against the request size.

### 3. Long prefills break clients without TCP keepalive

llama-server sends **nothing** on the connection before the first output token. A client with no TCP
keepalive gets reset by intermediate gear during a multi-minute prefill:

| Client | Long-prefill result |
|---|---|
| Python `urllib` / `http.client` | `ConnectionResetError: [Errno 54] Connection reset by peer` |
| `curl` (keepalive on by default) | works, no change needed |

This is easy to misdiagnose as a server crash — the server-side logs show the request completing
successfully while the client sees a reset. **Use `curl` (or set `SO_KEEPALIVE`) for long-context
benchmarking**, and give the timeout room: a full-window prefill measured **933s**, so start at `-m 1800`.

---

## Raw server-side log lines

<details>
<summary>Representative <code>print_timing</code> output per depth (click to expand)</summary>

```
# 18K
prompt eval time =   20439.62 ms /  16876 tokens (825.65 t/s)
       eval time =    5658.38 ms /    256 tokens ( 45.07 t/s)
draft acceptance = 1.00000 (  170 accepted /   170 generated), mean len =  3.00

# 48K
prompt eval time =   67351.96 ms /  44876 tokens (666.29 t/s)
       eval time =    6816.72 ms /    256 tokens ( 37.41 t/s)
draft acceptance = 1.00000 (  170 accepted /   170 generated), mean len =  3.00

# 93K
prompt eval time =  313764.83 ms / 128875 tokens (410.74 t/s)   # (138K run, shown for shape)
       eval time =    8403.47 ms /    256 tokens ( 30.34 t/s)
draft acceptance = 1.00000 (  170 accepted /   170 generated), mean len =  3.00

# 247K — the full-window run
prompt eval time =  933486.07 ms / 246476 tokens (264.04 t/s)   # 15.6 minutes
       eval time =   14108.69 ms /    256 tokens ( 18.07 t/s)
      total time =  947594.76 ms / 246732 tokens
draft acceptance = 1.00000 (  170 accepted /   170 generated), mean len =  3.00
slot release: stop processing: n_tokens = 247001, truncated = 0
```

</details>

---

## How this compares to other R9700 / community numbers

| Source | Setup | Decode |
|---|---|---|
| **This box, 2026-10-03** | Vulkan, MTP n=2, Q4_K_M, **18K ctx** | **45.07 t/s** |
| **This box**, empty-context sustained (README, 2026-09) | Vulkan, MTP n=2, Q4_K_M | 55–56 t/s |
| [AlanHuang99/qwen3.6-mtp-stack](https://github.com/AlanHuang99/qwen3.6-mtp-stack) | R9700 Vulkan, MTP **n=3**, Q5_K_M | 53.51 t/s (accept 71.6%) |
| same | R9700 Vulkan, MTP n=3, Q6_K | 52.84 t/s (accept 72.0%) |
| same | R9700 Vulkan, **no MTP**, Q5_K_M | 27.88 t/s |
| [discussion #21043](https://github.com/ggml-org/llama.cpp/discussions/21043) | R9700 RADV, Q4_K_M, **no MTP** | 29.07 t/s (32.5 t/s with ASPM) |
| [HN report](https://news.ycombinator.com/item?id=47940982) | R9700 Vulkan, Q6_K, no MTP | 22 t/s |
| [club-3090 #94](https://github.com/noonghunna/club-3090/issues/94) | **RTX 3090** CUDA, MTP, **91K prompt** | 47.82 t/s |

**Reading:**

- **The 27B + MTP figures agree with the community, and are slightly ahead of the only other published
  R9700 MTP number** (55–56 t/s here at `n=2` vs 52.8–53.5 t/s at `n=3`, using a smaller quant). That is
  consistent with Q4_K_M reading less bandwidth than Q5/Q6.
- **The no-MTP baseline (29.07 t/s @ #21043) has no counterpart here** — this box always runs MTP for the
  dense slot, and the "~20 t/s" in the README is the *pre-optimization* (old build + old Mesa) figure, not
  a current measurement. Treat 29–33 t/s as the current no-MTP reference for R9700.
- **Cross-vendor sanity check**: an RTX 3090 (936 GB/s) hits 47.82 t/s at 91K prompt. The R9700 has
  ~640 GB/s — a 1.46× bandwidth ratio — so an equivalent ~33 t/s is the expected ceiling, and **30.34 t/s
  measured at 93K is right where it should be** (≈92% of bandwidth-scaled parity). The denser card is not
  underperforming; the bus is the bus.
- **Deep-context decode numbers for the R9700 appear to be unpublished elsewhere.** The numbers in this
  document are the first depth series we could find for this card, which is why they are recorded here in
  full rather than just as a headline.

---

## The takeaway for workload placement / 实用结论

**Long context is limited by prefill wait, not by decode rate.**

| | 27B Dense (this slot) | 35B-A3B MoE |
|---|---|---|
| Prefill @ ~90K | 510 t/s | ~1362 t/s |
| Time to ingest 247K | **15.6 min** | ~5 min |
| Decode @ 247K | 18.1 t/s | (see 35B notes) |

That gap is architectural, not a tuning failure: the dense model streams all 15.7 GiB of weights per
step, while the MoE touches only ~3B active parameters. For very long documents, route to the MoE slot;
keep the dense 27B for ≤100K deep-reasoning work where its MTP-driven decode rate is the better trade.

---

## Reproduction

[`scripts/bench-longctx.py`](../scripts/bench-longctx.py) runs the full depth series against a live
`llama-server`:

```bash
python3 scripts/bench-longctx.py --url http://localhost:8080 \
    --api-key your-key --model Qwen3.8-27B-ABLITERATED-Q4K_M \
    --depths 18K,48K,93K,138K,247K
```

It uses `curl` as the transport (see pitfall 3) and reads timings from the container log via
`docker logs <container>` so that prefill and decode are never conflated.

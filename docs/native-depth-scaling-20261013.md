# Native 27B depth sweep: standard vs abliterated, 8K–128K

**Measured 2026-10-13**, native Ubuntu Linux, one Radeon AI PRO R9700.
This is a separate follow-up to the [four-slot migration](native-vllm-four-slots-20261013.md).
The pinned stack, weights, 163840 context limit and GPU coordinator are unchanged.

## Results / 四档结果

All **16 speed samples** completed with exact input lengths, no truncation, at least 256 output
tokens, matching telemetry counts and **zero prefix-cache hit tokens**. Tables show means of two runs.
In this synthetic long-document summary task, the abliterated model generated **22–43% faster**
than the standard model. Engine PP differed by only about **1–3%**; these small samples do not
establish a reliable prefill-performance advantage for either checkpoint.

### Standard 27B / 标准版

| Input | Engine PP (token/s) | TTFT (s) | Decode (token/s) | Whole response (s) | Actual output tokens | Six facts correct | Raw JSON valid |
|---|---:|---:|---:|---:|---|---:|---:|
| 8K | 3282.12 | 2.67 | 59.20 | 11.74 | 503 / 560 | 2/2 | 0/2 |
| 32K | 2928.81 | 11.45 | 53.37 | 21.60 | 576 / 501 | 2/2 | 1/2 |
| 64K | 2613.10 | 25.35 | 49.07 | 36.59 | 538 / 565 | 2/2 | 1/2 |
| 128K | 2129.95 | 61.84 | 46.34 | 72.93 | 556 / 468 | 2/2 | 2/2 |

### Abliterated 27B / 无审查版

| Input | Engine PP (token/s) | TTFT (s) | Decode (token/s) | Whole response (s) | Actual output tokens | Six facts correct | Raw JSON valid |
|---|---:|---:|---:|---:|---|---:|---:|
| 8K | 3377.03 | 2.70 | 72.39 | 10.07 | 476 / 593 | 1/2 | 2/2 |
| 32K | 2957.58 | 11.32 | 76.08 | 17.47 | 401 / 532 | 2/2 | 1/2 |
| 64K | 2680.44 | 24.73 | 66.49 | 32.00 | 475 / 492 | 2/2 | 2/2 |
| 128K | 2169.89 | 60.71 | 62.78 | 67.56 | 453 / 404 | 2/2 | 1/2 |

“Six facts correct” uses the separate semantic postcheck described below. “Raw JSON valid”
requires a directly parsable response with no enclosing Markdown fence. Both scores are retained.
Generated summaries are different texts/lengths, so whole-response differences are not an
identical-output latency comparison. These results do not replace the earlier prose/code samples.

### Observations and answer checks

- From 8K to 128K, standard engine PP fell **35.1%** (3282 → 2130 token/s), and abliterated PP
  fell **35.7%** (3377 → 2170 token/s). Sixteen times more input took roughly 23 times the TTFT.
- Both checkpoints reached roughly **61 s TTFT at 128K**. Native context handling is still
  length-dependent; there is no flat-speed claim across the full window.
- Standard: **8/8 responses recovered all six facts**, but **4/8 had enclosing JSON code fences**.
- Abliterated: **7/8 recovered all six facts**. One 8K answer expanded the short order ID
  `BENCH-a073d672` to `BENCH-a073d672c1bc404aa3039d3a13173185`, borrowing the system UUID suffix.
  The other five fields were correct. **2/8 had enclosing JSON code fences**; their facts were correct.
- The semantic postcheck strips **only an enclosing Markdown JSON fence** and compares the
  six fields exactly. It does not repair the wrong order ID. Original responses, original strict
  validator flags and derived semantic flags are all retained in the
  [machine-readable record](../results/2026-10-13-native-depth-scaling.json).
- Order IDs intentionally use the first eight characters of their paired system test UUID.
  This fixture characteristic limits generalisation of the observed order-ID error. Summary
  prose itself has not received a comprehensive factual/semantic review.

### Runtime observations

- standard: sampled peak VRAM **29.301 GiB**, minimum free **2.558 GiB**, minimum available host RAM **3.751 GiB**.
- abliterated: sampled peak VRAM **29.301 GiB**, minimum free **2.558 GiB**, minimum available host RAM **15.548 GiB**.

Memory sampling interval was 0.5 s over each model's preparation/load/sweep, not a guarantee
that every transient spike was captured. Final audit: **kernel boot OOM count 0**, all inspected containers non-OOM, standard model
clean exit 0, original abliterated slot restored through the coordinator (189.333 s including
loading and its short reply). Exactly four catalog entries remain and the coordinator source
hash is unchanged; no production configuration or model-limit change was made.
The standard-model warm-up included a **193.269 s cold switch**, excluded from the depth table.
The abliterated model was already loaded at the start; its short warm-up took 0.125 s.
The two checkpoints were tested in grouped order, abliterated first and standard second,
not alternating at every depth; thermal/background effects are not fully controlled.

## Method / 测试口径

- **K=1024.** Input is calibrated to exactly 8192, 32768, 65536 and 131072 tokens,
  including the system message and chat template. These are input lengths, not total windows.
- Two runs per depth/model, temperature 0, thinking disabled, one sequence and **1536 output budget**.
  Actual output lengths are retained. This differs from the earlier 1024-budget workload series.
- Both checkpoints receive **identical user text and system UUID at each paired point**.
  Each other point has a different UUID at the beginning to avoid prefix-cache reuse.
  Input is separately tokenized against each checkpoint; mismatches reject the comparison.
- Synthetic construction-document filler is arranged around six project facts at the start,
  middle and end. The model must produce JSON with the six facts and a roughly 650-character
  Chinese summary. It must summarise, not copy the filler. Summary length is an instruction,
  not a fixed generated-token count or a semantic-quality score.
- All tokenization, inference and telemetry requests go through the coordinator. Each model
  is warmed first, and loading is excluded from the depth timings. Work is grouped by model
  to avoid a cold load between every paired point.

Three separate metrics are reported:

1. **Engine PP:** actual input tokens divided by the increment in vLLM's
   `request_prefill_time_seconds_sum`. Before/after telemetry must show exactly one finished
   request, matching input/output token deltas and **zero prefix-cache hit tokens**.
   This measures the engine's request prefill phase; it is not a hardware-kernel-only timer.
2. **TTFT:** client request start → first generated content. It includes processing, routing
   and prefill, so input/TTFT is a different effective rate and is retained separately in JSON.
3. **Decode:** actual completion tokens divided by client time after the first generated
   content until SSE stream end. At least 256 completion tokens are required for a speed sample.
   A rate is computed per run, then the two rates are averaged, rather than dividing pooled totals.

Raw JSON validity and exact matches for all six factual fields are checked separately from
speed. The runtime validator is strict about raw JSON; the exported record additionally retains
a semantic postcheck after removing only an enclosing Markdown JSON fence. No claim is made about general summary quality, arbitrary OCR, video or multi-image input.
This sweep tests **text only** and stays below the fixed 163840 total input/output capacity.

## Reproduce

Set `LLAMA_API_KEY` in the environment. With your existing coordinator serving these exact
canonical model IDs, run the two checkpoints with the **same fixture file**:

```bash
python3 scripts/bench-native-depths.py \
  --model Huihui-Qwen3.8-27B-abliterated-MXFP4-DFlash \
  --fixtures /tmp/native-depth-fixtures.json --out /tmp/abliterated-depths.json

python3 scripts/bench-native-depths.py \
  --model Qwen3.8-27B-MXFP4-DFlash \
  --fixtures /tmp/native-depth-fixtures.json --out /tmp/standard-depths.json
```

The first invocation saves calibrated prompt recipes; the second reconstructs the same prompts
and rechecks token counts. Recipe hashes, UUIDs, expected synthetic facts, output lengths and
telemetry deltas are saved. The script never starts/stops containers, flushes caches or executes
generated code. Output files are private and existing evidence is not overwritten.
The coordinator must support model-selected read-only GET `/metrics` bodies and POST `/tokenize`.
These are requirements of this measured coordinator, not assumptions about every OpenAI-compatible API.

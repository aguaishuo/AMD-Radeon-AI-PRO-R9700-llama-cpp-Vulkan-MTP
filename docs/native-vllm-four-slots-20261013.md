# R9700: native vLLM/Radiance + DFlash and four model slots

**Current HP production path for the two 27B slots (since 2026-10-13).**
This is **not** the 2026-10-04 Vulkan/`--spec-type draft-mtp` rollback, and **not** root
`docker-compose.yml` (that file remains a historical Vulkan template).

**Measured 2026-10-13, Ubuntu native Linux, one R9700 (`gfx1201`).**
The final routing checks completed around 11:45 China Standard Time (UTC+8).
This extends the earlier Vulkan reports; it does not erase their dated measurements.

**Follow-up:** [exact 8K / 32K / 64K / 128K native depth sweep](native-depth-scaling-20261013.md),
16 additional samples with engine PP, paired prompts and retained answer/format errors.

本轮两个 27B 已切换到 vLLM/Radiance+DFlash，35B 和 VL-30B 保留 Vulkan。
中文、代码、长文分别比较；失败输出、冷加载耗时和视觉读数错误也保留。

## Runtime and checkpoints / 运行环境与权重

| Item | Measured configuration |
|---|---|
| GPU / memory | Radeon AI PRO R9700, 31.86 GiB usable VRAM; 32 GB host RAM |
| OS / kernel | Ubuntu 24.04; Linux 7.0.0-38-generic |
| Native implementation | [mike2153/mbea-qwen38-dflash](https://github.com/mike2153/mbea-qwen38-dflash/tree/353a6baf35aaf792066b6a1a1e48caac9cf1571d), commit `353a6baf35aaf792066b6a1a1e48caac9cf1571d` |
| Native image | Locally built `mbea-qwen38-dflash:2026.10.1-linux` |
| Native stack | vLLM 0.27.1, Torch 2.11, ROCm 7.14, Radiance/R4D target kernels; TRITON draft kernels |
| Standard target | AMD Qwen3.8-27B MXFP4 weights |
| Abliterated target | [legion666/Huihui-Qwen3.8-27B-abliterated-Quark-AWQ-MXFP4](https://huggingface.co/legion666/Huihui-Qwen3.8-27B-abliterated-Quark-AWQ-MXFP4/tree/f638acb3eabab87dab6fae99a1489bcb71e3b3b7), revision `f638acb3eabab87dab6fae99a1489bcb71e3b3b7` |
| Target / draft size | ~19.80 GB (~18.44 GiB) target; ~2.12 GB (~1.97 GiB) DFlash2 FP8 draft |
| KV / recurrent caches | FP8 KV; Mamba cache bfloat16, SSM float16 |
| Serving | One sequence; seven draft tokens; prefix cache enabled; async scheduling disabled |
| Capacity checks | Fixed 163840 total tokens; 0.5 GiB reserve and 2.2 GiB runtime overhead retained |
| Vision | Language-only flag disabled; one image; 1024×1024 profiling, max_pixels 1048576; video disabled |

The standard and community checkpoints have matching 1695 tensor names, shapes and dtypes,
and identical tokenizer files. The actual architecture has **64 layers**, despite the community
model card saying 48. These are different weights/quantizations from the old Q4_K_M GGUF:
this is a **serving-stack/checkpoint migration comparison**, not a controlled backend-only A/B.
The abliterated checkpoint SHA256 is
`13518d532b4234bb9121e4184a282f4eb9c11d1eef55bfbd28ece882c9020601`.
The MTP exclusion config was adapted from `.weight` names to module names; weights were unchanged.
No model-repository Python code or `trust_remote_code` was used. “Abliterated” is a community
checkpoint description, not a measured guarantee that it will never refuse a request.

The same official Vulkan image was used for the old 27B and retained 35B, image ID
`sha256:9b67d4b09614453f5511e655aa6a7792ddb16e93ae9116134c910bb1951da645`.
VL-30B retained image ID
`sha256:303ecd6c93300ac4f7a701c581c7cb929d1f6997e7fccae2a6ddd85d01b06c60`.
Their build numbers were not re-established in this session; the earlier build-10820 labels
apply to their dated reports. No driver/kernel or GPU performance-level tuning was performed here.
GPU runtime power control remained `on/active`. All weights and GPU tests were on the Linux host.

## Warm workload comparison / 热态同题对比

Two runs per model/task, `temperature=0`, thinking disabled, SSE through the common coordinator.
The user prompt is identical per task; every run has a new system UUID to avoid prefix-cache reuse.
The main comparison uses `max_tokens=1024`; actual output length and chat templates differ.
All samples generated at least 256 tokens. The document is synthetic, roughly 22.5K prompt tokens
(the old 27B template increases this to roughly 22.8K), with six facts embedded in repeated filler.

**Metric:** per-run completion-token count divided by `(stream end − first generated content)`;
table entries are arithmetic means of the two per-run rates. TTFT includes request processing
and prefill. This is an end-to-end client SSE estimate, **not** llama.cpp server-side `timings`.
Do not combine these values with the older server-timing tables as if they were one series.

| Model / stack | Chinese prose (token/s) | Code (token/s) | Long-doc decode (token/s) | Long-doc TTFT (s) | Long-doc total (s) |
|---|---:|---:|---:|---:|---:|
| Old abliterated 27B, Q4_K_M, Vulkan + MTP | 46.67 | 62.73 | 46.38 | 29.96 | 45.67 |
| Standard 27B, MXFP4, vLLM/Radiance + DFlash | 66.11 | 151.87 | 70.22 | 7.31 | 17.17 |
| Abliterated 27B, MXFP4, vLLM/Radiance + DFlash | 44.53 | **142.70 — invalid code, 0/2 passed** | 78.13 | 7.33 | 16.87 |
| Abliterated 27B, additional code runs, **4096 budget** | — | **114.82 — 2/2 passed** | — | — | — |
| Retained 35B-A3B MoE, Q4_K, Vulkan | 141.75 | 143.02 | 135.73 | 9.42 | 13.57 |

Code correctness was checked with `parse_amount` functional assertions in a CPU-only Docker
sandbox: no network, read-only filesystem, no host mounts, no capabilities, 256 MiB memory,
one CPU and 64 PID limit. Old 27B, standard 27B and 35B each passed 2/2 original cases.
The abliterated 27B failed both original cases: one was truncated at 1024 tokens; the other
finished at 753 tokens but had faulty currency parsing. **Not both failures were truncation.**
Two additional runs with a 4096 output budget finished at 742/783 tokens and passed 2/2.
The new budget and UUIDs change the experiment; this does not establish a universal fix.

Both native models and the Vulkan baselines recovered the four automatically checked document
facts (order, amount, deposit, delivery). This is not a full six-fact or business-quality score.

- Standard 27B prose: about **1.42×** the old 27B; valid code rate about **2.42×**.
- Abliterated 27B prose: about **5% slower** than the old 27B, so no blanket speedup claim.
- Whole-document response: old 45.67 s → standard 17.17 s / abliterated 16.87 s, about **2.66–2.71×** faster;
  much of this is the TTFT reduction from ~30 s to ~7.3 s.
- 35B remains the fastest prose and whole-document option in these samples. No 35B vLLM migration was tested.
- Two runs are a small sample, not a broad quality evaluation or an hour-long load qualification.

The [machine-readable record](../results/2026-10-13-native-four-slots.json) includes every
original and additional speed sample, output lengths, finish reasons, prompt fixtures and
validation counts. Private credentials, network addresses and container-inspection dumps are excluded.

### Repeat the speed measurement

With the endpoint already managed by your coordinator, set `LLAMA_API_KEY` in the shell environment
and use an exact canonical model ID from `/v1/models`:

```bash
python3 scripts/bench-native-sse.py \
  --model Qwen3.8-27B-MXFP4-DFlash \
  --out /tmp/standard-27b-new-run.json

# Separate additional-budget experiment, not mixed into the 1024-budget series:
python3 scripts/bench-native-sse.py \
  --model Huihui-Qwen3.8-27B-abliterated-MXFP4-DFlash \
  --only code --max-tokens 4096 \
  --out /tmp/abliterated-27b-code4096-new-run.json
```

The runner uses the published prompt fixtures, bypasses system proxies for the local endpoint,
does a warm-up outside the measurements, rejects unknown catalog IDs and mismatched response
identities, and writes a new private result file. It rejects incomplete streams/missing usage or
unexpected thinking output; fewer than 256 output tokens are marked insufficient. It **does not
execute generated code** or automatically assign a correctness pass. Preserve the response and
run functional assertions separately in an isolated sandbox before presenting “valid code” rates.

## Context and vision / 窗口与图片

| Fixed-window mixed text/image test | Abliterated 27B | Standard 27B |
|---|---:|---:|
| Actual input tokens, including one image | 157987 | 157987 |
| Completion / total tokens | 57 / 158044 | 57 / 158044 |
| Start/middle/end marker + image order ID | All correct | All correct |
| Total response (s) | 80.220 | 80.353 |
| Peak sampled VRAM (GiB) | 29.279 | 29.280 |
| Minimum sampled free VRAM (GiB) | 2.580 | 2.580 |
| GTT used (GiB) | 0.028 | 0.028 |
| Minimum available host memory (GiB) | 16.148 | 16.602 |

Production capacity is **163840 total tokens**, including prompt, image and output tokens.
The model's 262144 architectural limit is unchanged. Earlier experimental text-only 196608
settings are **not** the final production setting. Memory was sampled every 0.5 s, so transient
spikes may be missed. No kernel OOM events were observed in the checked boot.
Multi-image, video and full 262144 contexts were not qualified for this configuration.

After production reload, both native 27B models correctly read all seven fields in a synthetic
1024×1024 image (order ID, amount, deposit, date, red circle count, blue square count, green triangle).
Warm responses took 1.867 / 1.873 s. The retained VL-30B read amount **180000 as 1800000**
and deposit **60000 as 600000** on the same fixture. Its other fields were correct; its weights
and runtime were unchanged. This capability gap is retained in the record, not counted as a pass.
Exact financial OCR needs verification; one correct image does not prove general OCR accuracy.

## Four slots and switching / 四槽与切换

| Slot | Model / backend | Total context | Intended use |
|---|---|---:|---|
| A | Qwen3.6-35B-A3B abliterated, Vulkan | 131072 | Fast text and batch summarisation |
| B | Qwen3-VL-30B, Vulkan | 32768 | Scene and visual-element analysis; numeric OCR limitation above |
| C | Huihui Qwen3.8-27B abliterated MXFP4, vLLM/Radiance + DFlash | 163840 | Long documents, reasoning, single-image requests |
| D | Standard Qwen3.8-27B MXFP4, vLLM/Radiance + DFlash | 163840 | Code, long documents, single-image requests |

These are four selectable slots, **one GPU model active at a time**. ComfyUI image generation
and voice work share the GPU through a separate lease; they are not a fifth resident LLM.
The coordinator still unloads after five idle minutes. Old aliases resolve to the new native
27B model identities; the historical `code-dflash` alias now selects standard slot D.

| Load measurement: request through coordinator → short reply | Seconds |
|---|---:|
| Abliterated: first successful cached fixed-window configuration | 224.740 |
| Standard: first successful cached fixed-window configuration | 230.809 |
| Abliterated: production reload after releasing Comfy reservation | 204.945 |
| Standard: production reload from VL-30B | 177.713 |
| Old Vulkan 27B warm-up/switch reference | 35.262 |

**Native cold switching still takes about 3–3.5 minutes** in the production reload samples.
The sample paths differ and include loading plus the short reply, not just process startup.
Hot generation gains do not make model switching instantaneous; group work by model where practical.

Startup failures are also retained: an initial container name violated the existing control
identity rule and was rejected before loading. Fresh fixed-163840 compilation for each native
model then exposed only 2.25 GiB of available KV space versus 5.83 GiB required; capacity admission
rejected startup (exit 1, no OOM). A new stopped container using the populated compile cache
passed the **same** capacity/reserve checks. The guards were not relaxed to obtain a pass.
Fresh-cache first-start reliability remains a limitation of this pinned setup.

The native stop path requires authenticated quiescence: current boot identity and nonce,
closed admission, zero API requests, paused scheduler, device synchronization and one worker
must all be acknowledged before SIGTERM. Each slot has its own control endpoint.
This is custom coordination, not a claim that arbitrary vLLM containers can be stopped safely
by the old Vulkan path. Model mounts are read-only; persistent configs and caches are kept on the host.

Routing, aliases, capacity admission, private control isolation, unauthenticated rejection,
single-model exclusivity and Comfy lease begin/cancel were checked. The lease test drained
the standard native model cleanly and reloaded the abliterated model after release.
**No new Comfy image was generated in this final lease audit.** Old GGUF weights were retained
for coordinated rollback; one historical DFlash Vulkan container had exit 139 and was not retested.
Rollback must restore both scheduler and client model aliases; do not keep new names against
an old router that falls back to the 35B slot on unknown model IDs.

#!/usr/bin/env python3
"""Compare known OpenAI-compatible model IDs using the dated SSE prompt fixtures.

The endpoint/coordinator manages GPU loading. This script never starts/stops a container
and never executes model-generated code. Functional correctness requires separate checking.
API credentials are read only from LLAMA_API_KEY and are never included in result files.
"""

import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid


def measure(opener, url, key, model, name, prompt, maximum, timeout):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "内部测试编号 " + uuid.uuid4().hex},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": maximum,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(payload).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    start = time.monotonic()
    first = None
    usage = None
    actual = None
    finish = None
    done = False
    answer, reasoning = [], []
    with opener.open(request, timeout=timeout) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            item = json.loads(data)
            if "error" in item:
                raise RuntimeError("Endpoint returned an SSE error; sample rejected")
            actual = item.get("model") or actual
            usage = item.get("usage") or usage
            for choice in item.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                reason = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if first is None and (content or reason or delta.get("tool_calls")):
                    first = time.monotonic() - start
                answer.append(content)
                reasoning.append(reason)
    elapsed = time.monotonic() - start
    if actual != model:
        raise RuntimeError("Response model identity mismatch; sample rejected")
    if not done or not usage or first is None:
        raise RuntimeError("Incomplete stream/usage/content; sample rejected")
    if any(reasoning):
        raise RuntimeError("Thinking output present despite disabled thinking; sample rejected")
    count = usage["completion_tokens"]
    return {
        "case": name, "requested_model": model, "actual_model": actual,
        "ttft_s": round(first, 3), "wall_s": round(elapsed, 3),
        "prompt_tokens": usage["prompt_tokens"], "completion_tokens": count,
        "decode_tps": round(count / max(elapsed - first, .001), 2),
        "speed_sample_sufficient": count >= 256, "finish_reason": finish,
        "content": "".join(answer), "code_validation": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True, help="Exact canonical ID from /v1/models")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--only", choices=["prose", "code", "longdoc"])
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--out", type=Path, required=True, help="New output file; never overwrites")
    args = parser.parse_args()
    key = os.environ.get("LLAMA_API_KEY")
    if not key:
        parser.error("Set LLAMA_API_KEY in the environment")
    if args.runs < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("Runs, output budget and timeout must be positive")
    if args.out.exists():
        parser.error("Output already exists; choose a new file")
    url = args.url.rstrip("/")
    # Local/LAN benchmark: deliberately bypass system proxies, matching the measured run.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url + "/v1/models", headers={"Authorization": "Bearer " + key})
    with opener.open(request, timeout=30) as response:
        ids = {item["id"] for item in json.load(response)["data"]}
    if args.model not in ids:
        parser.error("Model is not in the advertised catalog; refusing unknown-name fallback")
    record = Path(__file__).resolve().parents[1] / "results/2026-10-13-native-four-slots.json"
    fixtures = json.loads(record.read_text())["prompt_fixtures"]
    # Exclude loading/first-request initialization from the warm workload samples.
    measure(opener, url, key, args.model, "warmup", "只回复：READY", 16, args.timeout)
    result = {"model": args.model, "max_tokens": args.max_tokens, "points": [],
              "note": "SSE speed only; generated code is not executed or automatically validated"}
    os.umask(0o077)
    with args.out.open("x", encoding="utf-8") as output:
        for name, prompt in fixtures.items():
            if args.only and name != args.only:
                continue
            for trial in range(1, args.runs + 1):
                point = measure(opener, url, key, args.model, name, prompt,
                                args.max_tokens, args.timeout)
                point["trial"] = trial
                result["points"].append(point)
                output.seek(0)
                json.dump(result, output, ensure_ascii=False, indent=2)
                output.truncate()
                output.flush()
                print(json.dumps({k: v for k, v in point.items() if k != "content"},
                                 ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

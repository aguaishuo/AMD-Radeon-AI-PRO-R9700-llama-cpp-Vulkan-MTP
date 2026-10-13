#!/usr/bin/env python3
"""Exact-input native depth sweep, through an existing GPU coordinator only.

Requires LLAMA_API_KEY. Shares saved prompt recipes across models. No container
management, backend-direct inference, cache flush, or generated-code execution.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import time
import urllib.request
import uuid

spec = importlib.util.spec_from_file_location("sse_bench", Path(__file__).with_name("bench-native-sse.py"))
sse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sse)
METRICS = [
    "request_prefill_time_seconds_sum", "request_prefill_time_seconds_count",
    "request_decode_time_seconds_sum", "request_decode_time_seconds_count",
    "request_queue_time_seconds_sum", "request_success_total",
    "prompt_tokens_total", "generation_tokens_total", "prefix_cache_hits_total",
    "num_requests_running", "num_requests_waiting",
]


def get_json(opener, url, key, path, body=None, method=None):
    request = urllib.request.Request(
        url + path, None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method=method,
    )
    with opener.open(request, timeout=2400) as response:
        return json.load(response)


def metrics(opener, url, key, model):
    # A GET body selects the model at the coordinator; this is read-only telemetry.
    request = urllib.request.Request(
        url + "/metrics", json.dumps({"model": model}).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key}, method="GET",
    )
    with opener.open(request, timeout=30) as response:
        text = response.read().decode()
    result = {name: 0.0 for name in METRICS}
    seen = set()
    for line in text.splitlines():
        match = re.match(r"vllm:([^\s{]+)(?:\{([^}]*)\})?\s+([-+\deE.]+)$", line)
        if not match or match.group(1) not in result:
            continue
        if 'model_name="' + model + '"' not in (match.group(2) or ""):
            continue
        result[match.group(1)] += float(match.group(3))
        seen.add(match.group(1))
    if not set(METRICS).issubset(seen):
        raise RuntimeError("Expected model metrics missing; refusing unverified PP figures")
    return result


def prompt(recipe):
    n = recipe["filler_lines"]
    expected = recipe["expected"]
    lines = [f"第{i:05d}条：施工资料模板，请核对现场尺寸、安装顺序、材料与客户确认记录，不以模板代替现场证据。" for i in range(n)]
    return (
        f"项目事实（开头）：订单号{expected['order_id']}；订单金额{expected['order_amount']}元。\n"
        + "\n".join(lines[:n // 2])
        + f"\n项目事实（中部）：已收定金{expected['deposit_amount']}元；预计交付{expected['delivery_date']}。\n"
        + "\n".join(lines[n // 2:])
        + "\n长度补齐段：" + " x" * recipe["padding_tokens"]
        + "\n项目事实（末尾）：费用审批尚未结束；公司是否已付款没有凭证。\n"
        + "请只输出一个JSON对象：order_id、order_amount、deposit_amount、delivery_date、approval_status、company_payment_status、summary。"
        + "两个金额用整数；审批状态用pending；公司付款状态用unverified。summary用中文写约650字的项目摘要，"
        + "说明已知事实、待核实事项和施工资料使用边界，不虚构现场情况、不逐行抄写模板。"
    )


def token_count(opener, url, key, model, recipe):
    result = get_json(opener, url, key, "/tokenize", {
        "model": model,
        "messages": [{"role": "system", "content": "内部测试编号 " + recipe["nonce"]},
                     {"role": "user", "content": prompt(recipe)}],
        "chat_template_kwargs": {"enable_thinking": False},
    })
    return result["count"]


def calibrate(opener, url, key, model, depth, trial):
    nonce = uuid.uuid4().hex
    rng = random.Random(nonce)
    recipe = {"depth_kib_tokens": depth, "target_input_tokens": depth * 1024,
              "trial": trial, "nonce": nonce, "filler_lines": 0, "padding_tokens": 0,
              "expected": {"order_id": "BENCH-" + nonce[:8], "order_amount": rng.randrange(150, 250) * 1000,
                           "deposit_amount": rng.randrange(30, 80) * 1000, "delivery_date": "2026-11-18",
                           "approval_status": "pending", "company_payment_status": "unverified"}}
    target = recipe["target_input_tokens"]
    low, high = 0, target // 10
    while low < high:
        middle = (low + high + 1) // 2
        recipe["filler_lines"] = middle
        if token_count(opener, url, key, model, recipe) <= target:
            low = middle
        else:
            high = middle - 1
    recipe["filler_lines"] = low
    for _ in range(8):
        count = token_count(opener, url, key, model, recipe)
        if count == target:
            break
        recipe["padding_tokens"] += target - count
        if recipe["padding_tokens"] < 0:
            raise RuntimeError("Padding calibration became negative")
    else:
        raise RuntimeError("Could not calibrate exact token count")
    recipe["prompt_sha256"] = hashlib.sha256(prompt(recipe).encode()).hexdigest()
    return recipe


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--depths", default="8,32,64,128", help="K=1024 input tokens, including chat template")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("LLAMA_API_KEY")
    if not key:
        parser.error("Set LLAMA_API_KEY in the environment")
    depths = [int(x) for x in args.depths.split(",")]
    if args.runs < 1 or args.max_tokens < 1 or not depths or min(depths) < 1 or len(set(depths)) != len(depths):
        parser.error("Depths, runs and output budget must be positive; depths must be unique")
    if args.out.exists():
        parser.error("Output already exists; preserve previous evidence and choose a new file")
    os.umask(0o077)
    url = args.url.rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    catalog = get_json(opener, url, key, "/v1/models")["data"]
    slot = next((m for m in catalog if m["id"] == args.model), None)
    if slot is None or max(depths) * 1024 + args.max_tokens > slot.get("max_model_len", 0):
        parser.error("Unknown model or requested input/output exceeds advertised capacity")
    result = {"schema_version": 1, "model": args.model, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "max_tokens": args.max_tokens, "context": slot["max_model_len"], "points": []}
    save(args.out, result)
    try:
        before_warm = metrics(opener, url, key, args.model)["request_prefill_time_seconds_count"]
    except Exception:
        before_warm = 0
    warm = sse.measure(opener, url, key, args.model, "warmup", "只回复：READY", 16, 2400)
    result["warmup"] = {k: v for k, v in warm.items() if k != "content"}
    save(args.out, result)
    print(json.dumps({"event": "warmup", "model": args.model, "wall_s": warm["wall_s"]}), flush=True)
    # Wait for telemetry from warm-up to arrive before taking the first baseline.
    for _ in range(120):
        baseline = metrics(opener, url, key, args.model)
        if baseline["request_prefill_time_seconds_count"] >= before_warm + 1 and not baseline["num_requests_running"] and not baseline["num_requests_waiting"]:
            break
        time.sleep(.25)
    else:
        raise RuntimeError("Warm-up telemetry did not become idle")
    if args.fixtures.exists():
        recipes = json.loads(args.fixtures.read_text())["recipes"]
        if [(r["depth_kib_tokens"], r["trial"]) for r in recipes] != [(k, t) for k in depths for t in range(1, args.runs + 1)]:
            raise RuntimeError("Existing fixture layout does not match requested sweep")
    else:
        recipes = []
        for depth in depths:
            for trial in range(1, args.runs + 1):
                recipes.append(calibrate(opener, url, key, args.model, depth, trial))
                print(json.dumps({"event": "calibrated", "input_tokens": depth * 1024, "trial": trial}), flush=True)
        save(args.fixtures, {"generator_version": 1, "recipes": recipes})
    result["fixture_recipes"] = recipes
    save(args.out, result)
    for recipe in recipes:
        text = prompt(recipe)
        if hashlib.sha256(text.encode()).hexdigest() != recipe["prompt_sha256"]:
            raise RuntimeError("Fixture hash mismatch")
        count = token_count(opener, url, key, args.model, recipe)
        if count != recipe["target_input_tokens"]:
            raise RuntimeError("Template/tokenizer differs; paired exact-input comparison rejected")
        before = metrics(opener, url, key, args.model)
        if before["num_requests_running"] or before["num_requests_waiting"]:
            raise RuntimeError("Concurrent work present; do not contaminate the sweep")
        # Use the same system nonce and user text for this point on both checkpoints.
        original_uuid = sse.uuid.uuid4
        sse.uuid.uuid4 = lambda: uuid.UUID(recipe["nonce"])
        try:
            point = sse.measure(opener, url, key, args.model, "depth_summary", text, args.max_tokens, 2400)
        finally:
            sse.uuid.uuid4 = original_uuid
        point.update(depth_kib_tokens=recipe["depth_kib_tokens"], trial=recipe["trial"],
                     expected=recipe["expected"], prompt_sha256=recipe["prompt_sha256"])
        try:
            answer = json.loads(point["content"])
            point["raw_json_valid"] = True
            point["facts_correct"] = {k: answer.get(k) == v for k, v in recipe["expected"].items()}
            point["summary_characters"] = len(answer.get("summary", ""))
        except (ValueError, TypeError, AttributeError):
            point["raw_json_valid"] = False
            point["facts_correct"] = {k: False for k in recipe["expected"]}
            point["summary_characters"] = 0
        result["points"].append(point)
        save(args.out, result)
        for _ in range(120):
            after = metrics(opener, url, key, args.model)
            requests = after["request_prefill_time_seconds_count"] - before["request_prefill_time_seconds_count"]
            if requests >= 1:
                break
            time.sleep(.25)
        delta = {k: after[k] - before[k] for k in METRICS if not k.startswith("num_requests_")}
        point["engine_metric_delta"] = delta
        valid = (requests == 1 and delta["request_success_total"] == 1
                 and delta["prompt_tokens_total"] == point["prompt_tokens"] == count
                 and delta["generation_tokens_total"] == point["completion_tokens"]
                 and delta["prefix_cache_hits_total"] == 0
                 and delta["request_prefill_time_seconds_sum"] > 0)
        point["engine_metrics_valid"] = valid
        point["engine_prefill_tps"] = round(count / delta["request_prefill_time_seconds_sum"], 2) if valid else None
        point["effective_input_per_ttft"] = round(count / point["ttft_s"], 2)
        save(args.out, result)
        print(json.dumps({k: v for k, v in point.items() if k not in ["content", "expected", "prompt_sha256", "engine_metric_delta"]}, ensure_ascii=False), flush=True)
        if not valid:
            raise RuntimeError("Telemetry count/token/cache mismatch; saved sample is not a valid PP measurement")
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.out, result)
    print("DEPTH_SWEEP_COMPLETE " + args.model, flush=True)


if __name__ == "__main__":
    main()

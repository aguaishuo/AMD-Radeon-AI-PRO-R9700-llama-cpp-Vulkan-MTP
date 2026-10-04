#!/usr/bin/env python3
"""Realistic-traffic A/B: is a speculative route faster on *production-like* prompts, not just on
the "copy the last 40 lines" filler that maximises drafter acceptance?

Four prompt types, temperature 0, >=256 generated tokens, server-side timings only:
  P1 long-form Chinese prose   P2 code generation   P3 filler copy (best case reference)   P4 step-by-step math

Usage: python3 bench_realistic.py --port 30003 --key yojohousing123 --container llama-server-nog --tag prod-dflash
"""
import argparse, json, re, subprocess, sys, time

def last(pat, text):
    f = None
    for f in re.finditer(pat, text):
        pass
    return f

RE_PREFILL = r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second"
RE_DECODE = r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) (?:tokens|runs).*?([\d.]+) tokens per second"
RE_ACC = r"draft acceptance =\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+) generated"
RE_MLEN = r"mean len =\s*([\d.]+)"
FILLER = "第%05d行：文档内容占位，用于测试预填充与解码速度，此行不含任何关键信息，仅用于构建长上下文。"

PROMPTS = [
    ("P1-prose", 400, "请用中文写一篇 600 字左右的说明文，介绍日式整体浴室（unit bath）的结构组成、"
                      "防水做法与现场安装顺序，要求分小节、有标题，内容具体到施工细节。", 0),
    ("P2-code", 400, "请写一个 Python 函数 lru_ttl(capacity, ttl)，实现带 TTL 的 LRU 缓存类，"
                     "要求线程安全、O(1) 读写，并给出三个断言测试。只输出代码，不要解释。", 0),
    ("P3-fillercopy", 256, "请把上面资料里最后 40 行逐行原样抄写出来，每行前面加 '- '，不要总结、不要解释、不要省略。", 128),
    ("P4-math", 400, "一个水池有两个进水管和一个出水管。甲管单独注满需 6 小时，乙管单独注满需 8 小时，"
                     "出水管单独排空需 12 小时。若三管同时开启，多久能注满？请分步推导，并检验答案。", 0),
    ("P5-xhs", 400, "你是小红书运营。写一篇 400 字左右的日式整体浴室种草笔记，带 emoji 和话题标签，"
                    "语气亲切，突出收纳、防水、易清洁三个卖点，不要写成硬广。", 0),
    ("P6-quote", 400, "用中文写一封给客户的报价说明邮件：某客户要给 90㎡ 三居室做日门内装整体改造，"
                      "预算 18 万。列出分项、工期、付款节点，语气专业礼貌，不要夸张。", 0),
    # P7 is the 27B slot's actual job: a real long-context task on a Chinese document, NOT a copy task.
    # Copying a document back verbatim is the friendliest possible workload for any drafter and
    # inflates every speculative route -- summarising is the honest one.
    ("P7-longdoc-summary", 400, "上面是一份施工项目资料。请用中文总结它的主要内容，分 5 点，每点 2-3 句，"
                                "再指出资料中重复出现的模板句有哪些。不要逐行抄写。", 700),
]

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=30003)
ap.add_argument("--key", default="yojohousing123")
ap.add_argument("--container", required=True)
ap.add_argument("--model", default="local", help="model id sent in the payload; MUST match the "
                                                 "slot's own alias when going through llama-proxy, "
                                                 "or an unknown name routes to slot A (35B)")
ap.add_argument("--tag", required=True)
ap.add_argument("--only", default="", help="comma-separated prompt names to run (default: all)")
ap.add_argument("--outdir", default="/tmp/bench")
a = ap.parse_args()

OUT = {"tag": a.tag, "points": []}


def run(name, max_tokens, text, filler_lines=0):
    if filler_lines:
        text = "【唯一编号 %d】\n" % time.time() + "\n".join(FILLER % i for i in range(filler_lines)) + "\n\n" + text
    payload = {"model": a.model, "messages": [{"role": "user", "content": text}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    open("/tmp/rp.json", "w").write(json.dumps(payload, ensure_ascii=False))
    t0 = time.time()
    subprocess.run(["curl", "-sN", "-m", "900", "-o", "/tmp/ro.txt",
                    "http://localhost:%d/v1/chat/completions" % a.port,
                    "-H", "Content-Type: application/json",
                    "-H", "Authorization: Bearer " + a.key, "-d", "@/tmp/rp.json"],
                   capture_output=True, text=True)
    wall = time.time() - t0
    lg = subprocess.run(["docker", "logs", "--tail", "2000", a.container],
                        capture_output=True, text=True)
    lg = lg.stdout + lg.stderr
    pp, ev, acc, ml = last(RE_PREFILL, lg), last(RE_DECODE, lg), last(RE_ACC, lg), last(RE_MLEN, lg)
    pt = {"prompt": name, "wall_s": round(wall)}
    if pp:
        pt["prompt_tokens"] = int(pp.group(2)); pt["prefill_tps"] = round(float(pp.group(3)))
    if ev:
        pt["decode_tokens"] = int(ev.group(2)); pt["decode_tps"] = round(float(ev.group(3)), 2)
    if acc:
        pt["accept"] = float(acc.group(1)); pt["accepted"] = int(acc.group(2)); pt["drafted"] = int(acc.group(3))
    if ml:
        pt["mean_len"] = float(ml.group(1))
    pt["trustworthy"] = bool(pt.get("decode_tokens", 0) >= 100)
    OUT["points"].append(pt)
    print(json.dumps(pt, ensure_ascii=False), flush=True)
    open("%s/real_%s.json" % (a.outdir, a.tag), "w").write(json.dumps(OUT, ensure_ascii=False, indent=1))


ONLY = [s for s in (a.only or "").split(",") if s]
for name, mt, text, fill in PROMPTS:
    if ONLY and name not in ONLY:
        continue
    run(name, mt, text, filler_lines=fill)
print("DONE " + a.tag)

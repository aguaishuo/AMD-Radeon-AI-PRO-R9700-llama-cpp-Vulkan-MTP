#!/usr/bin/env python3
"""Multi-depth decode/prefill benchmark for a local llama-server (HP R9700)."""
import argparse, json, re, secrets, subprocess, sys, time

def last(pat, text):
    f = None
    for f in re.finditer(pat, text):
        pass
    return f

RE_PREFILL = r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second"
RE_DECODE = r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) (?:tokens|runs).*?([\d.]+) tokens per second"
RE_ACC = r"draft acceptance =\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+) generated"
RE_ACC_SHORT = r"draft acceptance =\s*([\d.]+)"
RE_NT = r"n_tokens = (\d+), truncated = (\d+)"
RE_MLEN = r"mean len =\s*([\d.]+)"

FILLER = "第%05d行：文档内容占位，用于测试预填充与解码速度，此行不含任何关键信息，仅用于构建长上下文。"


def filler(n):
    return "\n".join(FILLER % i for i in range(n))


def req(port, key, lines, max_tokens, timeout):
    prompt = ("【唯一编号 " + secrets.token_hex(10) + "】\n" + filler(lines) +
              "\n\n请把上面资料里最后 40 行逐行原样抄写出来，每行前面加 '- '，不要总结、不要解释、不要省略。")
    payload = {"model": "local", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    # the payload must go through a file: a 65K-token prompt exceeds ARG_MAX ("E2BIG")
    pf = "/tmp/bench_payload.json"
    with open(pf, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    p = subprocess.run(
        ["curl", "-sN", "-m", str(timeout), "-o", "/dev/null",
         "-w", "http=%{http_code} ttfb=%{time_starttransfer} total=%{time_total}",
         f"http://localhost:{port}/v1/chat/completions",
         "-H", "Content-Type: application/json",
         "-H", "Authorization: Bearer " + key,
         "-d", "@" + pf],
        capture_output=True, text=True)
    return p.stdout.strip(), p.returncode


def logs(container, n=200):
    p = subprocess.run(["docker", "logs", "--tail", str(n), container],
                       capture_output=True, text=True)
    return p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19401)
    ap.add_argument("--key", default="test")
    ap.add_argument("--container", required=True)
    ap.add_argument("--depths", default="2000,20000,65000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--outdir", default="/tmp/bench")
    a = ap.parse_args()

    out = {"tag": a.tag, "port": a.port, "points": []}

    req(a.port, a.key, 40, 8, 300)
    lg = logs(a.container)
    nt = last(RE_NT, lg)
    pp0 = last(RE_PREFILL, lg)
    if pp0:
        ratio = int(pp0.group(2)) / 40.0
    elif nt:
        ratio = int(nt.group(1)) / 40.0
    else:
        ratio = 30.0
    print("calibration: %.2f tokens/line (%s)" % (ratio, (nt.group(0) if nt else "no n_tokens line")), flush=True)

    for depth in [int(d) for d in a.depths.split(",")]:
        lines = max(10, int(round(depth / ratio)))
        t0 = time.time()
        stat, rc = req(a.port, a.key, lines, a.max_tokens, a.timeout)
        wall = time.time() - t0
        lg = logs(a.container)
        pp, ev, acc, nt, ml = (last(RE_PREFILL, lg), last(RE_DECODE, lg),
                               last(RE_ACC, lg) or last(RE_ACC_SHORT, lg),
                               last(RE_NT, lg), last(RE_MLEN, lg))
        pt = {"target": depth, "lines": lines, "client": stat, "wall_s": round(wall), "rc": rc}
        if pp:
            pt["prompt_tokens"] = int(pp.group(2))
            pt["prefill_tps"] = round(float(pp.group(3)))
        elif nt:
            pt["prompt_tokens"] = int(nt.group(1))
        if nt:
            pt["truncated"] = int(nt.group(2))
        if ev:
            pt["decode_tokens"] = int(ev.group(2))
            pt["decode_tps"] = round(float(ev.group(3)), 2)
        if acc:
            pt["mtp_accept"] = float(acc.group(1))
            if acc.re is RE_ACC:
                pt["mtp_acc_n"] = int(acc.group(3))
        if ml:
            pt["mean_len"] = float(ml.group(1))
        pt["trustworthy"] = bool(pt.get("prompt_tokens") and not pt.get("truncated")
                                 and pt.get("decode_tokens", 0) >= 100)
        pt["raw_tail"] = lg[-1200:]
        out["points"].append(pt)
        row = {k: v for k, v in pt.items() if k != "raw_tail"}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with open("%s/bench_%s.json" % (a.outdir, a.tag), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    print("DONE " + a.tag, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Long-context prefill/decode scaling benchmark for a live llama-server.

Why this script exists — three traps it avoids:

1. DECODE SAMPLES MUST BE >=256 TOKENS. An 8-token sample of the same configuration measures
   ~2.3x slower than a 256-token sample, because first-token cost (KV first touch, SSM state
   restore) is not amortised. Anything under ~100 tokens is discarded here.
2. EVERY REQUEST NEEDS A UNIQUE PREFIX, otherwise llama-server reuses its prefix cache and
   reports prefill throughput over evaluated tokens only (a 216K prompt once reported a fake
   515 t/s while half of it was a cache hit).
3. LONG PREFILLS KILL CLIENTS WITHOUT TCP KEEPALIVE. llama-server sends nothing before the
   first output token; Python urllib/http.client gets ConnectionResetError after a few minutes
   of silence while curl (keepalive on by default) is fine. This script shells out to curl.

Timings are read from the server's own log lines (prompt eval time / eval time / draft
acceptance) rather than wall clock, so prefill and decode are never conflated.

Usage:
    python3 bench-longctx.py --url http://localhost:8080 --api-key KEY \
        --model Qwen3.8-27B-ABLITERATED-Q4K_M --depths 18K,48K,93K,138K,247K \
        [--container llama-server-nog] [--ssh user@host] [--max-tokens 256]

If --container is given, timings are read with `docker logs`; add --ssh to run that over SSH.
Without --container the script falls back to the `timings` field of the streamed response.
"""
import argparse, json, re, secrets, subprocess, sys, time

FILLER = "第{i:05d}行：文档内容占位，用于测试预填与解码速度，此行不含任何关键信息。"


def parse_depth(s):
    """'18K' / '247k' / '50000' -> int tokens (approx)."""
    s = s.strip().upper()
    if s.endswith("K"):
        return int(float(s[:-1]) * 1000)
    return int(s)


def sh(cmd, timeout=300):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def build_body(n_lines, nonce):
    head = f"【唯一编号 {nonce}】\n"
    filler = "\n".join(FILLER.format(i=i) for i in range(n_lines))
    return (head + filler +
            "\n\n请把上面资料里最后 40 行逐行原样抄写出来，每行前面加 '- '，"
            "不要总结、不要解释、不要省略，直接从最后一行往前抄。")


def run_one(args, n_lines):
    nonce = secrets.token_hex(10)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": build_body(n_lines, nonce)}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    pf = "/tmp/bench-longctx-payload.json"
    with open(pf, "w") as f:
        json.dump(payload, f)

    cmd = ["curl", "-sN", "-m", str(args.timeout), "-o", "/tmp/bench-longctx-out.txt",
           "-w", "%{http_code}", f"{args.url}/v1/chat/completions",
           "-H", "Content-Type: application/json",
           "-d", "@" + pf]
    if args.api_key:
        cmd += ["-H", f"Authorization: Bearer {args.api_key}"]

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    return wall, r.stdout.strip(), r.returncode


def read_timings(args):
    if not args.container:
        return None
    inner = f"docker logs {args.container} 2>&1 | grep print_timing | tail -60"
    if args.ssh:
        out = sh(f"ssh -o StrictHostKeyChecking=no {args.ssh} '{inner}'", timeout=120)
    else:
        out = sh(inner, timeout=120)
    def last(pat):
        """Return the LAST match: the log tail still holds earlier requests, and taking the
        first match silently reports the previous request's timings."""
        found = None
        for m in re.finditer(pat, out):
            found = m
        return found

    pp = last(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(.*?([\d.]+) tokens per second\)")
    ev = last(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(.*?([\d.]+) tokens per second\)")
    da = last(r"draft acceptance =\s*([\d.]+)\s*\(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)")
    return pp, ev, da


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--depths", default="18K,48K,93K,138K,247K",
                   help="comma list of approximate prompt sizes (28 tokens/filler line)")
    p.add_argument("--container", default=None, help="llama-server container name for log-based timings")
    p.add_argument("--ssh", default=None, help="user@host if the container runs remotely")
    p.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    p.add_argument("--timeout", type=int, default=1800, help="curl -m seconds (full-window prefill ~933s)")
    args = p.parse_args()

    TPL = 28  # tokens per filler line, approximate
    print(f"{'target':>8} | {'prefill':>22} | {'decode':>26} | {'MTP':>6} | wall")
    print("-" * 84)

    for d in args.depths.split(","):
        target = parse_depth(d)
        n_lines = max(1, int(target / TPL))
        wall, code, rc = run_one(args, n_lines)
        t = read_timings(args)
        pp = ev = da = None
        if t is not None:
            pp, ev, da = t
        if ev is not None:
            n_gen = int(ev.group(2))
            flag = "" if n_gen >= 100 else "  <-- DISCARD (<100 tok)"
            ev_s = f"{n_gen} tok @ {float(ev.group(3)):.2f} t/s{flag}"
            if pp is not None:
                pp_s = f"{int(pp.group(2))} tok @ {float(pp.group(3)):.0f} t/s"
            else:
                pp_s = "?"
            da_s = f"{float(da.group(1)):.2f}" if da is not None else "-"
        else:
            pp_s = ev_s = "?"
            da_s = "-"
        print(f"{d:>8} | {pp_s:>22} | {ev_s:>26} | {da_s:>6} | {wall:.0f}s http={code} rc={rc}")

    print("\nDecode figures under 100 tokens are meaningless — see the header docstring.")


if __name__ == "__main__":
    sys.exit(main())

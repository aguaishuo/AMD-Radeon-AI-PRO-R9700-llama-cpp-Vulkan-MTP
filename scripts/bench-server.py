#!/usr/bin/env python3
"""Benchmark a running llama-server via its OpenAI-compatible endpoint.

Reports llama.cpp's own timings (prompt_per_second / predicted_per_second) for
short/medium/long prefill and for sustained 256/512-token generation. Every prompt
carries a random nonce so no run reuses the KV cache.

Usage:
    LLAMA_URL=http://localhost:8080 LLAMA_API_KEY=... \
        python3 bench-server.py <model-alias> [<model-alias> ...]
"""
import json, os, time, urllib.request, sys, random, string

URL = os.environ.get('LLAMA_URL', 'http://localhost:8080').rstrip('/') + '/v1/chat/completions'
KEY = os.environ.get('LLAMA_API_KEY', '')
FILLER = 'The quick brown fox jumps over the lazy dog near the riverbank while the sun sets slowly behind distant mountains. '

def nonce(n=12):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))

def call(model, prompt, maxtok):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': maxtok, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    t = d.get('timings', {})
    return {'build': d.get('system_fingerprint'), 'cache_n': t.get('cache_n'),
            'pp_n': t.get('prompt_n'), 'pp_tps': t.get('prompt_per_second'),
            'tg_n': t.get('predicted_n'), 'tg_tps': t.get('predicted_per_second'), 'wall': wall}

# each prompt gets a unique nonce so no KV-cache reuse across runs
def prompts():
    return [
        ('short',  lambda: f'[{nonce()}] Hello, briefly introduce yourself.'),
        ('medium', lambda: f'[{nonce()}] Summarize the following text in one sentence:\n\n' + FILLER * 7),
        ('long',   lambda: f'[{nonce()}] Summarize the following text in one sentence:\n\n' + FILLER * 21),
    ]
# long-generation prompts: force the model to keep producing tokens
LONGGEN = lambda: f'[{nonce()}] Write a detailed 1500-word technical essay about how GPU memory bandwidth limits large language model inference. Do not stop early.'

models = sys.argv[1:] or ['Qwen3.8-27B-Q4K_M']
for m in models:
    print('#### ' + m, flush=True)
    try:
        call(m, 'hi ' + nonce(), 4)
    except Exception as e:
        print('  warmup failed:', e, flush=True); continue
    for name, pf in prompts():
        r = call(m, pf(), 16)
        print(f'  {name:7s} gen=16    pp_n={r["pp_n"]:5d} cache={r["cache_n"]:5d} pp={r["pp_tps"]:8.1f} t/s   tg_n={r["tg_n"]:4d} tg={r["tg_tps"]:6.1f} t/s  wall={r["wall"]:5.1f}s', flush=True)
    for gen in (256, 512):
        r = call(m, LONGGEN(), gen)
        print(f'  longgen gen={gen:<4d}  pp_n={r["pp_n"]:5d} cache={r["cache_n"]:5d} pp={r["pp_tps"]:8.1f} t/s   tg_n={r["tg_n"]:4d} tg={r["tg_tps"]:6.1f} t/s  wall={r["wall"]:5.1f}s', flush=True)
    print('  build=' + str(r['build']), flush=True)

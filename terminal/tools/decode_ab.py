#!/usr/bin/env python3
"""A/B speed check of one engine config: the arms differ by binary and environment, nothing else.

    python3 tools/decode_ab.py --profile base.json --arm base \\
        --arm cap:STRATA_BATCH_DF2_CAP=1 --arm new@/path/to/strata:STRATA_PCIE_NEVER=1 --runs 3 --out ab-results

An arm is NAME, NAME:ENV=VALUE,ENV=VALUE or NAME@EXE:ENV=VALUE,...  (EXE replaces the profile's "exe"; ENV is added
to the profile's "env", and ENV= with no value removes that variable).  An item --flag=value sets an engine argument
instead: it replaces the flag's value in the profile's "args", or is appended.  For each arm the tool writes a copy of the
profile (its exe and env, its own engine log, no vision reader), starts the server with it, waits until the model is
loaded, warms up, and measures:

  prefill   a fresh prompt of about 2,300 tokens (its own nonce each run: no cache help): the engine's prompt tok/s
  prose     one request, an essay, --max-tokens out: the engine's decode tok/s
  code      one request, a coding task, --max-tokens out
  batchN    N requests at once (--batch 3,4): all generated tokens over the burst's wall time, and per request

Medians of --runs rounds, per arm, and the arms side by side.  Every answer is kept, so two arms can be compared text
for text (greedy decoding; an expert that moves between the CPU and the GPU cache between runs can still flip a
near-tie, so a difference is a lead, not a proof).  Thinking is off.  The server is the profile's own engine root
(--engine-root, default the profile's "cwd"), run the way the production service runs it.  Standard library only.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ESSAYS = [
    "the history of lighthouses and how their lenses worked",
    "how a mechanical watch keeps time",
    "why bridges are built as arches and as suspension spans",
    "the water cycle and what a drought does to a river valley",
    "how bread rises, from yeast to crust",
    "the design of a medieval castle and why its walls were stepped",
]
CODE = [
    "Write a Python class LRUCache with get and put in O(1), using a dict and a doubly linked list. Include type hints, "
    "docstrings and five unit tests with unittest.",
    "Write a Python function that parses an INI-style config file into nested dicts, handling comments, quoted values and "
    "line continuations. Include docstrings and unit tests.",
    "Write a C function that reverses a singly linked list in place and a second one that detects a cycle with Floyd's "
    "algorithm. Include a main() that tests both.",
    "Write a TypeScript function that debounces another function, with cancel and flush methods, full types, and Jest tests.",
    "Write a Rust function that tokenizes simple arithmetic expressions (numbers, + - * / and parentheses) into an enum, "
    "with error handling and tests.",
    "Write a Go HTTP handler that accepts JSON {\"items\": [...]} and returns the items sorted and deduplicated, with tests "
    "using httptest.",
]


def prefill_prompt(words: int, nonce: str) -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz "
             "meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"[{nonce}] Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in one sentence."


def post(base: str, body: dict, key: str, timeout: float) -> dict:
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_wall_s"] = time.perf_counter() - t0
    return out


def health(base: str, key: str):
    try:
        req = urllib.request.Request(base + "/health", headers={"Authorization": f"Bearer {key}"} if key else {})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None


def chat(base, key, model, text, max_tokens, timeout=900.0):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}
    r = post(base, body, key, timeout)
    t = r.get("timings") or {}
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    return {"text": msg.get("content") or "", "n": (r.get("usage") or {}).get("completion_tokens") or t.get("predicted_n") or 0,
            "decode_tps": t.get("predicted_per_second"), "prompt_tps": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
            "draft_n": t.get("draft_n"), "draft_ok": t.get("draft_n_accepted"), "wall_s": r["_wall_s"]}


def parse_arm(spec: str):
    name, _, envs = spec.partition(":")
    name, _, exe = name.partition("@")
    env, args = {}, []
    for kv in filter(None, envs.split(",")):
        k, _, v = kv.partition("=")
        if k.strip().startswith("--"):
            args.append((k.strip(), v.strip()))
        else:
            env[k.strip()] = v.strip()
    return {"name": name, "exe": exe or None, "env": env, "args": args}


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 2) if xs else None


def run_arm(arm, a, base_profile, out_dir: Path, log):
    prof = json.loads(json.dumps(base_profile))
    if arm["exe"]:
        prof["exe"] = arm["exe"]
    env = dict(prof.get("env") or {})
    for k, v in arm["env"].items():
        if v == "":
            env.pop(k, None)
        else:
            env[k] = v
    prof["env"] = env
    pargs = list(prof.get("args") or [])
    for flag, val in arm.get("args", []):
        if flag in pargs:
            i = pargs.index(flag)
            if val and i + 1 < len(pargs):
                pargs[i + 1] = val
        else:
            pargs += [flag] + ([val] if val else [])
    prof["args"] = pargs
    prof.pop("vision", None)
    engine_log = out_dir / f"{arm['name']}.engine.log"
    prof["log"] = str(engine_log)
    prof["port"] = a.port
    cfg = out_dir / f"{arm['name']}.profile.json"
    cfg.write_text(json.dumps(prof, indent=1))
    root = a.engine_root or prof.get("cwd") or "."
    boot = ("import sys; r=sys.argv.pop(1); sys.path.insert(0, r); sys.path.insert(0, r + '/tools'); "
            "from serve import server; sys.exit(server.main())")
    cmd = [sys.executable, "-c", boot, root, "--engine", a.engine, "--config", str(cfg), "--host", "127.0.0.1",
           "--port", str(a.port), "--no-fit-max-tokens"]
    if not a.key:
        cmd.append("--no-api-key")
    srv_log = open(out_dir / f"{arm['name']}.server.log", "w")
    proc = subprocess.Popen(cmd, cwd=root, stdout=srv_log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{a.port}"
    res = {"arm": arm, "tests": {}, "answers": {}}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"the server ended while loading (exit {proc.returncode}); see {srv_log.name}")
            h = health(base, a.key)
            if h and h.get("loaded"):
                break
            if time.time() - t0 > a.ready_timeout:
                raise RuntimeError("the model did not load in time")
            time.sleep(3)
        res["load_s"] = round(time.time() - t0, 1)
        model = (h or {}).get("model") or prof.get("model_name") or "strata"
        log(f"[{arm['name']}] loaded in {res['load_s']} s")
        # warm-up: one of each, then one burst (graphs, the expert cache, the drafter)
        chat(base, a.key, model, f"Write about {ESSAYS[-1]}.", 64)
        chat(base, a.key, model, CODE[-1], 64)
        with cf.ThreadPoolExecutor(max(a.batch)) as ex:
            list(ex.map(lambda i: chat(base, a.key, model, CODE[i % len(CODE)], 64), range(max(a.batch))))
        for rnd in range(a.runs):
            nonce = f"{arm['name']}-{rnd}-{random.randrange(1 << 30)}"
            pf = chat(base, a.key, model, prefill_prompt(a.prefill_words, nonce), 8)
            res["tests"].setdefault("prefill", []).append(pf)
            for i in range(a.solo):
                p = chat(base, a.key, model, f"Write a clear, well organised essay on {ESSAYS[i % len(ESSAYS)]}. Use plain prose, no lists.", a.max_tokens)
                res["tests"].setdefault("prose", []).append(p)
                c = chat(base, a.key, model, CODE[i % len(CODE)], a.max_tokens)
                res["tests"].setdefault("code", []).append(c)
            for n in a.batch:
                for kind in a.batch_kinds:
                    texts = ([CODE[(rnd + i) % len(CODE)] for i in range(n)] if kind == "code" else
                             [f"Write a clear, well organised essay on {ESSAYS[(rnd + i) % len(ESSAYS)]}. Use plain prose, no lists." for i in range(n)])
                    t_b = time.perf_counter()
                    with cf.ThreadPoolExecutor(n) as ex:
                        rs = list(ex.map(lambda t: chat(base, a.key, model, t, a.max_tokens), texts))
                    wall = time.perf_counter() - t_b
                    res["tests"].setdefault(f"batch{n}_{kind}", []).append(
                        {"total_tps": sum(r["n"] for r in rs) / wall, "per_req": [r["decode_tps"] for r in rs],
                         "texts": [r["text"] for r in rs], "wall_s": wall})
            log(f"[{arm['name']}] round {rnd + 1}/{a.runs} done")
    finally:
        proc.terminate()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.kill()
        srv_log.close()
    s = {}
    for k, rows in res["tests"].items():
        if k.startswith("batch"):
            s[k] = {"total_tps": med([r["total_tps"] for r in rows]),
                    "per_req_tps": med([x for r in rows for x in r["per_req"]])}
        elif k == "prefill":
            s[k] = {"prompt_tps": med([r["prompt_tps"] for r in rows]), "prompt_n": med([r["prompt_n"] for r in rows])}
        else:
            acc = [r["draft_ok"] / r["draft_n"] for r in rows if r.get("draft_n")]
            s[k] = {"decode_tps": med([r["decode_tps"] for r in rows]), "accept": med(acc)}
    try:   # the engine's own batch rate (the pipelined loop prints rows/s per burst, admissions included)
        import re
        rates = [float(m) for m in re.findall(r"rows in \d+ ms = ([0-9.]+) rows/s", engine_log.read_text(errors="replace"))]
        s["engine_batch_rows_s"] = {"median": med(rates), "n": len(rates)}
    except OSError:
        pass
    res["summary"] = s
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="the engine config the arms start from (exe, env, args, cwd, ...)")
    ap.add_argument("--arm", action="append", required=True, help="NAME[@EXE][:ENV=VALUE,...]; repeat for each arm")
    ap.add_argument("--engine-root", help="the folder holding serve/ (default: the profile's cwd)")
    ap.add_argument("--port", type=int, default=11533)
    ap.add_argument("--engine", default="strata", choices=["strata", "mock"], help="mock: a smoke test of this tool without a GPU")
    ap.add_argument("--key", default="", help="API key the server should want (default: none, --no-api-key)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--solo", type=int, default=2, help="solo prose and code requests per round")
    ap.add_argument("--batch", default="3,4", help="burst sizes, comma separated")
    ap.add_argument("--batch-kinds", default="prose,code")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prefill-words", type=int, default=1500)
    ap.add_argument("--ready-timeout", type=float, default=1200)
    ap.add_argument("--out", default="ab-results")
    a = ap.parse_args(argv)
    a.batch = [int(x) for x in a.batch.split(",") if x]
    a.batch_kinds = [x for x in a.batch_kinds.split(",") if x]
    out_dir = (Path(a.out) / time.strftime("%Y%m%d-%H%M%S")).resolve()   # absolute: the server runs in the engine root
    out_dir.mkdir(parents=True, exist_ok=True)
    base_profile = json.loads(Path(a.profile).read_text())
    log = lambda m: print(m, flush=True)
    results = []
    for spec in a.arm:
        arm = parse_arm(spec)
        try:
            results.append(run_arm(arm, a, base_profile, out_dir, log))
        except Exception as e:   # one arm failing keeps the others
            log(f"[{arm['name']}] FAILED: {e}")
            results.append({"arm": arm, "error": str(e)})
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    ref = next((r for r in results if "summary" in r), None)
    keys = sorted({k for r in results for k in r.get("summary", {})})
    log("\narm".ljust(14) + "".join(k.rjust(16) for k in keys))
    for r in results:
        if "summary" not in r:
            log(r["arm"]["name"].ljust(13) + "  FAILED")
            continue
        cells = []
        for k in keys:
            v = r["summary"].get(k, {})
            x = v.get("total_tps", v.get("decode_tps", v.get("prompt_tps")))
            cells.append(f"{x:.1f}".rjust(16) if isinstance(x, (int, float)) else "-".rjust(16))
        log(r["arm"]["name"].ljust(13) + "".join(cells))
    if ref is not None:
        for r in results:
            if r is ref or "tests" not in r:
                continue
            same = tot = 0
            for k in ("prose", "code"):
                for x, y in zip(ref["tests"].get(k, []), r["tests"].get(k, [])):
                    tot += 1
                    same += x["text"] == y["text"]
            log(f"answers identical to {ref['arm']['name']}: {r['arm']['name']} {same}/{tot} solo")
    log(f"\nresults: {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

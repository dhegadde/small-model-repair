#!/usr/bin/env python3
"""frontier-on-a-budget pilot 4: public low-bit releases (GGUF, run by llama.cpp's llama-server) on HumanEval+ with the
SAME prompt text as cc_code / fb_packed (HF chat template + EvalPlus response prefix), greedy, <= 768 new tokens,
stopping at the closing fence. One request at a time (batch 1), so llama.cpp's own timings give a speed baseline.
Writes humaneval_samples.jsonl in the cc_code format plus stats.json. Protocol: plans/pilot4_protocol.md.

  python -m repair.fb_gguf_eval --server http://127.0.0.1:8080 --out runs/fb_pilot4/gguf_UD-Q2_K_XL
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def post(server, path, body):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    from transformers import AutoTokenizer

    from repair.cc_code import chat_prompt, extract_code, problems
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="only the first N problems (smoke tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    todo = sorted(problems("humaneval").items(), key=lambda kv: -len(kv[1]["prompt"]))
    todo = todo[: args.limit] if args.limit else todo
    # the server must see the same tokens as the HF runs (chat-template special tokens parsed, no extra BOS)
    text = chat_prompt(tok, todo[0][1]["prompt"])
    ours = tok(text, add_special_tokens=False)["input_ids"]
    theirs = post(args.server, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
    stats = {"prompt_tokens_match": ours == theirs, "prompt_tokens_hf": len(ours), "prompt_tokens_server": len(theirs)}
    print(stats, flush=True)
    gen_tokens, gen_s, rates = 0, 0.0, []
    with (out / "humaneval_samples.jsonl").open("w") as fh:
        for k, (tid, p) in enumerate(todo):
            r = post(args.server, "/completion", {"prompt": chat_prompt(tok, p["prompt"]), "n_predict": 768,
                                                  "temperature": 0.0, "top_k": 1, "stop": ["```"], "cache_prompt": False,
                                                  "n_probs": 1, "post_sampling_probs": False})
            t = r.get("timings", {})
            gen_tokens += t.get("predicted_n", 0)
            gen_s += t.get("predicted_ms", 0.0) / 1000
            if t.get("predicted_n", 0) >= 32:
                rates.append(t["predicted_per_second"])
            lps = [c["logprob"] for c in r.get("completion_probabilities", []) if "logprob" in c]
            closed = r.get("stop_type") == "word" or r.get("stopping_word") == "```"
            fh.write(json.dumps({"task_id": tid, "solution": extract_code(r["content"]),
                                 "mean_logprob": sum(lps) / len(lps) if lps else 0.0, "min_logprob": min(lps) if lps else 0.0,
                                 "n_tokens": t.get("predicted_n", 0), "closed_fence": closed}) + "\n")
            if k % 16 == 15:
                print(f"eval {k + 1}/{len(todo)} {gen_tokens / max(gen_s, 1e-9):.1f} tok/s", flush=True)
    stats["tokens_per_s_batch1"] = round(gen_tokens / gen_s, 1)
    stats["tokens_per_s_batch1_median"] = round(sorted(rates)[len(rates) // 2], 1) if rates else None
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(stats, flush=True)


if __name__ == "__main__":
    main()

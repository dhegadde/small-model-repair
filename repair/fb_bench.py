#!/usr/bin/env python3
"""frontier-on-a-budget pilot 5: does the 2-bit model with Python-trained corrections stay good OUTSIDE Python coding?
Same arms as pilot 4, on four tasks:
  humaneval / mbpp : EvalPlus HumanEval+ / MBPP+ (Python; prompt as cc_code.chat_prompt), graded by EvalPlus
  js / cpp         : MultiPL-E HumanEval translations (other languages), completion inside the assistant turn as in
                     MultiPL-E, executed here with node / g++ against the dataset's tests
  mmlu             : 1,000 fixed MMLU test questions (general knowledge), answer = most likely letter after
                     "The answer is"
Backends: vllm (a local HF-format model directory, e.g. from fb_fold) or llamacpp (a running llama-server, GGUF).
Protocol: goals/frontier-on-a-budget/plans/pilot5_protocol.md.

  python -m repair.fb_bench prep                          # once, in the main venv (needs `datasets`)
  python -m repair.fb_bench run --backend vllm --model_dir /root/folded --out runs/fb_pilot5/ours
  python -m repair.fb_bench run --backend llamacpp --server http://127.0.0.1:8080 --out runs/fb_pilot5/q2
"""
import argparse
import json
import random
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("processed_data/fb_bench")
LANG = {"js": ("JavaScript", "javascript"), "cpp": ("C++", "cpp")}
LETTERS = ("A", "B", "C", "D")
STD_HEADERS = ("algorithm", "bitset", "cassert", "cctype", "climits", "cmath", "cstdint", "cstdio", "cstdlib",
               "cstring", "deque", "functional", "iostream", "iterator", "list", "map", "numeric", "optional",
               "queue", "set", "sstream", "stack", "string", "tuple", "unordered_map", "unordered_set", "utility",
               "variant", "vector")


def prep():
    from datasets import load_dataset
    DATA.mkdir(parents=True, exist_ok=True)
    for lang in LANG:
        rows = load_dataset("nuprl/MultiPL-E", f"humaneval-{lang}", split="test")
        json.dump([{k: r[k] for k in ("name", "prompt", "tests", "stop_tokens")} for r in rows],
                  open(DATA / f"multipl_e_{lang}.json", "w"))
        print(lang, len(rows))
    rows = list(load_dataset("cais/mmlu", "all", split="test"))
    pick = random.Random(0).sample(range(len(rows)), 1000)
    json.dump([{k: rows[i][k] for k in ("question", "subject", "choices", "answer")} for i in pick],
              open(DATA / "mmlu_1000.json", "w"))
    print("mmlu", len(pick))


def with_prefix(tok, user, prefix):
    msgs = [{"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + prefix


def mmlu_prompt(tok, q):
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, q["choices"]))
    return with_prefix(tok, f"{q['question']}\n\n{opts}\n\nAnswer with the letter of the correct option.",
                       "The answer is")


def multipl_prompt(tok, lang, p):
    name, tag = LANG[lang]
    return with_prefix(tok, f"Please complete the following {name} function in a markdown code block:\n```{tag}\n"
                            f"{p['prompt']}```", f"```{tag}\n{p['prompt']}")


def run_program(lang, code, timeout=15):
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / ("main.js" if lang == "js" else "main.cpp")
        src.write_text(code)
        try:
            if lang == "js":
                return subprocess.run(["node", str(src)], capture_output=True, timeout=timeout).returncode == 0
            # MultiPL-E C++ uses GCC's <bits/stdc++.h>; -idirafter only supplies a stand-in where it is missing (macOS)
            shim = Path(d) / "shim" / "bits"
            shim.mkdir(parents=True)
            (shim / "stdc++.h").write_text("".join(f"#include <{h}>\n" for h in STD_HEADERS))
            if subprocess.run(["g++", "-std=c++17", "-O1", "-idirafter", str(shim.parent), str(src), "-o", f"{d}/main"],
                              capture_output=True, timeout=120).returncode != 0:
                return False
            return subprocess.run([f"{d}/main"], capture_output=True, timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False


class VLLM:
    def __init__(self, model_dir):
        from vllm import LLM, SamplingParams
        self.llm, self.SP = LLM(model=model_dir, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9), SamplingParams

    def generate(self, prompts, stop, max_tokens=768):
        outs = self.llm.generate(prompts, self.SP(temperature=0.0, max_tokens=max_tokens, stop=stop, logprobs=0))
        res = []
        for o in outs:
            c = o.outputs[0]
            lps = [next(iter(d.values())).logprob for d in (c.logprobs or [])]
            res.append({"text": c.text, "closed": isinstance(c.stop_reason, str), "logprobs": lps})  # hit a stop string
        return res

    def next_token_logprobs(self, prompts, k=20):
        outs = self.llm.generate(prompts, self.SP(temperature=0.0, max_tokens=1, logprobs=k))
        return [{lp.decoded_token: lp.logprob for lp in o.outputs[0].logprobs[0].values()} for o in outs]


class LlamaCpp:
    def __init__(self, server, workers=8):
        self.server, self.workers = server, workers

    def _post(self, body):
        req = urllib.request.Request(self.server + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())

    def generate(self, prompts, stop, max_tokens=768):
        def one(p):
            r = self._post({"prompt": p, "n_predict": max_tokens, "temperature": 0.0, "top_k": 1, "stop": stop,
                            "cache_prompt": False, "n_probs": 1, "post_sampling_probs": False})
            return {"text": r["content"], "closed": r.get("stop_type") == "word",
                    "logprobs": [c["logprob"] for c in r.get("completion_probabilities", []) if "logprob" in c]}
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(one, prompts))

    def next_token_logprobs(self, prompts, k=20):
        def one(p):
            r = self._post({"prompt": p, "n_predict": 1, "temperature": 0.0, "n_probs": k, "cache_prompt": False,
                            "post_sampling_probs": False})
            return {t["token"]: t["logprob"] for t in r["completion_probabilities"][0]["top_logprobs"]}
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(one, prompts))


def mmlu_choice(top):
    """Most likely of the four letters among the top next tokens (' A' or 'A'); None if no letter is there."""
    scores = {l: max(top.get(" " + l, -1e9), top.get(l, -1e9)) for l in LETTERS}
    best = max(scores, key=scores.get)
    return LETTERS.index(best) if scores[best] > -1e9 else None


def run(args):
    from transformers import AutoTokenizer

    from repair.cc_code import chat_prompt, extract_code, problems
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    be = VLLM(args.model_dir) if args.backend == "vllm" else LlamaCpp(args.server)
    summary = {}
    for task in args.tasks.split(","):
        if task in ("humaneval", "mbpp"):
            todo = list(problems(task).items())[: args.limit or None]
            res = be.generate([chat_prompt(tok, p["prompt"]) for _, p in todo], stop=["```"])
            with (out / f"{task}_samples.jsonl").open("w") as fh:
                for (tid, _), r in zip(todo, res):
                    lp = r["logprobs"]
                    fh.write(json.dumps({"task_id": tid, "solution": extract_code(r["text"]),
                                         "mean_logprob": sum(lp) / len(lp) if lp else 0.0, "closed_fence": r["closed"]}) + "\n")
            summary[task] = {"n": len(todo), "closed_fence": round(100 * sum(r["closed"] for r in res) / len(res), 1)}
        elif task in LANG:
            todo = json.loads((DATA / f"multipl_e_{task}.json").read_text())[: args.limit or None]
            res = be.generate([multipl_prompt(tok, task, p) for p in todo], stop=["```"] + todo[0]["stop_tokens"])
            programs = [p["prompt"] + r["text"] + "\n" + p["tests"] for p, r in zip(todo, res)]
            with ThreadPoolExecutor(16) as ex:
                passed = list(ex.map(lambda c: run_program(task, c), programs))
            (out / f"{task}_results.jsonl").write_text("".join(
                json.dumps({"name": p["name"], "passed": ok, "completion": r["text"]}) + "\n"
                for p, r, ok in zip(todo, res, passed)))
            summary[task] = {"n": len(todo), "pass": round(100 * sum(passed) / len(passed), 1)}
        elif task == "mmlu":
            todo = json.loads((DATA / "mmlu_1000.json").read_text())[: args.limit or None]
            picks = [mmlu_choice(t) for t in be.next_token_logprobs([mmlu_prompt(tok, q) for q in todo])]
            summary[task] = {"n": len(todo), "acc": round(100 * sum(p == q["answer"] for p, q in zip(picks, todo)) / len(todo), 1),
                             "no_letter": sum(p is None for p in picks)}
        print(task, summary[task], flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "run"))
    ap.add_argument("--backend", choices=("vllm", "llamacpp"), default="vllm")
    ap.add_argument("--model_dir")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--tasks", default="humaneval,mbpp,js,cpp,mmlu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    prep() if args.cmd == "prep" else run(args)


if __name__ == "__main__":
    main()

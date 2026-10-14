#!/usr/bin/env python3
"""frontier-on-a-budget pilot 7: repair a PUBLIC low-bit file of a small model with a trained LoRA "fix" that ships as a
llama.cpp add-on (--lora). Protocol: goals/frontier-on-a-budget/plans/pilot7_protocol.md.

  teacher : the near-lossless file (Q8_0, served by llama-server) answers a BROAD prompt mix (Python, 8 other
            programming languages, general chat), so the fix does not narrow the model to Python
  train   : student = the low-bit GGUF loaded by transformers, dequantized: exactly the weights llama.cpp uses
            (Mac check, pilot 6: identical top tokens); LoRA r16 on every linear layer; --loss ce: cross-entropy on
            the teacher's answers (code cut after the first closed block; pilot 7) or --loss kd: KL to the
            full-precision model's next-token distributions on those answers (pilot 8)
  (then)  : llama.cpp's convert_lora_to_gguf.py turns the adapter into a .gguf add-on; fb_bench evaluates it

  python -m repair.fb_small_recover teacher --server http://127.0.0.1:8080 --out runs/fb_pilot7/teacher.jsonl
  python -m repair.fb_small_recover train --gguf_dir /root/gguf --gguf_file X-UD-IQ2_M.gguf \
      --teacher runs/fb_pilot7/teacher.jsonl --out runs/fb_pilot7/fix_s0
"""
import argparse
import json
import random
import time
from pathlib import Path

MIX = {"python": 1000, "other_code": 600, "general": 400}
OTHER_LANGS = ("cpp", "java", "typescript", "rust", "csharp", "php", "swift", "shell")


def prompts(seed=0):
    from datasets import load_dataset
    rng = random.Random(seed)
    mc = [r for r in load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train") if len(r["problem"]) < 3000]
    py = [r for r in mc if r["lang"] == "python"]
    other = [r for r in mc if r["lang"] in OTHER_LANGS]
    rng.shuffle(py)
    rng.shuffle(other)
    out = [{"kind": "code", "lang": "python", "prompt": r["problem"]} for r in py[: MIX["python"]]]
    out += [{"kind": "code", "lang": r["lang"], "prompt": r["problem"]} for r in other[: MIX["other_code"]]]
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    chat = [r["prompt"] for r, _ in zip(chat, range(4 * MIX["general"])) if len(r["prompt"]) < 2000][: MIX["general"]]
    out += [{"kind": "general", "lang": "", "prompt": p} for p in chat]
    rng.shuffle(out)
    return out


def user_text(p):
    if p["kind"] == "code":
        return p["prompt"].strip() + f"\n\nWrite the solution in {p['lang']} in a markdown code block."
    return p["prompt"].strip()


def chat(tok, text):
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def code_only(text):
    s = text.find("```")
    e = text.find("```", s + 3) if s >= 0 else -1
    return None if s < 0 or e < 0 else text[: e + 3]


def teacher(args):
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoTokenizer

    from repair.fb_bench import LlamaCpp
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ps = prompts()[: args.limit or None]
    be = LlamaCpp(args.server, workers=args.workers)

    def one(p):
        r = be._post({"prompt": chat(tok, user_text(p)), "n_predict": args.max_new, "temperature": 0.0, "top_k": 1,
                      "cache_prompt": False})
        return {**p, "response": r["content"], "stop_type": r.get("stop_type")}
    t0 = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh, ThreadPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(one, ps)):
            fh.write(json.dumps(r) + "\n")
            if k % 100 == 99:
                print(f"teacher {k + 1}/{len(ps)} {time.time() - t0:.0f}s", flush=True)


def kd_targets(rows):
    """For logit distillation every answer is usable: code -> up to the first closed block if there is one, else the
    whole (possibly cut) answer; general -> the whole answer. The teacher supplies a full distribution per token."""
    return [(r, (code_only(r["response"]) if r["kind"] == "code" else None) or r["response"]) for r in rows
            if r["response"].strip()]


def kd_loss(student_logits, teacher_logits, chunk=256):
    """Forward KL(teacher || student) per response token, averaged; computed in fp32 chunks to bound memory."""
    import torch
    tot, n = 0.0, student_logits.shape[0]
    for i in range(0, n, chunk):
        t = torch.log_softmax(teacher_logits[i : i + chunk].float(), -1)
        s = torch.log_softmax(student_logits[i : i + chunk].float(), -1)
        tot = tot + (t.exp() * (t - s)).sum()
    return tot / n


def targets(rows):
    """(prompt row, target text): code -> up to the end of the first closed code block; general -> full answer if
    the teacher finished it (not cut by the length limit)."""
    out = []
    for r in rows:
        t = code_only(r["response"]) if r["kind"] == "code" else (r["response"] if r["stop_type"] == "eos" else None)
        if t:
            out.append((r, t))
    return out


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:  # the pod's peft rejects its torchao version (same fix as the Gemma runs)
        from repair.helpers import peft_without_torchao as _peft_without_torchao
        _peft_without_torchao()
    except ImportError:
        pass
    from peft import LoraConfig, get_peft_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # the student: the public low-bit file's own weights (dequantized), not a re-quantization of ours
    model = AutoModelForCausalLM.from_pretrained(args.gguf_dir, gguf_file=args.gguf_file, device_map=dev,
                                                 dtype=torch.bfloat16 if dev == "cuda" else torch.float32)  # CPU: fp32
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    teacher_model = None
    if args.loss == "kd":  # the full-precision original model, frozen, next to the student
        teacher_model = AutoModelForCausalLM.from_pretrained(args.tokenizer, dtype=torch.bfloat16, device_map=dev).eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
    rows = [json.loads(l) for l in Path(args.teacher).read_text().splitlines()][: args.limit or None]
    pairs = kd_targets(rows) if args.loss == "kd" else targets(rows)
    eos = tok.eos_token or ""
    seqs = [(tok(chat(tok, user_text(r)), add_special_tokens=False)["input_ids"],
             tok(t + eos, add_special_tokens=False)["input_ids"]) for r, t in pairs]
    seqs = [(p, r[: max(1, args.max_len - len(p))]) for p, r in seqs if len(p) < args.max_len - 16]
    print(f"training examples: {len(seqs)} of {len(rows)} teacher answers ({args.loss})", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    order, log, t0, step = list(range(len(seqs))), [], time.time(), 0
    total = args.epochs * (len(seqs) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(1, total)))
    for ep in range(args.epochs):
        random.shuffle(order)
        for b in range(0, len(order) - args.batch + 1, args.batch):
            model.train()
            opt.zero_grad()
            tot = 0.0
            for k in order[b : b + args.batch]:
                p_ids, r_ids = seqs[k]
                ids = torch.tensor([p_ids + r_ids], device=dev)
                labels = torch.tensor([[-100] * len(p_ids) + r_ids], device=dev)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    if teacher_model is None:
                        loss = model(input_ids=ids, labels=labels).loss / args.batch
                    else:  # logits at positions that predict response tokens: len(p_ids)-1 ... end-1
                        sl = slice(len(p_ids) - 1, ids.shape[1] - 1)
                        with torch.no_grad():
                            t_logits = teacher_model(input_ids=ids).logits[0, sl]
                        loss = kd_loss(model(input_ids=ids).logits[0, sl], t_logits) / args.batch
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0 or step == 1:
                log.append({"epoch": ep, "step": step, "loss": round(tot, 4), "seconds": round(time.time() - t0),
                            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if dev == "cuda" else None})
                print(log[-1], flush=True)
    model.save_pretrained(out / "lora_hf")
    (out / "train.json").write_text(json.dumps({"examples": len(seqs), "loss": args.loss, "rank": args.rank, "lr": args.lr,
                                                "epochs": args.epochs, "seed": args.seed, "log": log}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("teacher", "train"))
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--gguf_dir")
    ap.add_argument("--gguf_file")
    ap.add_argument("--loss", choices=("ce", "kd"), default="ce",
                    help="ce: cross-entropy on the teacher's text; kd: KL to the full-precision model's distributions")
    ap.add_argument("--max_len", type=int, default=1536, help="truncate prompt+answer to this many tokens")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts / rows only (smoke tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    teacher(args) if args.cmd == "teacher" else train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""frontier-on-a-budget pilot 2: recover a mixture-of-experts coder whose EXPERTS are at 2 bits.
Protocol: goals/frontier-on-a-budget/plans/pilot2_protocol.md.

Teacher: the bf16 model itself answers Python problems from Magicoder-OSS-Instruct-75K; answers are cut after the
first closed code block (the code-only recipe confirmed in compression-confidence Phase 3b).
Student: the same model with experts fake-quantized to N bits (cc_code.fake_quantize_experts_), plus ONE of
  attn    : LoRA r16 on attention q/k/v/o only (experts and router frozen): can the rest compensate?
  experts : a rank-r correction per expert matrix, W_e + A_e @ B_e, applied to the fused 3-D expert tensors with
            torch parametrizations (experts' 2-bit values frozen); adapters ~1% of parameters
trained with cross-entropy on the teacher's code. Then HumanEval+ samples are written in-process for EvalPlus.

  python -m repair.fb_recover --arm experts --out runs/fb_pilot2/experts_r8
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P


class LowRankDelta(nn.Module):
    """Parametrization for a fused expert tensor [E, out, in]: returns W + A @ B (A [E, out, r], B [E, r, in])."""

    def __init__(self, shape, rank, dtype, device):
        super().__init__()
        e, o, i = shape
        self.A = nn.Parameter(torch.zeros(e, o, rank, dtype=torch.float32, device=device))   # zero init: no-op start
        self.B = nn.Parameter(torch.randn(e, rank, i, dtype=torch.float32, device=device) / (i ** 0.5))
        self.dtype = dtype

    def forward(self, w):
        return w + torch.bmm(self.A, self.B).to(self.dtype)


def add_expert_deltas(model, rank):
    n = 0
    for name, mod in list(model.named_modules()):
        for pname, p in list(mod.named_parameters(recurse=False)):
            if ".experts" in f"{name}." and p.dim() == 3:
                p.requires_grad_(False)
                P.register_parametrization(mod, pname, LowRankDelta(p.shape, rank, p.dtype, p.device))
                n += 1
    return n


def code_only(text):
    s = text.find("```")
    e = text.find("```", s + 3) if s >= 0 else -1
    return None if s < 0 or e < 0 else text[: e + 3]


def main():
    from repair.cc_code import (chat_prompt, extract_code, fake_quantize_experts_, load_model, problems,
                                         token_logprobs)
    from repair.cc_code_recover import chat
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--arm", choices=("attn", "experts"), required=True)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--teacher_max_new", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--teacher", default=None, help="cached teacher.jsonl (shared between arms)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tok = load_model(args.model, "bf16")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = next(model.parameters()).device

    # 1) teacher answers from the bf16 model (cached and shared)
    tfile = Path(args.teacher) if args.teacher else out / "teacher.jsonl"
    if not tfile.exists():
        from datasets import load_dataset
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        py = [r["problem"] for r in ds if r.get("lang") == "python" and len(r["problem"]) < 3000]
        random.Random(0).shuffle(py)
        tok.padding_side = "left"
        t0 = time.time()
        with tfile.open("w") as fh:
            for b in range(0, args.n, 16):
                chunk = py[b : b + 16]
                enc = tok([chat(tok, p) for p in chunk], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(dev)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=args.teacher_max_new, do_sample=False)
                for p, s in zip(chunk, g[:, enc["input_ids"].shape[1] :].tolist()):
                    text = tok.decode([t for t in s if t != tok.pad_token_id], skip_special_tokens=True)
                    fh.write(json.dumps({"problem": p, "response": text}) + "\n")
                print(f"teacher {min(b + 16, args.n)}/{args.n} {time.time() - t0:.0f}s", flush=True)
    data = [json.loads(l) for l in tfile.read_text().splitlines()]
    pairs = [(r["problem"], code_only(r["response"])) for r in data]
    pairs = [(p, c) for p, c in pairs if c]
    print(f"teacher examples with a closed code block: {len(pairs)} / {len(data)}", flush=True)

    # 2) student: experts at N bits + the arm's trainable correction
    print(f"fake-quantized {fake_quantize_experts_(model, args.bits)} expert matrices to {args.bits} bits", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    if args.arm == "attn":
        from repair.helpers import peft_without_torchao as _peft_without_torchao
        _peft_without_torchao()
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                                 target_modules=r".*self_attn\.(q_proj|k_proj|v_proj|o_proj)"))
    else:
        print(f"added low-rank corrections to {add_expert_deltas(model, args.rank)} fused expert tensors", flush=True)
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in train_params) / 1e6:.1f}M", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    eos = tok.eos_token or ""
    seqs = [(tok(chat(tok, p), add_special_tokens=False)["input_ids"], tok(c + eos, add_special_tokens=False)["input_ids"])
            for p, c in pairs]
    order, t0, step, log = list(range(len(seqs))), time.time(), 0, []
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
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, labels=labels).loss / args.batch
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                log.append({"epoch": ep, "step": step, "loss": round(tot, 4), "seconds": round(time.time() - t0)})
                print(log[-1], flush=True)
    (out / "train.json").write_text(json.dumps({"arm": args.arm, "bits": args.bits, "rank": args.rank,
                                                "examples": len(seqs), "log": log}, indent=2))
    if args.arm == "experts":  # kept for pilot 3 (fb_packed: the same model stored at real 2 bits)
        from repair.fb_packed import save_deltas
        save_deltas(model, out / "deltas.pt")

    # 3) HumanEval+ samples from the recovered model, written in the cc_code format for EvalPlus
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    f = out / "humaneval_samples.jsonl"
    todo = sorted(problems("humaneval").items(), key=lambda kv: -len(kv[1]["prompt"]))
    torch.cuda.reset_peak_memory_stats()
    gen_s, gen_tokens = 0.0, 0
    with f.open("w") as fh:
        for b in range(0, len(todo), 16):
            chunk = todo[b : b + 16]
            enc = tok([chat_prompt(tok, p["prompt"]) for _, p in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            t1 = time.time()
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=768, do_sample=False, stop_strings=["```"], tokenizer=tok)
            gen_s += time.time() - t1
            plen = enc["input_ids"].shape[1]
            for row, ((tid, _), full) in enumerate(zip(chunk, g.tolist())):
                prompt = [t for t, m in zip(full[:plen], enc["attention_mask"][row].tolist()) if m]
                gen = [t for t in full[plen:] if t != tok.pad_token_id]
                gen_tokens += len(gen)
                text = tok.decode(gen, skip_special_tokens=True)
                n_code = max(1, min(len(gen), len(tok(text.split("```", 1)[0], add_special_tokens=False)["input_ids"])))
                vals = token_logprobs(model, prompt, gen[:n_code], dev)
                fh.write(json.dumps({"task_id": tid, "solution": extract_code(text),
                                     "mean_logprob": sum(vals) / max(1, len(vals)), "min_logprob": min(vals) if vals else 0.0,
                                     "n_tokens": len(gen), "closed_fence": "```" in text}) + "\n")
            print(f"eval {min(b + 16, len(todo))}/{len(todo)}", flush=True)
    # speed / memory of this SIMULATED model (experts in bf16), for comparison with fb_packed's real 2-bit one
    from repair.fb_packed import speed_test
    stats = {"peak_gpu_gb_eval_batch16": round(torch.cuda.max_memory_allocated() / 1e9, 2),
             "eval_tokens_per_s_batch16": round(gen_tokens / gen_s, 1)}
    stats["tokens_per_s_batch1"] = round(speed_test(model, tok, [chat_prompt(tok, p["prompt"]) for _, p in todo[:4]]), 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(stats, flush=True)


if __name__ == "__main__":
    main()

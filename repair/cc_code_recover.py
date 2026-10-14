#!/usr/bin/env python3
"""compression-confidence Phase 3b: recover a 3-bit coding model by sequence-level distillation from its own bf16
version (Kim & Rush 2016), the code analogue of Phase 2's R2. Protocol: goals/compression-confidence/plans/
phase3b_protocol.md.

  1) teacher: the bf16 model answers N Python problems from Magicoder-OSS-Instruct-75K (problem texts only;
     decontaminated against HumanEval/MBPP by its authors), greedy, <= max_new tokens -> teacher.jsonl
  2) student: the same model HQQ-quantized, plus LoRA r16, trained with next-token cross-entropy on the teacher's
     responses (prompt tokens masked) -> adapter/
Evaluate with: python -m repair.cc_code --model <m> --quant hqq3 --adapter <out>/adapter ...

  python -m repair.cc_code_recover --model Qwen/Qwen3.5-9B --quant hqq3 --out runs/cc_phase3b/r_9b_hqq3
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkvz|in_proj_ba|out_proj)"


def code_only(response):
    """Text up to and including the closing fence of the first ``` block; None if no closed block."""
    start = response.find("```")
    if start < 0:
        return None
    end = response.find("```", start + 3)
    return None if end < 0 else response[: end + 3]


def user_text(problem):
    return problem.strip() + "\n\nWrite the solution in Python in a markdown code block."


def chat(tok, problem):
    msgs = [{"role": "user", "content": user_text(problem)}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    from repair.cc_code import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="hqq3")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code_only", action="store_true", help="train on the teacher's first code block only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) teacher responses (bf16), cached
    tfile = out / "teacher.jsonl"
    if not tfile.exists() or sum(1 for _ in tfile.open()) < args.n:
        from datasets import load_dataset
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        py = [r["problem"] for r in ds if r.get("lang") == "python" and len(r["problem"]) < 3000]
        random.Random(args.seed).shuffle(py)
        probs = py[: args.n]
        model, tok = load_model(args.model, "bf16")
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        dev = next(model.parameters()).device
        t0 = time.time()
        with tfile.open("w") as fh:
            for b in range(0, len(probs), 16):
                chunk = probs[b : b + 16]
                enc = tok([chat(tok, p) for p in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
                for p, s in zip(chunk, g[:, enc["input_ids"].shape[1] :].tolist()):
                    fh.write(json.dumps({"problem": p, "response": tok.decode([t for t in s if t != tok.pad_token_id],
                                                                              skip_special_tokens=True)}) + "\n")
                print(f"teacher {min(b + 16, len(probs))}/{len(probs)} {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    data = [json.loads(l) for l in tfile.read_text().splitlines()][: args.n]

    # 2) student: quantized model + LoRA, cross-entropy on the teacher's response tokens only
    from repair.helpers import peft_without_torchao as _peft_without_torchao
    model, tok = load_model(args.model, args.quant)
    dev = next(model.parameters()).device
    _peft_without_torchao()
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    eos = tok.eos_token or ""
    seqs, skipped = [], 0
    for r in data:
        resp = r["response"]
        if args.code_only:  # Phase 3b rerun: keep the teacher's text only up to the END of its first code block
            resp = code_only(resp)
            if resp is None:
                skipped += 1
                continue
        p_ids = tok(chat(tok, r["problem"]), add_special_tokens=False)["input_ids"]
        r_ids = tok(resp + eos, add_special_tokens=False)["input_ids"]
        seqs.append((p_ids, r_ids))
    print(f"training examples: {len(seqs)} (skipped {skipped} without a closed code block)", flush=True)
    order, log, t0, step = list(range(len(seqs))), [], time.time(), 0
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                log.append({"epoch": ep, "step": step, "loss": round(tot, 4), "seconds": round(time.time() - t0)})
                print(log[-1], flush=True)
    model.save_pretrained(out / "adapter")
    (out / "result.json").write_text(json.dumps({"n": len(seqs), "epochs": args.epochs, "log": log,
                                                 "quant": args.quant, "model": args.model}, indent=2))


if __name__ == "__main__":
    main()

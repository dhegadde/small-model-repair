#!/usr/bin/env python3
"""frontier-on-a-budget pilot 5: write a 2-bit-expert variant as an ordinary bf16 checkpoint so a fast engine (vLLM)
can evaluate its QUALITY on many tasks. Each expert weight is replaced by exactly what fb_packed computes at run time
(2-bit dequantized + correction), so outputs equal the packed model's up to engine numerics (pilot 3: packed 85.4 vs
simulated 84.1 on HumanEval+). Memory is NOT saved by this file: it is an evaluation device only.

  python -m repair.fb_fold --deltas runs/fb_pilot3/sim/deltas.pt --save_dir /root/folded   # trained (ours)
  python -m repair.fb_fold --svd_rank 8 --save_dir /root/folded                             # data-free SVD
  python -m repair.fb_fold --save_dir /root/folded                                          # plain 2-bit
"""
import argparse

import torch


@torch.no_grad()
def fold_experts_(model, nbits=2, deltas=None, svd_rank=0, device="cuda"):
    from repair.fb_packed import PackedExperts
    n = 0
    for name, mod in model.named_modules():
        if not (isinstance(getattr(mod, "gate_up_proj", None), torch.Tensor) and mod.gate_up_proj.dim() == 3):
            continue
        d = {p: deltas[f"{name}.{p}"] for p in ("gate_up_proj", "down_proj")} if deltas is not None else None
        pe = PackedExperts(mod, nbits, 64, d, device, svd_rank=svd_rank)
        idx = torch.arange(pe.num_experts, device=device)
        for pname in ("gate_up_proj", "down_proj"):
            getattr(mod, pname).data.copy_(pe.weight(pname, idx).to(getattr(mod, pname).device))
        del pe
        n += 1
    return n


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--deltas", default=None)
    ap.add_argument("--svd_rank", type=int, default=0)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--save_dir", required=True)
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)  # on CPU
    deltas = torch.load(args.deltas) if args.deltas else None
    print(f"folded {fold_experts_(model, args.bits, deltas, args.svd_rank)} expert modules", flush=True)
    model.save_pretrained(args.save_dir)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()

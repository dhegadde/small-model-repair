#!/usr/bin/env python3
"""frontier-on-a-budget pilot 3: the REAL low-bit model. Pilot 2 only simulated 2-bit experts (values rounded to 2 bits
but stored in bf16, so no memory was saved). Here every expert matrix is stored as packed 2-bit codes (4 per byte) plus
an fp16 scale and zero per group of 64 weights (2.5 bits per weight), and the trained rank-r corrections are kept as
separate bf16 add-ons: folding them into the weights and re-quantizing would erase them. At run time only the experts a
batch actually uses are unpacked, a chunk at a time, layer by layer. Protocol:
goals/frontier-on-a-budget/plans/pilot3_protocol.md.

  python -m repair.fb_packed --deltas runs/fb_pilot3/sim/deltas.pt --out runs/fb_pilot3/packed
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def hqq_codes(w, nbits, group_size):
    """The same HQQ call as cc_code.fake_quantize_experts_, without bit packing: integer codes [G, group_size] and
    float32 scale / zero [G, 1], with dequantized = (codes - zero) * scale."""
    from hqq.core.quantize import Quantizer
    q, meta = Quantizer.quantize(w.float(), nbits=nbits, group_size=group_size, optimize=True, axis=1, bitpack=False,
                                 compute_dtype=torch.float32, device=str(w.device))
    return q.to(torch.uint8), meta["scale"], meta["zero"]


def pack(codes, nbits):
    """[..., n] integer codes < 2**nbits -> [..., n * nbits / 8] uint8 (lowest bits first)."""
    per = 8 // nbits
    c = codes.reshape(*codes.shape[:-1], -1, per).to(torch.uint8)
    out = torch.zeros(c.shape[:-1], dtype=torch.uint8, device=c.device)
    for k in range(per):
        out |= c[..., k] << (k * nbits)
    return out


def unpack(packed, nbits):
    per = 8 // nbits
    shifts = torch.arange(0, 8, nbits, dtype=torch.uint8, device=packed.device)
    return ((packed[..., None] >> shifts) & (2 ** nbits - 1)).reshape(*packed.shape[:-1], packed.shape[-1] * per)


class PackedExperts(nn.Module):
    """Drop-in for transformers' fused experts module (gate_up_proj [E, 2I, H], down_proj [E, H, I]) with the experts
    stored at `nbits` and optional low-rank corrections W_e + A_e @ B_e (as trained by fb_recover)."""

    def __init__(self, experts, nbits=2, group_size=64, deltas=None, device="cuda", delta_dtype=torch.bfloat16,
                 chunk=32, svd_rank=0):
        super().__init__()
        self.num_experts, self.act_fn = experts.num_experts, experts.act_fn
        self.nbits, self.group_size, self.chunk = nbits, group_size, chunk
        self.dtype = experts.gate_up_proj.dtype
        self.shapes = {}
        for pname in ("gate_up_proj", "down_proj"):
            w = getattr(experts, pname).data
            codes, scales, zeros, svd_a, svd_b = [], [], [], [], []
            for e in range(w.shape[0]):
                q, s, z = hqq_codes(w[e].to(device), nbits, group_size)
                codes.append(pack(q, nbits))
                scales.append(s.half())
                zeros.append(z.half())
                if svd_rank:  # data-free baseline (MiLo / Liu et al. 2512.17073): best rank-r fit of the rounding error
                    deq = ((q.float() - z.half().float()) * s.half().float()).reshape(w.shape[1:]).to(w.dtype)
                    u, sv, vh = torch.linalg.svd(w[e].to(device).float() - deq.float(), full_matrices=False)
                    svd_a.append(u[:, :svd_rank] * sv[:svd_rank])
                    svd_b.append(vh[:svd_rank])
            if svd_rank:
                deltas = deltas or {}
                deltas[pname] = {"A": torch.stack(svd_a), "B": torch.stack(svd_b)}
            self.shapes[pname] = tuple(w.shape[1:])
            self.register_buffer(f"{pname}_codes", torch.stack(codes))   # [E, G, group_size * nbits / 8] uint8
            self.register_buffer(f"{pname}_scale", torch.stack(scales))  # [E, G, 1] fp16
            self.register_buffer(f"{pname}_zero", torch.stack(zeros))
            if deltas is not None:
                self.register_buffer(f"{pname}_A", deltas[pname]["A"].to(device, delta_dtype))  # [E, out, r]
                self.register_buffer(f"{pname}_B", deltas[pname]["B"].to(device, delta_dtype))  # [E, r, in]
        self.has_deltas = deltas is not None

    def weight(self, pname, idx):
        """Dequantized (+ corrected) weights of experts `idx`, [len(idx), out, in], in the model dtype. Same arithmetic
        as training: bf16 dequantized weight + (A @ B in fp32) cast to bf16."""
        q = unpack(getattr(self, f"{pname}_codes")[idx], self.nbits).float()
        w = (q - getattr(self, f"{pname}_zero")[idx].float()) * getattr(self, f"{pname}_scale")[idx].float()
        w = w.reshape(len(idx), *self.shapes[pname]).to(self.dtype)
        if self.has_deltas:
            w = w + torch.bmm(getattr(self, f"{pname}_A")[idx].float(), getattr(self, f"{pname}_B")[idx].float()).to(self.dtype)
        return w

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten()
        for c in range(0, len(hit), self.chunk):
            idx = hit[c : c + self.chunk]
            gate_up, down = self.weight("gate_up_proj", idx), self.weight("down_proj", idx)
            for j, e in enumerate(idx.tolist()):
                top_k_pos, token_idx = torch.where(mask[e])
                gate, up = nn.functional.linear(hidden_states[token_idx], gate_up[j]).chunk(2, dim=-1)
                h = nn.functional.linear(self.act_fn(gate) * up, down[j]) * top_k_weights[token_idx, top_k_pos, None]
                out.index_add_(0, token_idx, h.to(out.dtype))
        return out


def save_deltas(model, path):
    """fb_recover's trained corrections, {"<experts module>.<param>": {"A": [E, out, r], "B": [E, r, in]}} in fp32."""
    import torch.nn.utils.parametrize as P
    torch.save({f"{name}.{pname}": {"A": d.A.detach().cpu(), "B": d.B.detach().cpu()}
                for name, mod in model.named_modules() if P.is_parametrized(mod)
                for pname, plist in mod.parametrizations.items() for d in plist}, path)


@torch.no_grad()
def pack_experts_(model, nbits=2, group_size=64, deltas=None, device="cuda", delta_dtype=torch.bfloat16, svd_rank=0):
    """Replace every fused experts module (3-D gate_up_proj and down_proj) with a PackedExperts on `device`; with
    svd_rank > 0 (and no deltas) each expert gets the rank-r SVD of its own rounding error instead of trained ones."""
    targets = [(name, mod) for name, mod in model.named_modules()
               if isinstance(getattr(mod, "gate_up_proj", None), torch.Tensor) and mod.gate_up_proj.dim() == 3]
    for name, mod in targets:
        d = None
        if deltas is not None:
            d = {p: deltas[f"{name}.{p}"] for p in ("gate_up_proj", "down_proj")}
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child,
                PackedExperts(mod, nbits, group_size, d, device, delta_dtype, svd_rank=svd_rank))
    return len(targets)


def tensor_gb(module):
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers())) / 1e9


@torch.no_grad()
def speed_test(model, tok, prompts, max_new=256):
    """Batch-1 greedy decoding speed (tokens/s) over a few prompts; the first prompt is a warm-up."""
    dev = next(model.parameters()).device
    rates = []
    for k, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to(dev)
        torch.cuda.synchronize()
        t0 = time.time()
        g = model.generate(**enc, max_new_tokens=max_new, min_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        if k:
            rates.append((g.shape[1] - enc["input_ids"].shape[1]) / (time.time() - t0))
    return sum(rates) / max(1, len(rates))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from repair.cc_code import chat_prompt, extract_code, problems, token_logprobs
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--deltas", default=None, help="deltas.pt from fb_recover (omit: plain 2-bit experts)")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--svd_rank", type=int, default=0, help="data-free SVD corrections of this rank (no --deltas)")
    ap.add_argument("--save_dir", default="/root/packed_model", help="write the packed weights here to measure file size")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model, info = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, output_loading_info=True)  # on CPU
    if info.get("missing_keys"):
        raise RuntimeError(f"weights missing: {info['missing_keys'][:3]}")
    deltas = torch.load(args.deltas) if args.deltas else None
    n = pack_experts_(model, args.bits, deltas=deltas, svd_rank=args.svd_rank)
    del deltas
    model.to("cuda").eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    stats = {"packed_modules": n, "build_seconds": round(time.time() - t0),
             "weights_gb_counted": round(tensor_gb(model), 2),
             "gpu_allocated_gb_after_load": round(torch.cuda.memory_allocated() / 1e9, 2)}
    print(stats, flush=True)
    if args.save_dir:
        from safetensors.torch import save_file
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        f = Path(args.save_dir) / "model.safetensors"
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(f))
        stats["file_gb"] = round(f.stat().st_size / 1e9, 2)
        f.unlink()
        print(f"packed weights file: {stats['file_gb']} GB", flush=True)

    # HumanEval+ samples, same format and settings as cc_code / fb_recover
    tok.padding_side = "left"
    todo = sorted(problems("humaneval").items(), key=lambda kv: -len(kv[1]["prompt"]))
    gen_s, gen_tokens = 0.0, 0
    with (out / "humaneval_samples.jsonl").open("w") as fh:
        for b in range(0, len(todo), 16):
            chunk = todo[b : b + 16]
            enc = tok([chat_prompt(tok, p["prompt"]) for _, p in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            t1 = time.time()
            g = model.generate(**enc, max_new_tokens=768, do_sample=False, stop_strings=["```"], tokenizer=tok)
            gen_s += time.time() - t1
            plen = enc["input_ids"].shape[1]
            for row, ((tid, _), full) in enumerate(zip(chunk, g.tolist())):
                prompt = [t for t, m in zip(full[:plen], enc["attention_mask"][row].tolist()) if m]
                gen = [t for t in full[plen:] if t != tok.pad_token_id]
                gen_tokens += len(gen)
                text = tok.decode(gen, skip_special_tokens=True)
                n_code = max(1, min(len(gen), len(tok(text.split("```", 1)[0], add_special_tokens=False)["input_ids"])))
                vals = token_logprobs(model, prompt, gen[:n_code], "cuda")
                fh.write(json.dumps({"task_id": tid, "solution": extract_code(text),
                                     "mean_logprob": sum(vals) / max(1, len(vals)), "min_logprob": min(vals) if vals else 0.0,
                                     "n_tokens": len(gen), "closed_fence": "```" in text}) + "\n")
            print(f"eval {min(b + 16, len(todo))}/{len(todo)}", flush=True)
    stats["peak_gpu_gb_eval_batch16"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    stats["eval_tokens_per_s_batch16"] = round(gen_tokens / gen_s, 1)
    torch.cuda.reset_peak_memory_stats()
    stats["tokens_per_s_batch1"] = round(speed_test(model, tok, [chat_prompt(tok, p["prompt"]) for _, p in todo[:4]]), 1)
    stats["peak_gpu_gb_batch1"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(stats, flush=True)


if __name__ == "__main__":
    main()

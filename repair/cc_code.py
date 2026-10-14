#!/usr/bin/env python3
"""compression-confidence Phase 3a: open coding models under weight quantization, on HumanEval+ and MBPP+
(EvalPlus). Protocol: goals/compression-confidence/plans/phase3_protocol.md.

For each problem the model writes a self-contained Python solution (chat template, thinking OFF, greedy, <= 768
new tokens); we also record its confidence in the code = mean log-probability of the generated tokens, so we can
ask whether a compressed coder still "knows" when its code is wrong. Grading (running the hidden tests) is done
separately with EvalPlus in its own venv: scripts/runpod/cc_phase3_grade.sh.

  python -m repair.cc_code --model Qwen/Qwen3.5-4B --quant hqq4 --out runs/cc_phase3/qwen35_4b_hqq4
"""
import argparse
import json
import time
from pathlib import Path

import torch

INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
RESPONSE_PREFIX = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n"


def problems(dataset, data_dir="processed_data/evalplus"):
    """EvalPlus problems {task_id: {"prompt": ...}}, exported to JSON by the grading venv (evalplus is NOT installed
    in the shared venv: an extra library there broke LoRA training on 2026-10-03)."""
    return json.loads((Path(data_dir) / f"{dataset}.json").read_text())


def extract_code(text):
    """The generation continues after an opened ```python fence: keep everything up to the closing fence."""
    return text.split("```", 1)[0].rstrip() + "\n"


@torch.no_grad()
def fake_quantize_experts_(model, nbits, group_size=64):
    """Simulated quantization of mixture-of-experts weights stored as fused 3-D tensors (transformers 5.x, e.g.
    Qwen3MoeExperts.gate_up_proj [E, out, in] and down_proj): each expert's matrix is quantized with HQQ's optimizer
    (group-wise along the input dim) and immediately dequantized back to bf16 in place. Measures the quality of
    low-bit experts exactly; memory is not saved in this experiment."""
    from hqq.core.quantize import Quantizer
    n = 0
    for name, p in model.named_parameters():
        if ".experts." not in name or p.dim() != 3:
            continue
        for e in range(p.shape[0]):
            w = p[e]
            wq, meta = Quantizer.quantize(w.float(), nbits=nbits, group_size=group_size, optimize=True, axis=1,
                                          compute_dtype=torch.float32, device=str(w.device))
            p[e].copy_(Quantizer.dequantize(wq, meta).reshape(w.shape).to(p.dtype))
            n += 1
    torch.cuda.empty_cache()
    return n


def load_model(name, quant, adapter=None):
    """Full (multimodal-capable) model class so no weights are silently re-initialized; text-only use."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    cfg = AutoConfig.from_pretrained(name)
    cls = AutoModelForImageTextToText if "ConditionalGeneration" in (cfg.architectures or [""])[0] else AutoModelForCausalLM
    multi = torch.cuda.device_count() > 1
    extra = {}
    if (getattr(cfg, "quantization_config", None) or {}).get("quant_method") == "fp8":
        # FP8 releases (e.g. Devstral Small 2): expand to bf16 WITH the per-layer scales before any further
        # quantization. Without this, HQQ re-quantized raw FP8 values and the model produced gibberish (2026-10-04).
        # Verified on Ministral-3-3B: dequantized FP8 == the official BF16 release (100% next-token agreement).
        from transformers import FineGrainedFP8Config
        extra["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    model, info = cls.from_pretrained(name, dtype=torch.bfloat16, device_map="auto" if multi else "cuda",
                                      output_loading_info=True, **extra)
    tied = getattr(model.config.get_text_config(), "tie_word_embeddings", False) or getattr(model.config, "tie_word_embeddings", False)
    missing = [k for k in info.get("missing_keys", []) if "rotary" not in k and not (tied and "lm_head" in k)]
    if missing:  # the Gemma lesson: a text-only class can silently re-initialize weights
        raise RuntimeError(f"{len(missing)} weights missing when loading {name}, e.g. {missing[:3]}")
    if quant.startswith("experts"):  # frontier pilot: simulated low-bit EXPERTS only (e.g. experts2, experts3g64)
        nbits, _, gs = quant[len("experts"):].partition("g")
        print(f"{quant}: fake-quantized {fake_quantize_experts_(model, int(nbits), int(gs or 64))} expert matrices "
              "(attention, router, embeddings stay bf16)", flush=True)
    elif quant.startswith("hqq"):
        from repair.helpers import hqq_quantize_
        nbits, _, gs = quant[3:].partition("g")
        print(f"{quant}: quantized {hqq_quantize_(model, int(nbits), int(gs or 64))} decoder Linear layers", flush=True)
    if adapter:  # Phase 3b: a recovery LoRA trained on the quantized model
        from repair.helpers import peft_without_torchao as _peft_without_torchao
        _peft_without_torchao()
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


@torch.no_grad()
def token_logprobs(model, prompt, gen, dev, chunk=128):
    """Log-probability of each generated token, from one teacher-forced pass; the vocabulary softmax is done in
    chunks of positions so a 248k vocabulary never materializes for the whole sequence at once."""
    ids = torch.tensor([prompt + gen], device=dev)
    logits = model(input_ids=ids, logits_to_keep=len(gen) + 1).logits[0, :-1]
    tgt = torch.tensor(gen, device=logits.device)
    out = []
    for i in range(0, len(gen), chunk):
        lp = torch.log_softmax(logits[i : i + chunk].float(), dim=-1)
        out += lp.gather(1, tgt[i : i + chunk, None]).squeeze(1).tolist()
    return [v for v in out if v == v and v > -1e9]


def chat_prompt(tok, prompt):
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\n```\n{prompt.strip()}\n```"}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + RESPONSE_PREFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="bf16", help="bf16 | hqq8 | hqq4 | hqq3 (| hqq3g32 ...)")
    ap.add_argument("--datasets", default="humaneval,mbpp")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only the first N problems (speed tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args.model, args.quant, args.adapter)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = next(model.parameters()).device
    for ds in args.datasets.split(","):
        f = out / f"{ds}_samples.jsonl"
        done = {json.loads(l)["task_id"] for l in f.read_text().splitlines()} if f.exists() else set()
        todo = [(tid, p["prompt"]) for tid, p in problems(ds).items() if tid not in done]
        todo.sort(key=lambda x: -len(x[1]))
        if args.limit:
            todo = todo[: args.limit]
        print(f"{ds}: {len(todo)} problems to do", flush=True)
        t0 = time.time()
        with f.open("a") as fh:
            for b in range(0, len(todo), args.batch):
                chunk = todo[b : b + args.batch]
                enc = tok([chat_prompt(tok, p) for _, p in chunk], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(dev)
                with torch.no_grad():  # no output_scores: 248k-vocab scores for every step ran a 32 GB GPU out of memory
                    g = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, stop_strings=["```"],
                                       tokenizer=tok)
                plen = enc["input_ids"].shape[1]
                for row, ((tid, _), full) in enumerate(zip(chunk, g.tolist())):
                    prompt = [t for t, m in zip(full[:plen], enc["attention_mask"][row].tolist()) if m]
                    gen = [t for t in full[plen:] if t != tok.pad_token_id]
                    text = tok.decode(gen, skip_special_tokens=True)
                    # confidence over the code tokens only (stop at the closing fence if present)
                    n_code = max(1, min(len(gen), len(tok(text.split("```", 1)[0], add_special_tokens=False)["input_ids"])))
                    vals = token_logprobs(model, prompt, gen[:n_code], dev)
                    fh.write(json.dumps({"task_id": tid, "solution": extract_code(text),
                                         "mean_logprob": sum(vals) / max(1, len(vals)),
                                         "min_logprob": min(vals) if vals else 0.0, "n_tokens": len(gen),
                                         "closed_fence": "```" in text}) + "\n")
                fh.flush()
                print(f"{ds} {min(b + args.batch, len(todo))}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

# Pilot 6: where do public low-bit files of a SMALL model break? (pre-registered 2026-10-07, before running)
Why: pilot 4b showed the 30B MoE coder barely degrades even at 8-9 GB, so repair has little room there. Small dense
models lost far more in our own runs (Qwen3.5-4B 76.8 -> 59.8 at 3 bits). The human's priority is small models that
run anywhere. Approved: "lets do it" (the Mac check, then the cheap measurement); this runs FREE on the Mac.

Mac check (done 2026-10-07, scratchpad lora_check.sh / lora_compare.py, Qwen3-0.6B):
- llama.cpp applies a PEFT LoRA converted with convert_lora_to_gguf.py exactly like PyTorch: top-1 next token 20/20,
  mean |d logprob| 0.001 (f16 base); the LoRA also loads on a Q2_K base and changes outputs.
- transformers loads low-bit GGUFs dequantized (Q2_K, Unsloth UD-IQ1_M, UD-IQ2_XXS) and matches llama.cpp on that
  same file: 20/20, 19/20, 20/20 (|d logprob| 0.003-0.011). => fixes can be TRAINED against the exact weights of a
  public low-bit file and SHIPPED as a llama.cpp --lora add-on. Works for architectures transformers can read from
  GGUF (qwen2, qwen3, llama, gemma3/4, ...; not qwen3.5).
Model: Qwen3-4B-Instruct-2507 (qwen3 architecture; Unsloth GGUFs, ~234k downloads).
Files (llama.cpp on the Mac, Metal, batch 1, HumanEval+ 164, harness fb_gguf_eval): Q8_0 (reference, ~lossless),
Q4_K_M 2.5 GB, UD-Q2_K_XL 1.7, UD-IQ2_M 1.5, UD-IQ2_XXS 1.3, UD-IQ1_M 1.1.
Predictions: a clear cliff for the small model, unlike the 30B MoE: UD-IQ2_XXS loses >= 10 points vs Q8_0 and
UD-IQ1_M loses >= 25. Q4_K_M within 3 points of Q8_0.
Decision rule: the smallest file that loses >= 8 points but still closes its code >= 80% of the time is the repair
target (train LoRA fixes against its exact weights, broad teacher mix, ship as --lora).

## PARTIAL RESULT (2026-10-07, Mac, free; first attempt stopped by mac_guard, see failures.md "prompt cache")
Graded from the problems each file finished before the stop (same problem order for all files: longest prompt
first; EVALPLUS_MAX_MEMORY_BYTES=-1, --min_time_limit 5 --gt_time_limit_factor 20 for every file, 0 timeouts left).
Q8_0's partial file was overwritten by the rerun, so the reference here is Q4_K_M (Q8 full run in progress).
| Qwen3-4B-Instruct-2507 (Unsloth) | size | same 86 problems: HumanEval / HumanEval+ / closes code | same 35 problems: HE+ |
|---|---|---|---|
| Q4_K_M | 2.5 GB | 80.2 / 73.3 / 93.0% | 71.4 |
| UD-Q2_K_XL | 1.7 GB | 62.8 / 59.3 / 84.9% | 60.0 |
| UD-IQ2_M | 1.5 GB | 52.3 / 50.0 / 82.6% | 51.4 |
| UD-IQ2_XXS | 1.3 GB | (35 done) 0.0 / 0.0 / 14.3% | 0.0 |
UD-IQ2_XXS is broken: it re-copies the prompt and runs to the 768-token limit without finishing code.
- Prediction "a clear cliff for the small model": **supported** (vs Q4_K_M: -14 at 1.7 GB, -23 at 1.5 GB, collapse at
  1.3 GB), unlike the 30B MoE (pilot 4b). Exact numbers to be replaced by the full runs.
- Repair target by the pre-registered rule (smallest file losing >= 8 pts that still closes code >= 80%):
  **UD-IQ2_M (1.5 GB)**: ~50% vs ~73% at 2.5 GB. Stretch target: UD-IQ2_XXS (1.3 GB, broken).

## FULL RESULT (2026-10-07, Mac M2 Pro, llama.cpp Metal, free; all 164 problems; graded with
## EVALPLUS_MAX_MEMORY_BYTES=-1 --min_time_limit 5 --gt_time_limit_factor 20, 0 timeouts)
| Qwen3-4B-Instruct-2507 (Unsloth) | size | HumanEval | HumanEval+ | vs Q8_0 | closes code | AUROC |
|---|---|---|---|---|---|---|
| Q8_0 (reference) | 4.3 GB | 89.0 | 82.3 | - | 97.6% | 0.743 |
| Q4_K_M | 2.5 GB | 86.6 | 80.5 | -1.8 | 96.3% | 0.749 |
| UD-Q2_K_XL | 1.7 GB | 73.2 | 68.9 | -13.4 | 90.9% | 0.775 |
| UD-IQ2_M | 1.5 GB | 61.0 | 57.9 | -24.4 | 87.8% | 0.781 |
| UD-IQ2_XXS | 1.3 GB | (35 problems) 0.0 | 0.0 | broken | 14% | - |
UD-IQ1_M (1.1 GB) not run: the run was stopped after UD-IQ2_M because UD-IQ2_XXS (larger) already collapsed and each
broken file takes ~45 min (runs to the token limit on every problem).
- Q4_K_M within 3 points of Q8_0: supported (-1.8). UD-IQ2_XXS loses >= 10: supported (collapse).
  UD-IQ1_M >= 25 loss: not measured (implied by the larger file's collapse).
- Repair target confirmed: UD-IQ2_M, 1.5 GB, 57.9 vs 82.3 (Q8) / 80.5 (Q4_K_M, 2.5 GB). Secondary: UD-Q2_K_XL.

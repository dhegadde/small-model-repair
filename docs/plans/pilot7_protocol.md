# Pilot 7: repair Unsloth's 1.5 GB file of Qwen3-4B-Instruct-2507 with a trained llama.cpp add-on (pre-registered
# 2026-10-07, before running)
Target (pilot 6, Mac): UD-IQ2_M 1.5 GB, HumanEval+ 57.9 vs Q8_0 82.3 and Q4_K_M 80.5 (2.5 GB); UD-Q2_K_XL 1.7 GB 68.9.
Method (repair/fb_small_recover.py; smoke-tested end to end on the Mac with Qwen3-0.6B: teacher via
llama-server, training against the GGUF's dequantized weights, convert_lora_to_gguf, llama-server --lora; the trained
fix gives the same top tokens in llama.cpp as in PyTorch, 20/20, |d logprob| 0.004):
- teacher: Q8_0 (llama.cpp) answers 2,000 prompts: 1,000 Python + 600 other languages (cpp, java, typescript, rust,
  csharp, php, swift, shell) from Magicoder-OSS-Instruct (decontaminated vs HumanEval/MBPP by its authors) + 400
  general chat prompts (UltraChat). Greedy, <= 512 tokens. Broad on purpose (side-agent note: avoid narrowing).
- student: UD-IQ2_M loaded by transformers (dequantized; = llama.cpp's weights). LoRA r16 on all 7 linear projections
  (~33M params, ~66 MB as f16 GGUF), CE on teacher answers (code: up to the first closed block; general: full answer if
  finished), 2 epochs, batch 8, LR 2e-4 (warm-up 20 steps, linear decay), seed 0.
- shipped as: UD-IQ2_M.gguf + fix.gguf (llama-server --lora).
Evaluation (fb_bench, llama.cpp, same GPU, same harness for all arms): HumanEval+, MBPP+, MultiPL-E JS and C++, MMLU
1,000. Arms: Q8_0 (reference), UD-IQ2_M, UD-IQ2_M + fix, UD-Q2_K_XL.
Predictions:
- F1: UD-IQ2_M + fix >= UD-IQ2_M + 10 points HumanEval+.
- F2: no damage elsewhere: on MMLU, JS, C++ the fixed model >= UD-IQ2_M - 2 points (expect gains on JS/C++).
- F3 (aspirational): UD-IQ2_M + fix (~1.57 GB) >= UD-Q2_K_XL (1.7 GB) on HumanEval+.
If F1 holds: 2 more seeds before any "confirmed" claim. If F1 fails: record, and try the larger target (UD-Q2_K_XL)
or a stronger loss (logit KL) only with a new protocol.
Cost: one 24 GB GPU (L4, $0.49/h): teacher ~45 min, training ~35-45 min, evaluation ~60 min: ~2.5-3 h, ~$1.25-1.50.

## RESULT (2026-10-08; teacher + training on A100 PCIe owlgzeqms0kjiq 17:04-17:50 UTC; eval on RTX 4090 pods
## x895ufuhlki8d4 17:55-19:00 and xh69akk89ygxk1 19:28-20:48; total ~$3.50 vs $1.25-1.50 estimated: see failures.md
## (peft/torchao, converter index file, unset $3 -> 45-min stall, idle pod))
Training: 1,387 of 2,000 teacher answers usable (code answers without a closed block and general answers cut at 512
tokens dropped); 33.0M LoRA params; 2 epochs ~16 min on A100; add-on file 64 MB (f16 GGUF).
All arms on the same RTX 4090, llama.cpp, fb_bench (greedy; HumanEval+/MBPP+ graded by EvalPlus on Linux):
| Qwen3-4B-Instruct-2507 | size | HumanEval+ | MBPP+ | JS | C++ | MMLU | HE closes code |
|---|---|---|---|---|---|---|---|
| Q8_0 (reference) | 4.3 GB | 82.3 | 63.2 | 84.5 | 82.6 | 71.4 | 97.6% |
| UD-Q2_K_XL (Unsloth) | 1.7 GB | 69.5 | 48.9 | 67.7 | 60.9 | 60.5 | 90.2% |
| UD-IQ2_M (Unsloth) | 1.5 GB | 62.8 | 45.0 | 65.8 | 52.2 | 58.9 | 87.8% |
| **UD-IQ2_M + trained fix** | 1.5 GB + 64 MB | **68.3** | **47.6** | **70.8** | **65.8** | **61.6** | **98.2%** |
| fix gain over UD-IQ2_M | | +5.5 | +2.6 | +5.0 | +13.6 | +2.7 | +10.4 |
| share of the gap to Q8_0 closed | | 28% | 14% | 27% | 45% | 22% | |
(UD-IQ2_M HumanEval+ was 57.9 on the Mac/Metal in pilot 6 and 62.8 here: engine/hardware numerics move greedy
outputs by a few points; within-run comparisons only.)
- F1 (fix >= UD-IQ2_M + 10 on HumanEval+): **falsified** (+5.5).
- F2 (no damage on MMLU/JS/C++, >= -2): **supported**, and the fix HELPED on all five tasks (5/5 positive): the broad
  teacher mix did not narrow the model. Largest gain on C++ (+13.6) and on finishing code (87.8 -> 98.2%).
- F3 (fixed 1.57 GB >= UD-Q2_K_XL 1.7 GB on HumanEval+): **not met** (68.3 vs 69.5, -1.2, within noise); across the
  five tasks the fixed file is ahead on 3 (JS +3.1, C++ +4.9, MMLU +1.1) and behind on 2 (HE+ -1.2, MBPP+ -1.3).
Claim status: suggestive / likely (one seed; per-task gains of 2.6-5.5 are within ~1-1.5 SE; C++ +13.6 and the
closes-code jump are well outside noise; consistent sign on all 5 tasks). Not confirmed.
Reading: the add-on works mechanically and helps everywhere, but recovers only ~15-45% of the loss with this recipe
(sequence-level KD on greedy teacher text, rank 16, ~1.4k examples, 2 epochs). Literature (Recover-LoRA, RILQ) uses
logit-level / model-output distillation, which gives far more signal per token: the obvious next step.

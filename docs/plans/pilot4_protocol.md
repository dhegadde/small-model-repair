# Pilot 4: is the trained per-expert repair better than what already exists? (pre-registered 2026-10-06, before running)
Approved by the human: "Lets do step 1, then step three, and then step 2 after" (step 1 = this comparison, ~$3-4 quoted).

Our model (pilot 3): Qwen3-Coder-30B-A3B, experts at real 2 bits + trained rank-8 per-expert corrections, 12.77 GB,
HumanEval+ 85.4 (packed) / 84.15 (simulated, mean of 4 runs); bf16 86.6.
Arms (all HumanEval+ 164, same prompt text, greedy, <= 768 tokens, stop at closing fence):
- svd_r8: same 2-bit experts + rank-8 corrections from the SVD of each expert's rounding error, NO data, NO training
  (the data-free compensators of MiLo, MLSys 2025, and Liu et al. 2512.17073). Same size as ours (12.77 GB).
  repair/fb_packed.py --svd_rank 8 (unit-tested: SVD corrections shrink the weight error).
- gguf UD-Q2_K_XL: Unsloth's public 2-bit release (11.79 GB), llama.cpp, batch 1.
- gguf UD-IQ3_XXS: Unsloth's public release closest in size to ours (12.85 GB), llama.cpp, batch 1.
  repair/fb_gguf_eval.py (smoke-tested on a local llama-server: prompt tokens identical to HF).
Predictions:
- C1: trained corrections beat SVD corrections of the same rank by >= 5 points HumanEval+ (ours 84.15 mean, so
  SVD <= 79.2). Falsified if SVD is within 5 points: then training adds little over the published data-free method.
- C2: ours beats UD-Q2_K_XL by >= 3 points.
- C3 (equal size): ours is not worse than UD-IQ3_XXS by more than 2.5 points. Falsified if IQ3_XXS > ours + 2.5:
  then at equal size an existing public file already does as well, and our method only matters at 2-bit experts.
Also recorded (step-3 baseline): llama.cpp batch-1 tokens/s and GPU memory for each GGUF.
Caveats stated with results: single runs (simulated spread 82.3-85.4); GGUF arms run a different inference engine
(llama.cpp kernels, its own numerics) and different quantizer formats (k-quants / i-quants with an importance
matrix) — the comparison is "what a user can download today", not a controlled ablation.
Cost: one 80-96 GB GPU ~2 h (SVD build ~15 min + packed eval ~60 min; llama.cpp build + 2 GGUF evals ~30 min): ~$4-5.

SCOPE NOTE (added 2026-10-07, while running, before any pilot-4 result was seen): our corrections were trained on
~800 Python answers and every test so far is HumanEval+ (Python). The GGUF releases and the SVD arm are general-purpose
(Unsloth's importance matrix is calibrated on general data; SVD uses no data). So C1-C3 compare on OUR home ground.
Any claim from this pilot must say "on Python coding (HumanEval+)". A generality check (non-Python code, general
knowledge, MBPP+) on the same arms is needed before a general claim; planned as pilot 5.

## RESULT (2026-10-07; RTX PRO 6000 Server Ed., pod sxfqh9047dcwb9 00:15-02:03 UTC, ~$3.75, auto-stopped)
| Qwen3-Coder-30B-A3B, HumanEval+ (164) | size | HumanEval | HumanEval+ | closes code | batch-1 speed |
|---|---|---|---|---|---|
| bf16 (pilot 1) | 61.1 GB | 91.5 | 86.6 | 96.3% | - |
| plain 2-bit experts, no repair (pilot 1, simulated) | 12.1 GB | 75.0 | 69.5 | 79.9% | - |
| svd_r8: 2-bit experts + data-free SVD rank-8 (real storage) | 12.77 GB | 80.5 | 78.0 | 85.4% | 10.4 tok/s |
| ours: 2-bit experts + trained rank-8 (real storage, pilot 3) | 12.77 GB | 89.0 | 85.4 | 96.3% | 10.4 tok/s |
| ours, simulated, mean of 4 runs | - | - | 84.15 | - | - |
| Unsloth UD-Q2_K_XL (llama.cpp) | 11.79 GB | 91.5 | 85.4 | 95.7% | 259 tok/s |
| Unsloth UD-IQ3_XXS (llama.cpp) | 12.85 GB | 91.5 | 89.0 | 97.6% | 252 tok/s |
GPU memory with llama.cpp server loaded (4k context): 12.2 GB (Q2_K_XL), 13.2 GB (IQ3_XXS). Prompt tokens identical
to the HF runs (441/441).
- C1 (trained beats SVD by >= 5): **supported**: 84.15 (mean of 4) / 85.4 (packed) vs 78.0 (+6.2 / +7.4). SVD repairs
  about half the 2-bit loss (69.5 -> 78.0), training about 86%. Every trained run (82.3-85.4) is above SVD.
- C2 (ours >= UD-Q2_K_XL + 3): **falsified**: tie at 85.4 (ours is 1 GB larger).
- C3 (ours not worse than UD-IQ3_XXS by > 2.5): **falsified**: IQ3_XXS 89.0 vs ours 85.4 / 84.15 at the same size.
- Speed: llama.cpp's real 2-bit/3-bit kernels are ~25x faster than our unoptimized PyTorch path.
Why the public files win at equal size: they spend bits better. Ours keeps attention, embeddings, router (3.08 GB)
and corrections (0.63 GB) in bf16 and gives every expert the same plain 2.5 bits; Unsloth's dynamic files quantize
those too and give more bits to sensitive tensors (importance matrix), so their experts get more than 2.5 bits on
average within the same ~12 GB (UD-Q2_K_XL ~3.1 bits/weight overall, ours ~3.35).
Claims: confirmed "trained per-expert corrections beat data-free SVD corrections of the same rank by ~6 points on
HumanEval+ (Qwen3-Coder-30B-A3B, 2-bit HQQ experts)" (4 trained runs vs a deterministic baseline; gap ~4x the run SD).
Withdrawn as a practical claim: "our 12.8 GB model is better than what users can download": it is not, even on Python.
Gate (agreed plan): we did not beat the public files on Python -> stop and rethink before pilot 5.

## PILOT 4b: where do the public files break? (pre-registered 2026-10-07, before running; approved "Do the cheap
## measurement", ~$1-1.50)
Score Unsloth's smaller dynamic files on HumanEval+ with the same harness (fb_gguf_eval, llama.cpp, batch 1):
UD-IQ2_M 10.84 GB, UD-IQ2_XXS 10.33 GB, UD-IQ1_M 9.63 GB, UD-IQ1_S 8.91 GB, UD-TQ1_0 8.01 GB.
Purpose: find the size where quality drops; that is where trained repair could add something nobody offers.
Prediction: a cliff below ~10.5 GB: IQ2_M >= 82; IQ1_M <= 75; TQ1_0 <= 60. Interesting target for repair = the
smallest file that loses >= 8 points (e.g. 85.4 -> <= 77) while still producing mostly working code.
Cheap 24-48 GB GPU (files <= 11 GB), ~45 min.

## PILOT 4b RESULT (2026-10-07; L4 24 GB pod 5cxr19rfzk3mpi 15:10-16:41 UTC, ~$0.75, auto-stopped)
| Unsloth dynamic GGUF (llama.cpp, L4) | file | GPU mem (4k ctx) | HumanEval | HumanEval+ | closes code | tok/s b1 |
|---|---|---|---|---|---|---|
| UD-IQ3_XXS (pilot 4, PRO 6000) | 12.85 GB | 13.2 GB | 91.5 | 89.0 | 97.6% | 252* |
| UD-Q2_K_XL (pilot 4, PRO 6000) | 11.79 GB | 12.2 GB | 91.5 | 85.4 | 95.7% | 259* |
| UD-IQ2_M | 10.84 GB | 10.9 GB | 89.0 | 86.6 | 97.0% | 96 |
| UD-IQ2_XXS | 10.33 GB | 10.4 GB | 89.0 | 85.4 | 98.2% | 97 |
| UD-IQ1_M | 9.63 GB | 9.7 GB | 87.8 | 84.8 | 96.3% | 100 |
| UD-IQ1_S | 8.91 GB | 9.0 GB | 82.9 | 79.9 | 93.3% | 107 |
| UD-TQ1_0 | 8.01 GB | 8.2 GB | 86.0 | 81.7 | 92.7% | 111 |
(* faster GPU; L4 speeds are not comparable to the PRO 6000 ones.) bf16 reference 86.6.
- Prediction (cliff below ~10.5 GB; IQ1_M <= 75; TQ1_0 <= 60): **falsified**. No cliff: down to 9.6 GB the public
  files equal bf16 within noise; at 8-9 GB they lose only ~5-7 points.
- Reading: for this MoE coder on HumanEval+, Unsloth's dynamic quantization is already near-lossless; the headroom
  for any repair is <= ~5-7 points at 8-9 GB, about 2x the run-to-run noise (SE ~3 pt on 164 problems).
  HumanEval+ may also be too easy to separate them (a harder benchmark could show bigger gaps).
- Implication for the program: competing on this model gives little room. Our earlier small dense models lost far
  more (Qwen3.5-4B 76.8 -> 59.8 at 3 bits; 9B 79.3 -> 51.2), so repair has more room there; to check against public
  low-bit files of a small dense coder first.

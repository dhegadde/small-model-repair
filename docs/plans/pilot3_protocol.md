# Pilot 3: the recovered 2-bit-expert coder stored at REAL 2 bits (pre-registered 2026-10-06, before running)

Why: pilots 1-2 SIMULATED 2-bit experts (values rounded, storage still bf16). The confirmed result (84.2 ± 1.6
HumanEval+, 3 seeds) says nothing yet about real memory or speed. Nothing goes public until this is measured.

Size correction (made while designing this): the "~11.6 GB" quoted in pilot 2 left out the per-group scale and zero.
Exact count for Qwen3-Coder-30B-A3B: experts 28.99B weights at 2.5 bits (2-bit codes + fp16 scale & zero per 64) =
9.06 GB; attention, router, embeddings, lm_head 1.54B in bf16 = 3.08 GB; rank-8 corrections 314.6M in bf16 = 0.63 GB;
total 12.77 GB (bf16 model 61.1 GB). Correct to ~12.8 GB everywhere.

Method (repair/fb_packed.py; tests/test_fb_packed.py passes on a toy Qwen3-MoE: packed weights match the
simulated ones to < 0.5%, 2.5 bits per weight exactly):
1. Retrain the experts arm, seed 0, exactly as pilot 2 (the trained corrections were not saved then); this run now
   saves deltas.pt and also measures the SIMULATED model's speed and memory. Graded on HumanEval+.
2. Load bf16 to CPU, quantize each expert with the same HQQ call, store packed 2-bit codes + fp16 scale/zero on the GPU,
   keep the corrections as separate bf16 add-ons (not folded in). Measure: GPU memory after load, packed file size,
   peak memory while generating, tokens/s (batch 16 eval and batch 1). Generate + grade HumanEval+ as before.
Predictions:
- R1 memory: weights on GPU <= 13.0 GB; file size <= 13.0 GB.
- R2 quality: packed HumanEval+ within 2.5 points of the simulated score from the same training run (step 1), and
  >= 80%. Falsified if more than 2.5 points lower: then the rounding of scales/corrections matters, and the
  "12.8 GB" claim cannot be made yet.
- R3 speed: no prediction; this is unoptimized PyTorch (Python-level unpacking, no fused 2-bit kernel). Reported as
  measured, next to the simulated model's speed on the same GPU. A fast kernel / llama.cpp port is a later step.
Cost: one 80-96 GB GPU, ~2.5-3 h (retrain ~1.3 h incl. eval; packed build + eval ~1-1.5 h): ~$4-6.

## RESULT (2026-10-06; RTX PRO 6000 Server Ed., pod ejsvf8tfdts8we 21:03-23:45 UTC ~$5.65, + ~$0.90 on a throttled
## Workstation-Ed. pod deleted after 25 min; total ~$6.55, auto-stopped)
| Qwen3-Coder-30B-A3B, same trained corrections | HumanEval | HumanEval+ | AUROC | closes code |
|---|---|---|---|---|
| simulated 2-bit (bf16 storage), this run | 87.8 | 84.1 | 0.634 | 97.0% |
| **REAL 2-bit storage (packed)** | **89.0** | **85.4** | 0.737 | 96.3% |
| (bf16 original, pilot 1) | 91.5 | 86.6 | 0.644 | 96.3% |
Memory (measured): weights on GPU 12.77 GB = packed file 12.77 GB = the pre-registered count (bf16: 61.1 GB, 4.8x
smaller). Peak GPU memory while generating: 14.35 GB at batch 1 (256 new tokens), 15.89 GB at batch 16 (<= 768 new).
Speed (unoptimized PyTorch, Python-level unpacking): 10.4 tokens/s at batch 1, 17.7 tokens/s aggregate at batch 16.
The simulated model measured 3.1 / 21.0 tokens/s, but it recomputes every expert's correction on each forward, so it
is NOT a fair bf16 speed baseline; a plain bf16 speed was not measured.
- R1 (weights <= 13.0 GB, file <= 13.0 GB): supported (12.77 / 12.77).
- R2 (packed within 2.5 pt of simulated, >= 80%): supported: 85.4 vs 84.1 (+1.3; 5 tasks flip to pass, 3 to fail;
  only 39/164 solutions are character-identical: tiny numeric differences change greedy text, not the pass rate).
- Seed-0 rerun of the simulated model: 84.1 (pilot 2 seed 0: 84.8); 4-run simulated mean now 84.15.
Claim status: confirmed for memory (measured, matches the count exactly); likely for "real storage keeps the
quality" (one run, but the difference is within noise and in the favourable direction).
Not yet shown: speed competitive with real 2-bit kernels (llama.cpp / Marlin-style); comparison with SVD
compensators and with public 2-bit releases (e.g. GGUF Q2 variants).

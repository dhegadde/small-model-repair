# Pilot 2: recovering 2-bit experts in Qwen3-Coder-30B-A3B (pre-registered 2026-10-05; needs spend OK)

From pilot 1 (HumanEval+): bf16 86.6, experts 3-bit 84.8, experts 2-bit 69.5 (-17.1; 20% of answers do not finish
their code). Goal: recover the 2-bit-expert model (~11 GB if stored) toward the bf16 / 3-bit level.
Teacher: the bf16 model's own answers to 1,000 Magicoder-OSS-Instruct Python problems (<= 512 tokens), cut after the
first closed code block (code-only recipe, confirmed in compression-confidence Phase 3b).
Arms (one seed each; experts at 2 bits, HQQ group 64, simulated):
- attn: LoRA r16 on attention q/k/v/o only; experts and router frozen. Can the non-expert layers compensate?
- experts: rank-8 correction per expert matrix (W_e + A_e B_e on the fused tensors; ~315M trainable, ~1% of params).
Both: cross-entropy on teacher code, 2 epochs, batch 4, LR 1e-4. Eval HumanEval+ (164), as pilot 1.
Code: repair/fb_recover.py (toy-MoE test: no-op at init, adapters learn, frozen 2-bit experts untouched).
Predictions: P1 attn recovers < 30% of the gap (<= 74.9%); P2 experts recovers >= 50% (>= 78.1%);
P3 the recovered model finishes its code >= 90% of the time.
Cost: one 80-96 GB GPU, ~4 h (teacher ~1 h shared; each arm ~1.3-1.8 h incl. eval): ~$7-10.
Caveat to state with results: the experts arm stores small fp32 corrections next to the 2-bit experts (~0.6 GB),
so the deployable size is ~11.6 GB, not 11 GB.

## RESULT (2026-10-05/06; RTX PRO 6000, pod 21:56-00:55 UTC, ~$6.25, auto-stopped)
Teacher: 1,008 answers from bf16 (~66 min); 813 with a closed code block used for training.
| Qwen3-Coder-30B-A3B, HumanEval+ (164) | HumanEval | HumanEval+ | AUROC | closes code | approx. size |
|---|---|---|---|---|---|
| bf16 | 91.5 | 86.6 | 0.644 | 96.3% | ~61 GB |
| experts 2-bit, no recovery | 75.0 | 69.5 | 0.838 | 79.9% | ~11 GB |
| experts 2-bit + attention LoRA | 83.5 | 79.3 | 0.714 | 95.1% | ~11 GB |
| **experts 2-bit + per-expert rank-8 corrections** | **87.2** | **84.8** | 0.734 | **98.2%** | ~11.6 GB |
| (experts 3-bit, no recovery, pilot 1) | 90.2 | 84.8 | 0.614 | 95.7% | ~14 GB |
- P2 (experts arm >= 78.1%): **supported**: 84.8%, recovering 89% of the 17.1-point gap; equals the 3-bit model in
  ~2.4 GB less memory. P3 (>= 90% finished code): supported (98.2%).
- P1 (attention-only < 30% of the gap): **falsified**: attention-only LoRA recovered 57% (79.3%).
- Confidence remains informative (AUROC 0.73).
- Status: likely (one seed, simulated quantization, 164 problems; SE ~2.7pt at this accuracy).
- Training: experts arm ~25 min, 315M trainable params, 67 GB peak GPU memory.

## SEEDS (pre-registered 2026-10-06, before running)
Approved: "lets do the seeds" (~$4-6). Repeat ONLY the experts arm (rank 8, 2-bit, same settings) with seeds 1 and 2.
Same cached teacher (runs/fb_pilot2/teacher.jsonl); the seed changes the correction init (B) and the data order.
So the spread measures training noise, not teacher noise (stated as a limit).
Predictions: S1 mean HumanEval+ over seeds 0-2 >= 81.5% (>= 70% of the 17.1-pt gap recovered) and every seed
>= 78.1% (P2 holds per seed); S2 mean closes-code >= 95%.
Falsified if: the mean < 78.1% or any seed < 75%; then P2 is downgraded to "suggestive" and seed 0 is treated as lucky.
Bar for "confirmed" (claim-review): 3 seeds, mean recovery well above the noise (SE ~2.7 pt per run),
strongest objection (seed luck) tested by this run. Simulated quantization remains a stated limit.
Cost: one 80-96 GB GPU, ~2.7 h (each seed ~75 min incl. load, quantize, train, eval): ~$5.5-6.

## SEEDS RESULT (2026-10-06; A100-SXM4-80GB, pod lvan4w9khywz3s 17:03-~20:20 UTC, ~$5.3, auto-stopped)
| experts arm, 2-bit + rank-8 per-expert corrections | HumanEval | HumanEval+ | AUROC | closes code |
|---|---|---|---|---|
| seed 0 (pilot 2) | 87.2 | 84.8 | 0.734 | 98.2% |
| seed 1 | 86.6 | 82.3 | 0.714 | 97.0% |
| seed 2 | 89.0 | 85.4 | 0.711 | 95.7% |
| **mean ± SD** | 87.6 ± 1.3 | **84.2 ± 1.6** | 0.720 | 97.0% |
- S1 (mean >= 81.5, every seed >= 78.1): **supported**: mean 84.2; min 82.3. Mean recovery 86% of the 17.1-pt gap
  (69.5 -> 84.2; bf16 86.6). Effect over the unrepaired 2-bit model (+14.7) is ~9x the seed SD.
- S2 (mean closes-code >= 95%): supported (97.0%).
- Claim status (claim-review): **confirmed**: "with experts at 2 bits (simulated, HQQ g64), rank-8 per-expert
  corrections trained on ~800 self-distilled code answers recover ~86% of the HumanEval+ loss on Qwen3-Coder-30B-A3B
  (84.2 ± 1.6 vs 86.6 bf16), at ~11.6 GB of weights if stored." Limits stated with it: simulated quantization (no
  real low-bit storage or speed measured yet); one teacher set; one benchmark (164 problems); one model.
- Training on A100 ~47 min/seed (vs ~25 min on RTX PRO 6000); ~95 min per seed incl. load, quantize, eval.


CORRECTION (2026-10-06): '~11.6 GB' above omitted the per-group fp16 scale/zero; the real stored size is ~12.8 GB
(see pilot3_protocol.md). Measured in pilot 3.

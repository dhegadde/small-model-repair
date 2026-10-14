# Pilot 8: a stronger fix by logit-level distillation (pre-registered 2026-10-09, before running)
Approved: "Yes do it all on gpu" (build the stronger version; tests run on the GPU, not the Mac).
Pilot 7 (sequence-level KD = cross-entropy on greedy teacher text, 1,387 usable answers): the 64 MB fix helped on all
5 tasks but closed only 14-45% of the gap (HE+ 62.8 -> 68.3) and tied Unsloth's 1.7 GB UD-Q2_K_XL.
Change (one variable family: the training signal): --loss kd = KL(full-precision Qwen3-4B-Instruct-2507 || student)
on every response token's full next-token distribution, over all 2,000 teacher answers (same prompts and texts as
pilot 7; no answers dropped; <= 1,536 tokens). Same student (UD-IQ2_M dequantized), LoRA r16 on the 7 projections
(64 MB), 2 epochs, batch 8, LR 2e-4, warm-up 20, linear decay. Literature: RILQ (model-output discrepancy loss),
Recover-LoRA (logit distillation) report stronger 2-bit recovery than text-only training.
GPU smoke test first (same pod): KL(x||x)=0 check; 16-example KD run with peak-memory readout.
Arms (same GPU, llama.cpp, fb_bench: HumanEval+, MBPP+, MultiPL-E JS, C++, MMLU 1,000): Q8_0, UD-Q2_K_XL, UD-IQ2_M,
UD-IQ2_M + CE fix (pilot 7 adapter, re-measured here), UD-IQ2_M + KD fix seeds 0, 1, 2.
Predictions:
- K1: KD fix (mean of 3 seeds) >= UD-IQ2_M + 10 points on HumanEval+.
- K2: KD fix closes >= 40% of the gap to Q8_0 averaged over the 5 tasks (CE fix: ~27%).
- K3: KD fix (1.5 GB + 64 MB) >= UD-Q2_K_XL (1.7 GB) on >= 4 of 5 tasks (mean of seeds).
- K4: KD fix > CE fix on the 5-task average.
Falsified -> logit KD at rank 16 is not enough; next lever would be rank / data size (new protocol).
Cost: one 24-80 GB GPU (~3 h on an RTX 4090 at ~$0.89/h; ~2 h on an A100 at ~$1.8/h): ~$2.7-3.6.

## RESULT (2026-10-09; RTX 4090 pod x2g7vl6haop7ek 18:52-22:37 UTC, ~$3.35 of which ~$1.60 idle after ALL DONE
## 20:49 because pod_watch hung in rsync, see failures.md; work itself ~2 h / ~$1.75)
GPU smoke test: KL(x||x)=0 ok; KD training peak 18.2-18.8 GB on the 24 GB card; 2,000/2,000 answers used;
~16 min per seed. All arms on this GPU, llama.cpp, fb_bench; HumanEval+/MBPP+ by EvalPlus (plus = base+extra pass).
| Qwen3-4B-Instruct-2507 | size | HumanEval+ | MBPP+ | JS | C++ | MMLU | gap to Q8 closed (5-task mean) |
|---|---|---|---|---|---|---|---|
| Q8_0 | 4.3 GB | 82.9 | 63.2 | 84.5 | 84.5 | 71.2 | - |
| UD-Q2_K_XL (Unsloth) | 1.7 GB | 64.0 | 49.7 | 68.3 | 57.8 | 60.2 | 17% |
| UD-IQ2_M (Unsloth) | 1.5 GB | 61.0 | 44.2 | 65.2 | 53.4 | 58.6 | - |
| + CE fix (pilot 7) | +64 MB | 68.3 | 47.4 | 70.2 | 65.8 | 61.5 | 28% |
| + KD fix s0 / s1 / s2 | +64 MB | 70.1 / 72.0 / 70.7 | 51.1 / 51.1 / 48.7 | 72.7 / 68.3 / 69.6 | 68.9 / 65.2 / 64.6 | 61.8 / 61.5 / 61.3 | 38 / 33 / 30% |
| **+ KD fix, mean ± SD** | +64 MB | **70.9 ± 0.9** | **50.3 ± 1.4** | **70.2 ± 2.3** | **66.2 ± 2.3** | **61.5 ± 0.3** | **34%** |
- K1 (KD >= UD-IQ2_M + 10 HE+): **just missed** (+9.9). K2 (>= 40% of the gap): **falsified** (34%).
- K3 (KD >= UD-Q2_K_XL on >= 4/5 tasks): **supported, 5/5**: HE+ +6.9, C++ +8.4, JS +1.9, MMLU +1.3, MBPP+ +0.6
  (the last three within noise).
- K4 (KD > CE on the 5-task mean): supported, modestly (34% vs 28%; HE+ +2.6, MBPP+ +2.9, C++ +0.4, JS 0, MMLU 0).
Measurement noise: UD-Q2_K_XL scored 69.5 HE+ in pilot 7 and 64.0 here (same file, llama.cpp with 8 parallel slots
is not bit-reproducible): single-run baseline numbers move by up to ~5 points; KD seeds vary 0.3-2.3 pt by task.
Claims (claim-review):
- **confirmed**: the KD fix improves Unsloth's UD-IQ2_M on all five tasks (3 seeds; HE+ +9.9, C++ +12.8, JS +5.0,
  MBPP+ +6.1, MMLU +2.9; every seed above the base on every task) and makes it finish code (85 -> ~97%).
- **likely**: UD-IQ2_M + KD fix (1.5 GB + 64 MB) is better than Unsloth's larger UD-Q2_K_XL (1.7 GB) on code (HE+
  and C++ margins exceed the baseline's run-to-run swing; JS/MBPP+/MMLU are ties). Needs repeated baseline runs (or
  --parallel 1 deterministic evaluation) to be confirmed.
- not supported: closing most of the gap; recovery is ~1/3 of what 2-bit costs.

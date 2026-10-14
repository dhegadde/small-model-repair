# Small Model Repair

Research code and results on **repairing language models that were compressed to about 2 bits**, by training a
small add-on (a LoRA "fix") that loads in [llama.cpp](https://github.com/ggml-org/llama.cpp) with `--lora`.
Dheeraj Gadde, 2026. Work in progress, not peer reviewed.

## Main result so far

Unsloth's public 1.5 GB file of Qwen3-4B-Instruct-2507 (`UD-IQ2_M`) plus a **64 MB trained fix** scores higher than
their larger 1.7 GB file (`UD-Q2_K_XL`) on all five tests. Everything below ran in llama.cpp on the same GPU
(RTX 4090); the fix is the mean ± SD of 3 training seeds.

| Qwen3-4B-Instruct-2507 | size | HumanEval+ | MBPP+ | JavaScript | C++ | MMLU (1,000) |
|---|---|---|---|---|---|---|
| Q8_0 (near-original) | 4.3 GB | 82.9 | 63.2 | 84.5 | 84.5 | 71.2 |
| Unsloth UD-Q2_K_XL | 1.7 GB | 64.0 | 49.7 | 68.3 | 57.8 | 60.2 |
| Unsloth UD-IQ2_M | 1.5 GB | 61.0 | 44.2 | 65.2 | 53.4 | 58.6 |
| **UD-IQ2_M + trained fix** | **1.5 GB + 64 MB** | **70.9 ± 0.9** | **50.3 ± 1.4** | **70.2 ± 2.3** | **66.2 ± 2.3** | **61.5 ± 0.3** |

How sure we are:
- **Confirmed**: the fix improves the 1.5 GB file on every task (every seed, every task), most on code
  (HumanEval+ +9.9, C++ +12.8), and it fixes rambling: the model finishes its code ~97% of the time instead of 85%.
- **Likely, not yet confirmed**: the fixed 1.5 GB file beats the 1.7 GB file. The code margins (HumanEval+ +6.9,
  C++ +8.4) are larger than the run-to-run swing of the baseline, but the same 1.7 GB file scored 69.5 HumanEval+ in
  an earlier run (llama.cpp with parallel requests is not bit-reproducible). JS, MBPP+ and MMLU are ties.
- **Not achieved**: the fix recovers only about a third of what 2-bit compression loses vs the near-original model.

How the fix is trained (`repair/fb_small_recover.py`): the near-original model answers 2,000 prompts (1,000 Python,
600 in eight other languages, 400 general chat); the student is the public low-bit file itself, loaded by
`transformers` with its exact dequantized weights; a rank-16 LoRA on every linear layer learns to match the
full-precision model's next-token distributions (logit distillation, KL) on those answers. A text-only version
(cross-entropy on the teacher's answers) gave smaller gains (HumanEval+ 68.3).

## How we got here (all predictions were written down before each run: `docs/plans/`)

| Pilot | Question | Answer |
|---|---|---|
| 1 | Qwen3-Coder-30B-A3B (MoE): how low can the experts go? | 4 and 3 bits ~lossless; 2 bits drops HumanEval+ 86.6 -> 69.5 |
| 2 | Can small trained per-expert fixes repair 2-bit experts? | Yes: 84.2 ± 1.6 (3 seeds), ~86% of the loss recovered |
| 3 | Does it hold with real 2-bit storage? | Yes: 12.77 GB measured (61 GB original), HumanEval+ 85.4 |
| 4 | Better than existing options? | Trained fixes beat data-free SVD fixes (78.0) by ~6 points, **but** Unsloth's public files of the same size were as good or better (85.4-89.0) and ~25x faster |
| 4b | Where do public files of the 30B MoE break? | They don't, down to ~8-9 GB: little room for any repair |
| 6 | Where do public files of a *small* model break? | Qwen3-4B: 82.3 (8-bit) -> 68.9 (1.7 GB) -> 57.9 (1.5 GB) -> broken (1.3 GB) |
| 7 | Text-trained fix on the 1.5 GB file | Helps on all 5 tasks; ties the 1.7 GB file |
| 8 | Logit-distilled fix, 3 seeds | The table above |

The big MoE result is a negative one worth keeping: on a 30B mixture-of-experts coder, existing public quantization is
already near-lossless at 2-3 bits, so the room for repair is on small dense models.

## Layout

| Path | What |
|---|---|
| `repair/fb_small_recover.py` | teacher answers (via llama-server), fix training (`--loss ce` or `--loss kd`) against a public GGUF's own weights |
| `repair/fb_bench.py` | HumanEval+/MBPP+ (EvalPlus), MultiPL-E JS/C++ (executed with node/g++), MMLU; vLLM or llama.cpp backend |
| `repair/fb_gguf_eval.py` | HumanEval+ for any GGUF through llama-server, same prompts as the PyTorch runs |
| `repair/fb_recover.py`, `fb_packed.py`, `fb_fold.py` | MoE pilots: per-expert low-rank corrections, real 2-bit packed experts, folding for fast evaluation |
| `repair/cc_code.py`, `cc_code_analyze.py` | HumanEval+ generation with confidence (mean token log-prob), pass@1 / AUROC tables |
| `scripts/runpod/` | the exact GPU run scripts, one per pilot |
| `scripts/pod_watch.sh` | watches a GPU pod, copies results, and stops it when the job ends, hangs, or fills the disk |
| `docs/` | the research mission and every pilot's pre-registered protocol with its results and failed predictions |
| `results/` | per-problem outputs, grades and summaries behind every number above |

## Reproduce

Tests (CPU, a few seconds): `python -m unittest discover tests`
(the MultiPL-E harness tests need `python -m repair.fb_bench prep` first).

Pilot 8 on one 24 GB GPU (~2 h): run `scripts/runpod/fb_pilot7.sh` (teacher answers) then
`scripts/runpod/fb_pilot8.sh`. They expect a Python venv with `torch`, `transformers`, `peft`, `datasets`,
`huggingface_hub`, a separate venv with `evalplus`, `node` and `g++`; they download the GGUF files and build
llama.cpp themselves. Pilot 8 cost about $1.75 of GPU time.

## Limits

One small model family (Qwen3-4B) for the repair result; HumanEval+ and MultiPL-E have 161-164 problems each
(~3-4 points of sampling noise); MMLU is scored by the most likely answer letter on a fixed 1,000-question sample;
greedy decoding only. The trained fix files are not in this repository yet.

License: MIT (see `LICENSE`; third-party notes in `NOTICE.md`).

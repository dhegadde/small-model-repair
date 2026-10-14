# Pilot 5: does the Python-trained 2-bit repair hold up OUTSIDE Python coding? (pre-registered 2026-10-07, written
# before any pilot-4 result was seen)
Why: our corrections were trained on ~800 Python answers and only tested on HumanEval+ (side-agent note, accepted).
Public releases are general-purpose. This pilot decides whether the claim is "general" or "Python specialist".
Order agreed with the human: pilot 4 -> pilot 5 -> speed (step 3) -> rotation (step 2). Run only if pilot 4 shows we
beat the alternatives on Python (otherwise rethink first); needs a spend OK.

Arms (Qwen3-Coder-30B-A3B):
- bf16 (reference); ours = 2-bit experts + trained rank-8 corrections (pilot 3 deltas); svd_r8 = data-free rank-8;
  plain2 = 2-bit experts, no corrections. These four run in vLLM from bf16 checkpoints written by fb_fold (folded
  weights = exactly the packed model's weights; unit test: identical logits on a toy MoE).
- UD-Q2_K_XL (11.79 GB) and UD-IQ3_XXS (12.85 GB, same size as ours), llama.cpp.
Tasks (repair/fb_bench.py; harness unit-tested, smoke-tested end to end on a local llama-server):
- humaneval (HumanEval+, sanity: ours in vLLM should land in 82-86), mbpp (MBPP+ 378, Python, different style),
- js, cpp (MultiPL-E HumanEval translations, 161 each, executed with node / g++),
- mmlu (1,000 fixed test questions, general knowledge, most likely letter).
Predictions:
- G1 no damage outside Python: on js, cpp and mmlu, ours >= plain2 - 2 points each.
- G2 code transfers: on js and cpp, ours >= plain2 + 3 points (the repair is about code, not only Python).
- G3 MBPP+: ours >= plain2 + 5 points (not just HumanEval's style).
- G4 vs the equal-size public file (UD-IQ3_XXS): on mmlu, no prediction of a win (expect IQ3 >= ours: general
  calibration); on js/cpp ours >= IQ3_XXS - 3.
Reading: if G1 fails, the repair narrowed the model -> the step-3 retraining uses a broad teacher mix (code in several
languages + general questions) and the public claim says "Python coding" only until fixed.
Caveats: single runs; vLLM and llama.cpp numerics differ (engine effect is within the 2-3-point run-to-run spread seen
in pilots 2-3); MMLU by next-token letter only.
Cost: one 80-96 GB GPU ~2 h: ~$4-5.

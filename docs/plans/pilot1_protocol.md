# Pilot 1: low-bit experts in a modern MoE coder (pre-registered 2026-10-05; needs the human's spend OK)

Model: Qwen/Qwen3-Coder-30B-A3B-Instruct (30.5B total, 3.3B active, 48 layers x 128 experts, Apache 2.0).
Experts are ~95% of the parameters, stored as fused 3-D tensors in transformers 5.17, so they are compressed with
simulated quantization (`cc_code.fake_quantize_experts_`: HQQ optimizer per expert matrix, group 64, dequantized
back to bf16; tested on toy experts). Attention, router, embeddings stay bf16.
Memory if stored for real: bf16 ~61 GB; experts at 2 bits + rest bf16 ~11 GB; at 3 bits ~14 GB; at 4 bits ~18 GB.
Eval: HumanEval+ (164), as in compression-confidence Phase 3 (greedy, thinking off), pass@1 + confidence AUROC.
Arms: bf16 (reference), experts 4 / 3 / 2 bits.
Literature (2026-10-05): expert layers tolerate low bits better than dense FFNs; 2-bit experts can still beat dense
models (MoQE, arXiv 2310.02410; QuantMoE-Bench 2406.08155; MoPEQ 2509.02512); down projections need more precision.
Predictions:
- P1: 4-bit experts within 2 points of bf16.
- P2: 3-bit experts within 5 points (unlike the dense Qwen3.5 models' -17 to -28 at 3 bits).
- P3: 2-bit experts lose >= 10 points (the recovery target for pilot 2).
- P4: confidence AUROC within 0.05 of bf16 at 3 bits.
Cost: 1x RTX PRO 6000 96 GB (~$2.09/hr) or equivalent, ~2-2.5 h incl. a 61 GB download: ~$4-6.
Follow-up (separate OK): recovery for 2-bit experts.

## RESULT (2026-10-05; RTX PRO 6000, pod 20:04-21:12 UTC, ~$2.37)
| Qwen3-Coder-30B-A3B, HumanEval+ (164) | HumanEval | HumanEval+ | AUROC | closes code | approx. memory if stored |
|---|---|---|---|---|---|
| bf16 | 91.5 | 86.6 | 0.644 | 96.3% | ~61 GB |
| experts 4-bit | 90.2 | 85.4 | 0.687 | 95.7% | ~18 GB |
| experts 3-bit | 90.2 | 84.8 | 0.614 | 95.7% | ~14 GB |
| experts 2-bit | 75.0 | 69.5 | 0.838 | 79.9% | ~11 GB |
(EvalPlus's own pass@1 for bf16 printed 86.0; the table uses our per-task analysis, 86.6.)
- P1 (4-bit within 2pt): supported (-1.2). P2 (3-bit within 5pt): supported (-1.8). P3 (2-bit loses >= 10pt):
  supported (-17.1). P4 (AUROC within 0.05 at 3 bits): supported (-0.03).
- Expert compression is far gentler than dense compression: 3-bit experts cost 1.8 points here vs 17-28 points for
  dense Qwen3.5 models at 3 bits. A ~14 GB version scores 84.8% HumanEval+, above every dense model we tested at any
  precision (best: Qwen3.8-27B 8-bit, 82.9%).
- 2-bit experts are the cliff (and 20% of answers stop finishing their code): the recovery target for pilot 2.
- Status: likely (one run each; simulated quantization measures quality, not runtime memory).

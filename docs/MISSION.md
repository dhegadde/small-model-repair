# Near-frontier models on very little GPU (PROPOSED 2026-10-05, from the human's goal)

Human's goal: "research and find a way for us to run Opus-level LLMs with really tiny GPU usage."

## Honest framing (literature/web check 2026-10-05)
- The strongest open-weight models (GLM-5.x, Kimi K2.x, Qwen3.5-397B, DeepSeek V3.x, MiniMax M2.x) are within a few
  to ~10 points of Claude Opus on public coding/reasoning benchmarks (blog-reported numbers; verify before citing).
  They are mixture-of-experts models: 300B-1T total parameters, ~30-40B active per token.
- GPU size is already solved by expert offloading: KTransformers (SOSP'25) and llama.cpp run a 671B MoE on ONE 24 GB
  GPU plus system RAM at ~10-20 tok/s, keeping attention/KV/shared parts on GPU and routed experts in RAM.
  The bottleneck moves to SYSTEM MEMORY (~400-700 GB at 4-8 bits).
- Matching Opus at everything on a tiny GPU is not achievable today; the levers below can get close on average
  quality, or match it on narrow tasks.

## Research directions (our lab's comparative advantage: compression + recovery + routing)
1. Extreme expert compression + recovery: quantize routed experts to ~2-3 bits (attention/shared parts higher),
   then copy-the-full-model recovery (our confirmed method). Target: 671B-class at ~200 GB RAM, small quality loss.
   Open question: are experts more or less robust to low bits than dense layers? Does recovery work per expert?
2. Local-first cascades (typed-decision-heads Phase 7 result): a small local model answers what it is confident
   about; escalate the rest to a big model. Measure the accuracy-vs-compute frontier with an open near-frontier
   model as the fallback.
3. Narrow distillation: a small model trained on a big model's answers for one domain (e.g. the user's coding).
## First steps (to cost before running)
- Literature check: MoE quantization (expert sensitivity), MoE offloading (KTransformers, DALI arXiv 2602.03495,
  Mixtral offloading arXiv 2312.17238), speculative decoding.
- Pilot on a mid-size open MoE that fits our pods (e.g. Qwen3-Coder-30B-A3B or a ~100B-class MoE): expert bit-width
  sweep + recovery. Needs a spend OK.

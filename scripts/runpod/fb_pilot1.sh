#!/bin/bash
# Frontier-on-a-budget pilot 1: Qwen3-Coder-30B-A3B, experts at bf16 / 4 / 3 / 2 bits (simulated), HumanEval+.
. scripts/runpod/cc_phase3_common.sh
setup_eval
OUT=runs/fb_pilot1; mkdir -p $OUT
for q in bf16 experts4 experts3 experts2; do
  d=$OUT/qwen3coder30b_$q
  python -u -m repair.cc_code --model Qwen/Qwen3-Coder-30B-A3B-Instruct --quant $q --datasets humaneval \
    --out $d > $d.log 2>&1 && echo "$q generated $(date -u +%H:%M:%S)" || { echo "$q generation failed"; continue; }
  $EV/bin/evalplus.evaluate --dataset humaneval --samples $d/humaneval_samples.jsonl --parallel 8 > $d/humaneval_grade.log 2>&1 \
    && echo "$q graded" || echo "$q grading failed"
done
echo "ALL DONE $(date -u +%H:%M:%S)"

#!/bin/bash
# Frontier pilot 2: teacher once, then the 'experts' arm (main) and the 'attn' arm, each graded on HumanEval+.
. scripts/runpod/cc_phase3_common.sh
setup_eval
OUT=runs/fb_pilot2; mkdir -p $OUT
T=$OUT/teacher.jsonl
for arm in experts attn; do
  d=$OUT/$arm
  python -u -m repair.fb_recover --arm $arm --teacher $T --out $d > $d.log 2>&1 \
    && echo "$arm trained+generated $(date -u +%H:%M:%S)" || { echo "$arm failed $(date -u +%H:%M:%S)"; continue; }
  $EV/bin/evalplus.evaluate --dataset humaneval --samples $d/humaneval_samples.jsonl --parallel 8 > $d/humaneval_grade.log 2>&1 \
    && echo "$arm graded" || echo "$arm grading failed"
done
echo "ALL DONE $(date -u +%H:%M:%S)"

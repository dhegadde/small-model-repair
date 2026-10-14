#!/bin/bash
# Frontier pilot 2 seeds: the 'experts' arm with seeds 1 and 2, same cached teacher; graded on HumanEval+.
. scripts/runpod/cc_phase3_common.sh
setup_eval
OUT=runs/fb_pilot2; mkdir -p $OUT
T=$OUT/teacher.jsonl
[ -f $T ] || { echo "FAILED: no teacher at $T"; exit 1; }
for s in 1 2; do
  d=$OUT/experts_s$s
  python -u -m repair.fb_recover --arm experts --seed $s --teacher $T --out $d > $d.log 2>&1 \
    && echo "seed $s trained+generated $(date -u +%H:%M:%S)" || { echo "seed $s failed $(date -u +%H:%M:%S)"; continue; }
  $EV/bin/evalplus.evaluate --dataset humaneval --samples $d/humaneval_samples.jsonl --parallel 8 > $d/humaneval_grade.log 2>&1 \
    && echo "seed $s graded" || echo "seed $s grading failed"
done
echo "ALL DONE $(date -u +%H:%M:%S)"

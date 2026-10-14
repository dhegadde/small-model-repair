#!/bin/bash
# Frontier pilot 3: retrain the experts arm (seed 0, saving the corrections), then the same model at REAL 2 bits.
. scripts/runpod/cc_phase3_common.sh
setup_eval
T=runs/fb_pilot2/teacher.jsonl; OUT=runs/fb_pilot3; mkdir -p $OUT
[ -f $T ] || { echo "FAILED: no teacher at $T"; exit 1; }
grade() { $EV/bin/evalplus.evaluate --dataset humaneval --samples $1/humaneval_samples.jsonl --parallel 8 > $1/humaneval_grade.log 2>&1 \
  && echo "$1 graded" || echo "$1 grading failed"; }
python -u -m repair.fb_recover --arm experts --seed 0 --teacher $T --out $OUT/sim > $OUT/sim.log 2>&1 \
  && echo "sim trained+generated $(date -u +%H:%M:%S)" || { echo "FAILED: sim $(date -u +%H:%M:%S)"; exit 1; }
grade $OUT/sim
python -u -m repair.fb_packed --deltas $OUT/sim/deltas.pt --out $OUT/packed > $OUT/packed.log 2>&1 \
  && echo "packed generated $(date -u +%H:%M:%S)" || echo "packed failed $(date -u +%H:%M:%S)"
[ -f $OUT/packed/humaneval_samples.jsonl ] && grade $OUT/packed
echo "ALL DONE $(date -u +%H:%M:%S)"

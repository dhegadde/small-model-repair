#!/bin/bash
# Pilot 6 on the Mac (free): Unsloth GGUF files of a small model on HumanEval+ via llama.cpp (Metal), graded locally.
#   bash scripts/run_small_gguf_mac.sh unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507 "Q8_0 Q4_K_M ..."
set -u
REPO=$1; STEM=$2; TOK=$3; QS=$4
OUT=runs/fb_pilot6; G=${GGUF_DIR:-hf_cache/gguf}; mkdir -p $OUT $G
for q in $QS; do
  d=$OUT/${STEM}_$q; mkdir -p $d
  [ -f $d/humaneval_samples_eval_results.json ] && { echo "$q already done"; continue; }
  f=$G/$STEM-$q.gguf
  [ -f $f ] || curl -sfL -o $f "https://huggingface.co/$REPO/resolve/main/$STEM-$q.gguf" || { echo "$q download failed"; continue; }
  llama-server -m $f -ngl 99 -c 4096 --parallel 1 --cache-ram 0 --port 8090 > $d/server.log 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do curl -s localhost:8090/health | grep -q ok && break; sleep 2; done
  .venv/bin/python -u -m repair.fb_gguf_eval --server http://127.0.0.1:8090 --tokenizer $TOK --out $d > $d.log 2>&1 \
    && echo "$q generated $(date +%H:%M:%S)" || echo "$q failed $(date +%H:%M:%S)"
  kill $SRV; wait $SRV 2>/dev/null
  EVALPLUS_MAX_MEMORY_BYTES=-1 .venv_evalplus/bin/evalplus.evaluate --dataset humaneval --samples $d/humaneval_samples.jsonl --parallel 4 --min_time_limit 5 --gt_time_limit_factor 20 > $d/humaneval_grade.log 2>&1 \
    && echo "$q graded" || echo "$q grading failed"
done
echo "ALL DONE $(date +%H:%M:%S)"

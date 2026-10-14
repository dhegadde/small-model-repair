#!/bin/bash
# Shared helpers for compression-confidence Phase 3a. EvalPlus lives in its OWN venv (/workspace/venv_evalplus):
# never install extra libraries into the shared venv (2026-10-03 lesson).
set -u
export HF_HOME=/root/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/cc_phase3; mkdir -p $OUT processed_data/evalplus
EV=/workspace/venv_evalplus
setup_eval() {
  if [ ! -f $EV/.ready ]; then
    if mkdir /workspace/.evalplus_lock 2>/dev/null; then
      python3 -m venv $EV && $EV/bin/pip install -q evalplus && touch $EV/.ready && echo "eval venv ready"
      rmdir /workspace/.evalplus_lock
    else until [ -f $EV/.ready ]; do sleep 20; done; fi
  fi
  [ -f processed_data/evalplus/mbpp.json ] || $EV/bin/python - <<'PY'
import json
from evalplus.data import get_human_eval_plus, get_mbpp_plus
for name, f in (("humaneval", get_human_eval_plus), ("mbpp", get_mbpp_plus)):
    d = {k: {"prompt": v["prompt"]} for k, v in f().items()}
    json.dump(d, open(f"processed_data/evalplus/{name}.json", "w")); print(name, len(d))
PY
}
run_cfg() {  # $1 short name, $2 HF model id, $3 quant
  d=$OUT/$1_$3
  python -u -m repair.cc_code --model $2 --quant $3 --out $d > $d.log 2>&1 \
    && echo "$1 $3 generated $(date -u +%H:%M:%S)" || { echo "$1 $3 generation failed $(date -u +%H:%M:%S)"; return; }
  for ds in humaneval mbpp; do
    [ -f $d/${ds}_samples_eval_results.json ] && continue
    $EV/bin/evalplus.evaluate --dataset $ds --samples $d/${ds}_samples.jsonl --parallel 8 > $d/${ds}_grade.log 2>&1 \
      && echo "$1 $3 $ds graded" || echo "$1 $3 $ds grading failed"
  done
}

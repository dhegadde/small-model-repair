#!/bin/bash
# Frontier pilot 8: logit-distilled fix for Unsloth's UD-IQ2_M Qwen3-4B file; 3 seeds; all arms evaluated on this GPU.
. scripts/runpod/cc_phase3_common.sh
setup_eval
trap '[ "${DONE:-}" = 1 ] || echo "FAILED: driver exited early (see errors above)"' EXIT   # lets pod_watch stop at once
OUT=runs/fb_pilot8; mkdir -p $OUT; G=/root/gguf; STEM=Qwen3-4B-Instruct-2507
T=runs/fb_pilot7/teacher.jsonl; CE=runs/fb_pilot7/fix_s0/fix.gguf
[ -s $T ] && [ -s $CE ] || { echo "FAILED: pilot-7 teacher or CE fix missing"; exit 1; }
export PYTHONPATH=/root/gguf_pkgs:${PYTHONPATH:-}
( (apt-get update -qq && apt-get install -y -qq cmake nodejs g++) >/dev/null 2>&1
  [ -d /root/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp
  pip install -q --target /root/gguf_pkgs /root/llama.cpp/gguf-py > /root/gguf_pkgs.log 2>&1 \
  && cmake -S /root/llama.cpp -B /root/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /root/llama_build.log 2>&1 \
  && cmake --build /root/llama.cpp/build --target llama-server -j $(nproc) >> /root/llama_build.log 2>&1 && echo "llama.cpp built $(date -u +%H:%M:%S)" ) &
B=$!
python -c "
from huggingface_hub import hf_hub_download, snapshot_download
for q in ('Q8_0', 'UD-IQ2_M', 'UD-Q2_K_XL'):
    hf_hub_download('unsloth/$STEM-GGUF', f'$STEM-{q}.gguf', local_dir='$G')
snapshot_download('Qwen/$STEM', allow_patterns=['config.json', 'generation_config.json', 'tokenizer*', 'vocab.json', 'merges.txt'], local_dir='/root/hf_cfg')
snapshot_download('Qwen/$STEM')   # full-precision teacher for logit distillation (HF_HOME cache)
" > /root/dl.log 2>&1 && echo "downloads done $(date -u +%H:%M:%S)" || { tail -3 /root/dl.log; echo "FAILED: download"; exit 1; }
wait $B   # gguf_pkgs is needed by transformers' GGUF loader below
python -c "import gguf" || { echo "FAILED: gguf package"; exit 1; }
# GPU smoke test: the loss is zero for identical distributions, then a 16-example KD run with peak memory
python -c "
import torch; from repair.fb_small_recover import kd_loss
x = torch.randn(10, 151936); assert kd_loss(x, x).abs() < 1e-5; assert kd_loss(x, torch.randn(10, 151936)) > 0; print('kd_loss ok')"
python -u -m repair.fb_small_recover train --loss kd --gguf_dir $G --gguf_file $STEM-UD-IQ2_M.gguf --teacher $T \
  --limit 16 --epochs 1 --batch 4 --out $OUT/smoke > $OUT/smoke.log 2>&1 && echo "smoke ok: $(grep -o "'peak_gb': [0-9.]*" $OUT/smoke.log | tail -1)" \
  || { tail -15 $OUT/smoke.log; echo "FAILED: smoke test"; exit 1; }
[ -x /root/llama.cpp/build/bin/llama-server ] || { tail -20 /root/llama_build.log; echo "FAILED: llama.cpp build"; exit 1; }
SRV_ARGS="-ngl 99 -c 32768 --parallel 8 --cache-ram 0 --port 8080"
serve() {  # $1 gguf [$2 lora]
  /root/llama.cpp/build/bin/llama-server -m $1 ${2:+--lora $2} $SRV_ARGS > $OUT/server_$(basename $1 .gguf)${2:+_fix}.log 2>&1 &
  SRV=$!; for i in $(seq 1 120); do curl -s localhost:8080/health | grep -q ok && return; sleep 5; done; }
bench() {  # $1 arm name, $2 gguf, [$3 lora]
  d=$OUT/$1; [ -f $d/summary.json ] && grep -q mmlu $d/summary.json && return
  serve $2 ${3:-}
  python -u -m repair.fb_bench run --backend llamacpp --server http://127.0.0.1:8080 --tokenizer Qwen/$STEM --out $d > $d.log 2>&1 \
    && echo "$1 benchmarked $(date -u +%H:%M:%S)" || echo "$1 failed $(date -u +%H:%M:%S)"
  kill $SRV; wait $SRV 2>/dev/null
  for ds in humaneval mbpp; do $EV/bin/evalplus.evaluate --dataset $ds --samples $d/${ds}_samples.jsonl --parallel 8 > $d/${ds}_grade.log 2>&1; done
  echo "$1 graded"; }
fix() {  # $1 seed: train the KD fix, convert, evaluate
  F=$OUT/fix_kd_s$1
  [ -f $F/lora_hf/adapter_model.safetensors ] || { python -u -m repair.fb_small_recover train --loss kd --gguf_dir $G \
    --gguf_file $STEM-UD-IQ2_M.gguf --teacher $T --seed $1 --out $F > $F.log 2>&1 && echo "kd fix s$1 trained $(date -u +%H:%M:%S)" \
    || { tail -10 $F.log; echo "FAILED: train s$1"; exit 1; }; }
  [ -s $F/fix.gguf ] || python /root/llama.cpp/convert_lora_to_gguf.py $F/lora_hf --base /root/hf_cfg --outtype f16 --outfile $F/fix.gguf > $F/convert.log 2>&1 \
    || { tail -5 $F/convert.log; echo "FAILED: convert s$1"; exit 1; }
  bench iq2m_kd_s$1 $G/$STEM-UD-IQ2_M.gguf $F/fix.gguf; }
fix 0
bench iq2m $G/$STEM-UD-IQ2_M.gguf
bench iq2m_ce_s0 $G/$STEM-UD-IQ2_M.gguf $CE
bench q2kxl $G/$STEM-UD-Q2_K_XL.gguf
bench q8 $G/$STEM-Q8_0.gguf
fix 1
fix 2
DONE=1; echo "ALL DONE $(date -u +%H:%M:%S)"

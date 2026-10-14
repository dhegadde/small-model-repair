#!/bin/bash
# Frontier pilot 7: trained llama.cpp add-on that repairs Unsloth's UD-IQ2_M file of Qwen3-4B-Instruct-2507.
. scripts/runpod/cc_phase3_common.sh
setup_eval
trap '[ "${DONE:-}" = 1 ] || echo "FAILED: driver exited early (see errors above)"' EXIT   # lets pod_watch stop at once
OUT=runs/fb_pilot7; mkdir -p $OUT; G=/root/gguf; STEM=Qwen3-4B-Instruct-2507; SEED=${SEED:-0}
export PYTHONPATH=/root/gguf_pkgs:${PYTHONPATH:-}   # gguf (for transformers' GGUF loader + converter), NOT in the shared venv
( (apt-get update -qq && apt-get install -y -qq cmake nodejs g++) >/dev/null 2>&1
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp \
  && pip install -q --target /root/gguf_pkgs /root/llama.cpp/gguf-py > /root/gguf_pkgs.log 2>&1 \
  && cmake -S /root/llama.cpp -B /root/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-native} -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /root/llama_build.log 2>&1 \
  && cmake --build /root/llama.cpp/build --target llama-server -j $(nproc) >> /root/llama_build.log 2>&1 && echo "llama.cpp built $(date -u +%H:%M:%S)" ) &
B=$!
python -c "
from huggingface_hub import hf_hub_download, snapshot_download
for q in ('Q8_0', 'UD-IQ2_M', 'UD-Q2_K_XL'):
    hf_hub_download('unsloth/$STEM-GGUF', f'$STEM-{q}.gguf', local_dir='$G')
snapshot_download('Qwen/$STEM', allow_patterns=['config.json', 'generation_config.json', 'tokenizer*', 'vocab.json', 'merges.txt'], local_dir='/root/hf_cfg')  # no weight index: the converter then reads config only
" > /root/dl.log 2>&1 && echo "downloads done $(date -u +%H:%M:%S)" || { tail -3 /root/dl.log; echo "FAILED: download"; exit 1; }
wait $B
[ -x /root/llama.cpp/build/bin/llama-server ] || { tail -20 /root/llama_build.log; echo "FAILED: llama.cpp build"; exit 1; }
SRV_ARGS="-ngl 99 -c 32768 --parallel 8 --cache-ram 0 --port 8080"
serve() {  # $1 gguf [$2 lora]; waits until healthy
  /root/llama.cpp/build/bin/llama-server -m $1 ${2:+--lora $2} $SRV_ARGS > $OUT/server_$(basename $1 .gguf)${2:+_fix}.log 2>&1 &
  SRV=$!; for i in $(seq 1 120); do curl -s localhost:8080/health | grep -q ok && return; sleep 5; done; }
# 1) teacher answers from Q8_0
if [ ! -s $OUT/teacher.jsonl ]; then
  serve $G/$STEM-Q8_0.gguf
  python -u -m repair.fb_small_recover teacher --out $OUT/teacher.jsonl > $OUT/teacher.log 2>&1 \
    && echo "teacher done $(date -u +%H:%M:%S)" || { kill $SRV; echo "FAILED: teacher"; exit 1; }
  kill $SRV; wait $SRV 2>/dev/null
fi
# 2) train the fix against UD-IQ2_M's own weights, 3) convert it to a llama.cpp add-on
F=$OUT/fix_s$SEED
[ -f $F/lora_hf/adapter_model.safetensors ] || { python -u -m repair.fb_small_recover train --gguf_dir $G --gguf_file $STEM-UD-IQ2_M.gguf --teacher $OUT/teacher.jsonl \
  --seed $SEED --out $F > $F.log 2>&1 && echo "fix trained $(date -u +%H:%M:%S)" || { echo "FAILED: train"; exit 1; }; }
python /root/llama.cpp/convert_lora_to_gguf.py $F/lora_hf --base /root/hf_cfg --outtype f16 --outfile $F/fix.gguf > $F/convert.log 2>&1 \
  && echo "fix converted $(du -h $F/fix.gguf | cut -f1)" || { tail -5 $F/convert.log; echo "FAILED: convert"; exit 1; }
# 4) evaluate every arm with the same harness on this GPU
bench() {  # $1 arm name, $2 gguf, [$3 lora]
  d=$OUT/$1; [ -f $d/summary.json ] && grep -q mmlu $d/summary.json && return
  serve $2 ${3:-}
  python -u -m repair.fb_bench run --backend llamacpp --server http://127.0.0.1:8080 --tokenizer Qwen/$STEM --out $d > $d.log 2>&1 \
    && echo "$1 benchmarked $(date -u +%H:%M:%S)" || echo "$1 failed $(date -u +%H:%M:%S)"
  kill $SRV; wait $SRV 2>/dev/null
  for ds in humaneval mbpp; do $EV/bin/evalplus.evaluate --dataset $ds --samples $d/${ds}_samples.jsonl --parallel 8 > $d/${ds}_grade.log 2>&1; done
  echo "$1 graded"; }
bench iq2m_fix_s$SEED $G/$STEM-UD-IQ2_M.gguf $F/fix.gguf
bench iq2m $G/$STEM-UD-IQ2_M.gguf
bench q8 $G/$STEM-Q8_0.gguf
bench q2kxl $G/$STEM-UD-Q2_K_XL.gguf
DONE=1; echo "ALL DONE $(date -u +%H:%M:%S)"

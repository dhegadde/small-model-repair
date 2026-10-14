#!/bin/bash
# Frontier pilot 4: SVD (data-free) corrections at the same rank, and public GGUF releases via llama.cpp.
. scripts/runpod/cc_phase3_common.sh
setup_eval
OUT=runs/fb_pilot4; mkdir -p $OUT
grade() { $EV/bin/evalplus.evaluate --dataset humaneval --samples $1/humaneval_samples.jsonl --parallel 8 > $1/humaneval_grade.log 2>&1 \
  && echo "$1 graded" || echo "$1 grading failed"; }
# CPU-side work in the background while the GPU does the SVD arm: build llama.cpp, download the GGUF files
( (command -v cmake >/dev/null || (apt-get update -qq && apt-get install -y -qq cmake)) >/dev/null 2>&1
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp \
  && cmake -S /root/llama.cpp -B /root/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /root/llama_build.log 2>&1 \
  && cmake --build /root/llama.cpp/build --target llama-server -j 32 >> /root/llama_build.log 2>&1 && echo "llama.cpp built $(date -u +%H:%M:%S)" ) &
BUILD=$!
( python -c "
from huggingface_hub import hf_hub_download
for q in ('UD-Q2_K_XL', 'UD-IQ3_XXS'):
    hf_hub_download('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', f'Qwen3-Coder-30B-A3B-Instruct-{q}.gguf', local_dir='/root/gguf')
" > /root/gguf_dl.log 2>&1 && echo "gguf downloaded $(date -u +%H:%M:%S)" ) &
DL=$!
python -u -m repair.fb_packed --svd_rank 8 --save_dir "" --out $OUT/svd_r8 > $OUT/svd_r8.log 2>&1 \
  && echo "svd_r8 generated $(date -u +%H:%M:%S)" || echo "svd_r8 failed $(date -u +%H:%M:%S)"
[ -f $OUT/svd_r8/humaneval_samples.jsonl ] && grade $OUT/svd_r8
wait $BUILD $DL
[ -x /root/llama.cpp/build/bin/llama-server ] || { tail -20 /root/llama_build.log; echo "FAILED: llama.cpp build"; exit 1; }
for q in UD-Q2_K_XL UD-IQ3_XXS; do
  d=$OUT/gguf_$q; mkdir -p $d
  /root/llama.cpp/build/bin/llama-server -m /root/gguf/Qwen3-Coder-30B-A3B-Instruct-$q.gguf -ngl 99 -c 4096 --parallel 1 \
    --port 8080 > $d/server.log 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do curl -s localhost:8080/health | grep -q ok && break; sleep 5; done
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > $d/gpu_memory.txt
  python -u -m repair.fb_gguf_eval --server http://127.0.0.1:8080 --out $d > $d.log 2>&1 \
    && echo "$q generated $(date -u +%H:%M:%S)" || echo "$q failed $(date -u +%H:%M:%S)"
  kill $SRV; wait $SRV 2>/dev/null
  [ -f $d/humaneval_samples.jsonl ] && grade $d
done
echo "ALL DONE $(date -u +%H:%M:%S)"

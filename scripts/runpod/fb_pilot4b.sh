#!/bin/bash
# Frontier pilot 4b: Unsloth's smaller dynamic GGUF files on HumanEval+ (llama.cpp, batch 1).
. scripts/runpod/cc_phase3_common.sh
setup_eval
OUT=runs/fb_pilot4b; mkdir -p $OUT
QS="UD-IQ2_M UD-IQ2_XXS UD-IQ1_M UD-IQ1_S UD-TQ1_0"
( (command -v cmake >/dev/null || (apt-get update -qq && apt-get install -y -qq cmake)) >/dev/null 2>&1
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp \
  && cmake -S /root/llama.cpp -B /root/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-native} -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /root/llama_build.log 2>&1 \
  && cmake --build /root/llama.cpp/build --target llama-server -j $(nproc) >> /root/llama_build.log 2>&1 && echo "llama.cpp built $(date -u +%H:%M:%S)" ) &
BUILD=$!
python -c "
from huggingface_hub import hf_hub_download
for q in '$QS'.split():
    hf_hub_download('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', f'Qwen3-Coder-30B-A3B-Instruct-{q}.gguf', local_dir='/root/gguf')
" > /root/gguf_dl.log 2>&1 && echo "gguf downloaded $(date -u +%H:%M:%S)" || { echo "FAILED: download"; exit 1; }
wait $BUILD
[ -x /root/llama.cpp/build/bin/llama-server ] || { tail -20 /root/llama_build.log; echo "FAILED: llama.cpp build"; exit 1; }
for q in $QS; do
  d=$OUT/gguf_$q; mkdir -p $d
  /root/llama.cpp/build/bin/llama-server -m /root/gguf/Qwen3-Coder-30B-A3B-Instruct-$q.gguf -ngl 99 -c 4096 --parallel 1 \
    --port 8080 > $d/server.log 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do curl -s localhost:8080/health | grep -q ok && break; sleep 5; done
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > $d/gpu_memory.txt
  python -u -m repair.fb_gguf_eval --server http://127.0.0.1:8080 --out $d > $d.log 2>&1 \
    && echo "$q generated $(date -u +%H:%M:%S)" || echo "$q failed $(date -u +%H:%M:%S)"
  kill $SRV; wait $SRV 2>/dev/null
  [ -f $d/humaneval_samples.jsonl ] && $EV/bin/evalplus.evaluate --dataset humaneval --samples $d/humaneval_samples.jsonl \
    --parallel 8 > $d/humaneval_grade.log 2>&1 && echo "$q graded"
done
echo "ALL DONE $(date -u +%H:%M:%S)"

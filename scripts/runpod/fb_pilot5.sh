#!/bin/bash
# Frontier pilot 5: generality check. 4 arms in vLLM (bf16 + folded 2-bit variants), 2 public GGUFs in llama.cpp.
. scripts/runpod/cc_phase3_common.sh
setup_eval
OUT=runs/fb_pilot5; mkdir -p $OUT; VV=/root/venv_vllm
D=runs/fb_pilot3/sim/deltas.pt; [ -f $D ] || { echo "FAILED: no deltas at $D"; exit 1; }
[ -f processed_data/fb_bench/mmlu_1000.json ] || { echo "FAILED: run fb_bench prep and sync processed_data/fb_bench"; exit 1; }
grade() { for ds in humaneval mbpp; do [ -f $1/${ds}_samples.jsonl ] && $EV/bin/evalplus.evaluate --dataset $ds \
  --samples $1/${ds}_samples.jsonl --parallel 16 > $1/${ds}_grade.log 2>&1; done; echo "$1 graded $(date -u +%H:%M:%S)"; }
# CPU-side setup in the background: runtimes for JS/C++, vLLM in its own venv, llama.cpp + public GGUFs
( (apt-get update -qq && apt-get install -y -qq nodejs g++ cmake) > /root/apt.log 2>&1 && echo "node/g++ ready" ) &
A=$!
( python3 -m venv $VV && $VV/bin/pip install -q vllm > /root/vllm_install.log 2>&1 && echo "vllm ready $(date -u +%H:%M:%S)" ) &
V=$!
wait $A
( git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp \
  && cmake -S /root/llama.cpp -B /root/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /root/llama_build.log 2>&1 \
  && cmake --build /root/llama.cpp/build --target llama-server -j 32 >> /root/llama_build.log 2>&1 && echo "llama.cpp built $(date -u +%H:%M:%S)" ) &
B=$!
( python -c "
from huggingface_hub import hf_hub_download
for q in ('UD-Q2_K_XL', 'UD-IQ3_XXS'):
    hf_hub_download('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', f'Qwen3-Coder-30B-A3B-Instruct-{q}.gguf', local_dir='/root/gguf')
" > /root/gguf_dl.log 2>&1 && echo "gguf downloaded $(date -u +%H:%M:%S)" ) &
G=$!
wait $V
$VV/bin/python -c "import vllm" 2>/dev/null || { tail -5 /root/vllm_install.log; echo "FAILED: vllm install"; exit 1; }
for arm in bf16 ours svd_r8 plain2; do
  rm -rf /root/folded; M=/root/folded
  case $arm in
    bf16) M=Qwen/Qwen3-Coder-30B-A3B-Instruct ;;
    ours) python -u -m repair.fb_fold --deltas $D --save_dir $M > $OUT/fold_$arm.log 2>&1 ;;
    svd_r8) python -u -m repair.fb_fold --svd_rank 8 --save_dir $M > $OUT/fold_$arm.log 2>&1 ;;
    plain2) python -u -m repair.fb_fold --save_dir $M > $OUT/fold_$arm.log 2>&1 ;;
  esac
  $VV/bin/python -u -m repair.fb_bench run --backend vllm --model_dir $M --out $OUT/$arm > $OUT/$arm.log 2>&1 \
    && echo "$arm benchmarked $(date -u +%H:%M:%S)" || echo "$arm failed $(date -u +%H:%M:%S)"
  grade $OUT/$arm
done
rm -rf /root/folded
wait $B $G
[ -x /root/llama.cpp/build/bin/llama-server ] || { tail -20 /root/llama_build.log; echo "FAILED: llama.cpp build"; exit 1; }
for q in UD-Q2_K_XL UD-IQ3_XXS; do
  d=$OUT/gguf_$q; mkdir -p $d
  /root/llama.cpp/build/bin/llama-server -m /root/gguf/Qwen3-Coder-30B-A3B-Instruct-$q.gguf -ngl 99 -c 32768 --parallel 8 \
    --port 8080 > $d/server.log 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do curl -s localhost:8080/health | grep -q ok && break; sleep 5; done
  python -u -m repair.fb_bench run --backend llamacpp --server http://127.0.0.1:8080 --out $d > $d.log 2>&1 \
    && echo "$q benchmarked $(date -u +%H:%M:%S)" || echo "$q failed $(date -u +%H:%M:%S)"
  kill $SRV; wait $SRV 2>/dev/null
  grade $d
done
echo "ALL DONE $(date -u +%H:%M:%S)"

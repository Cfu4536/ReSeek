#!/bin/bash
set -euo pipefail

# Tuned defaults for high-throughput embedding with vLLM.
# Override any value from the shell, for example:
#   CUDA_VISIBLE_DEVICES=0,1 DATA_PARALLEL_SIZE=2 bash search/retrieval/start_vllm_api.sh
# For this heterogeneous cluster, prefer same-GPU pairs:
#   A6000: CUDA_VISIBLE_DEVICES=0,1 or CUDA_VISIBLE_DEVICES=3,4
#   A800:  CUDA_VISIBLE_DEVICES=2

MODEL_PATH="${MODEL_PATH:-/opt/data/TencentBAC/e5-base-v2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
if [ -z "${DATA_PARALLEL_SIZE:-}" ]; then
  DATA_PARALLEL_SIZE="$(awk -F',' '{print NF}' <<< "${VISIBLE_DEVICES}")"
fi
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2048}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"

CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" vllm serve "${MODEL_PATH}" \
  --task embed \
  --host "${HOST}" \
  --port "${PORT}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --disable-log-requests

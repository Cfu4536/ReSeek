#!/bin/bash
set -euo pipefail

# Tuned defaults for high-throughput embedding with vLLM.
# Override any value from the shell, for example:
#   CUDA_VISIBLE_DEVICES=0,1 DATA_PARALLEL_SIZE=2 bash search/retrieval/start_vllm_api.sh

MODEL_PATH="${MODEL_PATH:-/opt/data/TencentBAC/e5-base-v2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2048}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" vllm serve "${MODEL_PATH}" \
  --task embed \
  --host "${HOST}" \
  --port "${PORT}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --disable-log-requests

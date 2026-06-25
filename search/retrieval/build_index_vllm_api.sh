#!/bin/bash
set -euo pipefail

# Build a FAISS index through the vLLM embedding HTTP API.
# Start search/retrieval/start_vllm_api.sh first.
#
# Throughput knobs:
#   BATCH_SIZE: documents per API request
#   API_PARALLELISM: concurrent API requests kept in flight
#   MAX_LENGTH: truncate length sent to vLLM

CORPUS_PATH="${CORPUS_PATH:-/opt/datasets/TencentBAC/ReSeek-corpus/hot-wiki-18.jsonl}"
SAVE_DIR="${SAVE_DIR:-/opt/datasets/TencentBAC/ReSeek-corpus/}"
RETRIEVAL_METHOD="${RETRIEVAL_METHOD:-e5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
API_PARALLELISM="${API_PARALLELISM:-8}"
VLLM_API_URL="${VLLM_API_URL:-http://localhost:8000}"
MAX_LENGTH="${MAX_LENGTH:-256}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
CHUNK_SAVE_INTERVAL="${CHUNK_SAVE_INTERVAL:-2000}"
FAISS_TYPE="${FAISS_TYPE:-Flat}"

echo
echo "Starting vLLM API index build..."
echo "  corpus: ${CORPUS_PATH}"
echo "  save dir: ${SAVE_DIR}"
echo "  batch size: ${BATCH_SIZE}"
echo "  API parallelism: ${API_PARALLELISM}"
echo "  max length: ${MAX_LENGTH}"
echo

python search/retrieval/index_builder_api.py \
  --retrieval_method "${RETRIEVAL_METHOD}" \
  --corpus_path "${CORPUS_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --vllm_api_url "${VLLM_API_URL}" \
  --max_length "${MAX_LENGTH}" \
  --api_parallelism "${API_PARALLELISM}" \
  --request_timeout "${REQUEST_TIMEOUT}" \
  --chunk_save_interval "${CHUNK_SAVE_INTERVAL}" \
  --save_embedding \
  --faiss_type "${FAISS_TYPE}"

echo "Index build finished."

#!/bin/bash
set -euo pipefail

# Build a FAISS index from an existing embedding memmap when available.
# If the memmap does not exist, build embeddings through the vLLM HTTP API
# first, save them, and then build the index.
#
# Existing memmaps are expected to match index_builder_api.py:
#   dtype: float32
#   shape: (len(corpus), 768)

CORPUS_PATH="${CORPUS_PATH:-/opt/datasets/TencentBAC/ReSeek-corpus/hot-wiki-18.jsonl}"
SAVE_DIR="${SAVE_DIR:-/opt/datasets/TencentBAC/ReSeek-corpus/}"
RETRIEVAL_METHOD="${RETRIEVAL_METHOD:-e5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EMBEDDING_PATH="${EMBEDDING_PATH:-${SAVE_DIR%/}/emb_${RETRIEVAL_METHOD}.memmap}"
API_PARALLELISM="${API_PARALLELISM:-8}"
VLLM_API_URL="${VLLM_API_URL:-http://localhost:8000}"
MAX_LENGTH="${MAX_LENGTH:-256}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
CHUNK_SAVE_INTERVAL="${CHUNK_SAVE_INTERVAL:-2000}"
# FAISS factory string. By default, build an IVF index with exact vectors in
# each inverted list. IVF_NLIST controls the number of coarse clusters.
# Examples:
#   IVF_NLIST=16384 ./search/retrieval/build_index_vllm_api.sh
#   FAISS_TYPE=Flat ./search/retrieval/build_index_vllm_api.sh
#   FAISS_TYPE=IVF4096,PQ64 ./search/retrieval/build_index_vllm_api.sh
IVF_NLIST="${IVF_NLIST:-4096}"
FAISS_TYPE="${FAISS_TYPE:-IVF${IVF_NLIST},Flat}"
#FAISS_TYPE="${FAISS_TYPE:-Flat}"

COMMON_ARGS=(
  --retrieval_method "${RETRIEVAL_METHOD}"
  --corpus_path "${CORPUS_PATH}"
  --save_dir "${SAVE_DIR}"
  --batch_size "${BATCH_SIZE}"
  --faiss_type "${FAISS_TYPE}"
)

echo
echo "  corpus: ${CORPUS_PATH}"
echo "  save dir: ${SAVE_DIR}"
echo "  faiss type: ${FAISS_TYPE}"

if [ -f "${EMBEDDING_PATH}" ]; then
  echo "Starting memmap index build..."
  echo "  embedding: ${EMBEDDING_PATH}"
  echo

  python search/retrieval/index_builder_api.py \
    "${COMMON_ARGS[@]}" \
    --embedding_path "${EMBEDDING_PATH}"
else
  echo "Embedding memmap not found, starting vLLM API embedding + index build..."
  echo "  expected embedding: ${EMBEDDING_PATH}"
  echo "  batch size: ${BATCH_SIZE}"
  echo "  API parallelism: ${API_PARALLELISM}"
  echo "  vLLM API: ${VLLM_API_URL}"
  echo "  max length: ${MAX_LENGTH}"
  echo

  python search/retrieval/index_builder_api.py \
    "${COMMON_ARGS[@]}" \
    --vllm_api_url "${VLLM_API_URL}" \
    --max_length "${MAX_LENGTH}" \
    --api_parallelism "${API_PARALLELISM}" \
    --request_timeout "${REQUEST_TIMEOUT}" \
    --chunk_save_interval "${CHUNK_SAVE_INTERVAL}" \
    --embedding_save_path "${EMBEDDING_PATH}" \
    --save_embedding
fi

echo "Index build finished."

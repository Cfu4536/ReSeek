#!/bin/bash
set -euo pipefail


CORPUS_PATH="${CORPUS_PATH:-/opt/datasets/TencentBAC/ReSeek-corpus/hot-wiki-18.jsonl}"
SAVE_DIR="${SAVE_DIR:-/opt/datasets/TencentBAC/ReSeek-corpus3/}"
RETRIEVAL_METHOD="${RETRIEVAL_METHOD:-e5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EMBEDDING_PATH=""
VLLM_API_URL="${VLLM_API_URL:-http://localhost:8000}"
MAX_LENGTH="${MAX_LENGTH:-256}"
IVF_NLIST="${IVF_NLIST:-4096}"
FAISS_TYPE="${FAISS_TYPE:-IVF${IVF_NLIST},Flat}"
CHUNK_SIZE="${CHUNK_SIZE:-100000}"
TRAIN_SIZE="${TRAIN_SIZE:-262144}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
API_MODEL="${API_MODEL:-}"
EMBEDDING_DTYPE="${EMBEDDING_DTYPE:-float16}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SAVE_DIR}"

args=(
  "${SCRIPT_DIR}/index_builder_api.py"
  --corpus-path "${CORPUS_PATH}"
  --save-dir "${SAVE_DIR}"
  --retrieval-method "${RETRIEVAL_METHOD}"
  --api-url "${VLLM_API_URL}"
  --batch-size "${BATCH_SIZE}"
  --chunk-size "${CHUNK_SIZE}"
  --max-length "${MAX_LENGTH}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --embedding-dtype "${EMBEDDING_DTYPE}"
  --faiss-type "${FAISS_TYPE}"
  --train-size "${TRAIN_SIZE}"
)

if [[ -n "${API_MODEL}" ]]; then
  args+=(--api-model "${API_MODEL}")
fi

# EXTRA_ARGS can be used for operational switches such as:
#   EXTRA_ARGS="--delete-embeddings" bash build_index_vllm_api.sh
# shellcheck disable=SC2206
extra_args=(${EXTRA_ARGS:-})

echo "Building ${SAVE_DIR%/}/${RETRIEVAL_METHOD}_${FAISS_TYPE}.index"
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_ARGS})
    python3 "${args[@]}" "${extra_args[@]}"
else
    python3 "${args[@]}"
fi


echo "Index build finished."

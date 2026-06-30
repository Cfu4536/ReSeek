
file_path=${DATA_DIR}/ReSeek-corpus
index_file=$file_path/e5_IVF4096,Flat.index
corpus_file=$file_path/hot-wiki-18.jsonl
retriever_name=e5
retriever_path=${MODEL_DIR}/e5-base-v2
# Number of IVF lists searched per query. Increase for better recall at the
# cost of latency. Override at launch time, for example: NPROBE=64 bash ...
nprobe=${NPROBE:-64}

python search/retrieval/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --nprobe $nprobe \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path
#                                            --faiss_gpu

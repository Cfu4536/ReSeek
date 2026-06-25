CUDA_VISIBLE_DEVICES=3,4 vllm serve /opt/data/TencentBAC/e5-base-v2 \
  --task embed \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.5



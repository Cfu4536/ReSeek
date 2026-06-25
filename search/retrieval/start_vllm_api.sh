# export CUDA_DEVICE_ORDER='PCI_BUS_ID'

CUDA_VISIBLE_DEVICES=3 vllm serve /opt/data/TencentBAC/e5-base-v2 \
  --task embed \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.5



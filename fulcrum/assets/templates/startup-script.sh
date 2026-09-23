#!/bin/bash
# ==============================================================================
# /fulcrum: Production GCE GPU Startup Script for vLLM
# Automatically installs NVIDIA Container Toolkit, Docker, and runs vLLM
# ==============================================================================
set -euo pipefail

exec > >(tee -a /var/log/fulcrum-startup.log) 2>&1
echo "[fulcrum] Starting node configuration: $(date)"

# Install NVIDIA Container Toolkit and Docker if not already present
if ! command -v nvidia-ctk &> /dev/null; then
    echo "[fulcrum] Setting up NVIDIA Container Toolkit repository..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit docker.io
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# Configuration parameters (can be overridden via instance metadata)
MODEL_ID="${MODEL_ID:-google/gemma-3-12b-it}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
GPU_UTIL="${GPU_UTIL:-0.95}"

echo "[fulcrum] Serving model: ${MODEL_ID} on port: ${PORT}"

# Pull and start vLLM container
docker pull vllm/vllm-openai:latest
docker stop fulcrum-vllm || true
docker rm fulcrum-vllm || true

docker run -d \
    --name fulcrum-vllm \
    --restart unless-stopped \
    --gpus all \
    --ipc=host \
    --net=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model "${MODEL_ID}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --enable-chunked-prefill \
    --enable-prefix-caching

echo "[fulcrum] Awaiting server readiness..."
for i in {1..60}; do
    if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null; then
        echo "[fulcrum] vLLM is operational and healthy!"
        break
    fi
    sleep 5
done

#!/bin/bash


echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./setup/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: ${GPU_COUNT}"

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B" # Qwen2.5-1.5B
PT_CONFIG_NAME="beamrl"

PY_SCRIPT="./beamrl/grpo.py"
PY_CONFIG="./recipes/train_model_${PT_CONFIG_NAME}.yaml"
ACCELERATE_DS_CONFIG="./recipes/zero2.yaml"

echo ""
echo "Running ${PY_SCRIPT} on model ${BASE_MODEL_NAME} with dataset ${PT_CONFIG_NAME}"
echo ""

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_DS_CONFIG}" \
    --main_process_port=29500 \
    --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" --config "${PY_CONFIG}"

echo "END TIME: $(date)"
echo "DONE"
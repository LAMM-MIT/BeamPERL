#!/bin/bash

# Parse YAML config file (default: recipes/eval_baselines_lighteval.yaml)
EVAL_CONFIG="${1:-recipes/eval_baselines_lighteval.yaml}"

if [ ! -f "${EVAL_CONFIG}" ]; then
    echo "Error: Evaluation config file not found: ${EVAL_CONFIG}"
    echo "Usage: $0 [path_to_config.yaml]"
    exit 1
fi

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"
echo "Using config: ${EVAL_CONFIG}"

source "./setup/set_vars.sh"

# Parse YAML config using Python
eval $(python3 <<EOF
import yaml
import sys

with open('${EVAL_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)

# Handle models list
models = config.get('models', ['Qwen/Qwen2.5-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'])
model_str = ' '.join(f'"{m}"' for m in models)
print(f"MODEL_LIST=({model_str})")

# Handle tasks list
tasks = config.get('tasks', ['aime24', 'aime25', 'amc23'])
task_str = ' '.join(f'"{t}"' for t in tasks)
print(f"TASKS=({task_str})")

# Model-specific max lengths (defaults)
print(f"DEFAULT_MAX_MODEL_LENGTH={config.get('max_model_length', 32768)}")
print(f"DEFAULT_MAX_NEW_TOKENS={config.get('max_new_tokens', 32768)}")

# vLLM parameters
print(f"DTYPE=\"{config.get('dtype', 'bfloat16')}\"")
print(f"GPU_MEMORY_UTILIZATION={config.get('gpu_memory_utilization', 0.7)}")

# Generation parameters
print(f"TEMPERATURE={config.get('temperature', 0.6)}")
print(f"TOP_P={config.get('top_p', 0.95)}")

# GPU configuration
cuda_devices = config.get('cuda_visible_devices', '0,1')
print(f"CUDA_VISIBLE_DEVICES=\"{cuda_devices}\"")
EOF
)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: ${GPU_COUNT}"
echo ""

for MODEL_NAME in "${MODEL_LIST[@]}"; do

    # Set model-specific max lengths (override defaults from YAML if needed)
    if [ "${MODEL_NAME}" == "Qwen/Qwen2.5-Math-1.5B" ]; then
        MAX_MODEL_LENGTH=4096
        MAX_NEW_TOKENS=4096
    elif [ "${MODEL_NAME}" == "Qwen/Qwen2.5-1.5B" ]; then
        MAX_MODEL_LENGTH=32768 # 131072
        MAX_NEW_TOKENS=32768 # 131072
    else
        MAX_MODEL_LENGTH=${DEFAULT_MAX_MODEL_LENGTH:-32768}
        MAX_NEW_TOKENS=${DEFAULT_MAX_NEW_TOKENS:-32768}
    fi

    MODEL_ARGS="pretrained=${MODEL_NAME},dtype=${DTYPE},data_parallel_size=${GPU_COUNT},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:${TEMPERATURE},top_p:${TOP_P}}"

    for TASK in "${TASKS[@]}"; do
      echo "Evaluating task: ${TASK} on model ${MODEL_NAME}"
      lighteval vllm "${MODEL_ARGS}" "custom|${TASK}|0|0" \
          --custom-tasks ./scripts/eval/run_eval_custom_tasks.py \
          --use-chat-template \
          --output-dir "${OUTPUT_DIR}/${TASK}/42/${MODEL_NAME}"
    done

done

echo "END TIME: $(date)"
echo "DONE"
#!/bin/bash

# Parse YAML config file (default: recipes/eval_model_lighteval.yaml)
EVAL_CONFIG="${1:-recipes/eval_model_lighteval.yaml}"

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

# Extract values and create bash variables
print(f"MODEL_NAME=\"{config.get('model_name', 'DeepSeek-R1-Distill-Qwen-1.5B')}\"")
print(f"PT_TYPE=\"{config.get('pt_type', 'grpo')}\"")
print(f"PT_CONFIG_NAME=\"{config.get('pt_config_name', 'beamrl')}\"")
print(f"PT_DATASET_NAME=\"{config.get('pt_dataset_name', 'beamrl_train')}\"")
print(f"SAVE_NAME=\"{config.get('save_name', 'beamrl_260101')}\"")

# Handle checkpoints list
checkpoints = config.get('checkpoints', ['checkpoint-3', 'checkpoint-4'])
ckpt_str = ' '.join(f'"{c}"' for c in checkpoints)
print(f"CKPT_LIST=({ckpt_str})")

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

for CKPT in "${CKPT_LIST[@]}"; do
    echo "Running post-trained model ${PT_CONFIG_NAME} with ckpt ${CKPT}"
    
    # The adapter path structure is: {PT_TYPE}_{PT_DATASET_NAME}/{SAVE_NAME}/checkpoint-{N}
    # Based on grpo.py: training_args.output_dir = f"{ckpt_prefix}/{ckpt_postfix}/{save_bucket}"
    # where ckpt_postfix = f"{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"
    ADAPTER_TYPE="${PT_TYPE}_${PT_DATASET_NAME}"
    
    # The merge script expects: {ckpt_dir}/models/{model_name}/{adapter_type}/{ckpt}
    # So we pass adapter_type with save_name included: "{ADAPTER_TYPE}/{SAVE_NAME}"
    python  ./beamrl/merge_post_trained_models.py \
      --model_name "${MODEL_NAME}" \
      --adapter_type "${ADAPTER_TYPE}/${SAVE_NAME}" \
      --ckpt "${CKPT}" \

    # Set model-specific max lengths (override defaults from YAML if needed)
    if [ "${MODEL_NAME}" == "Qwen2.5-Math-1.5B" ]; then
        MAX_MODEL_LENGTH=4096
        MAX_NEW_TOKENS=4096
    elif [ "${MODEL_NAME}" == "Qwen2.5-1.5B" ]; then
        MAX_MODEL_LENGTH=32768 # 131072
        MAX_NEW_TOKENS=32768 # 131072
    else
        MAX_MODEL_LENGTH=${DEFAULT_MAX_MODEL_LENGTH:-32768}
        MAX_NEW_TOKENS=${DEFAULT_MAX_NEW_TOKENS:-32768}
    fi

    # Model path includes save_name: grpo_beamrl_train/beamrl_260101/checkpoint-3-merged
    MODEL_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${ADAPTER_TYPE}/${SAVE_NAME}/${CKPT}-merged"
    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE},data_parallel_size=${GPU_COUNT},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:${TEMPERATURE},top_p:${TOP_P}}"

    for TASK in "${TASKS[@]}"; do
        echo "Evaluating task: ${TASK} on model ${MODEL_NAME} post-trained with ${PT_CONFIG_NAME} (${CKPT})"
        lighteval vllm "${MODEL_ARGS}" "custom|${TASK}|0|0" \
            --custom-tasks ./scripts/eval/run_eval_custom_tasks.py \
            --use-chat-template \
            --output-dir "${OUTPUT_DIR}/${TASK}/42/${MODEL_NAME}_${PT_TYPE}_${PT_CONFIG_NAME}_${CKPT}"
    done
done

echo "END TIME: $(date)"
echo "DONE"

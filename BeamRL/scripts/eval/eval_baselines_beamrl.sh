#!/bin/bash

# Parse YAML config file (default: recipes/eval_baselines_beamrl.yaml)
EVAL_CONFIG="${1:-recipes/eval_baselines_beamrl.yaml}"

if [ ! -f "${EVAL_CONFIG}" ]; then
    echo "Error: Evaluation config file not found: ${EVAL_CONFIG}"
    echo "Usage: $0 [path_to_config.yaml]"
    exit 1
fi

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"
echo "Using config: ${EVAL_CONFIG}"

source "./setup/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: ${GPU_COUNT}"
echo ""

# Parse YAML config using Python
eval $(python3 <<EOF
import yaml
import sys
import json

with open('${EVAL_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)

# Handle models list
models = config.get('models', ['Qwen/Qwen2.5-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'])
model_str = ' '.join(f'"{m}"' for m in models)
print(f"MODEL_LIST=({model_str})")

# Evaluation parameters
print(f"MAX_PROMPT_LENGTH={config.get('max_prompt_length', 512)}")
print(f"MAX_GENERATION_LENGTH={config.get('max_generation_length', 4096)}")
print(f"BATCH_SIZE={config.get('batch_size', 8)}")
print(f"NUM_GENERATIONS={config.get('num_generations', 5)}")
print(f"TEMPERATURE={config.get('temperature', 0.6)}")

# Dataset config
print(f"EVAL_DATASET_NAME=\"{config.get('eval_dataset_name', 'beamrl_eval')}\"")
eval_dataset_config = config.get('eval_dataset_config')
if eval_dataset_config:
    print(f"EVAL_DATASET_CONFIG=\"{eval_dataset_config}\"")
else:
    print("EVAL_DATASET_CONFIG=\"\"")
print(f"EVAL_SPLIT=\"{config.get('eval_split', 'train')}\"")
max_eval_samples = config.get('max_eval_samples')
if max_eval_samples:
    print(f"MAX_EVAL_SAMPLES={max_eval_samples}")
else:
    print("MAX_EVAL_SAMPLES=\"\"")

# vLLM parameters
print(f"MAX_MODEL_LEN={config.get('max_model_len', 32768)}")
print(f"GPU_MEMORY_UTILIZATION={config.get('gpu_memory_utilization', 0.7)}")
print(f"DTYPE=\"{config.get('dtype', 'bfloat16')}\"")

# WandB config
print(f"LOG_WANDB={str(config.get('log_wandb', True)).lower()}")
print(f"WANDB_PROJECT=\"{config.get('wandb_project', 'beamrl-eval-baselines')}\"")

# Model-specific max lengths (for compatibility)
max_model_length = config.get('max_model_length', 32768)
max_new_tokens = config.get('max_new_tokens', 32768)
print(f"DEFAULT_MAX_MODEL_LENGTH={max_model_length}")
print(f"DEFAULT_MAX_NEW_TOKENS={max_new_tokens}")
EOF
)

for MODEL_NAME in "${MODEL_LIST[@]}"; do
    echo "Evaluating baseline model: ${MODEL_NAME}"
    
    # Set model-specific max lengths
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
    
    # Use model-specific max_model_len for vLLM (override YAML default if needed)
    MAX_MODEL_LEN=${MAX_MODEL_LENGTH}
    
    # Use model name directly (no merging needed for base models)
    MODEL_PATH="${MODEL_NAME}"
    
    # Sanitize model name for output directory (replace / with _)
    MODEL_NAME_SANITIZED=$(echo "${MODEL_NAME}" | sed 's/\//_/g')
    
    echo "Evaluating on beamrl_eval dataset for baseline model ${MODEL_NAME}"
    echo "  Dataset: ${EVAL_DATASET_NAME}"
    echo "  Split: ${EVAL_SPLIT}"
    echo "  Using vLLM with max_model_len=${MAX_MODEL_LEN}, gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    
    OUTPUT_DIR_TASK="${OUTPUT_DIR}/beamrl_eval/42/${MODEL_NAME_SANITIZED}_baseline"
    mkdir -p "${OUTPUT_DIR_TASK}"
    
    # Build evaluation command
    EVAL_CMD="python ./scripts/eval/run_dataset_eval.py \
        --model_path \"${MODEL_PATH}\" \
        --eval_dataset_name \"${EVAL_DATASET_NAME}\" \
        --eval_split \"${EVAL_SPLIT}\" \
        --max_prompt_length ${MAX_PROMPT_LENGTH} \
        --max_generation_length ${MAX_GENERATION_LENGTH} \
        --batch_size ${BATCH_SIZE} \
        --num_generations ${NUM_GENERATIONS} \
        --temperature ${TEMPERATURE} \
        --max_model_len ${MAX_MODEL_LEN} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --dtype \"${DTYPE}\" \
        --output_file \"${OUTPUT_DIR_TASK}/results.json\""
    
    # Add optional dataset config if specified
    if [ -n "${EVAL_DATASET_CONFIG}" ]; then
        EVAL_CMD="${EVAL_CMD} --eval_dataset_config \"${EVAL_DATASET_CONFIG}\""
    fi
    
    # Add optional max_eval_samples if specified
    if [ -n "${MAX_EVAL_SAMPLES}" ]; then
        EVAL_CMD="${EVAL_CMD} --max_eval_samples ${MAX_EVAL_SAMPLES}"
    fi
    
    # Add WandB flags if enabled
    if [ "${LOG_WANDB}" == "true" ]; then
        EVAL_CMD="${EVAL_CMD} --log_wandb --wandb_project \"${WANDB_PROJECT}\""
        EVAL_CMD="${EVAL_CMD} --wandb_run_name \"${MODEL_NAME_SANITIZED}_baseline_beamrl_eval\""
    fi
    
    eval ${EVAL_CMD}
done

echo "END TIME: $(date)"
echo "DONE"


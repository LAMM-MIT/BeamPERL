#!/bin/bash

# Parse YAML config file (default: recipes/eval_model_beamrl.yaml)
EVAL_CONFIG="${1:-recipes/eval_model_beamrl.yaml}"

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
print(f"WANDB_PROJECT=\"{config.get('wandb_project', 'beamrl-eval')}\"")

# Model-specific max lengths (for merge script compatibility)
max_model_length = config.get('max_model_length', 32768)
max_new_tokens = config.get('max_new_tokens', 32768)
print(f"DEFAULT_MAX_MODEL_LENGTH={max_model_length}")
print(f"DEFAULT_MAX_NEW_TOKENS={max_new_tokens}")
EOF
)

for CKPT in "${CKPT_LIST[@]}"; do
    echo "Running post-trained model ${PT_CONFIG_NAME} with ckpt ${CKPT}"
    
    # The adapter path structure is: {PT_TYPE}_{PT_DATASET_NAME}/{SAVE_NAME}/checkpoint-{N}
    # Based on grpo.py: training_args.output_dir = f"{ckpt_prefix}/{ckpt_postfix}/{save_bucket}"
    # where ckpt_postfix = f"{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"
    ADAPTER_TYPE="${PT_TYPE}_${PT_DATASET_NAME}"
    
    # Construct full paths including save_name
    ADAPTER_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${ADAPTER_TYPE}/${SAVE_NAME}/${CKPT}"
    MERGED_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${ADAPTER_TYPE}/${SAVE_NAME}/${CKPT}-merged"
    
    # Check if merged model already exists
    if [ ! -d "${MERGED_PATH}" ] || [ ! -f "${MERGED_PATH}/config.json" ]; then
        echo "Merging adapter from: ${ADAPTER_PATH}"
        # The merge script expects: {ckpt_dir}/models/{model_name}/{adapter_type}/{ckpt}
        # So we pass adapter_type with save_name included: "{ADAPTER_TYPE}/{SAVE_NAME}"
        python  ./beamrl/merge_post_trained_models.py \
          --model_name "${MODEL_NAME}" \
          --adapter_type "${ADAPTER_TYPE}/${SAVE_NAME}" \
          --ckpt "${CKPT}" || {
            echo "Warning: Merge failed. Check if adapter exists at ${ADAPTER_PATH}"
            echo "If merged model already exists elsewhere, you may need to copy it to ${MERGED_PATH}"
        }
    else
        echo "Merged model already exists at: ${MERGED_PATH}"
    fi

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

    MODEL_PATH="${MERGED_PATH}"
    
    echo "Evaluating on beamrl_eval dataset for model ${MODEL_NAME} post-trained with ${PT_CONFIG_NAME} (${CKPT})"
    echo "  Dataset: ${EVAL_DATASET_NAME}"
    echo "  Split: ${EVAL_SPLIT}"
    echo "  Using vLLM with max_model_len=${MAX_MODEL_LEN}, gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    
    OUTPUT_DIR_TASK="${OUTPUT_DIR}/beamrl_eval/42/${MODEL_NAME}_${PT_TYPE}_${PT_CONFIG_NAME}_${CKPT}"
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
        EVAL_CMD="${EVAL_CMD} --wandb_run_name \"${MODEL_NAME}_${PT_TYPE}_${PT_CONFIG_NAME}_${CKPT}_beamrl_eval\""
    fi
    
    eval ${EVAL_CMD}
done

echo "END TIME: $(date)"
echo "DONE"

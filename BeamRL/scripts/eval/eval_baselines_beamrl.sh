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

# Parse YAML config using Python script
eval $(python3 ./scripts/eval/parse_eval_config.py "${EVAL_CONFIG}" --mode baselines_beamrl)

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


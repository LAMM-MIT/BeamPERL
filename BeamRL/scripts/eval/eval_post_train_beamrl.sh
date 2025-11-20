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

# Parse YAML config using Python script
eval $(python3 ./scripts/eval/parse_eval_config.py "${EVAL_CONFIG}" --mode post_train_beamrl)

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

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

# Respect an externally-set CUDA_VISIBLE_DEVICES (e.g. pinning one ablation per
# GPU for parallel eval); default to both GPUs when unset (original behavior).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: ${GPU_COUNT}"
echo ""

# Parse YAML config using Python script
eval $(python3 ./scripts/eval/parse_eval_config.py "${EVAL_CONFIG}" --mode post_train_beamrl)

# Seeds for multi-seed evaluation. Override by setting EVAL_SEEDS env var before calling:
#   EVAL_SEEDS="42 123" bash eval_model_beamrl.sh
IFS=' ' read -ra SEEDS <<< "${EVAL_SEEDS:-42 123 456}"

# --- Phase 1: merge all adapters (only needs to happen once, not once per seed) ---
for CKPT in "${CKPT_LIST[@]}"; do
    ADAPTER_TYPE="${PT_TYPE}_${PT_DATASET_NAME}"
    ADAPTER_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${ADAPTER_TYPE}/${SAVE_NAME}/${CKPT}"
    MERGED_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${ADAPTER_TYPE}/${SAVE_NAME}/${CKPT}-merged"

    if [ ! -d "${MERGED_PATH}" ] || [ ! -f "${MERGED_PATH}/config.json" ]; then
        echo "Merging adapter: ${ADAPTER_PATH}"
        python ./beamrl/merge_post_trained_models.py \
          --model_name "${MODEL_NAME}" \
          --adapter_type "${ADAPTER_TYPE}/${SAVE_NAME}" \
          --ckpt "${CKPT}" || {
            echo "Warning: Merge failed. Check adapter at ${ADAPTER_PATH}"
        }
    else
        echo "Merged model already exists: ${MERGED_PATH}"
    fi
done

# --- Phase 2: evaluate each checkpoint × each seed ---
for CKPT in "${CKPT_LIST[@]}"; do
    echo "Running post-trained model ${PT_CONFIG_NAME} with ckpt ${CKPT}"

    ADAPTER_TYPE="${PT_TYPE}_${PT_DATASET_NAME}"
    MERGED_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${ADAPTER_TYPE}/${SAVE_NAME}/${CKPT}-merged"

    # Set model-specific max lengths
    if [ "${MODEL_NAME}" == "Qwen2.5-Math-1.5B" ]; then
        MAX_MODEL_LENGTH=4096
        MAX_NEW_TOKENS=4096
    elif [ "${MODEL_NAME}" == "Qwen2.5-1.5B" ]; then
        MAX_MODEL_LENGTH=32768
        MAX_NEW_TOKENS=32768
    else
        MAX_MODEL_LENGTH=${DEFAULT_MAX_MODEL_LENGTH:-32768}
        MAX_NEW_TOKENS=${DEFAULT_MAX_NEW_TOKENS:-32768}
    fi

    MODEL_PATH="${MERGED_PATH}"

    for SEED in "${SEEDS[@]}"; do
        echo "  Evaluating with seed ${SEED}..."
        echo "  Dataset: ${EVAL_DATASET_NAME} | Split: ${EVAL_SPLIT}"
        echo "  vLLM: max_model_len=${MAX_MODEL_LEN}, gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"

        OUTPUT_DIR_TASK="${OUTPUT_DIR}/beamrl_eval/${SEED}/${MODEL_NAME}_${PT_TYPE}_${PT_CONFIG_NAME}_${CKPT}"
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
            --seed ${SEED} \
            --output_file \"${OUTPUT_DIR_TASK}/results.json\""

        if [ -n "${EVAL_DATASET_CONFIG}" ]; then
            EVAL_CMD="${EVAL_CMD} --eval_dataset_config \"${EVAL_DATASET_CONFIG}\""
        fi

        if [ -n "${MAX_EVAL_SAMPLES}" ]; then
            EVAL_CMD="${EVAL_CMD} --max_eval_samples ${MAX_EVAL_SAMPLES}"
        fi

        if [ "${LOG_WANDB}" == "true" ]; then
            EVAL_CMD="${EVAL_CMD} --log_wandb --wandb_project \"${WANDB_PROJECT}\""
            EVAL_CMD="${EVAL_CMD} --wandb_run_name \"${MODEL_NAME}_${PT_TYPE}_${PT_CONFIG_NAME}_${CKPT}_seed${SEED}_beamrl_eval\""
        fi

        eval ${EVAL_CMD}
    done
done

echo "END TIME: $(date)"
echo "DONE"

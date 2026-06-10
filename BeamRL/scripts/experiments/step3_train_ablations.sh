#!/bin/bash
# =============================================================================
# STEP 3: Train reward ablation models (format-only and accuracy-only)
# =============================================================================
# Trains two models sequentially, each for 1 epoch, identical to the original
# BeamPERL run except for the reward configuration.
#
# Training takes ~same wall time as the original (~hours per model).
# The two jobs run sequentially (both need 2 GPUs). To run in parallel,
# launch each section on a separate machine.
#
# Run from: BeamPERL/BeamRL/
#   bash scripts/experiments/step3_train_ablations.sh
#
# To train only one ablation:
#   bash scripts/experiments/step3_train_ablations.sh format_only
#   bash scripts/experiments/step3_train_ablations.sh accuracy_only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAMRL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WHICH="${1:-both}"  # "format_only", "accuracy_only", or "both"

echo "============================================================"
echo "STEP 3: Training reward ablation models"
echo "Target: ${WHICH}"
echo "Started: $(date)"
echo "============================================================"

cd "${BEAMRL_DIR}"
source "./setup/set_vars.sh"

# --- Environment fixes for a 2x NVIDIA L4 node (torch 2.5.1 / vLLM 0.7.2) ---
# CUDA_LAUNCH_BLOCKING=1 (set in set_vars.sh) serializes CUDA launches and
# deadlocks vLLM's NCCL process-group init (new_group hangs). Disable it here.
unset CUDA_LAUNCH_BLOCKING
# NCCL works on this node over the socket transport on ens5 (P2P/SHM/IB stay
# disabled via set_vars.sh). set_vars.sh does not touch these two, so setting
# them here makes the launch self-contained.
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=ens5

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs available: ${GPU_COUNT}"

# Dedicated-vLLM-GPU layout: train on 1 GPU (cuda:0), run vLLM rollouts on the
# other GPU (cuda:1). The original 2-GPU vLLM-colocate topology deadlocks on such
# nodes at vLLM's NCCL communicator init (DeepSpeed needs NCCL P2P/SHM disabled, which
# is incompatible with a vLLM comm colocated on a training GPU). Both GPUs stay
# visible (CUDA_VISIBLE_DEVICES=0,1) so vLLM can claim cuda:1; only 1 training
# process is launched. Effective batch size is held at 32 via grad_accum 4->8 in
# the ablation configs (per_device 4 x grad_accum 8 x 1 GPU = 32, == original 4x4x2).
TRAIN_PROCS=1
echo "Training processes: ${TRAIN_PROCS} (vLLM uses the remaining GPU)"

ACCELERATE_DS_CONFIG="./recipes/zero2.yaml"
PY_SCRIPT="./beamrl/grpo.py"

run_training() {
    local config_name="$1"
    local config_file="./recipes/train_model_${config_name}.yaml"

    if [ ! -f "${config_file}" ]; then
        echo "ERROR: Config not found: ${config_file}"
        exit 1
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Training: ${config_name}"
    echo "Config:   ${config_file}"
    echo "Start:    $(date)"
    echo "------------------------------------------------------------"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${ACCELERATE_DS_CONFIG}" \
        --main_process_port=29501 \
        --num_processes="${TRAIN_PROCS}" \
        "${PY_SCRIPT}" --config "${config_file}"

    echo "Finished training ${config_name}: $(date)"
}

if [[ "${WHICH}" == "format_only" || "${WHICH}" == "both" ]]; then
    run_training "beamrl_format_only"
fi

if [[ "${WHICH}" == "accuracy_only" || "${WHICH}" == "both" ]]; then
    run_training "beamrl_accuracy_only"
fi

echo ""
echo "============================================================"
echo "STEP 3 COMPLETE: $(date)"
echo "Checkpoints saved in:"
echo "  \$CKPT_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_beamrl_train/beamrl_format_only/"
echo "  \$CKPT_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_beamrl_train/beamrl_accuracy_only/"
echo "Proceed to step4_eval_ablations.sh"
echo "============================================================"

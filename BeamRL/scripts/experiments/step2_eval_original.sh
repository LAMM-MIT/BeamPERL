#!/bin/bash
# =============================================================================
# STEP 2: Multi-seed evaluation of the original BeamPERL model on eval v2
# =============================================================================
# Evaluates checkpoints 5–45 of the original trained model across 3 seeds.
# Requires the original checkpoints to be on disk.
#
# Prerequisites:
#   - Step 1 complete (tphage/BeamRL-EvalData-v2 uploaded)
#   - Original BeamPERL checkpoints at:
#       $CKPT_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_beamrl_train/beamrl_260101/
#   - conda env activated
#
# Run from: BeamPERL/BeamRL/
#   bash scripts/experiments/step2_eval_original.sh
#
# Override seeds:  EVAL_SEEDS="42 123" bash scripts/experiments/step2_eval_original.sh
#
# Output: $OUTPUT_DIR/beamrl_eval/{seed}/DeepSeek-R1-..._grpo_beamrl_checkpoint-{N}/results.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAMRL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # BeamRL/

echo "============================================================"
echo "STEP 2: Multi-seed eval — original BeamPERL model (v2 dataset)"
echo "Started: $(date)"
echo "============================================================"

cd "${BEAMRL_DIR}"

# Load environment variables (sets CKPT_DIR, OUTPUT_DIR)
source "./setup/set_vars.sh"

# Verify checkpoint directory exists
CKPT_MODEL_DIR="${CKPT_DIR}/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_beamrl_train/beamrl_260101"
if [ ! -d "${CKPT_MODEL_DIR}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CKPT_MODEL_DIR}"
    echo "Make the original BeamPERL checkpoints available at this path first."
    exit 1
fi

echo "Checkpoint directory: ${CKPT_MODEL_DIR}"
echo "Output directory:     ${OUTPUT_DIR}"
echo "Seeds: ${EVAL_SEEDS:-42 123 456}"
echo ""

# Run the evaluation with multi-seed support
# The eval script reads seeds from EVAL_SEEDS env var
export EVAL_SEEDS="${EVAL_SEEDS:-42 123 456}"
bash scripts/eval/eval_model_beamrl.sh recipes/eval_model_beamrl_v2.yaml

echo ""
echo "============================================================"
echo "STEP 2 COMPLETE: $(date)"
echo "Results in: ${OUTPUT_DIR}/beamrl_eval/{seed}/"
echo "Proceed to step3_train_ablations.sh (or run step5_aggregate.sh now"
echo "to see partial results while waiting for ablation training)"
echo "============================================================"

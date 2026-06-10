#!/bin/bash
# =============================================================================
# BEAMPERL EXPERIMENT PIPELINE — Master Orchestration Script
# =============================================================================
# Runs all five steps in sequence. Each step can also be run independently.
# Typically you will NOT run this all at once (training takes days), but it
# serves as the definitive reference for the full sequence.
#
# BEFORE RUNNING:
#   1. Make the trained BeamPERL checkpoints available on this machine
#   2. Activate conda environment: conda activate beamrl
#   3. huggingface-cli login
#   4. Run from: BeamPERL/BeamRL/
#
# TYPICAL USAGE (run steps in separate terminal sessions):
#   bash scripts/experiments/step1_gen_eval_data.sh
#   bash scripts/experiments/step2_eval_original.sh          # needs ~N GPU-hours
#   bash scripts/experiments/step3_train_ablations.sh        # needs ~2N GPU-hours
#   bash scripts/experiments/step4_eval_ablations.sh         # needs ~2N GPU-hours
#   bash scripts/experiments/step5_aggregate.sh              # CPU only, seconds
#
# Override seeds (default: 42 123 456):
#   export EVAL_SEEDS="42 123"
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAMRL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "############################################################"
echo "#  BeamPERL Experiment Pipeline"
echo "#  Started: $(date)"
echo "############################################################"

cd "${BEAMRL_DIR}"

echo ""
echo ">>> Step 1: Generate expanded eval dataset (no GPU)"
bash scripts/experiments/step1_gen_eval_data.sh

echo ""
echo ">>> Step 2: Evaluate original BeamPERL model (GPU, ~hours)"
bash scripts/experiments/step2_eval_original.sh

echo ""
echo ">>> Step 3: Train reward ablation models (GPU, ~days)"
bash scripts/experiments/step3_train_ablations.sh

echo ""
echo ">>> Step 4: Evaluate ablation models (GPU, ~hours)"
bash scripts/experiments/step4_eval_ablations.sh

echo ""
echo ">>> Step 5: Aggregate all results (CPU)"
bash scripts/experiments/step5_aggregate.sh

echo ""
echo "############################################################"
echo "#  Pipeline COMPLETE: $(date)"
echo "#  Results are in the output directory printed by step 5."
echo "############################################################"

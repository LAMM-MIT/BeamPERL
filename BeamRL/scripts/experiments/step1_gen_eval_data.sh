#!/bin/bash
# =============================================================================
# STEP 1: Generate and upload the expanded evaluation dataset (v2)
# =============================================================================
# Generates beam configurations (symbeam, CPU), then uses the 7B question-
# generation LLM to produce 4 natural-language question variants per new sample
# (consistent with the original 24-sample eval set). Uploads the result as
# tphage/BeamRL-EvalData-v2.
#
# GPU required: yes (for LLM question generation, ~10 min on 1×L4)
#
# To skip the LLM and use template questions instead (no GPU needed):
#   bash scripts/experiments/step1_gen_eval_data.sh --no-llm
#
# Prerequisites:
#   huggingface-cli login
#   conda activate beamrl  (or whichever env has vllm, datasets, sympy)
#
# Run from: BeamPERL/BeamRL/
#   bash scripts/experiments/step1_gen_eval_data.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAMPERL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATAGEN_DIR="${BEAMPERL_ROOT}/DataGen"

NO_LLM_FLAG="${1:-}"  # pass --no-llm as first arg to skip LLM

echo "============================================================"
echo "STEP 1: Generating expanded eval dataset v2"
echo "LLM question generation: $([ "${NO_LLM_FLAG}" = "--no-llm" ] && echo DISABLED || echo ENABLED)"
echo "Started: $(date)"
echo "============================================================"

if [ ! -f "${DATAGEN_DIR}/generate_eval_v2.py" ]; then
    echo "ERROR: ${DATAGEN_DIR}/generate_eval_v2.py not found."
    exit 1
fi

# Check HuggingFace login
python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null || {
    echo "ERROR: Not logged in to HuggingFace. Run: huggingface-cli login"
    exit 1
}

cd "${DATAGEN_DIR}"
python generate_eval_v2.py ${NO_LLM_FLAG}

echo ""
echo "============================================================"
echo "STEP 1 COMPLETE: $(date)"
echo "Dataset: https://huggingface.co/datasets/tphage/BeamRL-EvalData-v2"
echo "Proceed to step2_eval_original.sh"
echo "============================================================"

#!/bin/bash
# =============================================================================
# STEP 5: Aggregate evaluation results across seeds
# =============================================================================
# Reads all results.json files produced by steps 2 and 4, computes
# mean ± std per metric per checkpoint, and produces:
#   - aggregated_results.csv   (for paper tables)
#   - aggregated_results.png   (for paper figures)
#   - Console summary
#
# Can be run at any time after step 2 (or step 4) to see partial results.
#
# Run from: BeamPERL/BeamRL/
#   bash scripts/experiments/step5_aggregate.sh
#
# Filter to a specific model:
#   bash scripts/experiments/step5_aggregate.sh "*beamrl_260101*"
#   bash scripts/experiments/step5_aggregate.sh "*beamrl_format_only*"
#   bash scripts/experiments/step5_aggregate.sh "*beamrl_accuracy_only*"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAMRL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PATTERN="${1:-}"  # Optional glob filter, e.g. "*beamrl_260101*"

echo "============================================================"
echo "STEP 5: Aggregating evaluation results"
echo "Started: $(date)"
echo "============================================================"

cd "${BEAMRL_DIR}"
source "./setup/set_vars.sh"

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR not set. source setup/set_vars.sh first."
    exit 1
fi

echo "Reading results from: ${OUTPUT_DIR}/beamrl_eval/"
echo "Seeds: ${EVAL_SEEDS:-42 123 456}"
[ -n "${PATTERN}" ] && echo "Filter pattern: ${PATTERN}"
echo ""

# Build command
SEEDS_ARGS=""
for s in ${EVAL_SEEDS:-42 123 456}; do
    SEEDS_ARGS="${SEEDS_ARGS} ${s}"
done

AGG_CMD="python ./scripts/eval/aggregate_eval_results.py \
    --output_dir \"${OUTPUT_DIR}\" \
    --seeds ${SEEDS_ARGS} \
    --out_csv \"${OUTPUT_DIR}/aggregated_results.csv\" \
    --out_fig \"${OUTPUT_DIR}/aggregated_results.png\" \
    --per_category \
    --eval_dataset tphage/BeamRL-EvalData-v2 \
    --eval_split train"

[ -n "${PATTERN}" ] && AGG_CMD="${AGG_CMD} --pattern \"${PATTERN}\""

eval ${AGG_CMD}

echo ""
echo "============================================================"
echo "STEP 5 COMPLETE: $(date)"
echo "Outputs:"
echo "  ${OUTPUT_DIR}/aggregated_results.csv             (overall mean ± std)"
echo "  ${OUTPUT_DIR}/aggregated_per_category.csv        (per-category mean ± std)"
echo "  ${OUTPUT_DIR}/aggregated_results.png             (overall summary figure)"
echo "============================================================"

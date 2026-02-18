#!/bin/bash
PRECISION=$1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to the project root (two levels up from scripts/submission_scripts)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"


precisions=(w4a16 w4a8)
model_names=(state-spaces/mamba2-1.3b state-spaces/mamba2-2.7b pretrained_models/state-spaces/mamba2-8b)
removal_names=(quamba2-1.3b quamba2-2.7b quamba2-8b)

# Iterate through both arrays in lockstep using array indices
for idx in "${!model_names[@]}"; do
    model_name="${model_names[$idx]}"
    removal_name="${removal_names[$idx]}"

    echo "Evaluating $model_name with precision fp16"
    "$PROJECT_ROOT/eval.sh" "$model_name" fp16 false false
    for precision in "${precisions[@]}"; do

        echo "Evaluating $model_name with precision $precision"
        
        for i in true false; do
            for j in true false; do
                "$PROJECT_ROOT/eval.sh" "$model_name" $precision $i $j
                rm -rf "$PROJECT_ROOT/pretrained_models/ut-enyac/${removal_name}-${precision}"
            done
        done
    done
done
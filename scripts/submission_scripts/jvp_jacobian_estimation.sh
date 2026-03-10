#!/bin/bash
#
# JVP Jacobian Estimation Script
# 
# This script runs Jacobian estimation for Mamba models using JVP sampling.
# It provides a clean, dedicated pathway separate from the evaluation pipeline.
#
# Usage:
#   ./jvp_jacobian_estimation.sh <model_path> [options]
#
# Example:
#   ./jvp_jacobian_estimation.sh state-spaces/mamba2-130m --w_bits 4 --nsamples 128
#

# Parse command-line arguments
MODEL=$1
shift  # Remove the first argument (model path) from the list

# Default values
W_BITS=4
NSAMPLES=128
SEQLEN=1024
OUTPUT_DIR="./jacobian_jvp_error"
PRETRAINED_DIR="./pretrained_models"
VERBOSE=""

# Check if model is provided
if [[ -z "$MODEL" ]]; then
  echo "Error: Model path is required"
  echo ""
  echo "Usage: $0 <model_path> [options]"
  echo ""
  echo "Arguments:"
  echo "  model_path         Path to the model (e.g., state-spaces/mamba2-130m)"
  echo ""
  echo "Options:"
  echo "  --w_bits N         Target bit-width for weights (default: 4)"
  echo "  --nsamples N       Number of calibration samples (default: 128)"
  echo "  --seqlen N         Sequence length for calibration (default: 1024)"
  echo "  --output_dir PATH  Output directory for results (default: ./jacobian_jvp_error)"
  echo "  --pretrained_dir PATH  Directory containing pretrained models (default: ./pretrained_models)"
  echo "  --group_heads      Group heads during reordering (for mamba2)"
  echo "  --verbose          Enable verbose logging"
  echo ""
  echo "Examples:"
  echo "  $0 state-spaces/mamba2-130m"
  echo "  $0 state-spaces/mamba2-130m --w_bits 4 --nsamples 256"
  echo "  $0 state-spaces/mamba2-2.7b --group_heads --verbose"
  exit 1
fi

# Build the command
CMD="python jacobian_estimation/main_jacobian.py $MODEL --pretrained_dir $PRETRAINED_DIR --w_bits $W_BITS --nsamples $NSAMPLES --seqlen $SEQLEN --output_dir $OUTPUT_DIR"

# Parse additional options
while [[ $# -gt 0 ]]; do
  case $1 in
    --w_bits)
      CMD+=" --w_bits $2"
      shift 2
      ;;
    --nsamples)
      CMD+=" --nsamples $2"
      shift 2
      ;;
    --seqlen)
      CMD+=" --seqlen $2"
      shift 2
      ;;
    --output_dir)
      CMD+=" --output_dir $2"
      shift 2
      ;;
    --pretrained_dir)
      CMD+=" --pretrained_dir $2"
      shift 2
      ;;
    --group_heads)
      CMD+=" --group_heads"
      shift
      ;;
    --verbose)
      CMD+=" --verbose"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0' without arguments for usage information"
      exit 1
      ;;
  esac
done

echo "========================================="
echo "JVP Jacobian Estimation"
echo "========================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "========================================="
echo ""
echo "Running: $CMD"
echo ""

eval $CMD


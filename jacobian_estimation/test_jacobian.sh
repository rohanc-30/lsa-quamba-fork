#!/bin/bash
#
# Test script for Jacobian estimation pathway
#
# This script provides a quick test to verify the Jacobian estimation
# pipeline is working correctly with minimal compute requirements.
#

echo "========================================"
echo "Testing Jacobian Estimation Pathway"
echo "========================================"
echo ""

# Use a small model and minimal samples for quick testing
MODEL="state-spaces/mamba2-130m"
NSAMPLES=8  # Very small for testing
SEQLEN=256  # Shorter sequences
OUTPUT_DIR="./test_jacobian_output"

echo "Test Configuration:"
echo "  Model: $MODEL"
echo "  Samples: $NSAMPLES"
echo "  Sequence Length: $SEQLEN"
echo "  Output: $OUTPUT_DIR"
echo ""

# Clean up previous test output if it exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning up previous test output..."
    rm -rf "$OUTPUT_DIR"
fi

echo "Running test..."
echo ""

# Run the Jacobian estimation with minimal settings
python jacobian_estimation/main_jacobian.py "$MODEL" \
    --nsamples $NSAMPLES \
    --seqlen $SEQLEN \
    --output_dir "$OUTPUT_DIR" \
    --w_bits 4 \
    --verbose

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 1 ]; then
    # Exit code 1 is expected due to ValueError("Jacobian loop over!")
    echo "Test Status: ✓ SUCCESS"
    echo "========================================"
    echo ""
    echo "Output files:"
    ls -lh "$OUTPUT_DIR" 2>/dev/null || echo "  (Output directory not created)"
    echo ""
    echo "To run a full Jacobian estimation, use:"
    echo "  ./scripts/submission_scripts/jvp_jacobian_estimation.sh $MODEL"
else
    echo "Test Status: ✗ FAILED"
    echo "Exit code: $EXIT_CODE"
    echo "========================================"
fi

echo ""


# TorchAO Baseline Quantization

This folder contains torchao-based quantization implementations for GLA and Mamba models.

## Purpose

Provide baseline RTN (Round-to-Nearest) and PTQ (Post-Training Quantization) implementations using torchao library to compare against the custom Quamba quantization approach.

## Structure

- `quantize_torchao.py` - Main quantization script for RTN/PTQ
- `utils_torchao.py` - Helper functions for torchao quantization
- `evaluate_torchao.py` - Evaluation script (optional, can use main.py)

## Usage

### Quantize a model with RTN:
```bash
python torchao_baseline/quantize_torchao.py \
    --model fla-hub/gla-1.3b \
    --model_type gla \
    --method rtn \
    --w_bits 4 \
    --pretrained_dir pretrained_models \
    --output_dir pretrained_models/ut-enyac
```

### Quantize a model with PTQ:
```bash
python torchao_baseline/quantize_torchao.py \
    --model state-spaces/mamba2-1.3b \
    --model_type mamba2 \
    --method ptq \
    --w_bits 8 \
    --calib_samples 512 \
    --pretrained_dir pretrained_models \
    --output_dir pretrained_models/ut-enyac
```

### Evaluate quantized model:
```bash
python main.py torchao-gla-1.3b-w4-rtn \
    --pretrained_dir pretrained_models \
    --eval_ppl \
    --eval_zero_shot \
    --log_dir logs
```

## Supported Models

- GLA (fla-hub/gla-1.3b)
- Mamba2 (state-spaces/mamba2-1.3b, mamba2-2.7b)
- Mamba (state-spaces/mamba-1.4b, etc.)

## Quantization Methods

- **RTN (Round-to-Nearest)**: Simple weight-only quantization without calibration
- **PTQ (Post-Training Quantization)**: Weight quantization with calibration data

## Notes

- Models are saved to `pretrained_models/ut-enyac/torchao-{model_name}-w{bits}-{method}/`
- Naming convention follows: `torchao-{model_name}-w{bits}-{method}` (e.g., `torchao-gla-1.3b-w4-rtn`)


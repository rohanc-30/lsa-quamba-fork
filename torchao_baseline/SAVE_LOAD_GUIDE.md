# TorchAO Quantized Model Save/Load Guide

## Overview

This guide explains how to save and load quantized models with TorchAO, with special handling for weight-only quantization modes.

## Quantization Modes

### W8A8 (8-bit weights + 8-bit activations)
- Uses `QuantizedLinear` modules
- Compatible with standard `transformers` loading
- Can be loaded with `AutoModelForCausalLM.from_pretrained()`

### W8A16 and W4A16 (Weight-Only Quantization)
- Uses `AffineQuantizedTensor` for weights
- Requires special loading mechanism
- Incompatible with standard `transformers` loading

## Saving Models

### For W8A8 Models
```python
from torchao_baseline.utils_torchao import save_quantized_model

save_path = save_quantized_model(
    model=model,
    tokenizer=tokenizer,
    output_dir='pretrained_models/fla-hub',
    model_name='gla_ptq-w8a8-1.3b',
    model_type='gla',
    quant_mode='w8a8',
    metadata={
        'calibration_samples': 512,
        'calibration_dataset': 'wikitext',
    }
)
```

### For W8A16/W4A16 Models (Weight-Only)
**Important:** You MUST provide `base_model_path` for weight-only models!

```python
save_path = save_quantized_model(
    model=model,
    tokenizer=tokenizer,
    output_dir='pretrained_models/fla-hub',
    model_name='gla_ptq-w8a16-1.3b',
    model_type='gla',
    quant_mode='w8a16',
    base_model_path='pretrained_models/fla-hub/gla-1.3B-100B',  # REQUIRED!
    metadata={
        'calibration_samples': 512,
        'calibration_dataset': 'wikitext',
        'group_size': 128,
    }
)
```

The `base_model_path` is stored in the quantization config and used during loading to reconstruct the model.

## Loading Models

### For W8A8 Models
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'pretrained_models/fla-hub/gla_ptq-w8a8-1.3b'
).to('cuda')
```

### For W8A16/W4A16 Models (Weight-Only)
```python
from torchao_baseline.utils_torchao import load_weight_only_quantized_model

model = load_weight_only_quantized_model(
    model_path='pretrained_models/fla-hub/gla_ptq-w8a16-1.3b',
    device='cuda'
)
```

### Automatic Loading in `utils.py`
The `build_mamba_and_tokenizer()` function in `utils.py` automatically detects weight-only quantization and uses the appropriate loader:

```python
# This automatically handles both W8A8 and W8A16/W4A16
python main.py --model gla_ptq-w8a16-1.3b \
    --pretrained_dir pretrained_models/fla-hub \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot
```

## How Weight-Only Loading Works

The `load_weight_only_quantized_model()` function performs three steps:

1. **Load Quantization Config**: Reads `quantization_config.json` to determine:
   - Quantization mode (w8a16 or w4a16)
   - Base model path
   - Group size

2. **Load Base Model**: Loads the unquantized base model from the stored path

3. **Apply Quantization Structure**: Applies the quantization transformation to create `AffineQuantizedTensor` placeholders

4. **Load Quantized Weights**: Loads the saved quantized state_dict into the prepared model

## Directory Structure

After saving, each quantized model directory contains:
```
gla_ptq-w8a16-1.3b/
├── pytorch_model.bin           # Quantized weights (with AffineQuantizedTensor)
├── quantization_config.json    # Quantization metadata
├── tokenizer_config.json       # Tokenizer files
├── special_tokens_map.json
└── ...
```

### quantization_config.json Example (Weight-Only)
```json
{
  "model_type": "gla",
  "model_name": "gla_ptq-w8a16-1.3b",
  "quantization": {
    "method": "torchao_w8a16_static",
    "weight_bits": 8,
    "activation_bits": 16,
    "weight_only": true,
    "requires_special_loading": true,
    "base_model_path": "pretrained_models/fla-hub/gla-1.3B-100B",
    "group_size": 128
  },
  "metadata": {
    "calibration_samples": 512,
    "calibration_dataset": "wikitext"
  }
}
```

## Why Weight-Only Needs Special Handling

### The Problem
Weight-only quantization replaces `nn.Linear.weight` with `AffineQuantizedTensor` subclasses. The `transformers` library's `AutoModelForCausalLM.from_pretrained()` uses `copy_()` to load weights, which fails with:
```
RuntimeError: Error(s) in loading state_dict...
'Not supported args for copy_ due to metadata mismatch'
```

This happens because `copy_()` cannot copy from a regular `Parameter` (model's expectation) to an `AffineQuantizedTensor` (saved checkpoint).

### The Solution
1. Load the base model first (creates regular `Parameter` objects)
2. Apply quantization (replaces `Parameter` with `AffineQuantizedTensor` structure)
3. Use `load_state_dict()` which supports loading into custom tensor subclasses

## Tips and Best Practices

1. **Always save `base_model_path`** for weight-only models
2. **Keep the base model** accessible at the saved path
3. **Use consistent group sizes** between save and load
4. **Test loading** immediately after saving to catch issues early

## Troubleshooting

### Error: "base_model_path not found in quantization config"
**Solution:** You forgot to pass `base_model_path` when saving. Re-quantize and save with the parameter.

### Error: "Base model not found at path"
**Solution:** The base model has been moved or deleted. Update the path in `quantization_config.json` or restore the base model.

### Error: "RuntimeError: Error(s) in loading state_dict"
**Solution:** You're trying to load a weight-only model with `AutoModelForCausalLM.from_pretrained()`. Use `load_weight_only_quantized_model()` instead.

## Evaluation with LM Eval

To evaluate quantized models with the main evaluation pipeline:

```bash
# For any quantization mode (auto-detects and uses correct loader)
python main.py \
    --model gla_ptq-w8a16-1.3b \
    --pretrained_dir pretrained_models/fla-hub \
    --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,winogrande \
    --eval_zero_shot
```

The `utils.py` automatically detects weight-only quantization by checking for `requires_special_loading` in the quantization config.


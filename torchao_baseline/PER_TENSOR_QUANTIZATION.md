# Per-Tensor Quantization with group_size=0

## ⚠️ Important Limitation

**Per-tensor quantization (`group_size=0`) is ONLY supported for W8A16, NOT for W4A16.**

- ✅ **W8A16** (8-bit weights): Supports `group_size=0` via `int8_weight_only(group_size=None)`
- ❌ **W4A16** (4-bit weights): Does NOT support `group_size=0` - TorchAO's `int4_weight_only` requires an explicit group_size

## Overview

Weight-only quantization (W8A16 only) now supports per-tensor quantization mode using `group_size=0` as a special sentinel value.

## Quantization Granularity Options

### 1. Per-Channel with Grouping (Default)
```python
model = convert_static_w8a16(model, group_size=128)  # 128 channels per group
model = convert_static_w4a16(model, group_size=64)   # 64 channels per group
```
- **Use case:** Best balance of accuracy vs model size
- **Group size:** Typically 64-128 for good accuracy
- **Quantization:** Different scale/zero-point per group of channels

### 2. Per-Tensor (NEW: group_size=0) - W8A16 ONLY
```python
model = convert_static_w8a16(model, group_size=0)  # ✅ Per-tensor quantization (WORKS)
# model = convert_static_w4a16(model, group_size=0)  # ❌ NOT SUPPORTED - will raise error
```
- **Use case:** Smallest model size, faster inference
- **Group size:** 0 (special value meaning "no grouping")
- **Quantization:** Single scale/zero-point for entire weight tensor
- **⚠️ Only works with W8A16**, not W4A16 (TorchAO limitation)

### 3. Auto-Detection (group_size=None)
```python
model = convert_static_w8a16(model, group_size=None)  # Auto-detect safe size
```
- **Use case:** Let the system choose based on layer dimensions
- **Behavior:** Finds largest safe group size compatible with all layers

## Implementation Details

### group_size=0 Conversion

When `group_size=0` is passed to the conversion functions:

1. **Detection**: The function detects `group_size == 0`
2. **Conversion**: Sets `group_size = None` (TorchAO's per-tensor mode)
3. **Logging**: Prints "PER-TENSOR quantization" messages
4. **Validation**: Skips group size validation (not needed for per-tensor)

### Code Flow

```python
# USAGE_EXAMPLE.py
model = convert_static_w8a16(model, group_size=0)

# ↓ Inside convert_static_w8a16()

if group_size == 0:
    print("🔧 group_size=0 detected → Using PER-TENSOR quantization")
    per_tensor_quant = True
    group_size = None  # TorchAO uses None for per-tensor

# ...

if per_tensor_quant:
    print("✓ Using PER-TENSOR quantization (no grouping)")
    # Skip validation
else:
    # Normal group size validation

# Apply quantization with group_size=None for per-tensor
quantize_(model, int8_weight_only(group_size=None), is_quantizable_linear)
```

## Usage Examples

### Example 1: Quantize with Per-Tensor

```python
from torchao_baseline.utils_torchao import (
    load_model_for_quantization,
    convert_static_w8a16,
    convert_static_w4a16,
    save_quantized_model
)

# Load model
model, tokenizer, config = load_model_for_quantization(
    model_path='pretrained_models/state-spaces/mamba2-1.3b',
    model_type='mamba2',
    device='cuda'
)

# Quantize with per-tensor mode
model = convert_static_w8a16(model, group_size=0)

# Save with group_size=0 in metadata
save_quantized_model(
    model=model,
    tokenizer=tokenizer,
    output_dir='pretrained_models/state-spaces',
    model_name='mamba2_ptq-w8a16-per-tensor-1.3b',
    model_type='mamba2',
    quant_mode='w8a16',
    base_model_path='pretrained_models/state-spaces/mamba2-1.3b',
    metadata={'group_size': 0}  # Saved as 0 in config
)
```

### Example 2: Compare Quantization Modes

```python
# Load three copies of the model
model_grouped = load_model_for_quantization(...)
model_per_tensor = load_model_for_quantization(...)
model_auto = load_model_for_quantization(...)

# Different quantization modes
convert_static_w8a16(model_grouped, group_size=128)     # Grouped
convert_static_w8a16(model_per_tensor, group_size=0)   # Per-tensor
convert_static_w8a16(model_auto, group_size=None)      # Auto-detect

# Compare results...
```

### Example 3: Load Per-Tensor Model

```python
from torchao_baseline.utils_torchao import load_weight_only_quantized_model

# Automatically handles group_size=0 from saved config
model = load_weight_only_quantized_model(
    'pretrained_models/state-spaces/mamba2_ptq-w8a16-per-tensor-1.3b',
    device='cuda'
)

# The loader:
# 1. Reads group_size=0 from config
# 2. Passes it to convert_static_w8a16()
# 3. Function converts 0 → None
# 4. Applies per-tensor quantization
```

## Saved Configuration

### With group_size=0 (Per-Tensor)
```json
{
  "model_type": "mamba2",
  "quantization": {
    "method": "torchao_w8a16_static",
    "weight_bits": 8,
    "activation_bits": 16,
    "weight_only": true,
    "requires_special_loading": true,
    "base_model_path": "pretrained_models/state-spaces/mamba2-1.3b",
    "group_size": 0
  },
  "metadata": {
    "group_size": 0,
    "quantization_framework": "torchao",
    "quantization_method": "PTQ"
  }
}
```

### With group_size=128 (Grouped)
```json
{
  "quantization": {
    "group_size": 128
  },
  "metadata": {
    "group_size": 128
  }
}
```

## Trade-offs

### Per-Tensor (group_size=0)

**Advantages:**
- ✅ Smallest model size (fewer scale/zero-point parameters)
- ✅ Faster inference (less overhead)
- ✅ Simpler quantization math

**Disadvantages:**
- ❌ Lower accuracy (single scale for entire tensor)
- ❌ Poor for tensors with high dynamic range
- ❌ May lose precision in outlier channels

### Per-Channel with Grouping (group_size=64/128)

**Advantages:**
- ✅ Better accuracy (separate scales per group)
- ✅ Handles varying channel distributions
- ✅ Good for tensors with outliers

**Disadvantages:**
- ❌ Slightly larger model size
- ❌ More computation during quantization
- ❌ Group size must be compatible with layer dimensions

## When to Use Per-Tensor

Use `group_size=0` when:
1. **Model size is critical** - Need smallest possible model
2. **Inference speed matters** - Want fastest possible inference
3. **Uniform weights** - Model weights have similar ranges across channels
4. **Baseline comparison** - Want to compare against grouped quantization

Avoid `group_size=0` when:
1. **Accuracy is critical** - Can't afford quantization loss
2. **Non-uniform weights** - Weights have varying ranges per channel
3. **Large dynamic range** - Weights have outliers or extremes

## Console Output

### With group_size=0 (Per-Tensor)
```
================================================================================
🔧 CONVERT_STATIC_W8A16 CALLED
================================================================================
🔧 group_size=0 detected → Using PER-TENSOR quantization
ℹ️  Skipping gk_proj layers (keeping them in BF16 for larger group_size)
📊 Model dtypes found: {torch.float32}
📊 First parameter dtype: torch.float32
🔄 Converting model from torch.float32 to bfloat16 for W8A16 quantization...
✓ Model converted! New dtype: torch.bfloat16
✓ Using PER-TENSOR quantization (no grouping)
🚀 Applying W8A16 quantization with PER-TENSOR mode...
✓ W8A16 quantization complete!
```

### With group_size=128 (Grouped)
```
================================================================================
🔧 CONVERT_STATIC_W8A16 CALLED
================================================================================
ℹ️  Skipping gk_proj layers (keeping them in BF16 for larger group_size)
📊 Model dtypes found: {torch.float32}
📊 First parameter dtype: torch.float32
🔄 Converting model from torch.float32 to bfloat16 for W8A16 quantization...
✓ Model converted! New dtype: torch.bfloat16
✓ Using specified group_size: 128
🔍 Validating group_size=128...
✓ group_size=128 validated successfully
🚀 Applying W8A16 quantization with group_size=128...
✓ W8A16 quantization complete!
```

## Modified Functions

### 1. convert_static_w8a16()
- Added `group_size=0` detection
- Converts `0 → None` for TorchAO
- Skips validation for per-tensor mode
- Updated logging to show "PER-TENSOR" mode

### 2. convert_static_w4a16()
- Same changes as w8a16
- Handles both INT8 and INT4 per-tensor quantization

### 3. load_weight_only_quantized_model()
- Reads `group_size=0` from config
- Passes directly to convert functions (automatic conversion)
- Updated logging to show "PER-TENSOR" when group_size=0

## Testing

```bash
# Quantize with per-tensor mode
cd torchao_baseline
python -c "
from utils_torchao import load_model_for_quantization, convert_static_w8a16

model, tokenizer, _ = load_model_for_quantization(
    'pretrained_models/state-spaces/mamba2-1.3b',
    'mamba2',
    device='cuda'
)

# Try per-tensor quantization
model = convert_static_w8a16(model, group_size=0)
print('✅ Per-tensor quantization successful!')
"

# Evaluate accuracy difference
python main.py --model mamba2_ptq-w8a16-per-tensor-1.3b \
    --pretrained_dir pretrained_models/state-spaces \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot
```

## Summary

- **group_size=0** → Per-tensor quantization (single scale per weight tensor)
- **group_size=N** → Grouped quantization (scale per N channels)
- **group_size=None** → Auto-detect safe group size
- Works with both W8A16 and W4A16 quantization
- Fully integrated with save/load mechanism
- Transparent to evaluation pipeline

Use `group_size=0` for smallest models and fastest inference, at the cost of some accuracy loss.


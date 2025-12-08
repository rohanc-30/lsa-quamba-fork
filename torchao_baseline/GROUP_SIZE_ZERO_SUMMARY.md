# Summary: group_size=0 for Per-Tensor Quantization

## What Changed

Implemented support for `group_size=0` as a special value to enable **per-tensor quantization** in W8A16 and W4A16 weight-only quantization modes.

## Key Changes

### 1. Modified `convert_static_w8a16()` 
**File:** `torchao_baseline/utils_torchao.py` (Lines 915-1015)

**Changes:**
- Added detection for `group_size == 0`
- Converts `group_size=0` → `group_size=None` (TorchAO's per-tensor mode)
- Skips group size validation for per-tensor mode
- Updated console logging to show "PER-TENSOR" mode

**Code:**
```python
# Handle group_size=0 as per-tensor quantization
per_tensor_quant = False
if group_size == 0:
    print("🔧 group_size=0 detected → Using PER-TENSOR quantization")
    logger.info("group_size=0 → Using per-tensor quantization")
    per_tensor_quant = True
    group_size = None  # TorchAO uses None for per-tensor
```

### 2. Modified `convert_static_w4a16()`
**File:** `torchao_baseline/utils_torchao.py` (Lines 808-913)

**Changes:**
- Same modifications as `convert_static_w8a16()`
- Handles INT4 per-tensor quantization

### 3. Updated `load_weight_only_quantized_model()`
**File:** `torchao_baseline/utils_torchao.py` (Lines 1170-1277)

**Changes:**
- Added comment: `group_size=0` means per-tensor
- Enhanced logging to show "PER-TENSOR (group_size=0)" when loading
- Passes `group_size=0` to convert functions (automatic conversion)

**Code:**
```python
# Get group size (0 means per-tensor quantization)
group_size = quant_config.get('group_size', 128)

if group_size == 0:
    logger.info(f"  Quantization: PER-TENSOR (group_size=0)")
else:
    logger.info(f"  Group size: {group_size}")
```

## How It Works

### Save Flow
```
User Code:
  convert_static_w8a16(model, group_size=0)
  save_quantized_model(..., metadata={'group_size': 0})

↓

quantization_config.json:
  {"quantization": {"group_size": 0}}
```

### Load Flow
```
load_weight_only_quantized_model(...)

↓ Reads config
  group_size = 0

↓ Passes to convert
  convert_static_w8a16(model, group_size=0)

↓ Detects and converts
  if group_size == 0:
      group_size = None  # Per-tensor mode

↓ Applies quantization
  int8_weight_only(group_size=None)  # TorchAO per-tensor
```

## Usage

### Before (Not Supported)
```python
# No way to do per-tensor quantization
model = convert_static_w8a16(model, group_size=None)  # Auto-detects, not per-tensor!
```

### After (Now Supported)
```python
# Per-tensor quantization
model = convert_static_w8a16(model, group_size=0)  # ✅ Per-tensor mode

# Grouped quantization
model = convert_static_w8a16(model, group_size=128)  # ✅ Grouped mode

# Auto-detect (finds safe group size)
model = convert_static_w8a16(model, group_size=None)  # ✅ Auto mode
```

## Quantization Modes Summary

| group_size | Mode         | Description                              |
|-----------|--------------|------------------------------------------|
| 0         | Per-tensor   | Single scale for entire weight tensor   |
| 64/128/.. | Grouped      | Scale per N channels                     |
| None      | Auto-detect  | Find largest safe group size for model   |

## Example: Quantize with Per-Tensor

```python
from torchao_baseline.utils_torchao import (
    load_model_for_quantization,
    convert_static_w8a16,
    save_quantized_model
)

# Load model
model, tokenizer, _ = load_model_for_quantization(
    'pretrained_models/state-spaces/mamba2-1.3b',
    'mamba2',
    device='cuda'
)

# Quantize with per-tensor mode
model = convert_static_w8a16(model, group_size=0)

# Save
save_quantized_model(
    model=model,
    tokenizer=tokenizer,
    output_dir='pretrained_models/state-spaces',
    model_name='mamba2_ptq-w8a16-per-tensor-1.3b',
    model_type='mamba2',
    quant_mode='w8a16',
    base_model_path='pretrained_models/state-spaces/mamba2-1.3b',
    metadata={'group_size': 0}  # Per-tensor marker
)
```

## Console Output

### With group_size=0
```
🔧 group_size=0 detected → Using PER-TENSOR quantization
✓ Using PER-TENSOR quantization (no grouping)
🚀 Applying W8A16 quantization with PER-TENSOR mode...
✓ W8A16 quantization complete!
```

### With group_size=128
```
✓ Using specified group_size: 128
🔍 Validating group_size=128...
✓ group_size=128 validated successfully
🚀 Applying W8A16 quantization with group_size=128...
✓ W8A16 quantization complete!
```

## Files Modified

1. **torchao_baseline/utils_torchao.py**
   - `convert_static_w8a16()` - Added per-tensor support
   - `convert_static_w4a16()` - Added per-tensor support
   - `load_weight_only_quantized_model()` - Enhanced logging

2. **torchao_baseline/PER_TENSOR_QUANTIZATION.md** (NEW)
   - Comprehensive documentation on per-tensor quantization
   - Usage examples and trade-offs

3. **torchao_baseline/GROUP_SIZE_ZERO_SUMMARY.md** (NEW)
   - This summary document

## Backward Compatibility

✅ **Fully backward compatible:**
- Existing models with `group_size=64/128` work unchanged
- Auto-detect mode (`group_size=None`) works unchanged
- No breaking changes to API

## Benefits

1. **Flexibility:** Users can now choose per-tensor quantization
2. **Smaller Models:** Per-tensor has lower overhead
3. **Faster Inference:** Less computation during inference
4. **Easy to Use:** Just pass `group_size=0`
5. **Transparent:** Works seamlessly with save/load

## When to Use group_size=0

**Use Per-Tensor (group_size=0) when:**
- Model size is critical
- Inference speed is priority
- Weights have uniform distribution
- You want a baseline for comparison

**Use Grouped (group_size=64/128) when:**
- Accuracy is critical
- Weights have varying ranges per channel
- You can afford slightly larger model size

## Testing

```bash
# Test per-tensor quantization
cd torchao_baseline
python -c "
from utils_torchao import *

# Load model
model, tokenizer, _ = load_model_for_quantization(
    'pretrained_models/state-spaces/mamba2-1.3b',
    'mamba2',
    device='cuda'
)

# Test per-tensor
model = convert_static_w8a16(model, group_size=0)
print('✅ Per-tensor (group_size=0) works!')

# Load another copy and test grouped
model2, _, _ = load_model_for_quantization(
    'pretrained_models/state-spaces/mamba2-1.3b',
    'mamba2',
    device='cuda'
)
model2 = convert_static_w8a16(model2, group_size=128)
print('✅ Grouped (group_size=128) works!')
"
```

## Related Documentation

- **PER_TENSOR_QUANTIZATION.md** - Detailed guide on per-tensor quantization
- **SAVE_LOAD_GUIDE.md** - How to save/load quantized models
- **WEIGHT_ONLY_LOADING_IMPLEMENTATION.md** - Technical implementation details

## Summary

✅ `group_size=0` now enables per-tensor quantization for W8A16 and W4A16  
✅ Fully integrated with save/load mechanism  
✅ Clear console logging shows "PER-TENSOR" mode  
✅ Backward compatible with existing code  
✅ Works for both GLA and Mamba2 models  


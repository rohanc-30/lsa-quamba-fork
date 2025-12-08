# Fix for Weight-Only Quantized Model Loading Issues

## Problem

When loading weight-only quantized models (W8A16, W4A16), the following error occurred:

```
RuntimeError: Error(s) in loading state_dict for MambaLMHeadModel:
While copying the parameter named "backbone.layers.0.mixer.in_proj.weight"...
an exception occurred: ("Not supported args for copy_ due to metadata mismatch:
(AffineQuantizedTensor... block_size=(1, 64)... vs AffineQuantizedTensor... block_size=(1, 256)...")
```

**Root Cause:**  
The saved model was quantized with `group_size=64` (block_size=64), but when reconstructing during load, a different group_size was used (resulting in block_size=256), causing a metadata mismatch that prevented `load_state_dict()` from copying the weights.

## Changes Made

### 1. Enhanced Group Size Detection (Lines 1223-1237)

**Before:**
```python
group_size = quant_config.get('group_size', 128)
```

**After:**
```python
# Check both quantization config and metadata
group_size_from_quant = quant_config.get('group_size')
group_size_from_meta = config.get('metadata', {}).get('group_size')

# Use the value from metadata if available (more reliable)
if group_size_from_meta is not None:
    group_size = group_size_from_meta
elif group_size_from_quant is not None:
    group_size = group_size_from_quant
else:
    group_size = 128  # default fallback
```

**Why:** Metadata is more reliable since it's user-provided. Checks both locations for maximum compatibility.

### 2. Simple and Correct (Lines 1267-1292)

**The Simple Solution - Just Use the Config:**

**Old approach (OVERCOMPLICATED):**
```python
# Try to detect group_size from saved weights by inspecting block_size
# Problem: Can't distinguish between group_size=2048 and per-tensor quantization
saved_temp = torch.load(state_dict_path)
for key, value in saved_temp.items():
    if hasattr(value, 'block_size'):
        detected_group_size = value.block_size[1]  # Could be wrong!
        break
```

**New approach (SIMPLE):**
```python
# Just read group_size from config (it's already correct!)
group_size = config['metadata']['group_size']  # 0 for per-tensor, 64/128/etc for per-channel

# Apply quantization with config group_size
model = convert_static_w8a16(model, group_size=group_size)

# Load weights (structures match!)
model.load_state_dict(saved_state_dict, strict=False)
```

**Why:** The config already stores the correct `group_size` (including `0` for per-tensor). Just use it directly! The `convert_static_w8a16` function already handles `group_size=0` correctly by converting it to `None` for TorchAO.

## How to Fix Existing Models

### Option 1: Re-Quantize with Correct group_size

If you changed the group_size (e.g., from 128 to 64), you need to re-quantize:

```bash
# Delete old quantized models
rm -rf pretrained_models/state-spaces/mamba2_ptq-w8a16-1.3b
rm -rf pretrained_models/state-spaces/mamba2_ptq-w4a16-1.3b

# Re-run quantization with new group_size
cd torchao_baseline
python USAGE_EXAMPLE.py
```

### Option 2: Manually Fix quantization_config.json

If the saved model is correct but the config is wrong:

```bash
# Check what's saved
cat pretrained_models/state-spaces/mamba2_ptq-w8a16-1.3b/quantization_config.json
```

Ensure it has:
```json
{
  "quantization": {
    "group_size": 64
  },
  "metadata": {
    "group_size": 64
  }
}
```

If not, edit the file to add the correct group_size.

### Option 3: Check Saved Model's Block Size

To see what group_size was actually used during quantization:

```python
import torch

# Load a saved weight
state_dict = torch.load('pretrained_models/state-spaces/mamba2_ptq-w8a16-1.3b/pytorch_model.bin')

# Check the first quantized weight
for key, value in state_dict.items():
    if hasattr(value, 'block_size'):
        print(f"{key}: block_size={value.block_size}")
        # block_size=(1, 64) means group_size=64
        # block_size=(1, 128) means group_size=128
        # block_size=(1, 256) means group_size=256
        break
```

## Diagnostic Flow

The updated loader now provides clear diagnostics:

```
INFO: Loading weight-only quantized model (W8A16)...
INFO:   Model path: pretrained_models/state-spaces/mamba2_ptq-w8a16-1.3b
INFO:   Base model: pretrained_models/state-spaces/mamba2-1.3b
INFO:   Config group_size (quant): 64
INFO:   Config group_size (metadata): 64
INFO:   → Using group_size from metadata: 64

INFO: Step 1: Loading base model...
INFO:   ✓ Base model loaded

INFO: Step 2: Applying W8A16 quantization structure...
INFO:   Using group_size=64 (per-channel quantization)
INFO:   ✓ Quantization structure applied

INFO: Step 3: Loading quantized weights...
INFO:   ✓ Quantized weights loaded successfully

✅ Model loaded successfully!
```

For per-tensor quantization (group_size=0):

```
INFO: Step 2: Applying W8A16 quantization structure...
INFO:   Using group_size=0 (per-tensor quantization)
INFO:   ✓ Quantization structure applied
```

## Prevention

To prevent this issue in the future:

1. **Always save group_size in metadata:**
```python
save_quantized_model(
    model=model,
    tokenizer=tokenizer,
    output_dir='...',
    model_name='...',
    model_type='mamba2',
    quant_mode='w8a16',
    base_model_path='...',
    metadata={
        'group_size': 64,  # CRITICAL: Include this!
        'quantization_framework': 'torchao',
        'quantization_method': 'PTQ',
    }
)
```

2. **Verify after saving:**
```python
# Immediately test loading after saving
model_loaded = load_weight_only_quantized_model(save_path, device='cuda')
print("✓ Model loads successfully!")
```

3. **Document group_size in model name:**
```python
model_name = f'mamba2_ptq-w8a16-g64-1.3b'  # g64 = group_size 64
```

## Summary of Fixes

✅ **Simple and correct** - just read group_size from config and use it  
✅ **Handles per-tensor quantization** - `group_size=0` works correctly  
✅ **Handles per-channel quantization** - `group_size=64/128/etc` works correctly  
✅ **No auto-detection complexity** - trust the saved config  
✅ **Proper quantization order** - apply quantization with config group_size, then load weights  
✅ **Detailed logging** - shows config values and quantization type  

## Testing

After applying these fixes:

```bash
# Test loading a quantized model
python -c "
from torchao_baseline.utils_torchao import load_weight_only_quantized_model

model = load_weight_only_quantized_model(
    'pretrained_models/state-spaces/mamba2_ptq-w8a16-1.3b',
    device='cuda'
)
print('✅ Model loaded successfully!')
"

# Test evaluation
python main.py --model mamba2_ptq-w8a16-1.3b \
    --pretrained_dir pretrained_models/state-spaces \
    --task_list lambada_openai \
    --eval_zero_shot --testing
```

## Related Files

- **utils_torchao.py** - Loading and saving functions
- **USAGE_EXAMPLE.py** - Example quantization script
- **GROUP_SIZE_ZERO_SUMMARY.md** - Per-tensor quantization guide
- **PER_TENSOR_QUANTIZATION.md** - Detailed quantization modes

## Important Notes

1. **Changing group_size requires re-quantization** - You can't just change the config
2. **Block size = (1, group_size)** - For per-channel quantization along input dimension
3. **Metadata is the source of truth** - Always include group_size in metadata when saving
4. **Old models may need re-quantization** - If saved before this fix, they might not load correctly



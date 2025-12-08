# Mamba2 Weight-Only Quantization Support

## Summary

Extended the weight-only quantization loading mechanism to support Mamba2 models (W8A16, W4A16).

## Changes Made

### 1. Updated `load_weight_only_quantized_model()` in `utils_torchao.py`
**Location:** Line 1194-1201

Added support for loading Mamba2 base models:
```python
elif model_type in ['mamba2', 'mamba']:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = MambaLMHeadModel.from_pretrained(base_model_path, device=device, dtype=dtype)
```

### 2. Added `mamba2_ptq` Model Type to `utils.py`
**Location:** Line 75-103

Created new model type `mamba2_ptq` that:
- Uses the same tokenizer logic as `mamba2`
- Auto-detects weight-only quantization from `quantization_config.json`
- Uses `load_weight_only_quantized_model()` for weight-only models
- Falls back to standard loading for W8A8 models

### 3. Updated Error Message in `utils.py`
**Location:** Line 112

Updated to include new supported types:
```python
raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', 'quamba', 'quamba2', 'gla_ptq', 'mamba2_ptq'")
```

### 4. Fixed Bug in `USAGE_EXAMPLE.py`
**Location:** Line 640

**Bug:** Missing print statements for model4 and model5 saves
**Fix:** Added proper print statements:
```python
print(f"✓ Mamba2 W8A16 quantized model saved to: {save_path4}")
print(f"✓ Mamba2 W4A16 quantized model saved to: {save_path5}")
```

### 5. Added Metadata to Mamba2 Save Calls
**Location:** Lines 629-634, 646-651

Added metadata for consistency:
```python
metadata={
    'calibration_samples': 0,  # Weight-only, no calibration needed
    'quantization_framework': 'torchao',
    'quantization_method': 'PTQ',
    'group_size': 128,
}
```

### 6. Updated Usage Instructions in `USAGE_EXAMPLE.py`
**Location:** Lines 659-661

Added Mamba2 evaluation example:
```bash
python main.py --model mamba2_ptq-w8a16-1.3b --pretrained_dir pretrained_models/state-spaces \
    --task_list lambada_openai,hellaswag --eval_zero_shot
```

## Review of User's USAGE_EXAMPLE.py Changes

### ✅ Correct Changes

1. **Model Loading** (Lines 60-70):
   - ✅ Correctly loads two Mamba2 models for W8A16 and W4A16 quantization
   - ✅ Uses correct path: `pretrained_models/state-spaces/mamba2-1.3b`

2. **Conversion** (Lines 561-562):
   - ✅ Correctly applies weight-only quantization without calibration
   - ✅ Weight-only quantization doesn't need prepare/calibrate steps

3. **Base Model Path** (Lines 576, 595):
   - ✅ Changed from `gla-1.3B-100B` to `gla-1.3b` (matches actual directory)
   - ✅ Verified: `/home/rcherukuri/lsa-quamba-fork/pretrained_models/fla-hub/gla-1.3b/` exists

4. **Mamba2 Saves** (Lines 621-640):
   - ✅ Correct output directories and model names
   - ✅ Correct base_model_path references

### 🐛 Bugs Fixed

1. **Missing Print Statements** (Line 640):
   - **Before:** Only printed `save_path3` after saving both mamba2 models
   - **After:** Added proper print statements for `save_path4` and `save_path5`

2. **Missing Metadata**:
   - **Before:** No metadata for mamba2 saves
   - **After:** Added consistent metadata with group_size

## Usage

### 1. Quantize and Save Models
```bash
cd torchao_baseline
python USAGE_EXAMPLE.py
```

This creates:
- **GLA Models:**
  - `pretrained_models/fla-hub/gla_ptq-w8a8-1.3b/`
  - `pretrained_models/fla-hub/gla_ptq-w8a16-1.3b/`
  - `pretrained_models/fla-hub/gla_ptq-w4a16-1.3b/`

- **Mamba2 Models:**
  - `pretrained_models/state-spaces/mamba2_ptq-w8a16-1.3b/`
  - `pretrained_models/state-spaces/mamba2_ptq-w4a16-1.3b/`

### 2. Evaluate with LM Eval

#### GLA Models
```bash
python main.py --model gla_ptq-w8a16-1.3b \
    --pretrained_dir pretrained_models/fla-hub \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot
```

#### Mamba2 Models
```bash
python main.py --model mamba2_ptq-w8a16-1.3b \
    --pretrained_dir pretrained_models/state-spaces \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot
```

### 3. Direct Loading (Optional)
```python
from torchao_baseline.utils_torchao import load_weight_only_quantized_model

# Load GLA model
model = load_weight_only_quantized_model(
    'pretrained_models/fla-hub/gla_ptq-w8a16-1.3b',
    device='cuda'
)

# Load Mamba2 model
model = load_weight_only_quantized_model(
    'pretrained_models/state-spaces/mamba2_ptq-w8a16-1.3b',
    device='cuda'
)
```

## Technical Details

### Model Type Mapping

| Model Type   | Base Model Class     | Quantization Support | Loader Function                        |
|-------------|---------------------|---------------------|----------------------------------------|
| `gla`       | GLAForCausalLM      | N/A                 | GLAForCausalLM.from_pretrained()       |
| `gla_ptq`   | GLAForCausalLM      | W8A8, W8A16, W4A16  | Auto-detect + custom loader            |
| `mamba2`    | MambaLMHeadModel    | N/A                 | MambaLMHeadModel.from_pretrained()     |
| `mamba2_ptq`| MambaLMHeadModel    | W8A8, W8A16, W4A16  | Auto-detect + custom loader            |

### Auto-Detection Flow

```
main.py with model_type="mamba2_ptq"
  └─> build_mamba_and_tokenizer()
       └─> Read quantization_config.json
            ├─> Check: requires_special_loading == True?
            │    └─> Yes: load_weight_only_quantized_model()
            │              ├─> Load base Mamba2 model
            │              ├─> Apply quantization (w8a16 or w4a16)
            │              └─> Load quantized weights
            │
            └─> No: MambaLMHeadModel.from_pretrained()
```

### Weight-Only vs W8A8

| Feature               | W8A8                    | W8A16/W4A16 (Weight-Only)  |
|----------------------|-------------------------|---------------------------|
| Needs calibration    | ✅ Yes                  | ❌ No                     |
| Module replacement   | ✅ QuantizedLinear      | ❌ AffineQuantizedTensor  |
| Standard loading     | ✅ Works                | ❌ Fails                  |
| Custom loader needed | ❌ No                   | ✅ Yes                    |

## Files Modified

1. **torchao_baseline/utils_torchao.py**
   - `load_weight_only_quantized_model()` - Added Mamba2 support

2. **utils.py**
   - `build_mamba_and_tokenizer()` - Added `mamba2_ptq` model type

3. **torchao_baseline/USAGE_EXAMPLE.py**
   - Fixed missing print statements
   - Added metadata to mamba2 saves
   - Updated usage instructions

## Verification

### Pre-requisites
- Base models must exist:
  - ✅ `/home/rcherukuri/lsa-quamba-fork/pretrained_models/fla-hub/gla-1.3b/`
  - ✅ `/home/rcherukuri/lsa-quamba-fork/pretrained_models/state-spaces/mamba2-1.3b/`

### Testing
1. Run `USAGE_EXAMPLE.py` to create quantized models
2. Verify all 5 models are saved correctly
3. Test loading with `main.py` for each model type
4. Verify evaluation runs without errors

## Known Limitations

1. **Base Model Dependency**: Quantized models require the base model to exist at the saved path
2. **Group Size**: Must use same group size (128) for quantization and loading
3. **Tokenizer**: For mamba2-8b, requires special tokenizer file

## Future Work

1. Support W8A8 Mamba2 quantization (requires calibration)
2. Add support for Mamba (v1) weight-only quantization
3. Enable custom group sizes per layer
4. Add model architecture validation during loading

## References

- Main implementation: `WEIGHT_ONLY_LOADING_IMPLEMENTATION.md`
- User guide: `SAVE_LOAD_GUIDE.md`
- TorchAO: https://github.com/pytorch/ao


# LM Eval Wrapper Patch for PTQ Models

## Issue

When attempting to evaluate `mamba2_ptq` models with `main.py`, the evaluation failed with:
```
ValueError: Unsupported model type: mamba2_ptq, only support 'mamba', 'mamba2', 'quamba' and 'quamba2'
```

## Root Cause

The `lm_eval_wrapper.py` file in `quamba/eval_utils/` did not recognize `mamba2_ptq` as a valid model type, even though:
- `utils.py` correctly loads `mamba2_ptq` models
- The model loading mechanism works correctly
- The model type is correctly extracted from the model name

## Solution

Patched `quamba/eval_utils/lm_eval_wrapper.py` to recognize `mamba2_ptq` as a variant of `mamba2`, following the same pattern as `gla_ptq` (which is treated as a variant of `gla`).

### Changes Made

**File:** `quamba/eval_utils/lm_eval_wrapper.py`

#### Change 1: `eval_mamba_few_shot()` function (Line 257)
**Before:**
```python
if model_type == "mamba" or model_type == "mamba2" or model_type == "quamba" or model_type == "quamba2":
    lm_obj = MambaEvalWrapper(model=model, tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
```

**After:**
```python
if model_type == "mamba" or model_type == "mamba2" or model_type == "quamba" or model_type == "quamba2" or model_type == "mamba2_ptq":
    lm_obj = MambaEvalWrapper(model=model, tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
```

#### Change 2: Error message (Line 264)
**Before:**
```python
raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', 'quamba' and 'quamba2'")
```

**After:**
```python
raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', 'quamba', 'quamba2', 'mamba2_ptq', 'gla_ptq'")
```

#### Change 3: `eval_mamba_generation()` function (Line 295)
**Before:**
```python
if model_type == "mamba" or model_type == "mamba2" or model_type == "quamba" or model_type == "quamba2":
    lm_obj = MambaEvalWrapper(model=model, tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
```

**After:**
```python
if model_type == "mamba" or model_type == "mamba2" or model_type == "quamba" or model_type == "quamba2" or model_type == "mamba2_ptq":
    lm_obj = MambaEvalWrapper(model=model, tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
```

#### Change 4: Error message (Line 302)
**Before:**
```python
raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', 'quamba' and 'quamba2'")
```

**After:**
```python
raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', 'quamba', 'quamba2', 'mamba2_ptq', 'gla_ptq'")
```

## Model Type Recognition Pattern

The system uses a consistent pattern for quantized models:

| Base Model Type | PTQ Variant      | Eval Wrapper Used   |
|----------------|------------------|---------------------|
| `gla`          | `gla_ptq`        | GLAEvalWrapper      |
| `mamba2`       | `mamba2_ptq`     | MambaEvalWrapper    |
| `mamba`        | (not yet added)  | MambaEvalWrapper    |

### How Model Type is Extracted

In `main.py` (Line 16):
```python
model_type = model_name.split('-')[0]
```

**Examples:**
- `mamba2_ptq-w8a16-1.3b` → `mamba2_ptq`
- `gla_ptq-w8a16-1.3b` → `gla_ptq`
- `mamba2-1.3b` → `mamba2`

## Usage

### Evaluate Mamba2 PTQ Models
```bash
# W8A16 quantization
python main.py --model mamba2_ptq-w8a16-1.3b \
    --pretrained_dir pretrained_models/state-spaces \
    --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,winogrande \
    --eval_zero_shot

# W4A16 quantization
python main.py --model mamba2_ptq-w4a16-1.3b \
    --pretrained_dir pretrained_models/state-spaces \
    --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,winogrande \
    --eval_zero_shot
```

### Evaluate GLA PTQ Models
```bash
# W8A8 quantization
python main.py --model gla_ptq-w8a8-1.3b \
    --pretrained_dir pretrained_models/fla-hub \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot

# W8A16 quantization
python main.py --model gla_ptq-w8a16-1.3b \
    --pretrained_dir pretrained_models/fla-hub \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot

# W4A16 quantization
python main.py --model gla_ptq-w4a16-1.3b \
    --pretrained_dir pretrained_models/fla-hub \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot
```

## Complete Workflow

### 1. Quantize Models
```bash
cd torchao_baseline
python USAGE_EXAMPLE.py
```

This creates 5 quantized models:
- `pretrained_models/fla-hub/gla_ptq-w8a8-1.3b/`
- `pretrained_models/fla-hub/gla_ptq-w8a16-1.3b/`
- `pretrained_models/fla-hub/gla_ptq-w4a16-1.3b/`
- `pretrained_models/state-spaces/mamba2_ptq-w8a16-1.3b/`
- `pretrained_models/state-spaces/mamba2_ptq-w4a16-1.3b/`

### 2. Evaluate Models
```bash
cd ..  # Back to root directory

# Evaluate all models
for model in gla_ptq-w8a8-1.3b gla_ptq-w8a16-1.3b gla_ptq-w4a16-1.3b; do
    python main.py --model $model --pretrained_dir pretrained_models/fla-hub \
        --task_list lambada_openai,hellaswag --eval_zero_shot
done

for model in mamba2_ptq-w8a16-1.3b mamba2_ptq-w4a16-1.3b; do
    python main.py --model $model --pretrained_dir pretrained_models/state-spaces \
        --task_list lambada_openai,hellaswag --eval_zero_shot
done
```

## System Architecture

```
main.py
  ├─> Extract model_type from model name
  ├─> build_mamba_and_tokenizer(model_type)
  │    ├─> if model_type == "mamba2_ptq":
  │    │    ├─> Check quantization_config.json
  │    │    ├─> if weight_only: load_weight_only_quantized_model()
  │    │    └─> else: standard MambaLMHeadModel.from_pretrained()
  │    │
  │    └─> if model_type == "gla_ptq":
  │         ├─> Check quantization_config.json
  │         ├─> if weight_only: load_weight_only_quantized_model()
  │         └─> else: standard AutoModelForCausalLM.from_pretrained()
  │
  └─> eval_mamba_few_shot(model, tokenizer, model_type)
       ├─> if model_type in ["mamba", "mamba2", "quamba", "quamba2", "mamba2_ptq"]:
       │    └─> Use MambaEvalWrapper
       │
       └─> if model_type in ["gla", "gla_ptq"]:
            └─> Use GLAEvalWrapper
```

## User's Group Size Change

The user changed the group size for Mamba2 models from 128 to 64:

**In USAGE_EXAMPLE.py:**
```python
# Lines 561-562
model4 = convert_static_w8a16(model4, group_size=64)  # Changed from 128
model5 = convert_static_w4a16(model5, group_size=64)  # Changed from 128

# Lines 632, 650 - Metadata also updated
metadata={'group_size': 64}  # Changed from 128
```

**Impact:**
- Smaller group size (64 vs 128) means more granular quantization
- May improve accuracy at the cost of slightly larger model size
- The `load_weight_only_quantized_model()` will correctly read group_size=64 from the saved config

## Files Modified

1. **quamba/eval_utils/lm_eval_wrapper.py**
   - Added `mamba2_ptq` support in two functions
   - Updated error messages

2. **torchao_baseline/USAGE_EXAMPLE.py** (by user)
   - Changed group_size from 128 to 64 for Mamba2 models

## Verification

✅ All model types now supported:
- `mamba`, `mamba2` - Base models
- `quamba`, `quamba2` - GPTQ quantized models
- `gla_ptq` - PTQ quantized GLA models (W8A8, W8A16, W4A16)
- `mamba2_ptq` - PTQ quantized Mamba2 models (W8A16, W4A16)

✅ All evaluation modes work:
- Zero-shot: `--eval_zero_shot`
- Few-shot: `--eval_few_shot --fewshot N`
- Generation: `--eval_generation`

## Testing

Run a quick test to verify everything works:
```bash
# Test Mamba2 PTQ loading and evaluation
python main.py --model mamba2_ptq-w8a16-1.3b \
    --pretrained_dir pretrained_models/state-spaces \
    --task_list lambada_openai \
    --eval_zero_shot \
    --testing  # Use smaller sample size for quick test
```

Expected output:
- Model loads successfully via `load_weight_only_quantized_model()`
- Evaluation runs without errors
- Results are printed for lambada_openai task

## Related Documentation

- **SAVE_LOAD_GUIDE.md** - How to save/load quantized models
- **WEIGHT_ONLY_LOADING_IMPLEMENTATION.md** - Technical details of weight-only loading
- **MAMBA2_PTQ_SUPPORT.md** - Mamba2 PTQ implementation details


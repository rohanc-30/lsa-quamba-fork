# Weight-Only Quantization Loading Implementation

## Summary

Implemented custom save/load mechanism for weight-only quantized models (W8A16, W4A16) that works with the LM evaluation pipeline.

## Problem

Weight-only quantization with TorchAO creates `AffineQuantizedTensor` weights that cannot be loaded via standard `transformers.AutoModelForCausalLM.from_pretrained()` due to metadata mismatch during the `copy_()` operation.

## Solution

### 1. Modified `save_quantized_model()` 
**File:** `torchao_baseline/utils_torchao.py`

Added:
- New parameter: `base_model_path` (required for weight-only models)
- Auto-detection of weight-only quantization modes
- Saves additional metadata in `quantization_config.json`:
  - `weight_only`: true
  - `requires_special_loading`: true
  - `base_model_path`: path to unquantized model
  - `group_size`: for reconstruction

### 2. Created `load_weight_only_quantized_model()`
**File:** `torchao_baseline/utils_torchao.py`

New function that:
1. Reads `quantization_config.json` to determine quantization parameters
2. Loads the base (unquantized) model from `base_model_path`
3. Applies quantization to create `AffineQuantizedTensor` structure
4. Loads the quantized weights via `load_state_dict()`

### 3. Updated `build_mamba_and_tokenizer()`
**File:** `utils.py`

Modified `gla_ptq` model loading to:
1. Check for `quantization_config.json`
2. Detect weight-only quantization via `requires_special_loading` flag
3. Use `load_weight_only_quantized_model()` for weight-only models
4. Use standard `AutoModelForCausalLM.from_pretrained()` for W8A8 models

### 4. Updated `USAGE_EXAMPLE.py`
**File:** `torchao_baseline/USAGE_EXAMPLE.py`

Added:
- `base_model_path` parameter to weight-only save calls
- `group_size` in metadata
- Usage instructions printed after quantization

## Files Changed

1. **torchao_baseline/utils_torchao.py**
   - Modified `save_quantized_model()` signature and implementation
   - Added `load_weight_only_quantized_model()` function

2. **utils.py**
   - Updated `build_mamba_and_tokenizer()` for `gla_ptq` model type
   - Added auto-detection of weight-only quantization

3. **torchao_baseline/USAGE_EXAMPLE.py**
   - Updated save calls for W8A16 and W4A16 models
   - Added usage instructions

4. **torchao_baseline/SAVE_LOAD_GUIDE.md** (NEW)
   - Comprehensive documentation on save/load mechanisms

5. **torchao_baseline/WEIGHT_ONLY_LOADING_IMPLEMENTATION.md** (NEW)
   - Technical implementation details

## Usage

### Quantizing and Saving
```bash
cd torchao_baseline
python USAGE_EXAMPLE.py
```

This will create:
- `pretrained_models/fla-hub/gla_ptq-w8a8-1.3b/`
- `pretrained_models/fla-hub/gla_ptq-w8a16-1.3b/`
- `pretrained_models/fla-hub/gla_ptq-w4a16-1.3b/`

### Evaluating with LM Eval
```bash
# W8A8 model (standard loading)
python main.py --model gla_ptq-w8a8-1.3b \
    --pretrained_dir pretrained_models/fla-hub \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot

# W8A16 model (automatic weight-only loading)
python main.py --model gla_ptq-w8a16-1.3b \
    --pretrained_dir pretrained_models/fla-hub \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot

# W4A16 model (automatic weight-only loading)
python main.py --model gla_ptq-w4a16-1.3b \
    --pretrained_dir pretrained_models/fla-hub \
    --task_list lambada_openai,hellaswag \
    --eval_zero_shot
```

The system automatically detects the quantization type and uses the appropriate loader!

## Technical Details

### Quantization Config Structure
```json
{
  "model_type": "gla",
  "model_name": "gla_ptq-w8a16-1.3b",
  "quantization": {
    "method": "torchao_w8a16_static",
    "weight_bits": 8,
    "activation_bits": 16,
    "weight_dtype": "int8",
    "activation_dtype": "float16",
    "weight_only": true,
    "requires_special_loading": true,
    "base_model_path": "pretrained_models/fla-hub/gla-1.3B-100B",
    "group_size": 128
  }
}
```

### Loading Flow

```
main.py 
  └─> build_mamba_and_tokenizer(model_type="gla_ptq")
       └─> Check quantization_config.json
            ├─> if requires_special_loading == True:
            │    └─> load_weight_only_quantized_model()
            │         ├─> Load base model
            │         ├─> Apply quantization (create AffineQuantizedTensor structure)
            │         └─> Load quantized state_dict
            │
            └─> else:
                 └─> AutoModelForCausalLM.from_pretrained()
```

### Why This Works

1. **Base Model Loading**: Creates standard `nn.Linear` modules with `Parameter` weights
2. **Quantization Application**: Replaces `Parameter` with `AffineQuantizedTensor` structure
3. **State Dict Loading**: `load_state_dict()` can load `AffineQuantizedTensor` into `AffineQuantizedTensor`

The key insight: We recreate the exact tensor structure before loading, avoiding the metadata mismatch.

## Testing

1. **Save Test**: Run `USAGE_EXAMPLE.py` to save all three quantization modes
2. **Load Test**: Use `main.py` with each model to verify loading works
3. **Evaluation Test**: Run actual LM eval tasks to ensure model functions correctly

## Limitations

1. **Base Model Dependency**: Weight-only models require the base model to be available at the saved path
2. **Group Size Consistency**: Must use the same group size for quantization and loading
3. **Model Type Support**: Currently only supports GLA models (easily extensible)

## Future Improvements

1. **Embed Base Model Config**: Store the base model's config.json to enable reconstruction without the base model
2. **Support More Model Types**: Add support for Mamba, Mamba2, DeltaNet
3. **Group Size Auto-Detection**: Infer group size from saved weights
4. **Validation**: Add checksums or validation to ensure correct loading

## References

- TorchAO Documentation: https://github.com/pytorch/ao
- Issue: `AffineQuantizedTensor` incompatibility with `transformers` loading
- Solution: Custom loader that recreates tensor structure before loading


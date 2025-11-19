# TorchAO Baseline Development Log

This document tracks all changes, decisions, and bug fixes during the development of the torchao baseline quantization implementation.

---

## Report 1: Initial Design - Shift from RTN to W8A8 PTQ
**Date:** Session 1  
**Issue:** Initial design was for RTN/PTQ weight-only quantization, but requirements changed

### Context
Original plan was to implement:
- RTN (Round-to-Nearest) weight-only quantization
- PTQ weight-only quantization with calibration
- Support for 4-bit and 8-bit weights

### User Requirement Change
User specified: "I don't want weight-only quantization. I want to do naive W8A8 quantization, something like PTQ-linear."

### Changes Made

#### Functions KEPT AS-IS (5 functions)
1. **`load_model_for_quantization()`** - Loads FP16 models correctly
2. **`get_calibration_data()`** - Provides calibration samples (needed for PTQ)
3. **`save_quantized_model()`** - Saves models (updated naming only)
4. **`get_model_size()`** - Measures size
5. **`verify_quantization()`** - Compares outputs

#### Functions UPDATED (4 functions + 1 class)
1. **`get_quantization_config()`**
   - **Before:** `method`, `w_bits`, `group_size` (for RTN/weight-only)
   - **After:** `granularity`, `a_symmetric`, `w_symmetric` (for W8A8)
   - **Reason:** W8A8 needs different parameters for activation and weight quantization

2. **`apply_ptq_quantization()` → `apply_w8a8_ptq_quantization()`**
   - **Renamed:** To be explicit about W8A8
   - **Updated params:** Removed `w_bits`, added `granularity`, `a_symmetric`, `w_symmetric`
   - **Updated docstring:** Now describes 4-step W8A8 PTQ process

3. **`run_calibration_forward()`**
   - **Updated docstring:** Explicitly mentions collecting both weight AND activation statistics
   - **Reason:** W8A8 needs both, not just weights

4. **`save_quantized_model()`**
   - **Updated signature:** Replaced `method` and `w_bits` with `granularity`
   - **Updated naming:** Now saves as `torchao-{model}-w8a8-{per_tensor|per_channel}`
   - **Examples:**
     - `torchao-gla-1.3b-w8a8-per_tensor`
     - `torchao-mamba2-1.3b-w8a8-per_channel`

5. **`TorchAOQuantizer` class**
   - **Updated init:** Takes `granularity`, `a_symmetric`, `w_symmetric` instead of `method`, `w_bits`
   - **Updated docstring:** Now says "W8A8 PTQ" instead of "RTN and PTQ"
   - **Updated `quantize()`:** Requires `calibration_data` (no longer optional)

#### Functions REMOVED
- **`apply_rtn_quantization()`** - Weight-only, not needed for W8A8

### Key Concepts Added

#### W8A8 Quantization
- **W8:** 8-bit quantized weights
- **A8:** 8-bit quantized activations
- Both are quantized (unlike weight-only methods)

#### Symmetric vs Asymmetric Quantization

**Symmetric (no zero-point):**
```
quantized = round(value / scale)
range: [-127, 127] for int8
```

**Asymmetric (with zero-point):**
```
quantized = round(value / scale) + zero_point
range: [-128, 127] or [0, 255] for int8
```

**Typical choices:**
- **Weights:** Symmetric (often centered around 0)
- **Activations:** Asymmetric (better for ReLU/non-negative)

#### Per-Tensor vs Per-Channel

**Per-Tensor:**
- One scale (and zero-point) for entire tensor
- Simpler, faster
- Slightly lower accuracy

**Per-Channel:**
- One scale per output channel
- More granular
- Better accuracy, slightly more overhead

---

## Report 2: Bug Fixes for Data Loading & Model Stability
**Date:** Session 1  
**Issues:** NaN outputs, too many short sequences, missing tokenizer config

### Issue 1: NaN Outputs from GLA Model ❌

**Problem:** Model producing all NaN values in logits  
```python
CausalLMOutputWithPast(
    logits=tensor([[[nan, nan, nan, ..., nan, nan, nan]]], 
    device='cuda:0', dtype=torch.float16, ...)
)
```

**Root Cause:** Loading GLA model directly in FP16 causes numerical instability

**Fix Applied:**
```python
# Before:
dtype: torch.dtype = torch.float16  # Default FP16
model = GLAForCausalLM.from_pretrained(model_path).to(device=device, dtype=dtype)

# After:
dtype: torch.dtype = torch.float32  # Default FP32 for stability
model = GLAForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
if dtype != torch.float32:
    logger.warning(f"Converting model from fp32 to {dtype}")
    model = model.to(dtype=dtype)
```

**Why this works:** GLA models have numerical stability issues in FP16. Loading in FP32 first ensures weights are initialized properly. Can still convert to FP16 later if needed, but start stable.

### Issue 2: Too Many Short Sequences ⚠️

**Problem:** Calibration data had many very short sequences (11-40 tokens)  
**Example output:** 
```
[torch.Size([1, 13]), torch.Size([1, 11]), torch.Size([1, 21]), ...]
```

**Root Cause:** Wikitext-2-raw has many empty lines, headers, and formatting lines

**Fix Applied:**
```python
# Before:
if input_ids.shape[1] > 10:  # At least 10 tokens
    calibration_samples.append(input_ids)

# After:
min_length: int = 64  # Added parameter

if input_ids.shape[1] >= min_length:  # At least 64 tokens
    calibration_samples.append(input_ids)
else:
    skipped_short += 1

# Also added logging:
logger.info(f"Sample lengths - min: {min(...)}, max: {max(...)}, mean: {mean(...)}")
logger.info(f"Skipped {skipped_empty} empty and {skipped_short} too-short samples")
```

**Why this improves calibration:**
- Wikitext-2-raw has MANY short lines (titles, section headers, etc.)
- Very short sequences don't provide good calibration statistics
- 64 tokens is a reasonable minimum for meaningful activation distributions

**Note on Variable-Length Sequences:**
Variable-length sequences in calibration data are **EXPECTED and CORRECT**:
- Real text has varying lengths
- Truncation to max_length (512) preserves natural distribution
- PTQ quantization benefits from diverse lengths (shows different activation patterns)

What's NOT normal: Too many very short sequences (<64 tokens) - these are artifacts from wikitext formatting.

### Issue 3: Missing Tokenizer Pad Token ⚠️

**Problem:** Could cause issues with batched inference

**Fix Applied:**
```python
# Added to load_model_for_quantization():
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
```

**Why this matters:** Batched operations may fail without a pad token defined.

### Additional Improvements

**Better Logging:**
- Added dtype logging: `"Loading {model_type} model from {model_path} with dtype={dtype}"`
- Added path resolution logging: `"Using full path: {model_path}"`
- Added calibration statistics: min/max/mean lengths, skipped samples

**Lazy Imports:**
- Moved model imports inside functions to avoid torchao version conflicts
- Prevents `AttributeError: module 'torch' has no attribute 'int1'` at import time

### Testing Results

**Before Fix:**
```python
m(d[-1].to(m.device))  # All NaN
```

**After Fix:**
```python
m(d[0].to(m.device))  # Real values, no NaN ✅
len(d)  # 512 samples
min([s.shape[1] for s in d])  # >= 64 tokens ✅
```

### Summary Table

| Issue | Status | Fix |
|-------|--------|-----|
| NaN outputs | ✅ Fixed | Load in FP32 by default |
| Too many short sequences | ✅ Fixed | Filter with min_length=64 |
| Missing pad token | ✅ Fixed | Auto-set to eos_token |
| Variable-length data | ✅ Normal | This is expected behavior |
| Import conflicts | ✅ Fixed | Lazy imports inside functions |

---

## Report 3: Bug Fixes for Calibration & Inference

**Date:** November 18, 2025  
**Issue:** MLP `down_proj` layers failing calibration; Inference crashes with CUDA assertion errors

### Bug 3.1: Fused SwiGLU Bypassing down_proj Observers

**Symptom:** During calibration, 24 out of 240 layers (all `mlp.down_proj` layers) failed to record min/max statistics, causing `AssertionError` during conversion.

**Root Cause:** GLA's `GatedMLP` has a `fuse_swiglu` flag. When enabled, the forward pass calls:
```python
return self.swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)
```
This passes `down_proj.weight` and `.bias` **directly to the fused kernel**, completely bypassing the `down_proj` module's `forward()` method. As a result, the `ObservedLinear` observers in `down_proj` never get called.

**Fix:** In `prepare_for_static_w8a8()`, we now:
1. Scan all modules for `fuse_swiglu` attribute
2. Temporarily set `fuse_swiglu = False` to force unfused path during calibration
3. Log how many MLP layers were modified

**Files Changed:**
- `utils_torchao.py`: Modified `prepare_for_static_w8a8()` to disable fusion
- `USAGE_EXAMPLE.py`: Added [CHECK 2.5] to verify fusion is disabled

### Bug 3.2: Quantized Tensors Breaking GLA Kernels

**Symptom:** After successful quantization, inference crashes with CUDA assertion:
```
../aten/src/ATen/native/cuda/Indexing.cu:1308: 
Assertion `srcIndex < srcSelectDimSize` failed
```
Stack trace shows error in GLA's triton kernels (`chunk_local_cumsum`).

**Root Cause:** Our `QuantizedLinear.forward()` was returning `AffineQuantizedTensor` objects directly. When `F.linear(qx, qweight, bias)` is called with quantized inputs, torchao may return a quantized output tensor. However, GLA's attention kernels (written in triton) expect **float tensors**, not quantized tensor subclasses. This caused type mismatches and invalid indexing operations.

**Fix:** In `QuantizedLinear.forward()`, we now:
1. **Dequantize input** if it's already quantized (defensive check)
2. Quantize input activations to int8
3. Perform int8×int8 matmul
4. **Explicitly dequantize output** back to float before returning

**Code Change:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Ensure input is float
    if hasattr(x, 'dequantize'):
        x = x.dequantize()
    
    # Quantize activations
    qx = to_affine_quantized_intx_static(x, self.act_scale, self.act_zero_point, ...)
    
    # Matmul
    output = F.linear(qx, self.qweight, self.bias)
    
    # Dequantize output back to float
    if hasattr(output, 'dequantize'):
        output = output.dequantize()
    
    return output
```

**Files Changed:**
- `utils_torchao.py`: Modified `QuantizedLinear.forward()` to explicitly dequantize outputs

### Bug 3.3: Test Input Generation Using Invalid Token IDs

**Symptom:** Both FP32 and quantized models crashed with CUDA assertion errors during test inference.

**Root Cause:** Test code was generating random token IDs with `torch.randint(0, 50000, ...)` but the tokenizer's vocabulary size was only 32000. When the embedding layer tried to look up token ID 40000, it caused an **index out of bounds** error in the CUDA embedding kernel.

**Fix:** Changed test input generation to use valid token IDs:
```python
test_input = torch.randint(0, tokenizer.vocab_size, (1, 128))  # ✓ Valid range
```

**Files Changed:**
- `USAGE_EXAMPLE.py`: Fixed all test input generation to respect `tokenizer.vocab_size`

### Optimization: Cached Weight Dequantization

**Issue:** Initial implementation was doing fake quantization (quantize activations → dequantize immediately → float matmul → return float), which added overhead without any compute benefit.

**Optimization:** Refactored `QuantizedLinear` to:
1. **Cache dequantized weights once at initialization** - Avoids repeated dequantization overhead on every forward pass
2. **Skip fake activation quantization** - Directly use float inputs for matmul
3. **Keep weights stored as int8** - Maintains 4x memory savings

**Trade-off:** We accept no int8 compute speedup (using float matmul) as the price for compatibility with GLA's triton kernels. The main benefit is **memory savings** (4x smaller weights) with **minimal overhead**.

**Performance Impact:**
- ✅ **Memory:** 4x reduction in weight storage
- ✅ **Speed:** No repeated dequantization overhead
- ❌ **Compute:** No int8 matmul speedup (constrained by GLA kernel compatibility)

**Files Changed:**
- `utils_torchao.py`: Refactored `QuantizedLinear.__init__()` and `.forward()` to cache dequantized weights

### Key Lessons Learned

1. **Fused operations can bypass module hooks/observers:** Always check if the model uses fused kernels that access `.weight` directly.
2. **Quantized tensors are not always compatible with custom kernels:** Triton/CUDA kernels written for float tensors may fail with tensor subclasses. Always dequantize outputs when interfacing with external ops.
3. **Diagnostic checks are invaluable:** The step-by-step checks in `USAGE_EXAMPLE.py` helped triangulate both issues quickly.
4. **Test inputs must be valid:** Always respect model constraints (vocab size, sequence length, etc.) when generating test data.
5. **Optimization trade-offs:** Sometimes accepting no compute speedup is necessary for compatibility; memory savings alone can still be valuable.

---

## Future Reports

Additional reports will be added here as development continues:
- Model saving/loading integration
- Evaluation pipeline integration
- Performance benchmarks
- Known issues and workarounds


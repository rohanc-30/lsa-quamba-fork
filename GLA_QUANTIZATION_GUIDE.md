# GLA Quantization Implementation Guide

## Summary

I've added initial support for quantizing GLA (Gated Linear Attention) models to the Quamba codebase. The implementation follows the same patterns as Mamba/Mamba2 quantization but accounts for GLA's different architecture.

## Files Created/Modified

### 1. **NEW: `quamba/qGLALayer.py`**
Contains quantized GLA layer implementations:
- `GLASimple` - Simplified GLA for calibration (wraps projections with HadLinear)
- `W4A16QGLA` - 4-bit weight, 16-bit activation quantized GLA
- `W4A8QGLA` - 4-bit weight, 8-bit activation quantized GLA  
- `W8A8QGLA` - 8-bit weight, 8-bit activation quantized GLA
- `GLAMLPSimple` - Placeholder for MLP quantization (not yet implemented)

### 2. **MODIFIED: `quamba/modelutils_mamba.py`**
Updated to support GLA throughout the quantization pipeline:

#### Changes made:
1. **Imports** - Added GLA layer imports
2. **`configure_model()`** - Added GLA case that:
   - Handles `model.model` structure instead of `model.backbone`
   - Fuses `attn_norm` to `q_proj`
   - Replaces attention with `GLASimple`
3. **`run_quamba_calibration()`** - Added GLA calibration logic that:
   - Uses `model.model.layers` instead of `model.backbone.layers`
   - Monitors `q_proj` and `k_proj` as key operations
   - Skips SSM state handling (GLA doesn't have SSM)
4. **`quantize_mixer_*()` functions** - Added `"GLA"` to all mixer dictionaries
5. **`quantize_fp16_model()`** - Added GLA case that:
   - Quantizes `model.model.embeddings` and `model.model.norm`
   - Handles `attn_norm` + `attn` quantization
   - Leaves MLP in FP16 for now
6. **LM head quantization** - Made it handle both `backbone.norm_f` (Mamba) and `model.norm` (GLA)

## Architecture Differences: GLA vs Mamba

| Component | Mamba | GLA |
|-----------|-------|-----|
| Root | `model.backbone` | `model.model` |
| Embedding | `backbone.embedding` | `model.embeddings` |
| Layers | `backbone.layers` | `model.layers` |
| Layer Type | `Block` with `mixer` | `GLABlock` with `attn` + `mlp` |
| Attention Projections | `in_proj`, `out_proj` | `q_proj`, `k_proj`, `v_proj`, `g_proj`, `gk_proj`, `o_proj` |
| Layer Norms | 1 per layer (`norm`) | 2 per layer (`attn_norm`, `mlp_norm`) |
| Final Norm | `backbone.norm_f` | `model.norm` |
| Has SSM State | Yes | No |
| Has MLP | No | Yes (GatedMLP per layer) |

## What's Working (In Theory)

✅ Model configuration with Hadamard transforms  
✅ Layer norm fusion to first projection  
✅ Calibration hook registration for all GLA projections  
✅ Quantized layer class structure (W4A16, W4A8, W8A8)  
✅ Integration into quantization pipeline  
✅ Embedding and LM head quantization  

## What Still Needs Implementation

### **CRITICAL: Forward Pass Implementation**

The quantized GLA classes currently raise `NotImplementedError` in their forward methods. You need to:

1. **Understand GLA's attention mechanism** from the `fla` package:
   ```python
   # Look at the original implementation
   from fla.modules import GatedLinearAttention
   ```

2. **Implement forward pass in `qGLALayer.py`** for each quantized class:
   - Handle Q, K, V, G projections with quantized linear layers
   - Implement the gated linear attention computation
   - Handle the `gk_proj` sequential bottleneck
   - Use `g_norm_swish_gate` correctly
   - Ensure output shapes match expectations

3. **Test forward pass** with a small input to verify:
   ```python
   # Pseudo-test
   x = torch.randn(1, 512, 2048).cuda()
   output = quantized_gla_attn(x)
   assert output.shape == x.shape
   ```

### **Optional Enhancements**

❌ **MLP Quantization** - Currently MLP blocks stay in FP16  
❌ **GPTQ Support** - `apply_gptq()` only works with Mamba structure  
❌ **Reordering** - GLA doesn't have head reordering like Mamba2  
❌ **Conv1D Handling** - If GLA uses convolutions, may need special handling  
❌ **g_norm_swish_gate Quantization** - Currently kept as FP16  

## Testing Steps

### Step 1: Test Configuration
```bash
python -c "
from fla.models import GLAForCausalLM
from quamba.modelutils_mamba import configure_model
import torch

model = GLAForCausalLM.from_pretrained('your-gla-model').cuda()
model = configure_model(model, model_type='gla', use_had_transform=True)
print('Configuration successful!')
"
```

### Step 2: Test Calibration
```bash
python -c "
from fla.models import GLAForCausalLM
from transformers import AutoTokenizer
from quamba.modelutils_mamba import configure_model, run_quamba_calibration

model = GLAForCausalLM.from_pretrained('your-gla-model').cuda()
tokenizer = AutoTokenizer.from_pretrained('fla-hub/gla-1.3B-100B')
model = configure_model(model, model_type='gla')
act_scales = run_quamba_calibration(model, 'gla', tokenizer, num_samples=16)
print(f'Calibration successful! Got {len(act_scales)} layer scales')
"
```

### Step 3: Test Quantization (will fail until forward pass implemented)
```bash
python -c "
# ... setup code ...
quantized_model = quantize_fp16_model(
    model, 'gla', act_scales, device='cuda',
    w_bits=4, a_bits=16
)
print('Quantization successful!')
"
```

### Step 4: Test Inference (will fail until forward pass implemented)
```bash
# Try generating text with quantized model
```

## Next Steps for You

1. **Run in your compute node + venv:**
   ```bash
   cd /home/rcherukuri/lsa-quamba-fork
   # Activate your venv
   python -c "from quamba.qGLALayer import GLASimple; print('Import successful!')"
   ```

2. **Examine the original GLA forward pass:**
   ```python
   from fla.modules import GatedLinearAttention
   import inspect
   print(inspect.getsource(GatedLinearAttention.forward))
   ```

3. **Share the original forward pass code with me** - I'll help implement the quantized versions

4. **Test configuration and calibration** (should work as-is)

5. **Implement forward passes** in the quantized classes

6. **Test end-to-end** quantization

## Key Design Decisions

1. **Fuse attn_norm to q_proj** - Since q_proj is the first linear layer in attention
2. **Skip MLP quantization initially** - Focus on attention first, MLP can be added later
3. **Use same observer pattern** - PercentileObserver for q/k projections, MinMaxObserver for others
4. **Keep g_norm_swish_gate in FP16** - Complex fused operation, quantize later if needed
5. **Handle gk_proj Sequential specially** - It's a 2-layer bottleneck, quantize both layers

## Questions to Answer

Before implementing forward passes, you might need to clarify:
- Does GLA use causal masking? How is it applied?
- What's the exact computation in gated linear attention?
- Are there any custom CUDA kernels we need to handle?
- Should we quantize the MLP blocks as well, or is attention-only sufficient?

## File Locations Reference

```
quamba/
├── qGLALayer.py          # NEW: Quantized GLA implementations
├── modelutils_mamba.py   # MODIFIED: Added GLA support throughout
├── qLinearLayer.py       # Used by GLA (W4A16B16O16Linear, etc.)
├── qNorm.py              # Used for attn_norm quantization
├── qEmbedding.py         # Used for embedding quantization
└── hadamard_utils.py     # Used for Hadamard transforms
```

## Contact/Debugging

If you hit issues:
1. Check which function is failing
2. Look at the error traceback
3. Share the specific error + relevant code section
4. I can help debug or add missing pieces

Good luck! The infrastructure is there, just needs the forward pass implementation.


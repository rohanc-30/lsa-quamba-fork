# GPTQ Implementation Trace: Why Mamba2 Works But GLA Doesn't

## Key Finding

**Mamba2 GPTQ (Quamba) performs GREAT, but GLA GPTQ performs TERRIBLY** despite both having:
- Similar quantization (W4A16 claimed)
- Similar scale "issues" (low ratios)
- Proper GPTQ format (INT32 weights + FP16 scales)

## Implementation Differences

### 1. **What Gets Quantized**

**GLA GPTQ (ut-enyac/gla-1.3b-w4a16):**
```
✓ Quantized: q_proj, k_proj, v_proj, o_proj (attention only)
✗ NOT Quantized: gate_proj, up_proj, down_proj (MLP layers)
```

**Mamba2 GPTQ (ut-enyac/quamba2-1.3b-w4a16):**
```
✓ Quantized: in_proj, out_proj (all main projections)
```

### 2. **GPTQ Application Code**

**File: `quamba/modelutils_mamba.py` lines 510-668**

```python
def apply_gptq(model, tokenizer, device, w_bits=4, model_type="mamba"):
    # Calibration setup
    nsamples = 128
    seqlen = 1024
    
    if model_type == "mamba2":
        # Quantize in_proj and out_proj
        for i in range(num_layers):
            gptq = {}
            gptq["in_proj"] = GPTQ(layer.mixer.in_proj)
            gptq["out_proj"] = GPTQ(layer.mixer.out_proj)
            
            # Run GPTQ optimization
            for name in gptq.keys():
                gptq[name].fasterquant(
                    percdamp=0.01, 
                    group_size=128, 
                    w_bits=bits
                )
    
    elif model_type == "gla":
        # Quantize attention projections only
        # (MLP layers skipped - code likely in qGLALayer.py)
```

### 3. **Quantized Layer Classes**

**Mamba2 uses: `W4A16QMamba2` (quamba/qMamba2.py)**
- Class for W4A16 (4-bit weights, FP16 activations)
- Line 381-532
- Uses custom Linear layers with GPTQ weights

**GLA uses: Custom attention layers (quamba/qGLALayer.py)**  
- Need to check this file for GLA-specific GPTQ implementation
- Likely only attention is implemented

## Code Flow Comparison

### Mamba2 GPTQ Loading:
```
utils.py (line 44)
  → QuambaLMHeadModel.from_pretrained()
    → Loads pytorch_model.bin with GPTQ weights
    → Uses W4A16QMamba2 layers
      → in_proj: W4A16Linear (GPTQ matmul)
      → out_proj: W4A16Linear (GPTQ matmul)
```

### GLA GPTQ Loading:
```
utils.py (line 52 or AutoModelForCausalLM)
  → GLAForCausalLM.from_pretrained()
    → Loads model.safetensors with GPTQ weights
    → Uses GPTQ attention layers
      → q/k/v/o_proj: GPTQ quantized
      → MLP: FULL PRECISION (not quantized!)
```

## Hypothesis: Why Mamba2 Works But GLA Doesn't

### Mamba2 Success Factors:
1. **Simpler architecture**: Only in_proj → SSM → out_proj
2. **No MLP needed**: Mamba2's d_mlp=0
3. **Full quantization**: Both projections quantized consistently
4. **Robust to quantization**: SSM operations may be more tolerant

### GLA Failure Factors:
1. **Hybrid quantization**: Attention quantized, MLP not
2. **Attention is critical**: GLA relies heavily on gated linear attention
3. **Quantization mismatch**: FP32 MLP outputs → INT4 attention inputs (range mismatch!)
4. **SwiGLU nonlinearity**: May amplify quantization errors

## Files to Investigate

### For Mamba2 GPTQ:
1. **`quamba/qMamba2.py`** - Lines 381-532 (`W4A16QMamba2` class)
   - Check `forward()` method to see how GPTQ weights are used
   - Look for activation handling

2. **`quamba/modelutils_mamba.py`** - Lines 510-668 (`apply_gptq`)
   - See how in_proj/out_proj are quantized
   - Check calibration process

### For GLA GPTQ:
1. **`quamba/qGLALayer.py`** - Need to find GPTQ GLA implementation
   - Check if MLP quantization exists but isn't being used
   - Look for why only attention is quantized

2. **`quamba/modelutils_mamba.py`** - Lines 537-540 (GLA branch)
   - See how GLA layers are accessed during GPTQ

## Questions to Answer

1. **Does GLA GPTQ code even support MLP quantization?**
   - Search qGLALayer.py for MLP/gate_proj/down_proj GPTQ classes
   
2. **Is the partial quantization intentional or a bug?**
   - Check if there's a W4A16QGLA class that only quantizes attention
   
3. **How does Mamba2 handle activation ranges?**
   - Does it use activation clipping/normalization that GLA doesn't?
   - Check the forward() method in W4A16QMamba2

4. **Are there different calibration strategies?**
   - Mamba2: Uses 128 samples, seqlen=1024
   - GLA: Same parameters but maybe different execution path?

## Next Steps for Debugging

1. Read `quamba/qGLALayer.py` to find GLA GPTQ classes
2. Compare `W4A16QMamba2.forward()` vs GLA GPTQ forward()
3. Check if GLA has activation range issues that Mamba2 doesn't
4. Look for normalization/clipping differences

The answer is in these implementation files! 🎯


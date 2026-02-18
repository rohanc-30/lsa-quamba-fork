# NaN/Inf Analysis Summary

## Root Cause: Float16 Overflow

### The Problem
The error at line 379 is actually caused by **Inf values** (not NaN), which are detected by the check:
```python
if torch.isnan(o).any() or torch.isinf(o).any():
```

### Key Findings

1. **All inputs to the attention kernel are valid** (no NaN/Inf):
   - q: range [-30.9, 30.9] ✓
   - k: range [-112.1, 112.4] ⚠️ (large but not extreme)
   - v: range [-691.5, 688.5] ❌ **EXTREMELY LARGE**
   - gk: range [-18.4, 0.0] ✓

2. **Output has 1.29% Inf values** (27,089 out of 2,097,152 elements)
   - Valid values span the entire float16 range: [-65504, 65504]
   - This indicates float16 overflow

3. **V tensor is the culprit**:
   - V values reach ±691, which is abnormally large
   - When these values go through the recurrent GLA attention kernel, intermediate computations overflow float16's max value (65504)

### Why It Happens at 55% Mark

The issue is **data-dependent**:
- First 55% of data likely has smaller/better-conditioned V values
- At 55%, the eval dataset contains examples that produce extreme V values
- Once V gets large enough, the attention computation overflows
- This could be due to:
  1. Longer sequences (more accumulation)
  2. Specific input patterns that activate quantized weights differently
  3. Accumulated numerical error in the quantized weights

### The Float16 Overflow Mechanism

In the GLA attention kernel (chunk_gla or fused_recurrent_gla):
```
o = attention(q, k, v, gk)
```

The recurrent computation involves operations like:
- Multiplying K and V: k @ v.T can produce values up to 112 * 691 ≈ 77,000 (already exceeds float16 max!)
- Exponentiating gk values: e^(-18.4) to e^0 = [1e-8 to 1.0]
- Accumulating over sequence length: compounds the overflow

## Solution Strategies

### 1. **Use Float32 for Attention Computation** (Recommended)
Convert inputs to float32 before the attention kernel:
```python
if mode == 'fused_recurrent':
    o, _ = fused_recurrent_gla(
        q=q.float(), k=k.float(), v=v.float(), gk=gk.float(),
        initial_state=None, output_final_state=False
    )
    o = o.half()  # Convert back to fp16
```

### 2. **Scale Down V Values**
Before the attention computation:
```python
v_scale = v.abs().max() / 100.0  # Scale to reasonable range
v_scaled = v / v_scale
o, _ = fused_recurrent_gla(...)
o = o * v_scale  # Scale back up
```

### 3. **Investigate V Projection Quantization**
The v_proj layer might have incorrect quantization scales:
- Check the quantization calibration for v_proj
- The GPTQ quantization might be producing extreme outputs for certain inputs
- Consider using higher precision for v_proj (keep it in fp16/fp32 instead of w4)

### 4. **Gradient Clipping in V Projection**
Add output clipping after v_proj:
```python
v = self.v_proj(hidden_states.half())
v = torch.clamp(v, -100, 100)  # Prevent extreme values
```

## Recommended Fix

The most robust solution is **Option 1**: Use float32 for the attention kernel computation.
This maintains numerical stability while keeping memory usage reasonable (only during computation).

The code should be modified around line 354-358 in qGLALayer.py.


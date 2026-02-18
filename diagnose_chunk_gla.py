"""
Diagnose why chunk_gla produces values that overflow float16.

The issue: V values are only ±691 (reasonable), but chunk_gla output 
in float32 exceeds ±65504, causing overflow when cast to float16.

IMPLEMENTATION NOTE:
This script implements the GLA recurrence naively (token-by-token) to verify 
the exact output values. The actual chunk_gla kernel uses a chunk-wise parallel
algorithm for efficiency, but computes mathematically equivalent results.

The recurrence formula (from FLA's fused_recurrent kernel) is:
  h_t = exp(gk_t) ⊙ h_{t-1} + k_t ⊗ v_t
  o_t = sum(q_t ⊙ h_t, axis=K)

where ⊙ is element-wise multiplication, ⊗ is outer product.
"""

import torch
import numpy as np

print("="*80)
print("CHUNK_GLA OVERFLOW DIAGNOSIS")
print("="*80)

# Load tensors
q = torch.load("debug_tensors/q.pt", map_location='cpu').float()
k = torch.load("debug_tensors/k.pt", map_location='cpu').float()
v = torch.load("debug_tensors/v.pt", map_location='cpu').float()
gk = torch.load("debug_tensors/gk.pt", map_location='cpu').float()

print("\nInput Statistics (converted to float32):")
print(f"q:  range=[{q.min():.2f}, {q.max():.2f}], mean={q.mean():.2f}, std={q.std():.2f}")
print(f"k:  range=[{k.min():.2f}, {k.max():.2f}], mean={k.mean():.2f}, std={k.std():.2f}")
print(f"v:  range=[{v.min():.2f}, {v.max():.2f}], mean={v.mean():.2f}, std={v.std():.2f}")
print(f"gk: range=[{gk.min():.2f}, {gk.max():.2f}], mean={gk.mean():.2f}, std={gk.std():.2f}")

print("\n" + "="*80)
print("MANUAL CHUNK_GLA COMPUTATION (simplified)")
print("="*80)

# Simulate what happens in chunk_gla
# The core operation is essentially: o = sum over time of (exp(cumsum(g)) * k * v)

batch, seq_len, num_heads, head_dim = q.shape
_, _, _, v_dim = v.shape

print(f"\nShape: batch={batch}, seq_len={seq_len}, heads={num_heads}, k_dim={head_dim}, v_dim={v_dim}")

# GK values after logsigmoid are log probabilities (negative)
# When exponentiated, they become decay factors
print(f"\nGK Analysis (log-space gate values):")
print(f"  Min: {gk.min():.4f} -> exp = {torch.exp(gk.min()):.6e}")
print(f"  Max: {gk.max():.4f} -> exp = {torch.exp(gk.max()):.6e}")
print(f"  Mean: {gk.mean():.4f} -> exp = {torch.exp(gk.mean()):.6e}")

# The dangerous operation: cumulative sum of gk (cumulative log probabilities)
# This creates cumulative decay that can become very large or very small
gk_cumsum = torch.cumsum(gk, dim=1)  # Cumsum along sequence

print(f"\nCumulative GK (cumsum along sequence):")
print(f"  Min: {gk_cumsum.min():.4f} -> exp = {torch.exp(gk_cumsum.min().clamp(min=-100)):.6e}")
print(f"  Max: {gk_cumsum.max():.4f} -> exp = {torch.exp(gk_cumsum.max().clamp(max=100)):.6e}")

# Check for extreme cumsum values
extreme_neg = (gk_cumsum < -100).sum().item()
extreme_pos = (gk_cumsum > 10).sum().item()
print(f"  Values < -100: {extreme_neg} ({100*extreme_neg/gk_cumsum.numel():.2f}%)")
print(f"  Values > 10: {extreme_pos} ({100*extreme_pos/gk_cumsum.numel():.2f}%)")

print("\n" + "="*80)
print("ATTENTION SCORE ANALYSIS")
print("="*80)

# Compute attention-like scores: q @ k.T
# Shape: [batch, seq_len, heads, head_dim] @ [batch, seq_len, heads, head_dim].T
#      = [batch, heads, seq_len, seq_len]

# Take a single sample for analysis
q_sample = q[0]  # [seq_len, heads, head_dim]
k_sample = k[0]
v_sample = v[0]

# Compute QK^T for one head
qk = torch.einsum('shd,thd->sth', q_sample, k_sample)  # [seq_len, seq_len, heads]
print(f"\nQ @ K.T statistics:")
print(f"  Range: [{qk.min():.2f}, {qk.max():.2f}]")
print(f"  Mean: {qk.mean():.2f}, Std: {qk.std():.2f}")

# The problematic product: attention scores * v values
# In chunk_gla, this is roughly: attn_weights @ v
# Max possible: max(qk) * max(v) ≈ {qk.max() * v_sample.max()}

max_intermediate = qk.max() * v_sample.max()
min_intermediate = qk.min() * v_sample.max()

print(f"\nWorst-case intermediate values:")
print(f"  max(QK) * max(V) = {qk.max():.2f} * {v_sample.max():.2f} = {max_intermediate:.2f}")
print(f"  min(QK) * max(V) = {qk.min():.2f} * {v_sample.max():.2f} = {min_intermediate:.2f}")

if abs(max_intermediate) > 65504 or abs(min_intermediate) > 65504:
    print(f"\n⚠️  OVERFLOW DETECTED: Intermediate values exceed float16 max (±65504)!")
else:
    print(f"\n✓ Intermediate values within float16 range")

print("\n" + "="*80)
print("NAIVE GLA COMPUTATION (HIGH PRECISION)")
print("="*80)

print("\nImplementing full GLA recurrence in float64 for accurate simulation...")
print("GLA formula (from FLA fused_recurrent kernel):")
print("  1. h_t = exp(gk_t) ⊙ h_{t-1}")
print("  2. h_t = h_t + k_t ⊗ v_t")
print("  3. o_t = sum(q_t ⊙ h_t, axis=K)")
print("\nNote: In FLA kernel, q is scaled by 1/sqrt(K) by default")

# Convert to float64 for maximum precision
q_f64 = q.double()
k_f64 = k.double()
v_f64 = v.double()
gk_f64 = gk.double()

# We'll compute the output for the first sample to analyze
batch, seq_len, num_heads, head_dim = q_f64.shape
_, _, _, v_dim = v_f64.shape

# Apply scale factor to q (as done in FLA kernel line 91)
scale = head_dim ** -0.5
q_f64 = q_f64 * scale

print(f"\nComputing for sample 0, all {num_heads} heads, seq_len={seq_len}")
print(f"Query scale factor: {scale:.6f} (1/sqrt({head_dim}))")

# Initialize output
o_computed = torch.zeros(seq_len, num_heads, v_dim, dtype=torch.float64)

# Process each head independently
for h in range(num_heads):
    # Initialize hidden state: [head_dim, v_dim]
    # This matches b_h = tl.zeros([BK, BV], dtype=tl.float32) in FLA kernel
    state = torch.zeros(head_dim, v_dim, dtype=torch.float64)
    
    # Recurrence over sequence (matches lines 90-108 in fused_recurrent_fwd_kernel)
    for t in range(seq_len):
        # Get current timestep values
        q_t = q_f64[0, t, h, :]  # [head_dim] (already scaled by 1/sqrt(K))
        k_t = k_f64[0, t, h, :]  # [head_dim]
        v_t = v_f64[0, t, h, :]  # [v_dim]
        g_t = gk_f64[0, t, h, :]  # [head_dim] - in log space (logsigmoid output)
        
        # Step 1: Apply gating/decay (line 101: b_h = b_h * exp(b_gk[:, None]))
        # exp(g_t) is the decay factor for each dimension of K
        decay = torch.exp(g_t)  # [head_dim]
        state = state * decay.unsqueeze(1)  # [head_dim, v_dim] * [head_dim, 1]
        
        # Step 2: Add new contribution (line 105: b_h += b_k[:, None] * b_v[None, :])
        state = state + torch.outer(k_t, v_t)  # [head_dim, v_dim]
        
        # Step 3: Compute output (lines 106-107: b_o = b_h * b_q[:, None]; b_o = tl.sum(b_o, axis=0))
        # This is equivalent to: o_t = sum(state * q_t[:, None], axis=0) = q_t @ state
        o_t = torch.matmul(q_t, state)  # [v_dim]
        o_computed[t, h, :] = o_t
        
        # Track extremes
        if h == 0 and (t < 5 or t % 10 == 0 or state.abs().max() > 65504):
            state_max = state.abs().max().item()
            o_max = o_t.abs().max().item()
            print(f"  Head 0, t={t:2d}: |state|_max={state_max:12.2f}, |o_t|_max={o_max:12.2f}")

print("\n" + "="*80)
print("COMPUTED OUTPUT STATISTICS (Float64 precision)")
print("="*80)

o_computed_flat = o_computed.flatten()
print(f"\nFull output statistics:")
print(f"  Min:  {o_computed_flat.min().item():12.2f}")
print(f"  Max:  {o_computed_flat.max().item():12.2f}")
print(f"  Mean: {o_computed_flat.mean().item():12.2f}")
print(f"  Std:  {o_computed_flat.std().item():12.2f}")

fp16_max = 65504.0
exceeds_fp16 = (o_computed_flat.abs() > fp16_max).sum().item()
total = o_computed_flat.numel()

print(f"\nFloat16 overflow analysis:")
print(f"  Float16 max: ±{fp16_max}")
print(f"  Values exceeding float16 max: {exceeds_fp16}/{total} ({100*exceeds_fp16/total:.3f}%)")

if exceeds_fp16 > 0:
    extreme_vals = o_computed_flat[o_computed_flat.abs() > fp16_max]
    print(f"  Extreme values range: [{extreme_vals.min().item():.2f}, {extreme_vals.max().item():.2f}]")
    print(f"\n  ⚠️  OUTPUT WILL OVERFLOW TO INF WHEN CONVERTED TO FLOAT16!")
    
    # Show distribution of extreme values
    percentiles = [50, 75, 90, 95, 99, 100]
    extreme_abs = extreme_vals.abs()
    print(f"\n  Percentiles of extreme values:")
    for p in percentiles:
        val = torch.quantile(extreme_abs.float(), p/100.0).item()
        print(f"    {p:3d}th: {val:12.2f}")
else:
    print(f"  ✓ All values within float16 range!")

# Compare with actual saved output (first sample only)
print("\n" + "="*80)
print("COMPARISON WITH SAVED OUTPUT")
print("="*80)

o_saved_sample = o.half()[0]  # First sample from saved data
o_computed_fp16 = o_computed.half()

print(f"\nSaved output (o.pt, sample 0):")
print(f"  Has Inf: {torch.isinf(o_saved_sample).any()}")
print(f"  Inf count: {torch.isinf(o_saved_sample).sum().item()}")
print(f"  Shape: {o_saved_sample.shape}")

print(f"\nComputed output (converted to fp16):")
print(f"  Has Inf: {torch.isinf(o_computed_fp16).any()}")
print(f"  Inf count: {torch.isinf(o_computed_fp16).sum().item()}")
print(f"  Shape: {o_computed_fp16.shape}")

# Check if patterns match
valid_saved = o_saved_sample[~torch.isinf(o_saved_sample)]
valid_computed = o_computed_fp16[~torch.isinf(o_computed_fp16)]

if len(valid_saved) > 0 and len(valid_computed) > 0:
    print(f"\nValid (non-Inf) values comparison:")
    print(f"  Saved range:    [{valid_saved.min().item():.2f}, {valid_saved.max().item():.2f}]")
    print(f"  Computed range: [{valid_computed.min().item():.2f}, {valid_computed.max().item():.2f}]")

print("\n" + "="*80)
print("ROOT CAUSE HYPOTHESIS")
print("="*80)

print("""
Based on the analysis:

1. V values (±691) are large but not extreme
2. Q@K^T can produce values in range [min_qk, max_qk]  
3. The recurrent state accumulation: h = decay*h + k⊗v
   amplifies the large V values over the sequence

Possible causes:
A. Quantization of K/Q projections produces unnaturally large QK dot products
B. The cumulative gating (exp(cumsum(gk))) doesn't decay fast enough
C. Sequence length (64) allows too much accumulation
D. V projection quantization scales are incorrect, producing outlier values

To fix:
- Clamp V values: v = torch.clamp(v, -100, 100) after v_proj
- Scale V down before attention, scale output back up
- Use better quantization calibration for v_proj
- Add layer normalization after v_proj
""")

print("\n" + "="*80)
print("CHECKING FOR OUTLIERS")
print("="*80)

# Check if the large V values are outliers or systematic
v_flat = v.flatten()
v_sorted, _ = torch.sort(v_flat.abs(), descending=True)
print("\nTop 20 largest |V| values:")
for i in range(min(20, len(v_sorted))):
    print(f"  {i+1}. {v_sorted[i]:.2f}")

# Check percentage above thresholds
above_100 = (v.abs() > 100).sum().item()
above_200 = (v.abs() > 200).sum().item()
above_500 = (v.abs() > 500).sum().item()

print(f"\nV value distribution:")
print(f"  |V| > 100: {above_100}/{v.numel()} ({100*above_100/v.numel():.3f}%)")
print(f"  |V| > 200: {above_200}/{v.numel()} ({100*above_200/v.numel():.3f}%)")
print(f"  |V| > 500: {above_500}/{v.numel()} ({100*above_500/v.numel():.3f}%)")

if above_500 / v.numel() > 0.001:  # More than 0.1%
    print("\n⚠️  Significant portion of V values are extremely large!")
    print("     This suggests quantization issues in v_proj")


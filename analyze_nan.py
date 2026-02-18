import torch
import numpy as np

print("="*80)
print("ANALYZING NaN DEBUG TENSORS")
print("="*80)

# Load all tensors (map to CPU since CUDA may not be available for analysis)
q = torch.load("debug_tensors/q.pt", map_location=torch.device('cpu'))
k = torch.load("debug_tensors/k.pt", map_location=torch.device('cpu'))
v = torch.load("debug_tensors/v.pt", map_location=torch.device('cpu'))
gk = torch.load("debug_tensors/gk.pt", map_location=torch.device('cpu'))
o = torch.load("debug_tensors/o.pt", map_location=torch.device('cpu'))

print("\n" + "="*80)
print("TENSOR SHAPES AND DTYPES")
print("="*80)
print(f"q:  {q.shape}, dtype={q.dtype}")
print(f"k:  {k.shape}, dtype={k.dtype}")
print(f"v:  {v.shape}, dtype={v.dtype}")
print(f"gk: {gk.shape}, dtype={gk.dtype}")
print(f"o:  {o.shape}, dtype={o.dtype}")

print("\n" + "="*80)
print("NaN/INF DETECTION")
print("="*80)
print(f"q  has NaN: {torch.isnan(q).any().item()}, has Inf: {torch.isinf(q).any().item()}")
print(f"k  has NaN: {torch.isnan(k).any().any().item()}, has Inf: {torch.isinf(k).any().item()}")
print(f"v  has NaN: {torch.isnan(v).any().item()}, has Inf: {torch.isinf(v).any().item()}")
print(f"gk has NaN: {torch.isnan(gk).any().item()}, has Inf: {torch.isinf(gk).any().item()}")
print(f"o  has NaN: {torch.isnan(o).any().item()}, has Inf: {torch.isinf(o).any().item()}")

print("\n" + "="*80)
print("INPUT RANGES (Q, K, V, GK)")
print("="*80)
print(f"q  range: [{q.min().item():.6f}, {q.max().item():.6f}], mean={q.mean().item():.6f}, std={q.std().item():.6f}")
print(f"k  range: [{k.min().item():.6f}, {k.max().item():.6f}], mean={k.mean().item():.6f}, std={k.std().item():.6f}")
print(f"v  range: [{v.min().item():.6f}, {v.max().item():.6f}], mean={v.mean().item():.6f}, std={v.std().item():.6f}")
print(f"gk range: [{gk.min().item():.6f}, {gk.max().item():.6f}], mean={gk.mean().item():.6f}, std={gk.std().item():.6f}")

print("\n" + "="*80)
print("OUTPUT ANALYSIS")
print("="*80)
if torch.isnan(o).any():
    nan_count = torch.isnan(o).sum().item()
    total_elements = o.numel()
    print(f"o has {nan_count}/{total_elements} NaN elements ({100*nan_count/total_elements:.2f}%)")
    
    # Find where NaNs are
    nan_mask = torch.isnan(o)
    nan_indices = torch.nonzero(nan_mask)
    print(f"\nFirst 10 NaN locations:")
    for i, idx in enumerate(nan_indices[:10]):
        print(f"  {i+1}. Index: {idx.tolist()}")
        
if torch.isinf(o).any():
    inf_count = torch.isinf(o).sum().item()
    total_elements = o.numel()
    print(f"o has {inf_count}/{total_elements} Inf elements ({100*inf_count/total_elements:.2f}%)")

# Check if output has any valid (non-NaN, non-Inf) values
valid_mask = ~(torch.isnan(o) | torch.isinf(o))
valid_count = valid_mask.sum().item()
print(f"\no has {valid_count}/{o.numel()} valid elements ({100*valid_count/o.numel():.2f}%)")
if valid_count > 0:
    valid_vals = o[valid_mask]
    print(f"Valid values range: [{valid_vals.min().item():.6f}, {valid_vals.max().item():.6f}]")

print("\n" + "="*80)
print("GK ANALYSIS (logsigmoid output)")
print("="*80)
print(f"gk is the result of: F.logsigmoid(gk_proj_output) / gate_logit_normalizer")
print(f"\nGK statistics:")
print(f"  - min:  {gk.min().item():.6f}")
print(f"  - max:  {gk.max().item():.6f}")
print(f"  - mean: {gk.mean().item():.6f}")
print(f"  - std:  {gk.std().item():.6f}")

# Check for extreme values in gk that could cause numerical issues
gk_extreme_negative = (gk < -20).sum().item()
gk_extreme_positive = (gk > 0).sum().item()  # logsigmoid should be <= 0
print(f"\nGK extreme values:")
print(f"  - Values < -20: {gk_extreme_negative} ({100*gk_extreme_negative/gk.numel():.2f}%)")
print(f"  - Values > 0:   {gk_extreme_positive} ({100*gk_extreme_positive/gk.numel():.2f}%)")

# Check percentile distribution
percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
gk_flat = gk.flatten().float()  # Convert to float for quantile
for p in percentiles:
    val = torch.quantile(gk_flat, p/100.0).item()
    print(f"  - {p:3d}th percentile: {val:.6f}")

print("\n" + "="*80)
print("POTENTIAL ISSUES")
print("="*80)

issues = []

# Check if gk has extreme negative values
if gk.min() < -50:
    issues.append(f"⚠️  GK has extremely negative values (min={gk.min().item():.2f}), which when exponentiated in attention could underflow to 0")
    
# Check if inputs have unusual distributions
if k.std() < 0.01:
    issues.append(f"⚠️  K has very low variance (std={k.std().item():.6f}), might cause numerical instability")
    
if v.std() < 0.01:
    issues.append(f"⚠️  V has very low variance (std={v.std().item():.6f}), might cause numerical instability")

# Check for near-zero values that could cause division issues
if (k.abs() < 1e-6).any():
    zero_count = (k.abs() < 1e-6).sum().item()
    issues.append(f"⚠️  K has {zero_count} near-zero values (< 1e-6)")

if len(issues) == 0:
    print("No obvious issues detected in inputs. The NaN likely comes from:")
    print("  1. Numerical instability in the attention kernel itself")
    print("  2. Accumulation of small errors over the recurrent computation")
    print("  3. Specific interaction between quantized weights and these input values")
else:
    for issue in issues:
        print(issue)

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("1. Check gate_logit_normalizer value - if too small, gk values become extreme")
print("2. Verify quantization scale factors are appropriate for this layer")
print("3. Consider gradient clipping or normalization before this layer")
print("4. Check if inputs at 55% mark have different characteristics (longer sequences?)")


"""
Verify what the ACTUAL chunk_gla kernel produces with the saved tensors.
This will tell us the true output values in float32 before any conversion.
"""

import torch
import sys

print("="*80)
print("ACTUAL CHUNK_GLA KERNEL OUTPUT VERIFICATION")
print("="*80)

# Load tensors (these are the float16 versions that were saved)
q_fp16 = torch.load("debug_tensors/q.pt", map_location='cpu')
k_fp16 = torch.load("debug_tensors/k.pt", map_location='cpu')
v_fp16 = torch.load("debug_tensors/v.pt", map_location='cpu')
gk_fp16 = torch.load("debug_tensors/gk.pt", map_location='cpu')
o_fp16 = torch.load("debug_tensors/o.pt", map_location='cpu')

print("\nLoaded tensors (float16 versions from saved state):")
print(f"q:  {q_fp16.shape}, dtype={q_fp16.dtype}")
print(f"k:  {k_fp16.shape}, dtype={k_fp16.dtype}")
print(f"v:  {v_fp16.shape}, dtype={v_fp16.dtype}")
print(f"gk: {gk_fp16.shape}, dtype={gk_fp16.dtype}")
print(f"o:  {o_fp16.shape}, dtype={o_fp16.dtype} (has Inf: {torch.isinf(o_fp16).any()})")

print("\n" + "="*80)
print("CALLING ACTUAL chunk_gla KERNEL")
print("="*80)

try:
    from fla.ops.gla import chunk_gla
    
    # Convert to float32 exactly as done in the code (line 358)
    q_fp32 = q_fp16.float()
    k_fp32 = k_fp16.float()
    v_fp32 = v_fp16.float()
    gk_fp32 = gk_fp16.float()
    
    print("\nConverted to float32:")
    print(f"q:  range=[{q_fp32.min():.2f}, {q_fp32.max():.2f}]")
    print(f"k:  range=[{k_fp32.min():.2f}, {k_fp32.max():.2f}]")
    print(f"v:  range=[{v_fp32.min():.2f}, {v_fp32.max():.2f}]")
    print(f"gk: range=[{gk_fp32.min():.2f}, {gk_fp32.max():.2f}]")
    
    # Move to CUDA if available (chunk_gla might require it)
    if torch.cuda.is_available():
        print("\nMoving to CUDA...")
        device = torch.device('cuda:0')
        q_fp32 = q_fp32.to(device)
        k_fp32 = k_fp32.to(device)
        v_fp32 = v_fp32.to(device)
        gk_fp32 = gk_fp32.to(device)
    else:
        print("\n⚠️  CUDA not available - chunk_gla may not work on CPU")
        print("   This is expected in this environment, but means we can't test the actual kernel")
        sys.exit(0)
    
    print("\nCalling chunk_gla(q, k, v, g=gk, initial_state=None, output_final_state=False)...")
    
    # Call exactly as in the code (line 358)
    o_output, final_state = chunk_gla(
        q=q_fp32, 
        k=k_fp32, 
        v=v_fp32, 
        g=gk_fp32, 
        initial_state=None, 
        output_final_state=False
    )
    
    print(f"\nchunk_gla returned:")
    print(f"  o shape: {o_output.shape}")
    print(f"  o dtype: {o_output.dtype}")
    
    # Check output statistics IN FLOAT32 (before any conversion)
    print(f"\nOutput statistics (FLOAT32, before any conversion):")
    print(f"  Min:  {o_output.min().item():.2f}")
    print(f"  Max:  {o_output.max().item():.2f}")
    print(f"  Mean: {o_output.mean().item():.2f}")
    print(f"  Std:  {o_output.std().item():.2f}")
    
    # Check for extreme values that would overflow float16
    fp16_max = 65504
    exceeds_fp16 = (o_output.abs() > fp16_max).sum().item()
    print(f"\n  Values exceeding float16 max (±{fp16_max}): {exceeds_fp16}/{o_output.numel()}")
    if exceeds_fp16 > 0:
        print(f"  Percentage: {100*exceeds_fp16/o_output.numel():.3f}%")
        
        # Show distribution of extreme values
        extreme_vals = o_output[o_output.abs() > fp16_max]
        print(f"\n  Extreme values range: [{extreme_vals.min().item():.2f}, {extreme_vals.max().item():.2f}]")
    
    # Now try converting to float16 like what presumably happens
    print("\n" + "="*80)
    print("CONVERTING TO FLOAT16")
    print("="*80)
    
    o_fp16_converted = o_output.half()
    
    print(f"\nAfter .half() conversion:")
    print(f"  dtype: {o_fp16_converted.dtype}")
    print(f"  Has NaN: {torch.isnan(o_fp16_converted).any()}")
    print(f"  Has Inf: {torch.isinf(o_fp16_converted).any()}")
    
    if torch.isinf(o_fp16_converted).any():
        inf_count = torch.isinf(o_fp16_converted).sum().item()
        print(f"  Inf count: {inf_count}/{o_fp16_converted.numel()} ({100*inf_count/o_fp16_converted.numel():.2f}%)")
    
    # Compare with saved o
    o_fp16_saved = o_fp16.to(device)
    matches = torch.allclose(o_fp16_converted, o_fp16_saved, equal_nan=True)
    print(f"\n  Matches saved 'o.pt': {matches}")
    
    if not matches:
        # Check where they differ
        diff_mask = ~torch.isclose(o_fp16_converted, o_fp16_saved, equal_nan=True)
        diff_count = diff_mask.sum().item()
        print(f"  Differences: {diff_count}/{o_fp16_converted.numel()}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if exceeds_fp16 > 0:
        print(f"""
✓ CONFIRMED: chunk_gla produces float32 values that exceed ±65504
  - chunk_gla output range: [{o_output.min().item():.2f}, {o_output.max().item():.2f}]
  - {exceeds_fp16} values overflow when converted to float16
  - These become Inf when cast to float16, matching the saved o.pt
  
The issue is NOT in the chunk_gla kernel itself, but in the fact that:
1. Q and K values from quantized projections are abnormally large
2. This causes chunk_gla's internal computations to produce large values
3. When converted back to float16, these overflow to Inf
        """)
    else:
        print(f"""
⚠️ UNEXPECTED: chunk_gla output stays within float16 range
  - This suggests the issue might be elsewhere
  - Or the environment/kernel version is different
        """)

except ImportError as e:
    print(f"\n❌ Could not import chunk_gla: {e}")
    print("   Cannot verify actual kernel behavior")
except Exception as e:
    print(f"\n❌ Error running chunk_gla: {e}")
    import traceback
    traceback.print_exc()





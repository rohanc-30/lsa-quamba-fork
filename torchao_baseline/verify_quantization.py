#!/usr/bin/env python3
"""
Diagnostic script to verify that W8A8 quantized weights are correctly saved/loaded.
Compares original FP16 weights with dequantized INT8 weights.
"""

import torch
import sys
sys.path.insert(0, '/home/rcherukuri/lsa-quamba-fork')

from fla.models.gla import GLAForCausalLM
from torchao_baseline.utils_torchao import QuantizedLinear

print("="*80)
print("W8A8 QUANTIZATION VERIFICATION")
print("="*80)

# Load models
print("\n1. Loading models...")
original_model = GLAForCausalLM.from_pretrained('pretrained_models/fla-hub/gla-1.3b')
quantized_model = torch.load('pretrained_models/fla-hub/gla_ptq-w8a8-1.3b/pytorch_model.bin', 
                              map_location='cpu', weights_only=False)
print(f"   ✓ Original model loaded")
print(f"   ✓ Quantized model loaded")

# Find QuantizedLinear modules
quant_modules = []
for name, module in quantized_model.named_modules():
    if isinstance(module, QuantizedLinear):
        quant_modules.append((name, module))
        if len(quant_modules) >= 3:
            break

print(f"\n2. Found {len(quant_modules)} QuantizedLinear modules (showing first 3)")

for i, (name, quant_module) in enumerate(quant_modules, 1):
    print(f"\n{'='*80}")
    print(f"LAYER {i}: {name}")
    print('='*80)
    
    # Get original weight
    original_module = original_model
    for part in name.split('.'):
        original_module = getattr(original_module, part)
    orig_weight = original_module.weight
    
    print(f"\n📊 ORIGINAL WEIGHT (FP16/FP32):")
    print(f"   Shape: {orig_weight.shape}")
    print(f"   Dtype: {orig_weight.dtype}")
    print(f"   Min:   {orig_weight.min().item():>10.6f}")
    print(f"   Max:   {orig_weight.max().item():>10.6f}")
    print(f"   Mean:  {orig_weight.mean().item():>10.6f}")
    print(f"   Std:   {orig_weight.std().item():>10.6f}")
    
    print(f"\n📊 QUANTIZED WEIGHT (qweight):")
    print(f"   Shape: {quant_module.qweight.shape}")
    print(f"   Dtype: {quant_module.qweight.dtype} ← logical dtype (for dequant output)")
    print(f"   Type:  {type(quant_module.qweight).__name__}")
    
    print(f"\n📊 WEIGHT QUANTIZATION PARAMETERS:")
    # The qweight is an AffineQuantizedTensor that stores:
    # - Quantized INT8 data internally
    # - Scale and zero_point for dequantization
    
    # Access internal structure (avoid calling unsupported ops on INT8 data)
    if hasattr(quant_module.qweight, 'tensor_impl'):
        impl = quant_module.qweight.tensor_impl
        print(f"   tensor_impl type: {type(impl).__name__}")
        if hasattr(impl, 'data'):
            int8_data = impl.data
            print(f"   INT8 data shape: {int8_data.shape}")
            print(f"   INT8 data dtype: {int8_data.dtype} ← actual quantized data")
            print(f"   INT8 data type: {type(int8_data).__name__}")
            # Note: Cannot directly access INT8 values due to PlainAQTTensorImpl restrictions
        
        if hasattr(impl, 'scale'):
            scale = impl.scale
            print(f"\n   📏 WEIGHT SCALE (for dequantization):")
            print(f"      Shape: {scale.shape}")
            print(f"      Dtype: {scale.dtype}")
            num_scales = scale.numel()
            if num_scales == 1:
                print(f"      Value: {scale.item():.8f} (per-tensor)")
            elif num_scales <= 16:
                # Small enough to show all
                scale_list = [scale.flatten()[j].item() for j in range(num_scales)]
                print(f"      Values (per-channel): {scale_list}")
            else:
                # Per-channel quantization - show samples
                print(f"      Mode: Per-channel ({num_scales} scales)")
                scale_samples = [scale.flatten()[j].item() for j in range(min(8, num_scales))]
                print(f"      Sample scales (first 8): {scale_samples}")
                # Get min/max by iterating
                scale_vals = [scale.flatten()[j].item() for j in range(num_scales)]
                print(f"      Scale range: [{min(scale_vals):.6f}, {max(scale_vals):.6f}]")
        
        if hasattr(impl, 'zero_point'):
            zp = impl.zero_point
            print(f"\n   🎯 WEIGHT ZERO_POINT (for dequantization):")
            print(f"      Shape: {zp.shape}")
            print(f"      Dtype: {zp.dtype}")
            num_zp = zp.numel()
            if num_zp == 1:
                print(f"      Value: {zp.item()} (per-tensor)")
            elif num_zp <= 16:
                zp_list = [zp.flatten()[j].item() for j in range(num_zp)]
                print(f"      Values: {zp_list}")
            else:
                # Per-channel - show samples
                print(f"      Mode: Per-channel ({num_zp} zero-points)")
                zp_samples = [int(zp.flatten()[j].item()) for j in range(min(8, num_zp))]
                print(f"      Sample (first 8): {zp_samples}")
                # Check if all zeros (common for symmetric quantization)
                all_vals = [int(zp.flatten()[j].item()) for j in range(num_zp)]
                unique_vals = list(set(all_vals))
                print(f"      Unique values: {unique_vals[:10]}")  # Show first 10 unique
                if unique_vals == [0]:
                    print(f"      → All zeros (symmetric quantization)")
    
    print(f"\n📊 ACTIVATION QUANTIZATION PARAMETERS (from module):")
    if hasattr(quant_module, 'act_scale'):
        print(f"   Activation scale: {quant_module.act_scale}")
    if hasattr(quant_module, 'act_zero_point'):
        print(f"   Activation zero_point: {quant_module.act_zero_point}")
    if hasattr(quant_module, 'target_dtype'):
        print(f"   Target dtype: {quant_module.target_dtype}")
    
    print(f"\n📊 DEQUANTIZATION TEST:")
    if hasattr(quant_module.qweight, 'dequantize'):
        dequant_weight = quant_module.qweight.dequantize()
        print(f"   Dequantized shape: {dequant_weight.shape}")
        print(f"   Dequantized dtype: {dequant_weight.dtype}")
        print(f"   Dequantized min:   {dequant_weight.min().item():>10.6f}")
        print(f"   Dequantized max:   {dequant_weight.max().item():>10.6f}")
        print(f"   Dequantized mean:  {dequant_weight.mean().item():>10.6f}")
        print(f"   Dequantized std:   {dequant_weight.std().item():>10.6f}")
        
        print(f"\n   ✓ VERIFICATION:")
        diff = (dequant_weight - orig_weight).abs()
        print(f"     Mean abs error: {diff.mean().item():.6f}")
        print(f"     Max abs error:  {diff.max().item():.6f}")
        print(f"     Relative error: {(diff.mean() / orig_weight.abs().mean()).item():.2%}")
        
        if diff.mean().item() < 0.001:
            print(f"     ✅ WEIGHTS ARE CORRECT!")
        elif diff.mean().item() < 0.01:
            print(f"     ✓ Weights appear reasonable")
        else:
            print(f"     ❌ WEIGHTS MAY BE WRONG!")

print(f"\n{'='*80}")
print(f"SUMMARY")
print('='*80)
print(f"Total QuantizedLinear modules found: {len(list(m for n, m in quantized_model.named_modules() if isinstance(m, QuantizedLinear)))}")
print(f"\nThe qweight dtype shows as torch.float32 because:")
print(f"  - qweight is an AffineQuantizedTensor wrapper")
print(f"  - The .dtype property returns the OUTPUT dtype after dequantization")
print(f"  - The actual INT8 quantized data is stored internally in tensor_impl.data")
print('='*80)


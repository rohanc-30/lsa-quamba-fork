#!/usr/bin/env python3
"""
Comprehensive analysis of W8A8 quantization quality.
Analyzes both weight quantization errors and activation errors through the network.
"""

import torch
import sys
import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/rcherukuri/lsa-quamba-fork')

from fla.models.gla import GLAForCausalLM
from torchao_baseline.utils_torchao import QuantizedLinear

OUTPUT_DIR = "quantization_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("COMPREHENSIVE W8A8 QUANTIZATION ANALYSIS")
print("="*80)

# Load models
print("\n1. Loading models...")
original_model = GLAForCausalLM.from_pretrained('pretrained_models/fla-hub/gla-1.3b')
quantized_model = torch.load('pretrained_models/fla-hub/gla_ptq-w8a8-1.3b/pytorch_model.bin', 
                              map_location='cpu', weights_only=False)
original_model.eval()
quantized_model.eval()
print(f"   ✓ Models loaded")

# ============================================================================
# PART 1: WEIGHT QUANTIZATION ERRORS
# ============================================================================
print(f"\n{'='*80}")
print("PART 1: WEIGHT QUANTIZATION ERROR ANALYSIS")
print('='*80)

weight_errors = []
weight_error_log = []

for name, quant_module in quantized_model.named_modules():
    if isinstance(quant_module, QuantizedLinear):
        # Get original weight
        try:
            original_module = original_model
            for part in name.split('.'):
                original_module = getattr(original_module, part)
            orig_weight = original_module.weight
            
            # Dequantize
            if hasattr(quant_module.qweight, 'dequantize'):
                dequant_weight = quant_module.qweight.dequantize()
                
                # Calculate errors
                diff = (dequant_weight - orig_weight).abs()
                mean_abs_error = diff.mean().item()
                max_abs_error = diff.max().item()
                relative_error = (diff.mean() / orig_weight.abs().mean()).item()
                
                weight_errors.append({
                    'name': name,
                    'mean_abs_error': mean_abs_error,
                    'max_abs_error': max_abs_error,
                    'relative_error': relative_error,
                    'weight_min': orig_weight.min().item(),
                    'weight_max': orig_weight.max().item(),
                    'weight_std': orig_weight.std().item(),
                })
                
                weight_error_log.append(f"{name}: relative_error={relative_error:.4%}, mean_abs={mean_abs_error:.6f}")
        except Exception as e:
            print(f"   ⚠️  Could not analyze {name}: {e}")
            continue

print(f"\nAnalyzed {len(weight_errors)} quantized layers")

# Print summary statistics
if weight_errors:
    rel_errors = [w['relative_error'] for w in weight_errors]
    print(f"\nWeight Quantization Error Statistics:")
    print(f"  Average relative error: {np.mean(rel_errors):.4%}")
    print(f"  Median relative error:  {np.median(rel_errors):.4%}")
    print(f"  Min relative error:     {np.min(rel_errors):.4%}")
    print(f"  Max relative error:     {np.max(rel_errors):.4%}")
    
    # Show worst 5 layers
    sorted_errors = sorted(weight_errors, key=lambda x: x['relative_error'], reverse=True)
    print(f"\n  Worst 5 layers:")
    for w in sorted_errors[:5]:
        print(f"    {w['name']}: {w['relative_error']:.4%}")
    
    # Save detailed log
    log_path = os.path.join(OUTPUT_DIR, "weight_quantization_errors.txt")
    with open(log_path, 'w') as f:
        f.write("WEIGHT QUANTIZATION ERRORS (sorted by name)\n")
        f.write("="*80 + "\n\n")
        for line in weight_error_log:
            f.write(line + "\n")
    print(f"\n✓ Saved detailed log to: {log_path}")
    
    # Save as JSON
    json_path = os.path.join(OUTPUT_DIR, "weight_quantization_errors.json")
    with open(json_path, 'w') as f:
        json.dump(weight_errors, f, indent=2)
    print(f"✓ Saved JSON data to: {json_path}")
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Relative errors across layers
    layer_indices = range(len(weight_errors))
    rel_errors_pct = [w['relative_error'] * 100 for w in weight_errors]
    
    axes[0].plot(layer_indices, rel_errors_pct, 'b-', alpha=0.7, linewidth=1)
    axes[0].scatter(layer_indices, rel_errors_pct, c='blue', s=10, alpha=0.5)
    axes[0].axhline(y=np.mean(rel_errors_pct), color='r', linestyle='--', label=f'Mean: {np.mean(rel_errors_pct):.3f}%')
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Relative Error (%)')
    axes[0].set_title('Weight Quantization Relative Error Across All Layers')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Distribution of errors
    axes[1].hist(rel_errors_pct, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=np.mean(rel_errors_pct), color='r', linestyle='--', label=f'Mean: {np.mean(rel_errors_pct):.3f}%')
    axes[1].axvline(x=np.median(rel_errors_pct), color='g', linestyle='--', label=f'Median: {np.median(rel_errors_pct):.3f}%')
    axes[1].set_xlabel('Relative Error (%)')
    axes[1].set_ylabel('Number of Layers')
    axes[1].set_title('Distribution of Weight Quantization Errors')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "weight_quantization_errors.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {plot_path}")
    plt.close()

# ============================================================================
# PART 2: ACTIVATION ERROR ANALYSIS (Two-Stage Approach)
# ============================================================================
print(f"\n{'='*80}")
print("PART 2: ACTIVATION ERROR ANALYSIS")
print('='*80)

# Stage 1: Capture activations by running models on CUDA
print("\n📥 STAGE 1: Capturing activations on CUDA...")

activation_errors = []

# Create sample input
print("\n2. Creating test input...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("fla-hub/gla-1.3B-100B")
test_text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeat to get longer sequence
test_input = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
input_ids = test_input['input_ids']
print(f"   Test input shape: {input_ids.shape}")

# Use CUDA for forward pass (GLA requires it), but capture activations to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Using device: {device} for forward pass")
print(f"   Activations will be captured and moved to CPU for comparison")

# Move models to device
original_model = original_model.to(device)
quantized_model = quantized_model.to(device)
input_ids = input_ids.to(device)

# For QuantizedLinear, ensure act_scale and act_zero_point are on same device as model
print("\n   Ensuring quantization parameters are on correct device...")
fixed_count = 0
for name, module in quantized_model.named_modules():
    if isinstance(module, QuantizedLinear):
        # Get the device of the module's weights
        try:
            weight_device = next(module.parameters()).device
        except StopIteration:
            # If no parameters, use the device variable
            weight_device = torch.device(device)
        
        # Move act_scale to correct device
        if hasattr(module, 'act_scale'):
            if isinstance(module.act_scale, torch.Tensor):
                if module.act_scale.device != weight_device:
                    module.act_scale = module.act_scale.to(weight_device)
                    fixed_count += 1
            else:
                # If it's a scalar, convert to tensor on correct device
                module.act_scale = torch.tensor(module.act_scale, device=weight_device)
                fixed_count += 1
        
        # Move act_zero_point to correct device
        if hasattr(module, 'act_zero_point'):
            if isinstance(module.act_zero_point, torch.Tensor):
                if module.act_zero_point.device != weight_device:
                    module.act_zero_point = module.act_zero_point.to(weight_device)
                    fixed_count += 1
            else:
                # If it's a scalar, convert to tensor on correct device
                module.act_zero_point = torch.tensor(module.act_zero_point, device=weight_device)
                fixed_count += 1

if fixed_count > 0:
    print(f"   ✓ Fixed {fixed_count} quantization parameters to match model device")

activation_errors = []

# Create subdirectory for activation captures
CAPTURE_DIR = os.path.join(OUTPUT_DIR, "activation_captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Reload models fresh for activation capture
print("   Reloading models fresh on CUDA...")
del original_model, quantized_model  # Free memory
torch.cuda.empty_cache()

original_model = GLAForCausalLM.from_pretrained('pretrained_models/fla-hub/gla-1.3b').cuda()
original_model.eval()

# Hook to save INPUT activations (not output)
def make_input_hook_save(name, prefix):
    def hook(module, input, output):
        # Capture INPUT to the module
        # input is a tuple, first element is the actual tensor
        if isinstance(input, tuple):
            act = input[0]
        else:
            act = input
        
        # Save to disk immediately
        act_cpu = act.detach().cpu() if hasattr(act, 'detach') else act
        save_path = os.path.join(CAPTURE_DIR, f"{prefix}_{name}.pt")
        torch.save(act_cpu, save_path)
    return hook

# Register INPUT hooks for original model (72 capture points)
print("   Registering INPUT hooks for original model...")
layer_names = []
for i in range(24):  # 24 layers
    # 1. Input to attention block
    attn_input_name = f"layer_{i:02d}_attn_input"
    original_model.model.layers[i].attn.register_forward_hook(make_input_hook_save(attn_input_name, "original"))
    layer_names.append(attn_input_name)
    
    # 2. Input to MLP block  
    mlp_input_name = f"layer_{i:02d}_mlp_input"
    original_model.model.layers[i].mlp.register_forward_hook(make_input_hook_save(mlp_input_name, "original"))
    layer_names.append(mlp_input_name)
    
    # 3. Mid-MLP: Input to down_proj (after SwiGLU, entering down_proj)
    mid_mlp_name = f"layer_{i:02d}_mid_mlp"
    original_model.model.layers[i].mlp.down_proj.register_forward_hook(make_input_hook_save(mid_mlp_name, "original"))
    layer_names.append(mid_mlp_name)

print(f"   ✓ Registered {len(layer_names)} input hooks (24 layers × 3 points)")

# Run original model
print("   Running original FP32 model forward pass...")
with torch.no_grad():
    _ = original_model(input_ids)
print(f"   ✓ Captured and saved {len(layer_names)} activation tensors")

# Free memory
del original_model
torch.cuda.empty_cache()

# Load quantized model
print("\n   Loading quantized model...")
quantized_model = torch.load('pretrained_models/fla-hub/gla_ptq-w8a8-1.3b/pytorch_model.bin',
                              map_location='cuda', weights_only=False)
quantized_model.eval()

# Register INPUT hooks for quantized model (same 72 points)
print("   Registering INPUT hooks for quantized model...")
for i in range(24):
    # 1. Input to attention
    attn_input_name = f"layer_{i:02d}_attn_input"
    quantized_model.model.layers[i].attn.register_forward_hook(make_input_hook_save(attn_input_name, "quantized"))
    
    # 2. Input to MLP
    mlp_input_name = f"layer_{i:02d}_mlp_input"
    quantized_model.model.layers[i].mlp.register_forward_hook(make_input_hook_save(mlp_input_name, "quantized"))
    
    # 3. Mid-MLP: Input to down_proj  
    mid_mlp_name = f"layer_{i:02d}_mid_mlp"
    quantized_model.model.layers[i].mlp.down_proj.register_forward_hook(make_input_hook_save(mid_mlp_name, "quantized"))

print(f"   ✓ Registered {len(layer_names)} input hooks (72 total)")

# Run quantized model
print("   Running quantized W8A8 model forward pass...")
with torch.no_grad():
    _ = quantized_model(input_ids)
print(f"   ✓ Captured and saved {len(layer_names)} activation tensors")

# Free memory
del quantized_model
torch.cuda.empty_cache()

print(f"\n✓ STAGE 1 COMPLETE: All activations saved to {CAPTURE_DIR}/")

# ============================================================================
# Stage 2: Analyze saved activations offline
# ============================================================================
print(f"\n📊 STAGE 2: Analyzing captured activations...")

# Load and compare activations
print("   Loading saved activations and computing errors...")
for name in layer_names:
    orig_path = os.path.join(CAPTURE_DIR, f"original_{name}.pt")
    quant_path = os.path.join(CAPTURE_DIR, f"quantized_{name}.pt")
    
    if not os.path.exists(orig_path) or not os.path.exists(quant_path):
        print(f"     ⚠️  Skipping {name} - files not found")
        continue
    
    # Load from disk
    orig_act = torch.load(orig_path, map_location='cpu')
    quant_act = torch.load(quant_path, map_location='cpu')
    
    # Calculate errors
    diff = (quant_act - orig_act).abs()
    mean_abs_error = diff.mean().item()
    max_abs_error = diff.max().item()
    
    # Relative to activation magnitude
    act_magnitude = orig_act.abs().mean().item()
    relative_error = mean_abs_error / act_magnitude if act_magnitude > 0 else 0
    
    # Elementwise relative error
    elem_rel_errors = diff / (orig_act.abs() + 1e-8)
    avg_elem_rel_error = elem_rel_errors.mean().item()
    
    activation_errors.append({
        'name': name,
        'layer_type': 'attn' if 'attn' in name else 'mlp',
        'layer_num': int(name.split('_')[1]),
        'mean_abs_error': mean_abs_error,
        'max_abs_error': max_abs_error,
        'relative_error': relative_error,
        'avg_elementwise_rel_error': avg_elem_rel_error,
        'orig_min': orig_act.min().item(),
        'orig_max': orig_act.max().item(),
        'quant_min': quant_act.min().item(),
        'quant_max': quant_act.max().item(),
    })

print(f"   ✓ Analyzed {len(activation_errors)} layer outputs")

# Get implied ranges from CURRENT module's calibration
# (since we're now capturing INPUTS, use the SAME layer's act_scale/zp)
print("\n   Extracting implied ranges from current layer's calibration...")
implied_ranges = {}

# Reload quantized model to get calibration parameters
quantized_model_for_params = torch.load('pretrained_models/fla-hub/gla_ptq-w8a8-1.3b/pytorch_model.bin',
                                         map_location='cpu', weights_only=False)

# Map capture points to their corresponding modules
for i in range(24):
    # 1. Attention input - get scale/zp from first Linear in attn
    attn_input_name = f"layer_{i:02d}_attn_input"
    attn_module = quantized_model_for_params.model.layers[i].attn
    
    for submod_name, submod in attn_module.named_modules():
        if isinstance(submod, QuantizedLinear):
            # Found first QuantizedLinear in attention
            scale_val = submod.act_scale.item() if hasattr(submod.act_scale, 'item') else float(submod.act_scale)
            zp_val = submod.act_zero_point.item() if hasattr(submod.act_zero_point, 'item') else float(submod.act_zero_point)
            
            implied_min = (0 - zp_val) * scale_val
            implied_max = (255 - zp_val) * scale_val
            
            implied_ranges[attn_input_name] = {
                'min': implied_min,
                'max': implied_max,
                'span': implied_max - implied_min,
                'scale': scale_val,
                'zero_point': zp_val,
                'calibrated_layer': f"{attn_input_name} (from {submod_name})",
            }
            break
    
    # 2. MLP input - get scale/zp from first Linear in mlp (gate_proj or up_proj)
    mlp_input_name = f"layer_{i:02d}_mlp_input"
    mlp_module = quantized_model_for_params.model.layers[i].mlp
    
    for submod_name, submod in mlp_module.named_modules():
        if isinstance(submod, QuantizedLinear) and ('gate_proj' in submod_name or 'up_proj' in submod_name):
            scale_val = submod.act_scale.item() if hasattr(submod.act_scale, 'item') else float(submod.act_scale)
            zp_val = submod.act_zero_point.item() if hasattr(submod.act_zero_point, 'item') else float(submod.act_zero_point)
            
            implied_min = (0 - zp_val) * scale_val
            implied_max = (255 - zp_val) * scale_val
            
            implied_ranges[mlp_input_name] = {
                'min': implied_min,
                'max': implied_max,
                'span': implied_max - implied_min,
                'scale': scale_val,
                'zero_point': zp_val,
                'calibrated_layer': f"{mlp_input_name} (from {submod_name})",
            }
            break
    
    # 3. Mid-MLP: Input to down_proj
    mid_mlp_name = f"layer_{i:02d}_mid_mlp"
    down_proj = quantized_model_for_params.model.layers[i].mlp.down_proj
    
    if isinstance(down_proj, QuantizedLinear):
        scale_val = down_proj.act_scale.item() if hasattr(down_proj.act_scale, 'item') else float(down_proj.act_scale)
        zp_val = down_proj.act_zero_point.item() if hasattr(down_proj.act_zero_point, 'item') else float(down_proj.act_zero_point)
        
        implied_min = (0 - zp_val) * scale_val
        implied_max = (255 - zp_val) * scale_val
        
        implied_ranges[mid_mlp_name] = {
            'min': implied_min,
            'max': implied_max,
            'span': implied_max - implied_min,
            'scale': scale_val,
            'zero_point': zp_val,
            'calibrated_layer': f"{mid_mlp_name} (down_proj)",
        }

# Debug: Print first 10 to verify
print(f"\n   Sample implied ranges (first 10):")
for idx, (name, ir) in enumerate(list(implied_ranges.items())[:10]):
    print(f"     {name}: scale={ir['scale']:.6f}, zp={ir['zero_point']:.2f}, " +
          f"implied=[{ir['min']:.2f}, {ir['max']:.2f}] (span={ir['span']:.2f})")

del quantized_model_for_params

# Add implied range info to activation_errors
for err in activation_errors:
    if err['name'] in implied_ranges:
        err['implied_min'] = implied_ranges[err['name']]['min']
        err['implied_max'] = implied_ranges[err['name']]['max']
        err['implied_span'] = implied_ranges[err['name']]['span']
        err['calibration_scale'] = implied_ranges[err['name']]['scale']
        err['calibration_zero_point'] = implied_ranges[err['name']]['zero_point']
    else:
        err['implied_min'] = None
        err['implied_max'] = None
        err['implied_span'] = None

print(f"\n   ✓ Extracted implied ranges for {len(implied_ranges)} input points")

# Create activation distribution histograms
print("\n   Creating activation distribution histograms for each layer...")
HIST_DIR = os.path.join(OUTPUT_DIR, "activation_distributions")
os.makedirs(HIST_DIR, exist_ok=True)

for name in layer_names:
    orig_path = os.path.join(CAPTURE_DIR, f"original_{name}.pt")
    quant_path = os.path.join(CAPTURE_DIR, f"quantized_{name}.pt")
    
    if not os.path.exists(orig_path) or not os.path.exists(quant_path):
        continue
    
    # Load activations
    orig_act = torch.load(orig_path, map_location='cpu')
    quant_act = torch.load(quant_path, map_location='cpu')
    
    # Flatten to 1D for histogram
    orig_flat = orig_act.flatten().numpy()
    quant_flat = quant_act.flatten().numpy()
    
    # Check if ranges are similar (within 2x)
    orig_range = orig_flat.max() - orig_flat.min()
    quant_range = quant_flat.max() - quant_flat.min()
    ranges_similar = (max(orig_range, quant_range) / (min(orig_range, quant_range) + 1e-8)) < 2
    
    if ranges_similar:
        # Single combined histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.hist(orig_flat, bins=50, alpha=0.5, label='Original FP32', color='blue', density=True)
        ax.hist(quant_flat, bins=50, alpha=0.5, label='Quantized INT8', color='red', density=True)
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} - Activation Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        textstr = f'Original: μ={orig_flat.mean():.4f}, σ={orig_flat.std():.4f}\n' + \
                  f'Quantized: μ={quant_flat.mean():.4f}, σ={quant_flat.std():.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_path = os.path.join(HIST_DIR, f"{name}_combined.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        # Separate histograms (ranges too different)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Original distribution
        ax1.hist(orig_flat, bins=50, color='blue', alpha=0.7, density=True)
        ax1.set_xlabel('Activation Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{name} - Original FP32 Distribution')
        ax1.grid(True, alpha=0.3)
        textstr1 = f'μ={orig_flat.mean():.4f}, σ={orig_flat.std():.4f}\n' + \
                   f'range=[{orig_flat.min():.2f}, {orig_flat.max():.2f}]'
        ax1.text(0.02, 0.98, textstr1, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Quantized distribution
        ax2.hist(quant_flat, bins=50, color='red', alpha=0.7, density=True)
        ax2.set_xlabel('Activation Value')
        ax2.set_ylabel('Density')
        ax2.set_title(f'{name} - Quantized INT8 Distribution')
        ax2.grid(True, alpha=0.3)
        textstr2 = f'μ={quant_flat.mean():.4f}, σ={quant_flat.std():.4f}\n' + \
                   f'range=[{quant_flat.min():.2f}, {quant_flat.max():.2f}]'
        ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        plt.tight_layout()
        plot_path = os.path.join(HIST_DIR, f"{name}_separate.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

print(f"   ✓ Created {len(layer_names)} activation distribution histograms in {HIST_DIR}/")

# Print ALL 72 activation INPUT errors with implied ranges
print(f"\n📊 ACTIVATION INPUT ERRORS (All {len(activation_errors)} Capture Points):")
print('='*80)
print(f"NOTE: Now capturing INPUTS to layers (where observers measure), not outputs!")
print(f"  - attn_input: Input to attention block (what q_proj sees)")
print(f"  - mlp_input:  Input to MLP block (what gate_proj/up_proj see)")
print(f"  - mid_mlp:    Input to down_proj (post-SwiGLU activation)")
print('='*80)
print(f"{'Capture Point':<25} {'Elem Rel Err':<12} {'Rel Err':<10} {'Orig Range (Input)':<35} {'Quant Range (Input)':<35} {'Implied Range (Calibration)':<40}")
print('-'*170)

for err in activation_errors:
    orig_span = err['orig_max'] - err['orig_min']
    quant_span = err['quant_max'] - err['quant_min']
    
    orig_str = f"[{err['orig_min']:>8.2f}, {err['orig_max']:>8.2f}] ({orig_span:>8.2f})"
    quant_str = f"[{err['quant_min']:>8.2f}, {err['quant_max']:>8.2f}] ({quant_span:>8.2f})"
    
    if err['implied_min'] is not None:
        implied_span = err['implied_span']
        implied_str = f"[{err['implied_min']:>8.2f}, {err['implied_max']:>8.2f}] ({implied_span:>8.2f})"
    else:
        implied_str = "N/A"
    
    print(f"{err['name']:<20} {err['avg_elementwise_rel_error']:>10.4%}  " +
          f"{err['relative_error']:>8.4%}  " +
          f"{orig_str:<35} {quant_str:<35} {implied_str:<40}")

# Save activation error log
if activation_errors:
    rel_errors = [a['relative_error'] for a in activation_errors]
    print(f"\n  Activation Error Statistics:")
    print(f"    Average relative error: {np.mean(rel_errors):.4%}")
    print(f"    Median relative error:  {np.median(rel_errors):.4%}")
    print(f"    Max relative error:     {np.max(rel_errors):.4%}")
    
    # Save detailed log
    act_log_path = os.path.join(OUTPUT_DIR, "activation_errors.txt")
    with open(act_log_path, 'w') as f:
        f.write("ACTIVATION ERRORS (Layer-by-Layer)\n")
        f.write("="*80 + "\n\n")
        for err in activation_errors:
            f.write(f"{err['name']}:\n")
            f.write(f"  Relative error: {err['relative_error']:.4%}\n")
            f.write(f"  Mean abs error: {err['mean_abs_error']:.6f}\n")
            f.write(f"  Max abs error:  {err['max_abs_error']:.6f}\n")
            f.write(f"  Original range: [{err['orig_min']:.4f}, {err['orig_max']:.4f}]\n")
            f.write(f"  Quantized range: [{err['quant_min']:.4f}, {err['quant_max']:.4f}]\n")
            f.write("\n")
    print(f"\n✓ Saved activation error log to: {act_log_path}")
    
    # Save JSON
    act_json_path = os.path.join(OUTPUT_DIR, "activation_errors.json")
    with open(act_json_path, 'w') as f:
        json.dump(activation_errors, f, indent=2)
    print(f"✓ Saved activation error JSON to: {act_json_path}")
    
    # Create activation error plots (3 subplots)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Elementwise relative error progression
    indices = range(len(activation_errors))
    elem_rel_errors = [a['avg_elementwise_rel_error'] * 100 for a in activation_errors]
    colors = ['blue' if a['layer_type'] == 'attn' else 'green' for a in activation_errors]
    
    axes[0].scatter(indices, elem_rel_errors, c=colors, s=40, alpha=0.6)
    axes[0].plot(indices, elem_rel_errors, 'k-', alpha=0.3, linewidth=0.8)
    axes[0].axhline(y=np.mean(elem_rel_errors), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(elem_rel_errors):.2f}%', linewidth=2)
    axes[0].set_xlabel('Layer Output (Attn=Blue, MLP=Green)', fontsize=10)
    axes[0].set_ylabel('Avg Elementwise Relative Error (%)', fontsize=10)
    axes[0].set_title('Activation Error Progression (Layer 0 → Layer 23)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Activation range progression
    orig_ranges = [(a['orig_max'] - a['orig_min']) for a in activation_errors]
    axes[1].plot(indices, orig_ranges, 'b-', linewidth=2, alpha=0.7)
    axes[1].scatter(indices, orig_ranges, c=colors, s=40, alpha=0.6)
    axes[1].set_xlabel('Layer Output', fontsize=10)
    axes[1].set_ylabel('Activation Range (max - min)', fontsize=10)
    axes[1].set_title('Activation Range Progression (shows where values explode)', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Error by layer (attn vs mlp)
    layer_nums = list(range(24))
    attn_errors = [a['avg_elementwise_rel_error'] * 100 for a in activation_errors if a['layer_type'] == 'attn']
    mlp_errors = [a['avg_elementwise_rel_error'] * 100 for a in activation_errors if a['layer_type'] == 'mlp']
    
    axes[2].plot(layer_nums, attn_errors, 'b-o', label='Attention', linewidth=2, markersize=6)
    axes[2].plot(layer_nums, mlp_errors, 'g-s', label='MLP', linewidth=2, markersize=6)
    axes[2].set_xlabel('Layer Number', fontsize=10)
    axes[2].set_ylabel('Avg Elementwise Relative Error (%)', fontsize=10)
    axes[2].set_title('Error by Layer (Attention vs MLP)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    act_plot_path = os.path.join(OUTPUT_DIR, "activation_error_progression.png")
    plt.savefig(act_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved activation error plot to: {act_plot_path}")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("SUMMARY")
print('='*80)

if weight_errors:
    weight_rel_errors = [w['relative_error'] for w in weight_errors]
    print(f"\n📊 Weight Quantization:")
    print(f"   Layers analyzed: {len(weight_errors)}")
    print(f"   Average relative error: {np.mean(weight_rel_errors):.4%}")
    print(f"   → Weights are {'GOOD' if np.mean(weight_rel_errors) < 0.01 else 'ACCEPTABLE' if np.mean(weight_rel_errors) < 0.05 else 'POOR'}")

if activation_errors:
    elem_rel_errors_vals = [a['avg_elementwise_rel_error'] for a in activation_errors]
    rel_errors_vals = [a['relative_error'] for a in activation_errors]
    
    print(f"\n📊 Activation Propagation Statistics:")
    print(f"   Layer outputs analyzed: {len(activation_errors)}")
    print(f"   Average elementwise relative error: {np.mean(elem_rel_errors_vals):.4%}")
    print(f"   Average relative error: {np.mean(rel_errors_vals):.4%}")
    
    # Check if error accumulates
    early_errors = elem_rel_errors_vals[:12]  # First 6 layers (attn+mlp)
    late_errors = elem_rel_errors_vals[-12:]   # Last 6 layers
    print(f"\n   Error Progression:")
    print(f"     Early layers (0-5):   {np.mean(early_errors):.4%}")
    print(f"     Late layers (18-23):  {np.mean(late_errors):.4%}")
    print(f"     Ratio (late/early):   {np.mean(late_errors)/np.mean(early_errors):.2f}x")
    if np.mean(late_errors) > 2 * np.mean(early_errors):
        print(f"     ⚠️  Error ACCUMULATES significantly through the network!")
    else:
        print(f"     ✓ Error remains relatively stable through the network")
    
    # Find worst layers
    sorted_errors = sorted(activation_errors, key=lambda x: x['avg_elementwise_rel_error'], reverse=True)
    print(f"\n   Worst 10 Layer Outputs (highest error):")
    for i, err in enumerate(sorted_errors[:10], 1):
        print(f"     {i}. {err['name']}: {err['avg_elementwise_rel_error']:.4%}, " +
              f"range=[{err['orig_min']:.1f}, {err['orig_max']:.1f}]")
    
    # Find layers with extreme activations
    extreme_layers = sorted(activation_errors, key=lambda x: max(abs(x['orig_min']), abs(x['orig_max'])), reverse=True)
    print(f"\n   Layers with Most Extreme Activations:")
    for i, err in enumerate(extreme_layers[:10], 1):
        print(f"     {i}. {err['name']}: range=[{err['orig_min']:.1f}, {err['orig_max']:.1f}], " +
              f"error={err['avg_elementwise_rel_error']:.4%}")

print(f"\n📁 All outputs saved to: {OUTPUT_DIR}/")
print('='*80)


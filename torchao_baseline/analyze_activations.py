#!/usr/bin/env python3
"""
Stage 2: Analyze captured activations offline.
Loads saved activation tensors and computes error metrics.
"""

import torch
import os
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CAPTURE_DIR = "activation_captures"
OUTPUT_DIR = "quantization_analysis_output"

print("="*80)
print("STAGE 2: ANALYZING CAPTURED ACTIVATIONS")
print("="*80)

# Load metadata
print("\n1. Loading metadata...")
with open(os.path.join(CAPTURE_DIR, "metadata.pkl"), 'rb') as f:
    metadata = pickle.load(f)

layer_names = metadata['layer_names']
original_stats = metadata['original_stats']
quantized_stats = metadata['quantized_stats']

print(f"   ✓ Found {len(layer_names)} layer outputs to analyze")

# Compute errors for each layer
print("\n2. Computing activation errors...")
activation_errors = []

for name in layer_names:
    print(f"   Processing {name}...")
    
    # Load activations from disk
    orig_path = os.path.join(CAPTURE_DIR, f"original_{name}.pt")
    quant_path = os.path.join(CAPTURE_DIR, f"quantized_{name}.pt")
    
    if not os.path.exists(orig_path) or not os.path.exists(quant_path):
        print(f"     ⚠️  Skipping {name} - files not found")
        continue
    
    orig_act = torch.load(orig_path, map_location='cpu')
    quant_act = torch.load(quant_path, map_location='cpu')
    
    # Compute errors
    diff = (quant_act - orig_act).abs()
    mean_abs_error = diff.mean().item()
    max_abs_error = diff.max().item()
    
    # Relative error
    orig_magnitude = orig_act.abs().mean().item()
    relative_error = mean_abs_error / orig_magnitude if orig_magnitude > 0 else 0
    
    # Element-wise relative errors
    elem_rel_errors = (diff / (orig_act.abs() + 1e-8)).cpu()
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
        'orig_mean': orig_act.mean().item(),
        'orig_std': orig_act.std().item(),
        'quant_min': quant_act.min().item(),
        'quant_max': quant_act.max().item(),
        'quant_mean': quant_act.mean().item(),
        'quant_std': quant_act.std().item(),
    })

print(f"\n✓ Analyzed {len(activation_errors)} layer outputs")

# Print all 48 errors to console
print(f"\n{'='*80}")
print(f"ACTIVATION ERRORS (All {len(activation_errors)} Layer Outputs)")
print('='*80)

for err in activation_errors:
    print(f"{err['name']:20s}: relative_error={err['relative_error']:>8.4%}, " +
          f"elem_rel_err={err['avg_elementwise_rel_error']:>8.4%}, " +
          f"orig_range=[{err['orig_min']:>10.2f}, {err['orig_max']:>10.2f}]")

# Save detailed log
print(f"\n3. Saving detailed logs...")
log_path = os.path.join(OUTPUT_DIR, "activation_errors_detailed.txt")
with open(log_path, 'w') as f:
    f.write("ACTIVATION ERRORS (Layer-by-Layer)\n")
    f.write("="*80 + "\n\n")
    for err in activation_errors:
        f.write(f"{err['name']}:\n")
        f.write(f"  Relative error: {err['relative_error']:.4%}\n")
        f.write(f"  Avg elementwise relative error: {err['avg_elementwise_rel_error']:.4%}\n")
        f.write(f"  Mean abs error: {err['mean_abs_error']:.6f}\n")
        f.write(f"  Max abs error:  {err['max_abs_error']:.6f}\n")
        f.write(f"  Original - min: {err['orig_min']:.4f}, max: {err['orig_max']:.4f}, mean: {err['orig_mean']:.4f}, std: {err['orig_std']:.4f}\n")
        f.write(f"  Quantized - min: {err['quant_min']:.4f}, max: {err['quant_max']:.4f}, mean: {err['quant_mean']:.4f}, std: {err['quant_std']:.4f}\n")
        f.write("\n")

print(f"✓ Saved to: {log_path}")

# Save JSON
json_path = os.path.join(OUTPUT_DIR, "activation_errors.json")
with open(json_path, 'w') as f:
    json.dump(activation_errors, f, indent=2)
print(f"✓ Saved to: {json_path}")

# Create visualizations
print("\n4. Creating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Average elementwise relative error progression
indices = range(len(activation_errors))
elem_rel_errors = [a['avg_elementwise_rel_error'] * 100 for a in activation_errors]
colors = ['blue' if a['layer_type'] == 'attn' else 'green' for a in activation_errors]

axes[0].scatter(indices, elem_rel_errors, c=colors, s=40, alpha=0.6)
axes[0].plot(indices, elem_rel_errors, 'k-', alpha=0.3, linewidth=0.8)
axes[0].axhline(y=np.mean(elem_rel_errors), color='r', linestyle='--', 
                label=f'Mean: {np.mean(elem_rel_errors):.2f}%', linewidth=2)
axes[0].set_xlabel('Layer Output (Attn=Blue, MLP=Green)', fontsize=10)
axes[0].set_ylabel('Avg Elementwise Relative Error (%)', fontsize=10)
axes[0].set_title('Activation Error Progression Through Network (Front to Back)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Add vertical lines to separate layers
for i in range(0, len(indices), 2):
    axes[0].axvline(x=i, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

# Plot 2: Activation range comparison
orig_ranges = [(a['orig_max'] - a['orig_min']) for a in activation_errors]
axes[1].plot(indices, orig_ranges, 'b-', label='Original range', linewidth=2, alpha=0.7)
axes[1].scatter(indices, orig_ranges, c=colors, s=40, alpha=0.6)
axes[1].set_xlabel('Layer Output', fontsize=10)
axes[1].set_ylabel('Activation Range (max - min)', fontsize=10)
axes[1].set_title('Activation Range Progression (shows where activations explode)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')  # Log scale to see extreme values
axes[1].legend()

# Plot 3: Error accumulation (compare early vs late layers)
layer_nums = [a['layer_num'] for a in activation_errors if a['layer_type'] == 'attn']
attn_errors = [a['avg_elementwise_rel_error'] * 100 for a in activation_errors if a['layer_type'] == 'attn']
mlp_errors = [a['avg_elementwise_rel_error'] * 100 for a in activation_errors if a['layer_type'] == 'mlp']

axes[2].plot(layer_nums, attn_errors, 'b-o', label='Attention', linewidth=2, markersize=6)
axes[2].plot(layer_nums, mlp_errors, 'g-s', label='MLP', linewidth=2, markersize=6)
axes[2].set_xlabel('Layer Number', fontsize=10)
axes[2].set_ylabel('Avg Elementwise Relative Error (%)', fontsize=10)
axes[2].set_title('Error by Layer Number (Attention vs MLP)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "activation_error_progression.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to: {plot_path}")
plt.close()

# Statistics
print(f"\n{'='*80}")
print("ACTIVATION ERROR STATISTICS")
print('='*80)

elem_rel_errors_arr = np.array(elem_rel_errors)
print(f"\nAverage Elementwise Relative Error:")
print(f"  Mean:   {np.mean(elem_rel_errors_arr):.2f}%")
print(f"  Median: {np.median(elem_rel_errors_arr):.2f}%")
print(f"  Min:    {np.min(elem_rel_errors_arr):.2f}%")
print(f"  Max:    {np.max(elem_rel_errors_arr):.2f}%")

# Check accumulation
early_errors = elem_rel_errors[:12]  # First 6 layers
late_errors = elem_rel_errors[-12:]   # Last 6 layers
print(f"\nError Accumulation:")
print(f"  Early layers (0-5):   {np.mean(early_errors):.2f}%")
print(f"  Late layers (18-23):  {np.mean(late_errors):.2f}%")
print(f"  Ratio (late/early):   {np.mean(late_errors)/np.mean(early_errors):.2f}x")

if np.mean(late_errors) > 2 * np.mean(early_errors):
    print(f"  ⚠️  Error ACCUMULATES significantly through network!")
else:
    print(f"  ✓ Error remains relatively stable")

# Find worst layers
sorted_errors = sorted(activation_errors, key=lambda x: x['avg_elementwise_rel_error'], reverse=True)
print(f"\nWorst 10 Layer Outputs (highest error):")
for i, err in enumerate(sorted_errors[:10], 1):
    print(f"  {i}. {err['name']}: {err['avg_elementwise_rel_error']:.4%}, " +
          f"orig_range=[{err['orig_min']:.1f}, {err['orig_max']:.1f}]")

# Find layers with extreme activations
extreme_layers = sorted(activation_errors, key=lambda x: max(abs(x['orig_min']), abs(x['orig_max'])), reverse=True)
print(f"\nLayers with Most Extreme Activations:")
for i, err in enumerate(extreme_layers[:10], 1):
    print(f"  {i}. {err['name']}: range=[{err['orig_min']:.1f}, {err['orig_max']:.1f}], " +
          f"error={err['avg_elementwise_rel_error']:.4%}")

print(f"\n{'='*80}")
print(f"✅ ANALYSIS COMPLETE")
print('='*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print(f"  - activation_error_progression.png")
print(f"  - activation_errors_detailed.txt")
print(f"  - activation_errors.json")
print('='*80)


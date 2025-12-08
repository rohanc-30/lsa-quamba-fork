#!/usr/bin/env python3
"""
Stage 1: Capture activations from both FP32 and W8A8 models.
Runs models on CUDA normally and saves activations to disk.
"""

import torch
import sys
import os
import pickle

sys.path.insert(0, '/home/rcherukuri/lsa-quamba-fork')

from fla.models.gla import GLAForCausalLM
from transformers import AutoTokenizer

OUTPUT_DIR = "activation_captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("STAGE 1: CAPTURING ACTIVATIONS")
print("="*80)

# Create test input
print("\n1. Creating test input...")
tokenizer = AutoTokenizer.from_pretrained("fla-hub/gla-1.3B-100B")
test_text = "The quick brown fox jumps over the lazy dog. " * 10
test_input = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
input_ids = test_input['input_ids'].cuda()
print(f"   Test input shape: {input_ids.shape}")

# ============================================================================
# CAPTURE FP32 ACTIVATIONS
# ============================================================================
print("\n2. Loading and running FP32 model...")
original_model = GLAForCausalLM.from_pretrained('pretrained_models/fla-hub/gla-1.3b').cuda()
original_model.eval()

original_activations = {}

def make_hook_original(name):
    def hook(module, input, output):
        # Save to CPU and to disk immediately to save memory
        act_cpu = output.detach().cpu() if hasattr(output, 'detach') else output
        save_path = os.path.join(OUTPUT_DIR, f"original_{name}.pt")
        torch.save(act_cpu, save_path)
        # Also keep in dict for summary
        original_activations[name] = {
            'shape': act_cpu.shape,
            'min': act_cpu.min().item(),
            'max': act_cpu.max().item(),
            'mean': act_cpu.mean().item(),
            'std': act_cpu.std().item(),
        }
    return hook

# Register hooks
layer_names = []
for i in range(24):  # 24 layers
    attn_name = f"layer_{i:02d}_attn"
    mlp_name = f"layer_{i:02d}_mlp"
    
    original_model.model.layers[i].attn.register_forward_hook(make_hook_original(attn_name))
    original_model.model.layers[i].mlp.register_forward_hook(make_hook_original(mlp_name))
    
    layer_names.append(attn_name)
    layer_names.append(mlp_name)

print(f"   Registered {len(layer_names)} hooks")

# Run forward pass
print("   Running forward pass...")
with torch.no_grad():
    _ = original_model(input_ids)

print(f"   ✓ Captured {len(original_activations)} activation tensors")
print(f"   ✓ Saved to {OUTPUT_DIR}/original_*.pt")

# Free memory
del original_model
torch.cuda.empty_cache()

# ============================================================================
# CAPTURE W8A8 ACTIVATIONS
# ============================================================================
print("\n3. Loading and running W8A8 quantized model...")
quantized_model = torch.load('pretrained_models/fla-hub/gla_ptq-w8a8-1.3b/pytorch_model.bin', 
                              map_location='cuda', weights_only=False)
quantized_model.eval()

quantized_activations = {}

def make_hook_quantized(name):
    def hook(module, input, output):
        # Save to CPU and to disk immediately
        act_cpu = output.detach().cpu() if hasattr(output, 'detach') else output
        save_path = os.path.join(OUTPUT_DIR, f"quantized_{name}.pt")
        torch.save(act_cpu, save_path)
        # Also keep stats in dict
        quantized_activations[name] = {
            'shape': act_cpu.shape,
            'min': act_cpu.min().item(),
            'max': act_cpu.max().item(),
            'mean': act_cpu.mean().item(),
            'std': act_cpu.std().item(),
        }
    return hook

# Register hooks
for i in range(24):
    attn_name = f"layer_{i:02d}_attn"
    mlp_name = f"layer_{i:02d}_mlp"
    
    quantized_model.model.layers[i].attn.register_forward_hook(make_hook_quantized(attn_name))
    quantized_model.model.layers[i].mlp.register_forward_hook(make_hook_quantized(mlp_name))

print(f"   Registered {len(layer_names)} hooks")

# Run forward pass
print("   Running forward pass...")
with torch.no_grad():
    _ = quantized_model(input_ids)

print(f"   ✓ Captured {len(quantized_activations)} activation tensors")
print(f"   ✓ Saved to {OUTPUT_DIR}/quantized_*.pt")

# Save metadata
print("\n4. Saving metadata...")
metadata = {
    'layer_names': layer_names,
    'num_layers': len(layer_names),
    'original_stats': original_activations,
    'quantized_stats': quantized_activations,
}

with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), 'wb') as f:
    pickle.dump(metadata, f)

print(f"   ✓ Saved metadata to {OUTPUT_DIR}/metadata.pkl")

print(f"\n{'='*80}")
print("✅ ACTIVATION CAPTURE COMPLETE")
print('='*80)
print(f"\nCaptured files:")
print(f"  - {len(layer_names)} original activation tensors")
print(f"  - {len(layer_names)} quantized activation tensors")
print(f"  - metadata.pkl with statistics")
print(f"\nNext step: Run analyze_activations.py to compute errors and generate plots")
print('='*80)


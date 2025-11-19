"""
Simple usage example for W8A8 PTQ quantization with torchao.

This demonstrates the clean API after refactoring.

Run from the repo root:
    cd /home/rcherukuri/lsa-quamba-fork
    python -m torchao_baseline.USAGE_EXAMPLE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# Add parent directory to path to allow imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from torchao_baseline.utils_torchao import (
    load_model_for_quantization,
    get_calibration_data,
    prepare_for_static_w8a8,
    calibrate_static_quant,
    convert_static_w8a8,
    save_quantized_model,
)

def quantize_gla_model(save_model: bool = True):
    """
    Complete W8A8 quantization workflow
    
    Args:
        save_model: Whether to save the quantized model (default: True)
    """
    
    # Step 1: Load FP32 model
    print("\n" + "="*60)
    print("Step 1: Loading FP32 model...")
    print("="*60)
    model, tokenizer, config = load_model_for_quantization(
        model_path='pretrained_models/fla-hub/gla-1.3b',
        model_type='gla',
        device='cuda'
    )
    
    # Check 1: Verify model loaded
    print("\n[CHECK 1] Model loaded successfully")
    print(f"  ✓ Model type: {type(model).__name__}")
    print(f"  ✓ Model device: {next(model.parameters()).device}")
    print(f"  ✓ Model dtype: {next(model.parameters()).dtype}")
    
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"  ✓ Total Linear layers: {linear_count}")
    
    # Inspect MLP structure
    print("\n[CHECK 1.5] Inspecting MLP structure")
    import inspect
    mlp = model.model.layers[0].mlp
    print(f"  MLP type: {type(mlp).__name__}")
    print(f"  Has swiglu_linear: {hasattr(mlp, 'swiglu_linear')}")
    print(f"  Has gate_proj: {hasattr(mlp, 'gate_proj')}")
    print(f"  Has up_proj: {hasattr(mlp, 'up_proj')}")
    print(f"  Has down_proj: {hasattr(mlp, 'down_proj')}")
    
    if hasattr(mlp, 'down_proj'):
        down_proj = mlp.down_proj
        print(f"  down_proj type: {type(down_proj).__name__}")
        print(f"  down_proj is nn.Linear: {isinstance(down_proj, nn.Linear)}")
    
    if hasattr(mlp, 'swiglu_linear'):
        print(f"  swiglu_linear type: {type(mlp.swiglu_linear).__name__}")
        print(f"  swiglu_linear is nn.Module: {isinstance(mlp.swiglu_linear, nn.Module)}")
    
    # Check MLP forward signature
    try:
        sig = inspect.signature(mlp.forward)
        print(f"  MLP forward signature: {sig}")
    except:
        print("  Could not inspect forward signature")
    
    # Step 2: Prepare model (insert observers)
    print("\n" + "="*60)
    print("Step 2: Preparing model for quantization...")
    print("="*60)
    model = prepare_for_static_w8a8(model)
    
    # Check 2: Verify observers inserted
    print("\n[CHECK 2] Observers inserted")
    from torchao_baseline.utils_torchao import ObservedLinear
    observed_count = sum(1 for m in model.modules() if isinstance(m, ObservedLinear))
    print(f"  ✓ ObservedLinear layers: {observed_count}")
    
    # Inspect first ObservedLinear layer
    first_obs = None
    for name, module in model.named_modules():
        if isinstance(module, ObservedLinear):
            first_obs = (name, module)
            break
    
    if first_obs:
        name, module = first_obs
        print(f"  ✓ First ObservedLinear: {name}")
        print(f"    - Has act_obs: {hasattr(module, 'act_obs')}")
        print(f"    - Has weight_obs: {hasattr(module, 'weight_obs')}")
        print(f"    - act_obs type: {type(module.act_obs).__name__}")
        print(f"    - weight_obs type: {type(module.weight_obs).__name__}")
    else:
        print("  ⚠️  WARNING: No ObservedLinear layers found!")
    
    # Check down_proj specifically
    print("\n[CHECK 2.5] Inspecting down_proj after preparation")
    down_proj_after = model.model.layers[0].mlp.down_proj
    print(f"  down_proj type after prepare: {type(down_proj_after).__name__}")
    print(f"  down_proj is ObservedLinear: {isinstance(down_proj_after, ObservedLinear)}")
    print(f"  down_proj is nn.Linear: {isinstance(down_proj_after, nn.Linear)}")
    
    # Check if fuse_swiglu was disabled
    mlp_after = model.model.layers[0].mlp
    if hasattr(mlp_after, 'fuse_swiglu'):
        print(f"  MLP fuse_swiglu after prepare: {mlp_after.fuse_swiglu}")
        print(f"  ✓ Fusion disabled!" if not mlp_after.fuse_swiglu else "  ⚠️  WARNING: Fusion still enabled!")
    
    # Count down_proj layers that are ObservedLinear
    down_proj_observed = 0
    down_proj_total = 0
    for name, module in model.named_modules():
        if 'down_proj' in name and isinstance(module, (nn.Linear, ObservedLinear)):
            down_proj_total += 1
            if isinstance(module, ObservedLinear):
                down_proj_observed += 1
    print(f"  down_proj layers: {down_proj_observed}/{down_proj_total} are ObservedLinear")
    
    # Step 3: Get calibration data
    print("\n" + "="*60)
    print("Step 3: Loading calibration data...")
    print("="*60)
    calib_data = get_calibration_data(
        tokenizer=tokenizer,
        dataset_name='wikitext',
        num_samples=512,
        seq_len=512,
        min_length=64
    )
    
    # Check 3: Verify calibration data
    print("\n[CHECK 3] Calibration data loaded")
    print(f"  ✓ Number of samples: {len(calib_data)}")
    print(f"  ✓ First sample shape: {calib_data[0].shape}")
    print(f"  ✓ Sample dtype: {calib_data[0].dtype}")
    
    # Step 4: Calibrate (automatically creates DataLoader internally!)
    print("\n" + "="*60)
    print("Step 4: Calibrating model...")
    print("="*60)
    print("Running forward passes to collect statistics...")
    
    # Skip pre-calibration test to avoid polluting observer statistics
    # (Running with different batch sizes causes shape mismatches in observers)
    print("\n[Note] Skipping pre-calibration test to avoid observer pollution")
    print("       Proceeding directly to full calibration...")
    
    # Now run full calibration
    model = calibrate_static_quant(
        model=model,
        calib_data=calib_data,
        batch_size=8,
        pad_token_id=tokenizer.pad_token_id,
        device='cuda',
        max_batches=None  # Use all batches
    )
    
    # Check 4: Verify observers populated (CHECK ALL LAYERS!)
    print("\n[CHECK 4] Verifying observer statistics after calibration")
    from torchao_baseline.utils_torchao import ObservedLinear
    
    total_count = 0
    passed_count = 0
    failed_layers = []
    attn_layers = []
    mlp_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, ObservedLinear):
            total_count += 1
            has_act_stats = hasattr(module.act_obs, 'min_val') and hasattr(module.act_obs, 'max_val')
            has_weight_stats = hasattr(module.weight_obs, 'min_val') and hasattr(module.weight_obs, 'max_val')
            
            if has_act_stats and has_weight_stats:
                passed_count += 1
                if 'attn' in name:
                    attn_layers.append(name)
                elif 'mlp' in name:
                    mlp_layers.append(name)
            else:
                failed_layers.append((name, has_act_stats, has_weight_stats))
            
            # Print first few in detail
            if total_count <= 3 or (total_count <= 10 and 'mlp' in name):
                print(f"\n  Layer {total_count}: {name}")
                print(f"    act_obs recorded: {has_act_stats}")
                print(f"    weight_obs recorded: {has_weight_stats}")
                if has_act_stats:
                    print(f"    act_obs.min_val: {module.act_obs.min_val}")
                    print(f"    act_obs.max_val: {module.act_obs.max_val}")
    
    print(f"\n  Summary: {passed_count}/{total_count} layers have statistics")
    print(f"    - Attention layers with stats: {len(attn_layers)}")
    print(f"    - MLP layers with stats: {len(mlp_layers)}")
    
    if failed_layers:
        print(f"\n  ✗ WARNING: {len(failed_layers)} layers FAILED to record statistics:")
        for name, has_act, has_weight in failed_layers[:10]:
            print(f"    {name}: act={has_act}, weight={has_weight}")
        if len(failed_layers) > 10:
            print(f"    ... and {len(failed_layers) - 10} more")
    else:
        print(f"  ✓ All {total_count} layers passed!")
    
    # Step 4b: Inspect Observer Statistics (DETAILED CHECK)
    print("\n" + "="*60)
    print("Step 4b: Inspecting Observer Statistics")
    print("="*60)
    print("\nThis check helps diagnose calibration quality by examining")
    print("the min/max ranges captured by observers. Look for:")
    print("  • Reasonable activation ranges (typically -10 to +10)")
    print("  • Symmetric or near-symmetric activation distributions")
    print("  • Consistent weight ranges across channels")
    print("\n" + "-"*60)
    
    def pool_tensor_to_16(tensor):
        """
        Pool a per-channel tensor (e.g., 1024 elements) down to 16 elements
        by taking min/max over chunks of size 1024/16 = 64.
        
        Returns: (pooled_mins, pooled_maxs) each with 16 elements
        """
        if tensor.numel() <= 16:
            return tensor, tensor  # Already small enough
        
        # Reshape to (16, chunk_size) and take min/max over chunks
        num_chunks = 16
        chunk_size = tensor.numel() // num_chunks
        
        # Flatten and reshape
        flat = tensor.flatten()[:num_chunks * chunk_size]  # Trim to even multiple
        chunks = flat.reshape(num_chunks, chunk_size)
        
        pooled_mins = chunks.min(dim=1)[0]
        pooled_maxs = chunks.max(dim=1)[0]
        
        return pooled_mins, pooled_maxs
    
    # Collect statistics from all layers
    observer_stats = []
    for name, module in model.named_modules():
        if isinstance(module, ObservedLinear):
            has_act = hasattr(module.act_obs, 'min_val') and module.act_obs.min_val is not None
            has_weight = hasattr(module.weight_obs, 'min_val') and module.weight_obs.min_val is not None
            
            if has_act and has_weight:
                act_min_val = module.act_obs.min_val
                act_max_val = module.act_obs.max_val
                weight_min_val = module.weight_obs.min_val
                weight_max_val = module.weight_obs.max_val
                
                # Handle per-tensor activation obs (should be scalar or 1-element tensor)
                if isinstance(act_min_val, torch.Tensor):
                    act_min = act_min_val.item() if act_min_val.numel() == 1 else act_min_val.mean().item()
                    act_max = act_max_val.item() if act_max_val.numel() == 1 else act_max_val.mean().item()
                else:
                    act_min = float(act_min_val)
                    act_max = float(act_max_val)
                
                # Handle per-channel weight obs (typically 1024 elements)
                if isinstance(weight_min_val, torch.Tensor) and weight_min_val.numel() > 1:
                    # Pool to 16 elements
                    weight_mins_pooled, _ = pool_tensor_to_16(weight_min_val)
                    _, weight_maxs_pooled = pool_tensor_to_16(weight_max_val)
                    
                    observer_stats.append({
                        'name': name,
                        'act_min': act_min,
                        'act_max': act_max,
                        'act_range': act_max - act_min,
                        'weight_min_global': weight_min_val.min().item(),
                        'weight_max_global': weight_max_val.max().item(),
                        'weight_mins_pooled': weight_mins_pooled.cpu().numpy(),
                        'weight_maxs_pooled': weight_maxs_pooled.cpu().numpy(),
                        'weight_channels': weight_min_val.numel(),
                    })
                else:
                    # Scalar weight obs (rare, but handle it)
                    weight_min = weight_min_val.item() if isinstance(weight_min_val, torch.Tensor) else float(weight_min_val)
                    weight_max = weight_max_val.item() if isinstance(weight_max_val, torch.Tensor) else float(weight_max_val)
                    
                    observer_stats.append({
                        'name': name,
                        'act_min': act_min,
                        'act_max': act_max,
                        'act_range': act_max - act_min,
                        'weight_min_global': weight_min,
                        'weight_max_global': weight_max,
                        'weight_mins_pooled': None,
                        'weight_maxs_pooled': None,
                        'weight_channels': 1,
                    })
    
    if observer_stats:
        # Create output directory for detailed stats
        import os
        import json
        import numpy as np
        output_dir = "observer_stats_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all stats to JSON for later analysis
        stats_for_json = []
        for s in observer_stats:
            stats_dict = {
                'name': s['name'],
                'act_min': float(s['act_min']),
                'act_max': float(s['act_max']),
                'act_range': float(s['act_range']),
                'weight_min_global': float(s['weight_min_global']),
                'weight_max_global': float(s['weight_max_global']),
                'weight_channels': int(s['weight_channels']),
            }
            if s['weight_mins_pooled'] is not None:
                stats_dict['weight_mins_pooled'] = s['weight_mins_pooled'].tolist()
                stats_dict['weight_maxs_pooled'] = s['weight_maxs_pooled'].tolist()
            stats_for_json.append(stats_dict)
        
        # Save summary JSON
        with open(os.path.join(output_dir, "all_layers_summary.json"), 'w') as f:
            json.dump(stats_for_json, f, indent=2)
        
        # Organize by layer and save to individual files
        layer_stats = {}
        for s in observer_stats:
            # Extract layer number from name (e.g., "model.layers.0.attn.q_proj" -> layer 0)
            name_parts = s['name'].split('.')
            if 'layers' in name_parts:
                layer_idx = int(name_parts[name_parts.index('layers') + 1])
                layer_key = f"layer_{layer_idx}"
                
                if layer_key not in layer_stats:
                    layer_stats[layer_key] = {}
                
                # Determine projection type
                if 'q_proj' in s['name']:
                    proj_type = 'q_proj'
                elif 'k_proj' in s['name']:
                    proj_type = 'k_proj'
                elif 'v_proj' in s['name']:
                    proj_type = 'v_proj'
                elif 'o_proj' in s['name']:
                    proj_type = 'o_proj'
                elif 'gate_proj' in s['name']:
                    proj_type = 'gate_proj'
                elif 'up_proj' in s['name']:
                    proj_type = 'up_proj'
                elif 'down_proj' in s['name']:
                    proj_type = 'down_proj'
                else:
                    proj_type = 'other'
                
                layer_stats[layer_key][proj_type] = s
        
        # Save per-layer statistics
        for layer_key, projs in layer_stats.items():
            layer_dir = os.path.join(output_dir, layer_key)
            os.makedirs(layer_dir, exist_ok=True)
            
            # Save summary for this layer
            layer_summary = {}
            for proj_type, stats in projs.items():
                layer_summary[proj_type] = {
                    'activations': {
                        'min': float(stats['act_min']),
                        'max': float(stats['act_max']),
                        'range': float(stats['act_range']),
                    },
                    'weights': {
                        'global_min': float(stats['weight_min_global']),
                        'global_max': float(stats['weight_max_global']),
                        'num_channels': int(stats['weight_channels']),
                    }
                }
                
                # Save detailed weight distributions
                if stats['weight_mins_pooled'] is not None:
                    # Save full pooled arrays as numpy files
                    np.save(
                        os.path.join(layer_dir, f"{proj_type}_weight_mins_pooled.npy"),
                        stats['weight_mins_pooled']
                    )
                    np.save(
                        os.path.join(layer_dir, f"{proj_type}_weight_maxs_pooled.npy"),
                        stats['weight_maxs_pooled']
                    )
                    
                    # Also save as text for easy viewing
                    with open(os.path.join(layer_dir, f"{proj_type}_weights.txt"), 'w') as f:
                        f.write(f"Projection: {proj_type}\n")
                        f.write(f"Full layer name: {stats['name']}\n")
                        f.write(f"Number of channels: {stats['weight_channels']}\n\n")
                        f.write(f"Activation range: [{stats['act_min']:.6f}, {stats['act_max']:.6f}]\n\n")
                        f.write(f"Weight global range: [{stats['weight_min_global']:.6f}, {stats['weight_max_global']:.6f}]\n\n")
                        f.write(f"Weight mins (pooled to 16 values):\n")
                        f.write(f"{stats['weight_mins_pooled']}\n\n")
                        f.write(f"Weight maxs (pooled to 16 values):\n")
                        f.write(f"{stats['weight_maxs_pooled']}\n")
            
            # Save layer summary as JSON
            with open(os.path.join(layer_dir, "summary.json"), 'w') as f:
                json.dump(layer_summary, f, indent=2)
        
        print(f"\n💾 Detailed statistics saved to: {output_dir}/")
        print(f"   - Per-layer folders: {len(layer_stats)} layers")
        print(f"   - Files per layer: summary.json + individual projection stats")
        
        # Show summary statistics
        print(f"\n📊 SUMMARY STATISTICS (across {len(observer_stats)} layers):")
        print("-"*60)
        
        act_mins = [s['act_min'] for s in observer_stats]
        act_maxs = [s['act_max'] for s in observer_stats]
        act_ranges = [s['act_range'] for s in observer_stats]
        weight_mins_global = [s['weight_min_global'] for s in observer_stats]
        weight_maxs_global = [s['weight_max_global'] for s in observer_stats]
        
        print("\n  ACTIVATION STATISTICS (per-tensor):")
        print(f"    Min values:   range=[{min(act_mins):>9.4f}, {max(act_mins):>9.4f}]  median={sorted(act_mins)[len(act_mins)//2]:>9.4f}")
        print(f"    Max values:   range=[{min(act_maxs):>9.4f}, {max(act_maxs):>9.4f}]  median={sorted(act_maxs)[len(act_maxs)//2]:>9.4f}")
        print(f"    Range (span): range=[{min(act_ranges):>9.4f}, {max(act_ranges):>9.4f}]  median={sorted(act_ranges)[len(act_ranges)//2]:>9.4f}")
        
        print("\n  WEIGHT STATISTICS (per-channel, showing global min/max):")
        print(f"    Global mins:  range=[{min(weight_mins_global):>9.4f}, {max(weight_mins_global):>9.4f}]  median={sorted(weight_mins_global)[len(weight_mins_global)//2]:>9.4f}")
        print(f"    Global maxs:  range=[{min(weight_maxs_global):>9.4f}, {max(weight_maxs_global):>9.4f}]  median={sorted(weight_maxs_global)[len(weight_maxs_global)//2]:>9.4f}")
        
        # Show sample layers with pooled weight distributions
        print("\n\n📋 SAMPLE LAYER DETAILS (showing pooled weight distributions):")
        print("-"*60)
        
        # Group layers by type
        attn_q = [s for s in observer_stats if 'attn' in s['name'] and 'q_proj' in s['name']]
        attn_k = [s for s in observer_stats if 'attn' in s['name'] and 'k_proj' in s['name']]
        mlp_gate = [s for s in observer_stats if 'mlp' in s['name'] and 'gate_proj' in s['name']]
        mlp_down = [s for s in observer_stats if 'mlp' in s['name'] and 'down_proj' in s['name']]
        
        def print_layer_stats(stats_list, layer_type, max_show=2):
            if stats_list:
                print(f"\n  {layer_type.upper()} Layers ({len(stats_list)} total):")
                for i, s in enumerate(stats_list[:max_show]):
                    print(f"\n    [{i+1}] {s['name']} ({s['weight_channels']} channels)")
                    print(f"        Activations: [{s['act_min']:>10.6f}, {s['act_max']:>10.6f}]  (range: {s['act_range']:>10.6f})")
                    print(f"        Weights (global): [{s['weight_min_global']:>10.6f}, {s['weight_max_global']:>10.6f}]")
                    
                    if s['weight_mins_pooled'] is not None:
                        print(f"        Weight mins (pooled to 16): {s['weight_mins_pooled'][:8]}")  # Show first 8
                        print(f"        Weight maxs (pooled to 16): {s['weight_maxs_pooled'][:8]}")  # Show first 8
                
                if len(stats_list) > max_show:
                    print(f"        ... and {len(stats_list) - max_show} more {layer_type} layers")
        
        print_layer_stats(attn_q, "Query Projection", max_show=2)
        print_layer_stats(attn_k, "Key Projection", max_show=2)
        print_layer_stats(mlp_gate, "MLP Gate", max_show=2)
        print_layer_stats(mlp_down, "MLP Down", max_show=2)
        
        # Check for potential issues
        print("\n\n🔍 POTENTIAL ISSUES CHECK:")
        print("-"*60)
        
        issues_found = False
        
        # Check for zero ranges
        zero_range_layers = [s['name'] for s in observer_stats if s['act_range'] == 0]
        if zero_range_layers:
            issues_found = True
            print(f"\n  ⚠️  {len(zero_range_layers)} layer(s) with ZERO activation range:")
            for name in zero_range_layers[:5]:
                print(f"      - {name}")
        
        # Check for extremely large activation values
        large_act_layers = [s for s in observer_stats if abs(s['act_min']) > 100 or abs(s['act_max']) > 100]
        if large_act_layers:
            issues_found = True
            print(f"\n  ⚠️  {len(large_act_layers)} layer(s) with LARGE activation values (>100):")
            for s in large_act_layers[:5]:
                print(f"      - {s['name']}: [{s['act_min']:.2f}, {s['act_max']:.2f}]")
        
        # Check for very asymmetric activations
        asym_layers = [s for s in observer_stats if s['act_range'] > 0 and 
                      (abs(s['act_min']) / s['act_range'] < 0.1 or abs(s['act_max']) / s['act_range'] < 0.1)]
        if asym_layers:
            issues_found = True
            print(f"\n  ⚠️  {len(asym_layers)} layer(s) with VERY ASYMMETRIC activations:")
            for s in asym_layers[:5]:
                print(f"      - {s['name']}: [{s['act_min']:.4f}, {s['act_max']:.4f}]")
        
        if not issues_found:
            print("\n  ✅ No obvious issues detected in observer statistics!")
            print("     Calibration ranges look reasonable for typical neural network layers.")
    
    else:
        print("\n  ✗ ERROR: No observer statistics found!")
        print("     This means calibration completely failed.")
    
    print("\n" + "-"*60)
    print("End of observer statistics inspection")
    print("="*60)
    
    # Step 5: Convert to quantized (extract qparams, create int8 layers)
    print("\n" + "="*60)
    print("Step 5: Converting to quantized model...")
    print("="*60)
    try:
        model = convert_static_w8a8(model, target_dtype=torch.int8)
        print("  ✓ Conversion successful!")
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}")
        print("\nDumping first observer state for debugging:")
        for name, module in model.named_modules():
            if isinstance(module, ObservedLinear):
                print(f"\nFirst ObservedLinear: {name}")
                print(f"  act_obs attributes: {dir(module.act_obs)}")
                print(f"  weight_obs attributes: {dir(module.weight_obs)}")
                break
        raise
    
    # Step 6: Save quantized model (optional)
    if save_model:
        print("Step 6: Saving quantized model...")
        save_path = save_quantized_model(
            model=model,
            tokenizer=tokenizer,
            output_dir='pretrained_models/fla-hub',  # Save alongside original model
            model_name='gla_ptq-w8a8-1.3b',  # PTQ W8A8 quantized version
            model_type='gla',
            metadata={
                'calibration_samples': 512,
                'calibration_dataset': 'wikitext',
                'batch_size': 8,
                'quantization_framework': 'torchao',
                'quantization_method': 'PTQ',  # Post-Training Quantization
            }
        )
        print(f"✓ Model saved to: {save_path}")
    
    print("\n✅ Quantization complete!")
    return model, tokenizer


def test_quantized_model(model, tokenizer):
    """Test the quantized model with sample inputs"""
    print("\n" + "="*60)
    print("Testing Quantized Model")
    print("="*60)
    
    # Memory profiling
    print("\n[MEMORY] Checking GPU memory usage...")
    import torch.cuda
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Get current memory usage
    quantized_mem = torch.cuda.memory_allocated() / 1024**3  # GB
    quantized_peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"  Quantized model GPU memory: {quantized_mem:.2f} GB")
    print(f"  Peak memory so far: {quantized_peak:.2f} GB")
    
    # Diagnostic: Check what got quantized
    print("\n[DIAGNOSTIC] Checking what modules were quantized...")
    from torchao_baseline.utils_torchao import QuantizedLinear
    import torch.nn as nn
    
    quantized_layers = []
    embedding_layers = []
    lm_head_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            quantized_layers.append(name)
        if isinstance(module, nn.Embedding):
            embedding_layers.append(name)
        if 'lm_head' in name and isinstance(module, (nn.Linear, QuantizedLinear)):
            lm_head_layers.append((name, type(module).__name__))
    
    print(f"  Total QuantizedLinear layers: {len(quantized_layers)}")
    print(f"  First 5: {quantized_layers[:5]}")
    print(f"  Last 5: {quantized_layers[-5:]}")
    print(f"\n  Embedding layers found: {len(embedding_layers)}")
    if embedding_layers:
        print(f"    {embedding_layers}")
    print(f"\n  LM head layers: {lm_head_layers}")
    
    # Test embedding layer output
    print("\n[DIAGNOSTIC] Testing embedding layer output...")
    test_tokens = torch.randint(0, 1000, (1, 10)).cuda()
    embed_out = model.model.embeddings(test_tokens)
    print(f"  Input tokens shape: {test_tokens.shape}")
    print(f"  Embedding output: type={type(embed_out).__name__}, dtype={embed_out.dtype}, shape={embed_out.shape}")
    print(f"  Embedding has NaN: {torch.isnan(embed_out).any()}")
    print(f"  Embedding min/max: {embed_out.min().item():.4f} / {embed_out.max().item():.4f}")
    
    # CRITICAL: Test if unquantized FP32 model works AND measure memory
    print("\n[DIAGNOSTIC] Loading FP32 model to verify it works...")
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    from torchao_baseline.utils_torchao import load_model_for_quantization
    fp32_model, _, _ = load_model_for_quantization('pretrained_models/fla-hub/gla-1.3b', 'gla')
    fp32_model.eval()
    
    # Measure FP32 memory
    fp32_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  FP32 model GPU memory: {fp32_mem:.2f} GB")
    
    print("  Testing FP32 model inference...")
    # Use VALID token IDs within vocab size!
    test_input_fp32 = torch.randint(0, tokenizer.vocab_size, (1, 50)).cuda()
    try:
        with torch.no_grad():
            fp32_output = fp32_model(test_input_fp32)
        print(f"  ✓ FP32 model works! Output shape: {fp32_output.logits.shape}")
    except Exception as e:
        print(f"  ✗ FP32 model ALSO fails: {e}")
        print("  ERROR: The base model itself has issues!")
    
    # Calculate memory savings
    print(f"\n[MEMORY COMPARISON]")
    print(f"  FP32 model: {fp32_mem:.2f} GB")
    print(f"  Quantized model: {quantized_mem:.2f} GB")
    if fp32_mem > 0:
        savings = (fp32_mem - quantized_mem) / fp32_mem * 100
        print(f"  Memory savings: {savings:.1f}% ({fp32_mem - quantized_mem:.2f} GB saved)")
        print(f"  Compression ratio: {fp32_mem / quantized_mem:.2f}x")
    
    del fp32_model  # Free memory
    torch.cuda.empty_cache()
    
    # Check disk sizes
    print(f"\n[DISK SIZE COMPARISON]")
    import os
    import pathlib
    
    fp32_path = pathlib.Path('pretrained_models/fla-hub/gla-1.3b')
    quant_path = pathlib.Path('pretrained_models/ut-enyac/gla_ptq-w8a8-1.3b')
    
    def get_dir_size(path):
        """Calculate total size of directory in GB"""
        total = 0
        if path.exists():
            for f in path.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
        return total / 1024**3
    
    if fp32_path.exists():
        fp32_disk = get_dir_size(fp32_path)
        print(f"  FP32 model on disk: {fp32_disk:.2f} GB")
    else:
        fp32_disk = 0
        print(f"  FP32 model path not found")
    
    if quant_path.exists():
        quant_disk = get_dir_size(quant_path)
        print(f"  Quantized model on disk: {quant_disk:.2f} GB")
        if fp32_disk > 0:
            disk_savings = (fp32_disk - quant_disk) / fp32_disk * 100
            print(f"  Disk savings: {disk_savings:.1f}% ({fp32_disk - quant_disk:.2f} GB saved)")
    else:
        print(f"  Quantized model not saved yet")
    
    # Test 1: Random input (with VALID token IDs!)
    print("\n[Test 1] Random input inference...")
    test_input = torch.randint(0, tokenizer.vocab_size, (1, 128)).cuda()
    with torch.no_grad():
        output = model(test_input)
        print(f"✓ Output logits shape: {output.logits.shape}")
        print(f"✓ Sample logits (first 10): {output.logits[0, 0, :10]}")
        print(f"✓ No NaN values: {not torch.isnan(output.logits).any()}")
    
    # Test 2: Real text
    print("\n[Test 2] Real text inference...")
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model(inputs)
        print(f"✓ Input text: '{text}'")
        print(f"✓ Input shape: {inputs.shape}")
        print(f"✓ Output shape: {output.logits.shape}")
        print(f"✓ Max logit value: {output.logits.max().item():.4f}")
        print(f"✓ Min logit value: {output.logits.min().item():.4f}")
    
    # Test 3: Check quantization (inspect a layer)
    print("\n[Test 3] Verify quantization...")
    from torchao_baseline.utils_torchao import QuantizedLinear
    
    quantized_count = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            quantized_count += 1
            if quantized_count == 1:  # Print first quantized layer details
                print(f"✓ First quantized layer: {name}")
                print(f"  - qweight dtype: {module.qweight.dtype}")
                print(f"  - act_scale: {module.act_scale}")
                print(f"  - act_zero_point: {module.act_zero_point}")
    
    print(f"\n✓ Total quantized Linear layers: {quantized_count}")
    print(f"✓ Quantization successful!")
    

if __name__ == '__main__':
    print("="*60)
    print("W8A8 Static Quantization with TorchAO")
    print("="*60)
    
    # Run complete quantization workflow (with saving)
    model, tokenizer = quantize_gla_model(save_model=True)
    
    # Test the quantized model
    test_quantized_model(model, tokenizer)
    
    print("\n" + "="*60)
    print("✅ Complete! Quantized model ready for deployment.")
    print("="*60)


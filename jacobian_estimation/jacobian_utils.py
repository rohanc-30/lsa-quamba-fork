"""
Utilities for Jacobian estimation via JVP (Jacobian-Vector Product) sampling.

This module provides functionality to estimate Jacobians for Mamba models
using forward-mode automatic differentiation and sampling techniques.
"""

import os
import gc
import logging
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm

# Import from quamba modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quamba.gptq_utils import GPTQ
from quamba.data_loaders import get_loaders

# Import our own Jacobian estimator
from jacobian_estimation.jvp_estimator import MambaJacobianEstimator


logger = logging.getLogger(__name__)


def save_jacobian_samples(model, tokenizer, device, w_bits=4, model_type="mamba", 
                         nsamples=128, seqlen=1024, output_dir="./jacobian_jvp_error"):
    """
    Compute and save Jacobian samples for Mamba model layers using JVP estimation.

    This function performs forward passes on calibration data and captures
    Jacobian information at each layer using the SMGPTQ (State-space Model GPTQ)
    approach. The results are saved to disk for analysis.

    Parameters:
    -----------
    model : torch.nn.Module
        The Mamba model to analyze
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for creating calibration text data
    device : str or torch.device
        Processing device (CPU/GPU)
    w_bits : int, optional
        Target bit-width for weights (default: 4)
    model_type : str, optional
        Type of model ("mamba" or "mamba2", default: "mamba")
    nsamples : int, optional
        Number of calibration samples (default: 128)
    seqlen : int, optional
        Sequence length for calibration data (default: 1024)
    output_dir : str, optional
        Directory to save outputs (default: "./jacobian_jvp_error")

    Returns:
    --------
    torch.nn.Module
        The model (moved to device)

    Notes:
    ------
    - This function raises ValueError("Jacobian loop over!") when complete,
      which is the expected behavior to signal completion.
    - Results are saved to the output_dir in the format expected by
      SMGPTQ.stitch_plots() for visualization.
    """
    bits = w_bits
    assert bits in [4, 8], "Only support 4 or 8 bits weights for now"
    
    logging.info("=" * 80)
    logging.info("Starting Jacobian Estimation via JVP Sampling")
    logging.info("=" * 80)
    logging.info(f"Model type: {model_type}")
    logging.info(f"Number of samples: {nsamples}")
    logging.info(f"Sequence length: {seqlen}")
    logging.info(f"Target bit-width for weights: {bits}")
    logging.info(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build dataloader for calibration
    logging.info("Building calibration dataloader from wikitext2...")
    dataloader, _ = get_loaders("wikitext2", tokenizer, nsamples=nsamples, seqlen=seqlen)
    
    # Get model layers
    layers = model.backbone.layers
    model.backbone.embedding = model.backbone.embedding.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype

    # Prepare input and residual buffers
    inps = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )    
    residual = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )

    # Catch the first layer's input
    cache = {"i": 0}
    
    class Catcher(nn.Module):
        """Helper module to capture layer inputs."""
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, res=None, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError  # Stop forward pass after capturing input
    
    # Capture inputs from calibration data
    logging.info("Capturing calibration inputs from first layer...")
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass  # Expected - we use ValueError to stop the forward pass

    # Define hooks to collect inputs for in_proj, out_proj
    def add_batch(module, inp, out, gptq, is_out_layer=False):
        """Hook for standard GPTQ (not currently used)."""
        gptq.add_batch(inp[0], out)

    def add_batch_layer(module, inp, out, gptq, is_out_layer=False):
        """Hook for SMGPTQ to capture layer inputs and outputs."""
        if is_out_layer:
            # Capture outputs from norm layer (end of mixer)
            logging.debug(f"Capturing outputs - num inputs: {len(inp)}")
            for i, el in enumerate(inp):
                logging.debug(f"  Input {i}: {el.shape}")
            gptq.capture_outputs(inp[0], inp[1], out)
        else:
            # Capture inputs to mixer layer
            gptq.capture_inputs(inp[0], out)

    # Restore original first layer and prepare for layer-wise processing
    layers[0] = layers[0].module  # Remove Catcher wrapper
    layers[0] = layers[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()
    
    # Process each layer
    logging.info(f"Processing {len(layers)} layers for Jacobian estimation...")
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing Layer {i}/{len(layers)-1}")
        logging.info(f"{'='*60}")
        
        # Move layer to device
        layer = layers[i].to(device)

        if model_type in ["mamba", "mamba2"]:
            # Create MambaJacobianEstimator object for this layer's mixer
            jacobian_estimator = {
                "in_proj": MambaJacobianEstimator(layer.mixer, idx=i),
            }

            # Register hooks to capture inputs and outputs
            handles_jac = [
                layer.mixer.register_forward_hook(
                    partial(add_batch_layer, gptq=jacobian_estimator["in_proj"], is_out_layer=False)
                ),
                layer.mixer.norm.register_forward_hook(
                    partial(add_batch_layer, gptq=jacobian_estimator["in_proj"], is_out_layer=True)
                ),
            ]
            
            # Run forward pass through this layer
            layer(inps, residual=residual)
            
            # Remove hooks
            for h in handles_jac:
                h.remove()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Compute Jacobian estimates and save results
        logging.info(f"Computing Jacobian estimates for layer {i}...")
        for name in jacobian_estimator.keys():
            logging.debug(f"Processing layer.{i}.mixer.{name} with {bits} bits")
            jacobian_estimator[name].stitch_plots()  # This saves data to output_dir
            jacobian_estimator[name].free()
        del jacobian_estimator
        
        # Compute outputs for next layer
        inps, residual = layer(inps, residual=residual)

        # Detach gradients but keep differentiable for next layer
        inps = inps.detach()
        residual = residual.detach()
        inps.requires_grad = True
        residual.requires_grad = True
        
        # Clean up
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        gc.collect()

    # Process final lm_head layer
    logging.info("\n" + "="*60)
    logging.info("Processing lm_head (final layer)")
    logging.info("="*60)
    
    model = model.to("cpu")  # Move model to CPU to save memory
    model.lm_head = model.lm_head.to(device)
    model.backbone.norm_f = model.backbone.norm_f.to(device)
    
    gptq_lm_head = GPTQ(model.lm_head)
    handle = model.lm_head.register_forward_hook(partial(add_batch, gptq=gptq_lm_head))
    
    # Compute final hidden states
    if model_type in ["mamba", "mamba2"]:
        final_hidden_states = layer_norm_fn(
            x=inps,
            weight=model.backbone.norm_f.weight,
            bias=model.backbone.norm_f.bias,
            eps=model.backbone.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=model.backbone.residual_in_fp32,
            is_rms_norm=isinstance(model.backbone.norm_f, RMSNorm),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Process lm_head
    for j in range(nsamples):
        model.lm_head(final_hidden_states[j].unsqueeze(0))

    handle.remove()
    
    # Note: We use float16 to save memory during quantization
    gptq_lm_head.fasterquant(
        percdamp=0.01, group_size=128, dtype=torch.float16
    )
    gptq_lm_head.free()
    del gptq_lm_head
    
    torch.cuda.empty_cache()
    gc.collect()

    model = model.to(device)
    
    logging.info("\n" + "="*80)
    logging.info("Jacobian estimation complete!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info("="*80)
    
    # Raise ValueError to signal completion (expected behavior)
    raise ValueError("Jacobian loop over!")


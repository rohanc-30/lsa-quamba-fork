"""
Main script for Jacobian estimation via JVP sampling.

This script provides a dedicated pathway for computing and saving Jacobian samples
for Mamba models, completely independent of the evaluation pipeline.
"""

import json
import logging
import sys
import os
import argparse

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import build_mamba_and_tokenizer, set_deterministic
from jacobian_estimation.jacobian_utils import save_jacobian_samples


def parse_arguments():
    """Parse command-line arguments for Jacobian estimation."""
    parser = argparse.ArgumentParser(
        description='Compute Jacobian samples for Mamba models via JVP estimation'
    )
    
    # Model arguments
    parser.add_argument(
        'model', type=str,
        help='Model to load; pass location of huggingface converted checkpoint.'
    )
    parser.add_argument(
        '--pretrained_dir', type=str, default=None,
        help='Directory containing pretrained models (required for quamba models)'
    )
    
    # Jacobian estimation parameters
    parser.add_argument(
        '--w_bits', type=int, default=4,
        help='Target bit-width for weights (default: 4)'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration samples (default: 128)'
    )
    parser.add_argument(
        '--seqlen', type=int, default=1024,
        help='Sequence length for calibration data (default: 1024)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./jacobian_jvp_error',
        help='Directory to save Jacobian estimation outputs (default: ./jacobian_jvp_error)'
    )
    
    # Model-specific arguments
    parser.add_argument(
        '--group_heads', action='store_true', default=False,
        help='Whether to group heads during reordering (for mamba2 models)'
    )
    
    # General settings
    parser.add_argument(
        '--verbose', action='store_true',
        help='Whether to print debug level information'
    )
    parser.add_argument(
        '--seed', type=int, default=1234,
        help='Random seed for reproducibility (default: 1234)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for Jacobian estimation."""
    args = parse_arguments()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Set deterministic behavior
    set_deterministic(args.seed)
    
    # Extract model information
    model_name = args.model.lower().split('/')[-1]
    model_type = model_name.split('-')[0]  # Assume format: "model_type-<size/version>"
    
    logging.info(f"Starting Jacobian estimation for model: {model_name}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Model path: {args.model}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Validate model type
    if model_type not in ["mamba", "mamba2"]:
        raise ValueError(
            f"Jacobian estimation currently only supports mamba and mamba2 models. "
            f"Got: {model_type}"
        )
    
    # Build model and tokenizer
    logging.info("Loading model and tokenizer...")
    model, tokenizer, is_quamba = build_mamba_and_tokenizer(args, model_type)
    model.config.use_cache = False
    model.eval()
    
    if is_quamba:
        logging.warning(
            "Model appears to be a quantized quamba model. "
            "Jacobian estimation is typically performed on FP16 models."
        )
    
    # Run Jacobian estimation
    logging.info("Starting Jacobian sampling and estimation...")
    logging.info(f"Parameters: nsamples={args.nsamples}, seqlen={args.seqlen}, w_bits={args.w_bits}")
    
    try:
        save_jacobian_samples(
            model=model,
            tokenizer=tokenizer,
            device="cuda",
            w_bits=args.w_bits,
            model_type=model_type,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            output_dir=args.output_dir
        )
        logging.info("Jacobian estimation completed successfully!")
        logging.info(f"Results saved to: {args.output_dir}")
        
    except ValueError as e:
        # The save_jacobian_samples function raises ValueError("Jacobian loop over!")
        # when it completes successfully
        if "Jacobian loop over" in str(e):
            logging.info("Jacobian estimation completed successfully!")
            logging.info(f"Results saved to: {args.output_dir}")
        else:
            raise
    
    # Save metadata
    metadata = {
        'model': args.model,
        'model_name': model_name,
        'model_type': model_type,
        'w_bits': args.w_bits,
        'nsamples': args.nsamples,
        'seqlen': args.seqlen,
        'seed': args.seed,
        'group_heads': args.group_heads,
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()


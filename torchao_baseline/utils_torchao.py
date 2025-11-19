"""
Utility functions for TorchAO quantization of GLA and Mamba models.

This module provides helper functions for:
- Loading models for quantization
- Calibration data preparation
- Applying RTN and PTQ quantization
- Saving quantized models
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, Tuple, List, Callable
from functools import partial
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# Import model classes
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from fla.models import GLAForCausalLM

from torchao.quantization.granularity import PerAxis, PerTensor
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.dtypes import to_affine_quantized_intx_static

from dataclasses import dataclass
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_
from torchao.quantization.transform_module import register_quantize_module_handler

import copy

logger = logging.getLogger(__name__)


def load_model_for_quantization(
    model_path: str,
    model_type: str,
    pretrained_dir: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32  # Changed to fp32 by default to avoid NaN
):
    """
    Load a FP32 model (GLA, Mamba, or Mamba2) for quantization.
    
    Note: Models are loaded in FP32 by default to avoid numerical instability (NaN).
    FP16 can cause issues during calibration, especially for GLA models.
    
    Args:
        model_path: Path or HuggingFace model ID (e.g., "fla-hub/gla-1.3b")
        model_type: Type of model ("gla", "mamba", "mamba2")
        pretrained_dir: Optional base directory for pretrained models
        device: Device to load model on
        dtype: Data type for model weights (default: torch.float32 for stability)
        
    Returns:
        tuple: (model, tokenizer, config_dict)
            - model: The loaded model ready for quantization
            - tokenizer: Associated tokenizer
            - config_dict: Configuration dictionary with model metadata
    """
    logger.info(f"Loading {model_type} model from {model_path} with dtype={dtype}")
    
    # Construct full path if pretrained_dir is provided
    if pretrained_dir and not model_path.startswith('/'):
        full_path = os.path.join(pretrained_dir, model_path)
        if os.path.exists(full_path):
            model_path = full_path
            logger.info(f"Using full path: {model_path}")
    
    # Load model based on type
    if model_type == "gla":
        # Load GLA model - load in fp32 first for stability
        tokenizer = AutoTokenizer.from_pretrained("fla-hub/gla-1.3B-100B", resume_download=None)
        model = GLAForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
        if dtype != torch.float32:
            logger.warning(f"Converting model from fp32 to {dtype}")
            model = model.to(dtype=dtype)
        
    elif model_type in ["mamba", "mamba2"]:
        # Load Mamba/Mamba2 model
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", resume_download=None)
        model = MambaLMHeadModel.from_pretrained(model_path, device=device, dtype=dtype)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: 'gla', 'mamba', 'mamba2'")
    
    # Ensure tokenizer has pad token (important for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Extract model name for metadata
    model_name = model_path.split('/')[-1]
    
    # Create config dictionary
    config_dict = {
        "model_name": model_name,
        "model_type": model_type,
        "model_path": model_path,
        "dtype": str(dtype),
        "device": device
    }
    
    logger.info(f"Successfully loaded {model_type} model: {model_name}")
    model.eval()
    
    return model, tokenizer, config_dict


def get_calibration_data(
    tokenizer,
    dataset_name: str = "wikitext",
    num_samples: int = 512,
    seq_len: int = 512,
    split: str = "train",
    min_length: int = 64  # Added minimum length filter
):
    """
    Prepare calibration dataset for PTQ quantization.
    
    Note: Variable-length sequences are EXPECTED and normal. Different text samples
    naturally have different lengths. We filter out very short sequences (< min_length)
    to ensure calibration quality.
    
    Args:
        tokenizer: Tokenizer for the model
        dataset_name: HuggingFace dataset name (default: "wikitext")
        num_samples: Number of calibration samples
        seq_len: Maximum sequence length
        split: Dataset split to use
        min_length: Minimum sequence length to include (default: 64 tokens)
        
    Returns:
        List of tokenized input_ids tensors ready for calibration
    """
    logger.info(f"Loading calibration data from {dataset_name}")
    
    # Load dataset based on name
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text_key = "text"
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        text_key = "text"
    elif dataset_name == "pile":
        dataset = load_dataset("monology/pile-uncopyrighted", 
                             data_files="val.jsonl.zst", 
                             split=split)
        text_key = "text"
    else:
        # Generic dataset loading
        dataset = load_dataset(dataset_name, split=split)
        text_key = "text"
    
    logger.info(f"Tokenizing {num_samples} samples with max length {seq_len}, min length {min_length}")
    
    calibration_samples = []
    sample_count = 0
    skipped_short = 0
    skipped_empty = 0
    
    for data in tqdm(dataset, total=num_samples * 2, desc="Preparing calibration data"):
        if sample_count >= num_samples:
            break
        
        # Get text from the dataset
        text = data[text_key]
        
        # Skip empty texts
        if not text or len(text.strip()) == 0:
            skipped_empty += 1
            continue
        
        # Tokenize
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding=False
        ).input_ids
        
        # Only use samples that have reasonable length (avoid tiny/header lines)
        if input_ids.shape[1] >= min_length:
            calibration_samples.append(input_ids)
            sample_count += 1
        else:
            skipped_short += 1
    
    logger.info(f"Prepared {len(calibration_samples)} calibration samples")
    logger.info(f"Skipped {skipped_empty} empty and {skipped_short} too-short samples")
    logger.info(f"Sample lengths - min: {min([s.shape[1] for s in calibration_samples])}, "
                f"max: {max([s.shape[1] for s in calibration_samples])}, "
                f"mean: {sum([s.shape[1] for s in calibration_samples]) / len(calibration_samples):.1f}")
    
    return calibration_samples


class CalibrationDataset(Dataset):
    """
    Simple Dataset wrapper for calibration samples.
    Each sample is a tensor of shape [1, seq_len_i] from get_calibration_data().
    """
    def __init__(self, samples: List[torch.Tensor]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Remove the batch dimension [1, seq_len] -> [seq_len]
        return self.samples[idx].squeeze(0)


def collate_calibration_batch(batch: List[torch.Tensor], pad_token_id: int = 0):
    """
    Collate function for calibration batches.
    Pads sequences to the same length within each batch.
    
    Args:
        batch: List of 1D tensors [seq_len_i] with variable lengths
        pad_token_id: Token ID to use for padding (default: 0)
        
    Returns:
        Padded tensor of shape [batch_size, max_seq_len]
    """
    # Find max length in this batch
    max_len = max(seq.shape[0] for seq in batch)
    
    # Pad all sequences to max_len
    padded_batch = []
    for seq in batch:
        if seq.shape[0] < max_len:
            # Pad to max_len
            padding = torch.full((max_len - seq.shape[0],), pad_token_id, dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_batch.append(padded_seq)
    
    # Stack into [batch_size, max_seq_len]
    return torch.stack(padded_batch, dim=0)


def create_calibration_dataloader(
    calibration_samples: List[torch.Tensor],
    batch_size: int = 8,
    pad_token_id: int = 0,
    shuffle: bool = False
) -> DataLoader:
    """
    Create a DataLoader from calibration samples (output of get_calibration_data).
    
    Args:
        calibration_samples: List of tensors [1, seq_len_i] from get_calibration_data()
        batch_size: Batch size for calibration (default: 8)
        pad_token_id: Token ID for padding (default: 0, typically matches tokenizer.pad_token_id)
        shuffle: Whether to shuffle samples (default: False, not needed for calibration)
        
    Returns:
        DataLoader that yields batches of shape [batch_size, max_seq_len_in_batch]
    """
    dataset = CalibrationDataset(calibration_samples)
    
    # Create collate function with specific pad_token_id
    collate_fn = partial(collate_calibration_batch, pad_token_id=pad_token_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # Keep 0 for simplicity with CUDA tensors
        pin_memory=False  # Samples might already be on device
    )
    
    logger.info(f"Created calibration DataLoader: {len(dataset)} samples, "
                f"batch_size={batch_size}, ~{len(dataloader)} batches")
    
    return dataloader


def make_w8a8_observers():
    """
    W8A8 PTQ-linear:

    - Activations:  per-tensor, 8-bit, *asymmetric*  (uint8)
    - Weights:      per-channel, 8-bit, *symmetric*  (int8)
    """
    # A8: uint8, per-tensor, asymmetric
    act_obs = AffineQuantizedMinMaxObserver(
        MappingType.ASYMMETRIC,
        torch.uint8,
        granularity=PerTensor(),
        eps=torch.finfo(torch.float32).eps,
        scale_dtype=torch.float32,
        zero_point_dtype=torch.float32,
    )

    # W8: int8, per-channel (axis=0), symmetric
    weight_obs = AffineQuantizedMinMaxObserver(
        MappingType.SYMMETRIC,   # <- this makes weights symmetric
        torch.int8,                     # signed int8 is standard for symmetric W
        granularity=PerAxis(axis=0),
        eps=torch.finfo(torch.float32).eps,
        scale_dtype=torch.float32,
        zero_point_dtype=torch.float32,
    )

    return act_obs, weight_obs


class ObservedLinear(nn.Linear):
    """
    High-precision Linear with attached activation/weight observers.
    Used only during calibration (PREPARE+CALIBRATION).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_obs: nn.Module,
        weight_obs: nn.Module,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.act_obs = act_obs
        self.weight_obs = weight_obs

    def forward(self, x: torch.Tensor):
        # Record activation ranges (detached to avoid affecting gradients)
        self.act_obs(x.detach())
        # Record weight ranges (weights are already detached tensors)
        self.weight_obs(self.weight.detach())
        # Still pure FP matmul with original tensors
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_float(cls, float_linear: nn.Linear, act_obs: nn.Module, weight_obs: nn.Module):
        """Wrap an existing nn.Linear with observers."""
        observed = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            weight_obs,
            bias=float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed.weight = float_linear.weight
        observed.bias = float_linear.bias
        return observed


def insert_observers_(model: nn.Module,
                      act_obs: nn.Module,
                      weight_obs: nn.Module,
                      gla_only: bool = True) -> nn.Module:
    """
    Replace selected nn.Linear modules with ObservedLinear, each having
    its own copy of act/weight observers.

    If gla_only is True, you can narrow to attention/MLP linears
    by checking the fully-qualified name.
    """
    replaced_count = 0
    down_proj_count = 0
    down_proj_types = set()
    
    def filter_fn(m: nn.Module, fqn: str) -> bool:
        nonlocal down_proj_count, down_proj_types
        
        # Log down_proj modules specifically
        if "down_proj" in fqn:
            down_proj_count += 1
            down_proj_types.add(type(m).__name__)
            if down_proj_count <= 3:  # Log first few
                logger.info(f"  Found down_proj: {fqn}, type={type(m).__name__}, is_linear={isinstance(m, nn.Linear)}")
        
        if not isinstance(m, nn.Linear):
            return False

        if gla_only:
            # Example heuristics: quantize transformer blocks but skip lm_head/embeds
            if "lm_head" in fqn or "embed" in fqn:
                return False
        return True

    def replacement_fn(m: nn.Linear) -> ObservedLinear:
        nonlocal replaced_count
        replaced_count += 1
        # each module needs its own observers
        a = copy.deepcopy(act_obs)
        w = copy.deepcopy(weight_obs)
        return ObservedLinear.from_float(m, a, w)

    _replace_with_custom_fn_if_matches_filter(model, replacement_fn, filter_fn)
    
    logger.info(f"Replaced {replaced_count} Linear layers with ObservedLinear")
    logger.info(f"Found {down_proj_count} down_proj modules with types: {down_proj_types}")
    
    return model

def prepare_for_static_w8a8(model: nn.Module) -> nn.Module:
    """
    Prepare model for W8A8 static quantization by inserting observers.
    
    IMPORTANT: For GLA models with fused SwiGLU, we temporarily disable
    the fusion so that down_proj observers can be called during calibration.
    """
    model.eval()
    
    # Disable fused SwiGLU in MLP layers (if present) so observers get called
    mlp_layers_modified = 0
    for name, module in model.named_modules():
        if hasattr(module, 'fuse_swiglu') and module.fuse_swiglu:
            logger.info(f"Disabling fuse_swiglu for {name} during calibration")
            module.fuse_swiglu = False
            mlp_layers_modified += 1
    
    if mlp_layers_modified > 0:
        logger.info(f"Disabled fuse_swiglu in {mlp_layers_modified} MLP layers for calibration")
    
    act_obs, weight_obs = make_w8a8_observers()
    model = insert_observers_(model, act_obs, weight_obs, gla_only=True)
    return model


def calibrate_static_quant(
    model: nn.Module,
    calib_data: List[torch.Tensor],
    batch_size: int = 8,
    pad_token_id: int = 0,
    device: str = "cuda",
    max_batches: int | None = None,
):
    """
    Run CALIBRATE step for static W8A8 quantization.
    
    This function:
    1. Creates a DataLoader from raw calibration data
    2. Runs forward passes to populate observers with statistics
    3. Returns the calibrated model (ready for conversion)

    Args:
        model: Prepared model with ObservedLinear layers (from prepare_for_static_w8a8)
        calib_data: Raw calibration data from get_calibration_data() 
                   (list of tensors with shape [1, seq_len_i])
        batch_size: Batch size for calibration (default: 8)
        pad_token_id: Token ID for padding (default: 0)
        device: Device to run on (default: "cuda")
        max_batches: Optional cap on number of calibration batches (default: None, use all)
        
    Returns:
        Calibrated model with populated observers
    """
    logger.info(f"Starting calibration with {len(calib_data)} samples, batch_size={batch_size}")
    
    # Create DataLoader from raw calibration data
    calib_loader = create_calibration_dataloader(
        calib_data,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        shuffle=False
    )
    
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(calib_loader, desc="Calibrating")):
            if max_batches is not None and i >= max_batches:
                logger.info(f"Reached max_batches={max_batches}, stopping calibration")
                break

            # Move batch to device; handle dict vs tensor
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = model(**batch)
            else:
                batch = batch.to(device)
                _ = model(batch)

    logger.info(f"Calibration complete! Processed {min(i+1, len(calib_loader))} batches")
    
    # Validate that observers recorded statistics
    logger.info("Validating observer statistics...")
    observer_count = 0
    failed_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, ObservedLinear):
            observer_count += 1
            has_act_stats = hasattr(module.act_obs, 'min_val') and hasattr(module.act_obs, 'max_val')
            has_weight_stats = hasattr(module.weight_obs, 'min_val') and hasattr(module.weight_obs, 'max_val')
            
            # Log first few in detail
            if observer_count <= 5:
                logger.info(f"  {name}: act_obs recorded={has_act_stats}, weight_obs recorded={has_weight_stats}")
            
            # Track ALL failed layers
            if not has_act_stats or not has_weight_stats:
                failed_layers.append((name, has_act_stats, has_weight_stats))
    
    if failed_layers:
        logger.error(f"✗ {len(failed_layers)} out of {observer_count} layers failed to record statistics!")
        for name, has_act, has_weight in failed_layers[:10]:  # Show first 10
            logger.error(f"  {name}: act_obs={has_act}, weight_obs={has_weight}")
        if len(failed_layers) > 10:
            logger.error(f"  ... and {len(failed_layers) - 10} more")
        raise RuntimeError(
            f"Calibration failed: {len(failed_layers)} layers did not record min/max values. "
            f"First failed layer: {failed_layers[0][0]}"
        )
    
    logger.info(f"✓ Validated {observer_count} ObservedLinear layers")
    return model

class QuantizedLinear(nn.Module):
    """
    Static W8A8 linear:
      - W: int8, per-channel, symmetric (via weight observer mapping_type)
      - A: int8, per-tensor, asymmetric (via act observer mapping_type)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_obs: nn.Module,
        weight_obs: nn.Module,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        target_dtype: torch.dtype,
    ):
        super().__init__()
        # 1) Compute quantization parameters from observers
        self.act_scale, self.act_zero_point = act_obs.calculate_qparams()
        weight_scale, weight_zero_point = weight_obs.calculate_qparams()

        assert weight.dim() == 2  # [out_features, in_features]

        # 2) Quantize weights statically (for memory savings)
        block_size = (1, weight.shape[1])  # rowwise blocks
        self.target_dtype = target_dtype
        self.bias = bias

        # Store quantized weights (int8, saves 4x memory)
        self.qweight = to_affine_quantized_intx_static(
            weight,
            weight_scale,
            weight_zero_point,
            block_size,
            self.target_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        W8A8 quantized forward pass:
        1. Quantize input activations to int8 (dynamic)
        2. Perform int8 x int8 matmul (fast!)
        3. Dequantize output to float (for GLA kernel compatibility)
        
        The key fix: explicitly dequantize output to plain torch.Tensor,
        not an AffineQuantizedTensor subclass that breaks triton kernels.
        """
        # Ensure input is plain float tensor (defensive check)
        if hasattr(x, 'dequantize'):
            x = x.dequantize()
        if type(x) != torch.Tensor:
            x = x.detach().clone()
        
        # Quantize activations dynamically using calibrated scale/zero_point
        block_size = x.shape
        qx = to_affine_quantized_intx_static(
            x,
            self.act_scale,
            self.act_zero_point,
            block_size,
            self.target_dtype,
        )
        
        # Int8 x int8 matmul (this is where we get speedup!)
        output = F.linear(qx, self.qweight, self.bias)
        
        # CRITICAL: Explicitly dequantize output to plain float tensor
        # This ensures GLA's triton kernels receive standard tensors, not subclasses
        if hasattr(output, 'dequantize'):
            output = output.dequantize()
        
        # Ensure it's a plain torch.Tensor and contiguous
        if type(output) != torch.Tensor:
            output = output.detach().clone().contiguous()
        else:
            output = output.contiguous()
        
        return output

    @classmethod
    def from_observed(cls, observed_linear: "ObservedLinear", target_dtype: torch.dtype):
        """
        Build QuantizedLinear from an already calibrated ObservedLinear.
        """
        return cls(
            observed_linear.in_features,
            observed_linear.out_features,
            observed_linear.act_obs,
            observed_linear.weight_obs,
            observed_linear.weight,
            observed_linear.bias,
            target_dtype,
        )

@dataclass
class StaticQuantConfig(AOBaseConfig):
    target_dtype: torch.dtype


@register_quantize_module_handler(StaticQuantConfig)
def _apply_static_quant(
    module: nn.Module,
    config: StaticQuantConfig,
):
    """
    This is called internally by torchao.quantization.quantize_.
    For each candidate module, it returns the quantized replacement.
    """
    # We expect `module` to be an ObservedLinear (filtered by filter_fn)
    return QuantizedLinear.from_observed(module, config.target_dtype)


def convert_static_w8a8(model: nn.Module,
                        target_dtype: torch.dtype = torch.int8) -> nn.Module:
    """
    Run the CONVERT phase:
      - ObservedLinear -> QuantizedLinear
      - qparams are frozen
      - observers disappear from the forward path
    """

    # filter: only touch ObservedLinear modules
    is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)

    # This walks the tree, and for each ObservedLinear it calls
    # _apply_static_quant(...), which returns a QuantizedLinear.
    quantize_(model, StaticQuantConfig(target_dtype), is_observed_linear)

    return model


def save_quantized_model(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    model_name: str = "gla_ptq-w8a8-1.3b",
    model_type: str = "gla",
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save W8A8 quantized model with proper naming and metadata.
    
    Naming convention: {model}_ptq-w8a8-{size}
    Examples:
        - gla_ptq-w8a8-1.3b  (GLA with PTQ W8A8)
        - mamba2_ptq-w8a8-1.3b  (Mamba2 with PTQ W8A8)
        - gla_gptq-w4a16-1.3b  (GLA with GPTQ W4A16, for comparison)
    
    Saves:
    - model weights (state_dict)
    - tokenizer
    - config with quantization metadata
    
    Args:
        model: Quantized model (after convert_static_w8a8)
        tokenizer: Associated tokenizer
        output_dir: Output directory (e.g., "pretrained_models/fla-hub")
        model_name: Model name (e.g., "gla_ptq-w8a8-1.3b")
        model_type: Type of model ("gla", "mamba", "mamba2")
        metadata: Optional additional metadata
    """
    import json
    from pathlib import Path
    
    # Create output directory
    save_path = Path(output_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving quantized model to {save_path}")
    
    # Save model state dict
    model_save_path = save_path / "pytorch_model.bin"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"✓ Saved model weights to {model_save_path}")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    logger.info(f"✓ Saved tokenizer to {save_path}")
    
    # Save config with quantization info
    config_dict = {
        "model_type": model_type,
        "model_name": model_name,
        "quantization": {
            "method": "torchao_w8a8_static",
            "weight_bits": 8,
            "activation_bits": 8,
            "weight_dtype": "int8",
            "activation_dtype": "uint8",  # asymmetric
            "weight_granularity": "per_channel",
            "activation_granularity": "per_tensor",
            "weight_symmetric": True,
            "activation_symmetric": False,
        }
    }
    
    # Add original model config if available
    if hasattr(model, 'config'):
        config_dict['original_config'] = model.config.to_dict() if hasattr(model.config, 'to_dict') else str(model.config)
    
    # Add custom metadata
    if metadata:
        config_dict['metadata'] = metadata
    
    config_save_path = save_path / "quantization_config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"✓ Saved quantization config to {config_save_path}")
    
    # Count quantized layers
    from torchao_baseline.utils_torchao import QuantizedLinear
    quantized_count = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Model saved successfully!")
    logger.info(f"  Location: {save_path}")
    logger.info(f"  Quantized layers: {quantized_count}")
    logger.info(f"  Size: {get_directory_size(save_path):.2f} MB")
    logger.info(f"{'='*60}\n")
    
    return save_path


def get_directory_size(path: str) -> float:
    """Calculate total size of directory in MB."""
    from pathlib import Path
    total_size = 0
    for file in Path(path).rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size / (1024 * 1024)  # Convert to MB
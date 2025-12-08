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
from fla.models import GLAForCausalLM, DeltaNetForCausalLM, RetNetForCausalLM

from torchao.quantization.granularity import PerAxis, PerTensor
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.dtypes import to_affine_quantized_intx_static

from dataclasses import dataclass
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_
from torchao.quantization.transform_module import register_quantize_module_handler

from torchao.quantization.quant_api import int4_weight_only, int8_weight_only

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
    Load a FP32 model (GLA, Mamba, Mamba2, or DeltaNet) for quantization.
    
    Note: Models are loaded in FP32 by default to avoid numerical instability (NaN).
    FP16 can cause issues during calibration, especially for GLA models.
    Exception: DeltaNet models are loaded in bfloat16 as required by the architecture.
    
    Args:
        model_path: Path or HuggingFace model ID (e.g., "fla-hub/gla-1.3b")
        model_type: Type of model ("gla", "mamba", "mamba2", "delta_net", "delta_net_ptq")
        pretrained_dir: Optional base directory for pretrained models
        device: Device to load model on
        dtype: Data type for model weights (default: torch.float32 for stability, except delta_net uses bfloat16)
        
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
        
        # Ensure model config matches tokenizer vocab_size
        if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
            if model.config.vocab_size != tokenizer.vocab_size:
                logger.warning(
                    f"Model config vocab_size ({model.config.vocab_size}) != tokenizer vocab_size ({tokenizer.vocab_size}). "
                    f"Updating config to match tokenizer."
                )
                model.config.vocab_size = tokenizer.vocab_size

    elif model_type == "delta_net" or model_type == "delta_net_ptq":
        # Load DeltaNet model - must use bfloat16 for delta_net
        tokenizer = AutoTokenizer.from_pretrained("fla-hub/delta_net-1.3B-100B", resume_download=None)
        model = DeltaNetForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    
    elif model_type == "retnet" or model_type == "retnet_ptq":
        # Load RetNet model - must use bfloat16 for retnet
        tokenizer = AutoTokenizer.from_pretrained("fla-hub/retnet-1.3B-100B", resume_download=None)
        model = RetNetForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        
    else:
        print(model_type)
        raise ValueError(f"Unsupported model type: {model_type}. Supported: 'gla', 'mamba', 'mamba2', 'delta_net', 'delta_net_ptq', 'retnet', 'retnet_ptq'")
    
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
    
    For Mamba2 models, we disable use_mem_eff_path to prevent the fused
    CUDA kernel from bypassing out_proj observers.
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
    
    # Disable use_mem_eff_path in Mamba2 layers (if present) so out_proj observers get called
    mamba_layers_modified = 0
    for name, module in model.named_modules():
        # Check for Mamba2 mixer layers
        if hasattr(module, 'use_mem_eff_path') and module.use_mem_eff_path:
            # Likely a Mamba2 layer - the fused kernel bypasses out_proj
            logger.info(f"Disabling use_mem_eff_path for {name} during calibration")
            module.use_mem_eff_path = False
            mamba_layers_modified += 1
    
    if mamba_layers_modified > 0:
        logger.info(f"Disabled use_mem_eff_path in {mamba_layers_modified} Mamba2 layers for calibration")
    
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
        
        Important: Preserves input dtype (e.g., bfloat16 for DeltaNet) to avoid dtype mismatches.
        """
        # Store original input dtype to restore after quantization
        input_dtype = x.dtype
        
        # Ensure input is plain float tensor (defensive check)
        if hasattr(x, 'dequantize'):
            x = x.dequantize()
        if type(x) != torch.Tensor:
            x = x.detach().clone()
        
        # Determine the dtype to use for quantization based on qweight
        # This ensures compatibility with saved quantized weights
        qweight_dtype = self.qweight.dtype if hasattr(self.qweight, 'dtype') else torch.float32
        
        # Convert input to match qweight dtype for quantization compatibility
        if x.dtype != qweight_dtype:
            x = x.to(qweight_dtype)
        
        # Quantize activations dynamically using calibrated scale/zero_point
        block_size = x.shape
        qx = to_affine_quantized_intx_static(
            x,
            self.act_scale,
            self.act_zero_point,
            block_size,
            self.target_dtype,
        )
        
        # Ensure bias matches qweight dtype for quantized linear operation
        bias = self.bias
        if bias is not None and bias.dtype != qweight_dtype:
            bias = bias.to(qweight_dtype)
        
        # Int8 x int8 matmul (this is where we get speedup!)
        output = F.linear(qx, self.qweight, bias)
        
        # CRITICAL: Explicitly dequantize output to plain float tensor
        # This ensures GLA's triton kernels receive standard tensors, not subclasses
        if hasattr(output, 'dequantize'):
            output = output.dequantize()
        
        # Ensure it's a plain torch.Tensor and contiguous
        if type(output) != torch.Tensor:
            output = output.detach().clone().contiguous()
        else:
            output = output.contiguous()
        
        # Cast back to original input dtype (important for bfloat16 models like DeltaNet)
        if output.dtype != input_dtype:
            output = output.to(input_dtype)
        
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

def _find_safe_group_size(model: nn.Module, filter_fn, verbose: bool = True) -> int:
    """
    Find the largest group_size that is safe for all quantizable layers.
    
    Returns a power of 2 that divides all weight dimensions.
    
    Args:
        model: Model to analyze
        filter_fn: Function to filter which layers to quantize
        verbose: If True, prints detailed analysis of bottleneck layers
    """
    min_dim = float('inf')
    layer_dims = []
    
    for name, module in model.named_modules():
        if filter_fn(module, name):
            if isinstance(module, nn.Linear):
                # Weight shape is [out_features, in_features]
                out_dim, in_dim = module.weight.shape
                min_dim = min(min_dim, out_dim, in_dim)
                layer_dims.append((name, out_dim, in_dim, min(out_dim, in_dim)))
    
    if min_dim == float('inf'):
        logger.warning("No quantizable layers found, defaulting to group_size=128")
        return 128
    
    # Find largest group_size that ALL layers are divisible by
    safe_group_sizes = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    selected_gs = None
    
    for gs in safe_group_sizes:
        # Check if ALL layers are compatible with this group size
        all_compatible = True
        for name, out_dim, in_dim, _ in layer_dims:
            if out_dim % gs != 0 or in_dim % gs != 0:
                all_compatible = False
                break
        
        if all_compatible:
            selected_gs = gs
            break
    
    if selected_gs is None:
        # Fallback to smallest possible
        logger.warning("No standard group size works, falling back to 1")
        selected_gs = 1
    
    # Print summary
    logger.info(f"Found {len(layer_dims)} quantizable layers")
    logger.info(f"Smallest dimension: {min_dim}")
    logger.info(f"Selected safe group_size: {selected_gs}")
    
    if verbose:
        print("\n" + "="*80)
        print("📊 GROUP SIZE ANALYSIS")
        print("="*80)
        
        # Sort by min dimension to find bottlenecks
        layer_dims_sorted = sorted(layer_dims, key=lambda x: x[3])
        
        # Show the smallest 10 layers (the bottlenecks)
        print(f"\n🔍 BOTTLENECK LAYERS (smallest dimensions):")
        print("-"*80)
        for i, (name, out_dim, in_dim, min_d) in enumerate(layer_dims_sorted[:10], 1):
            print(f"{i:2d}. {name}")
            print(f"    Shape: [{out_dim:5d}, {in_dim:5d}]  →  Min dim: {min_d:5d}")
            
            # Show which group sizes this layer blocks
            blocked_sizes = []
            for gs in [256, 128, 64, 32, 16]:
                if min_d < gs or out_dim % gs != 0 or in_dim % gs != 0:
                    blocked_sizes.append(gs)
            
            if blocked_sizes:
                print(f"    Blocks group_sizes: {blocked_sizes}")
        
        # Analyze what each group size would require
        print(f"\n📈 GROUP SIZE FEASIBILITY:")
        print("-"*80)
        for gs in [256, 128, 64, 32, 16, 8]:
            # Count how many layers would fail with this group size
            incompatible = []
            for name, out_dim, in_dim, _ in layer_dims:
                if out_dim % gs != 0 or in_dim % gs != 0:
                    incompatible.append(name)
            
            if len(incompatible) == 0:
                status = "✓ SAFE"
                detail = f"All {len(layer_dims)} layers compatible"
            else:
                status = "✗ FAILS"
                detail = f"{len(incompatible)}/{len(layer_dims)} layers incompatible"
            
            print(f"group_size={gs:3d}:  {status:8s}  ({detail})")
        
        print("="*80 + "\n")
    
    return selected_gs


def _validate_group_size(model: nn.Module, group_size: int, filter_fn):
    """
    Validate that group_size works for all quantizable layers.
    Raises detailed error if any layer is incompatible.
    """
    incompatible_layers = []
    
    for name, module in model.named_modules():
        if filter_fn(module, name):
            if isinstance(module, nn.Linear):
                out_dim, in_dim = module.weight.shape
                
                # Check if both dimensions are divisible by group_size
                if out_dim % group_size != 0 or in_dim % group_size != 0:
                    incompatible_layers.append({
                        'name': name,
                        'out_dim': out_dim,
                        'in_dim': in_dim,
                        'out_divisible': out_dim % group_size == 0,
                        'in_divisible': in_dim % group_size == 0
                    })
    
    if incompatible_layers:
        logger.error(f"\n{'='*80}")
        logger.error(f"❌ GROUP SIZE VALIDATION FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"group_size={group_size} is incompatible with {len(incompatible_layers)} layers:")
        logger.error(f"")
        
        for i, layer_info in enumerate(incompatible_layers[:10], 1):  # Show first 10
            logger.error(f"{i}. {layer_info['name']}")
            logger.error(f"   Weight shape: [{layer_info['out_dim']}, {layer_info['in_dim']}]")
            logger.error(f"   Out dim divisible by {group_size}: {layer_info['out_divisible']}")
            logger.error(f"   In dim divisible by {group_size}: {layer_info['in_divisible']}")
            logger.error(f"")
        
        if len(incompatible_layers) > 10:
            logger.error(f"... and {len(incompatible_layers) - 10} more layers")
        
        logger.error(f"{'='*80}")
        logger.error(f"SUGGESTED FIXES:")
        logger.error(f"  1. Use a smaller group_size (try: 64, 32, or 16)")
        logger.error(f"  2. Call convert_static_w8a16(model, group_size=None) for auto-detection")
        logger.error(f"  3. Use per-tensor quantization with group_size=1")
        logger.error(f"{'='*80}\n")
        
        raise ValueError(
            f"group_size={group_size} is incompatible with {len(incompatible_layers)} layers. "
            f"Smallest layer dimension found: {min(layer_info['in_dim'] for layer_info in incompatible_layers)}. "
            f"Use a smaller group_size or set group_size=None for auto-detection."
        )
    
    logger.info(f"✓ group_size={group_size} validated successfully")


def convert_static_w4a16(model: nn.Module, group_size: int = None, skip_gk_proj: bool = True):
    """
    Convert model to W4A16 (4-bit weights, FP16 activations - weight-only).
    
    Args:
        model: Model to quantize
        group_size: Group size for per-channel quantization. 
                   - If None, will auto-detect safe group size based on smallest layer dimension
                   - If 0, will use per-tensor quantization (group_size=None to TorchAO)
                   - Otherwise, use the specified group size
        skip_gk_proj: If True, skips quantizing gk_proj layers (allows larger group_size).
                     These small layers (dim=16) would otherwise force group_size=16 for all layers.
    
    Returns:
        Quantized model
    """
    print("\n" + "="*80)
    print("🔧 CONVERT_STATIC_W4A16 CALLED")
    print("="*80)
    
    # Handle group_size=0 as per-tensor quantization
    per_tensor_quant = False
    if group_size == 0:
        print("🔧 group_size=0 detected → Using PER-TENSOR quantization")
        logger.info("group_size=0 → Using per-tensor quantization")
        per_tensor_quant = True
        group_size = None  # TorchAO uses None for per-tensor
    
    # Define which layers to quantize
    if skip_gk_proj:
        is_quantizable_linear = lambda m, fqn: (
            isinstance(m, nn.Linear) and 
            "lm_head" not in fqn and
            "gk_proj" not in fqn and
            "b_proj" not in fqn
        )
        print("ℹ️  Skipping gk_proj layers (keeping them in BF16 for larger group_size)")
        logger.info("Skipping gk_proj layers from quantization")
    else:
        is_quantizable_linear = lambda m, fqn: isinstance(m, nn.Linear) and "lm_head" not in fqn
    
    # Check ALL dtypes in model (parameters and buffers)
    dtypes_found = set()
    for name, param in model.named_parameters():
        dtypes_found.add(param.dtype)
    for name, buffer in model.named_buffers():
        dtypes_found.add(buffer.dtype)
    
    print(f"📊 Model dtypes found: {dtypes_found}")
    
    # Convert model to bfloat16 (required for TorchAO's int4 tinygemm kernel)
    model_dtype = next(model.parameters()).dtype
    print(f"📊 First parameter dtype: {model_dtype}")
    
    # TorchAO's int4_weight_only with TensorCoreTiledLayout requires bfloat16 weights
    # See: torchao/quantization/quant_api.py:1147
    if model_dtype != torch.bfloat16 or len(dtypes_found) > 1:
        print(f"🔄 Converting model from {model_dtype} to bfloat16 for W4A16 quantization...")
        logger.info(f"Converting model from {model_dtype} to bfloat16 for W4A16 quantization...")
        
        # Convert to bfloat16 (TorchAO requirement for GPU int4 quantization)
        model = model.to(dtype=torch.bfloat16)
        
        # Double-check conversion worked
        new_dtype = next(model.parameters()).dtype
        print(f"✓ Model converted! New dtype: {new_dtype}")
        logger.info(f"✓ Model converted to bfloat16 (verified: {new_dtype})")
    else:
        print(f"✓ Model is already in bfloat16")
        logger.info("✓ Model is already in bfloat16")
    
    # Handle group size logic
    if per_tensor_quant:
        # TorchAO's int4_weight_only doesn't support group_size=None (per-tensor)
        # It requires an explicit group_size value
        raise ValueError(
            "❌ Per-tensor quantization (group_size=0) is NOT supported for W4A16.\n"
            "   TorchAO's int4_weight_only requires an explicit group_size.\n"
            "   Options:\n"
            "   1. Use a specific group_size (e.g., 64, 128, 256)\n"
            "   2. Use W8A16 for per-tensor quantization (int8_weight_only supports it)"
        )
    elif group_size is None:
        # Auto-detect safe group size
        print(f"🔍 Auto-detecting safe group_size...")
        group_size = _find_safe_group_size(model, is_quantizable_linear)
        print(f"✓ Auto-detected safe group_size: {group_size}")
        logger.info(f"Auto-detected safe group_size: {group_size}")
    else:
        # Use specified group size
        print(f"✓ Using specified group_size: {group_size}")
        logger.info(f"Using specified group_size: {group_size}")
        
        # Validate group size against model
        print(f"🔍 Validating group_size={group_size}...")
        _validate_group_size(model, group_size, is_quantizable_linear)
    
    # Apply quantization
    print(f"🚀 Applying W4A16 quantization with group_size={group_size}...")
    logger.info(f"Applying W4A16 quantization with group_size={group_size}...")
    
    # Use default TensorCoreTiledLayout (optimized for GPU with tinygemm kernel)
    # Now that model is in bfloat16, zero_point_dtype will match scale.dtype
    quantize_(model, int4_weight_only(group_size=group_size), is_quantizable_linear)
    
    print("✓ W4A16 quantization complete!")
    logger.info("✓ W4A16 quantization complete")
    
    return model

def convert_static_w8a16(model: nn.Module, group_size: int = None, skip_gk_proj: bool = True):
    """
    Convert model to W8A16 (8-bit weights, FP16 activations - weight-only).
    
    Args:
        model: Model to quantize
        group_size: Group size for per-channel quantization. 
                   - If None, will auto-detect safe group size based on smallest layer dimension
                   - If 0, will use per-tensor quantization (group_size=None to TorchAO)
                   - Otherwise, use the specified group size
        skip_gk_proj: If True, skips quantizing gk_proj layers (allows larger group_size).
                     These small layers (dim=16) would otherwise force group_size=16 for all layers.
    
    Returns:
        Quantized model
    """
    print("\n" + "="*80)
    print("🔧 CONVERT_STATIC_W8A16 CALLED")
    print("="*80)
    
    # Handle group_size=0 as per-tensor quantization
    per_tensor_quant = False
    if group_size == 0:
        print("🔧 group_size=0 detected → Using PER-TENSOR quantization")
        logger.info("group_size=0 → Using per-tensor quantization")
        per_tensor_quant = True
        group_size = None  # TorchAO uses None for per-tensor
    
    # Define which layers to quantize
    if skip_gk_proj:
        is_quantizable_linear = lambda m, fqn: (
            isinstance(m, nn.Linear) and 
            "lm_head" not in fqn and
            "gk_proj" not in fqn and
            "b_proj" not in fqn
        )
        print("ℹ️  Skipping gk_proj layers (keeping them in BF16 for larger group_size)")
        logger.info("Skipping gk_proj layers from quantization")
    else:
        is_quantizable_linear = lambda m, fqn: isinstance(m, nn.Linear) and "lm_head" not in fqn
    
    # Check ALL dtypes in model (parameters and buffers)
    dtypes_found = set()
    for name, param in model.named_parameters():
        dtypes_found.add(param.dtype)
    for name, buffer in model.named_buffers():
        dtypes_found.add(buffer.dtype)
    
    print(f"📊 Model dtypes found: {dtypes_found}")
    
    # Convert model to bfloat16 for consistency with TorchAO best practices
    model_dtype = next(model.parameters()).dtype
    print(f"📊 First parameter dtype: {model_dtype}")
    
    if model_dtype != torch.bfloat16 or len(dtypes_found) > 1:
        print(f"🔄 Converting model from {model_dtype} to bfloat16 for W8A16 quantization...")
        logger.info(f"Converting model from {model_dtype} to bfloat16 for W8A16 quantization...")
        
        # Convert to bfloat16 (TorchAO best practice for GPU inference)
        model = model.to(dtype=torch.bfloat16)
        
        # Double-check conversion worked
        new_dtype = next(model.parameters()).dtype
        print(f"✓ Model converted! New dtype: {new_dtype}")
        logger.info(f"✓ Model converted to bfloat16 (verified: {new_dtype})")
    else:
        print(f"✓ Model is already in bfloat16")
        logger.info("✓ Model is already in bfloat16")
    
    # Handle group size logic
    if per_tensor_quant:
        # Skip group size detection and validation for per-tensor
        print(f"✓ Using PER-TENSOR quantization (no grouping)")
        logger.info("Using PER-TENSOR quantization (no grouping)")
    elif group_size is None:
        # Auto-detect safe group size
        print(f"🔍 Auto-detecting safe group_size...")
        group_size = _find_safe_group_size(model, is_quantizable_linear)
        print(f"✓ Auto-detected safe group_size: {group_size}")
        logger.info(f"Auto-detected safe group_size: {group_size}")
    else:
        # Use specified group size
        print(f"✓ Using specified group_size: {group_size}")
        logger.info(f"Using specified group_size: {group_size}")
        
        # Validate group size against model
        print(f"🔍 Validating group_size={group_size}...")
        _validate_group_size(model, group_size, is_quantizable_linear)
    
    # Apply quantization
    if per_tensor_quant:
        print(f"🚀 Applying W8A16 quantization with PER-TENSOR mode...")
        logger.info(f"Applying W8A16 quantization with PER-TENSOR mode...")
    else:
        print(f"🚀 Applying W8A16 quantization with group_size={group_size}...")
        logger.info(f"Applying W8A16 quantization with group_size={group_size}...")
    
    quantize_(model, int8_weight_only(group_size=group_size), is_quantizable_linear)
    print("✓ W8A16 quantization complete!")
    logger.info("✓ W8A16 quantization complete")
    
    return model

def save_quantized_model(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    model_name: str = None,
    model_type: str = "gla",
    quant_mode: str = None,
    metadata: Optional[Dict[str, Any]] = None,
    base_model_path: str = None,  # NEW: needed for weight-only models
):
    """
    Save quantized model with proper naming and metadata.
    Supports W8A8, W8A16, W4A16, W4A8 quantization modes.
    
    Naming convention: {model}_ptq-w{W}a{A}-{size}
    Examples:
        - gla_ptq-w8a8-1.3b   (8-bit weights, 8-bit activations)
        - gla_ptq-w8a16-1.3b  (8-bit weights, FP16 activations - weight-only)
        - gla_ptq-w4a16-1.3b  (4-bit weights, FP16 activations - weight-only)
        - gla_ptq-w4a8-1.3b   (4-bit weights, 8-bit activations)
    
    Saves:
    - model weights (state_dict)
    - tokenizer
    - config with quantization metadata
    
    For weight-only models (w8a16, w4a16):
    - Saves a marker indicating special loading is needed
    - Saves reference to base model path for reconstruction
    
    Args:
        model: Quantized model
        tokenizer: Associated tokenizer
        output_dir: Output directory (e.g., "pretrained_models/fla-hub")
        model_name: Model name (auto-generated if None, e.g., "gla_ptq-w8a8-1.3b")
        model_type: Type of model ("gla", "mamba", "mamba2")
        quant_mode: Quantization mode ("w8a8", "w8a16", "w4a16", "w4a8")
                   If None, will auto-detect from model structure
        metadata: Optional additional metadata
        base_model_path: Path to base (unquantized) model. Required for weight-only modes.
    """
    import json
    from pathlib import Path
    
    # Auto-detect quantization mode if not specified
    if quant_mode is None:
        logger.info("Auto-detecting quantization mode...")
        quant_mode = _detect_quantization_mode(model)
        logger.info(f"  Detected: {quant_mode}")
    
    # Validate quant_mode
    valid_modes = ["w8a8", "w8a16", "w4a16", "w4a8"]
    if quant_mode not in valid_modes:
        raise ValueError(f"Invalid quant_mode: {quant_mode}. Must be one of {valid_modes}")
    
    # Check if this is weight-only quantization
    is_weight_only = quant_mode in ["w8a16", "w4a16"]
    
    # For weight-only, we need the base model path
    if is_weight_only and base_model_path is None:
        logger.warning(
            f"⚠️  Weight-only quantization ({quant_mode}) requires base_model_path for proper loading. "
            "The model can be saved but may not load correctly without it."
        )
    
    # Auto-generate model name if not specified
    if model_name is None:
        model_name = f"{model_type}_ptq-{quant_mode}-1.3b"
        logger.info(f"Auto-generated model name: {model_name}")
    
    # Create output directory
    save_path = Path(output_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving quantized model ({quant_mode.upper()})...")
    logger.info(f"{'='*60}")
    logger.info(f"  Output: {save_path}")
    
    # Save model
    # For W8A8: Must save entire model object (QuantizedLinear doesn't serialize with state_dict)
    # For weight-only: Save state_dict (AffineQuantizedTensor works fine)
    model_save_path = save_path / "pytorch_model.bin"
    if quant_mode == "w8a8":
        logger.info("  Saving entire model object (W8A8 with QuantizedLinear)")
        torch.save(model, model_save_path)
    else:
        torch.save(model.state_dict(), model_save_path)
    logger.info(f"✓ Saved model to {model_save_path}")
    
    # Save tokenizer
    # Clean tokenizer to remove any non-JSON-serializable objects (e.g., torch.dtype)
    # This can happen if torch_dtype accidentally leaks into tokenizer config
    def clean_dict_for_json(d):
        """Recursively clean dictionary of non-JSON-serializable objects."""
        if not isinstance(d, dict):
            return d
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, torch.dtype):
                logger.warning(f"  Removing non-serializable torch.dtype from tokenizer config key '{k}'")
                # Convert to string representation
                cleaned[k] = str(v)
            elif isinstance(v, dict):
                cleaned[k] = clean_dict_for_json(v)
            elif isinstance(v, list):
                cleaned[k] = [clean_dict_for_json(item) if isinstance(item, dict) else item for item in v]
            else:
                cleaned[k] = v
        return cleaned
    
    # Clean init_kwargs if present (this is a mutable dict attribute)
    if hasattr(tokenizer, 'init_kwargs') and isinstance(tokenizer.init_kwargs, dict):
        tokenizer.init_kwargs = clean_dict_for_json(tokenizer.init_kwargs)
    
    tokenizer.save_pretrained(save_path)
    logger.info(f"  ✓ Saved tokenizer to {save_path}")
    
    # Save model config (needed for HuggingFace loading)
    if hasattr(model, 'config'):
        model_config_path = save_path / "config.json"
        # Update vocab_size to match tokenizer if mismatch
        if hasattr(tokenizer, 'vocab_size') and hasattr(model.config, 'vocab_size'):
            if model.config.vocab_size != tokenizer.vocab_size:
                logger.warning(f"  ⚠️  Model vocab_size ({model.config.vocab_size}) != tokenizer vocab_size ({tokenizer.vocab_size})")
                logger.info(f"  Updating model config to match tokenizer vocab_size: {tokenizer.vocab_size}")
                model.config.vocab_size = tokenizer.vocab_size
        
        # Try to save config using save_pretrained (works for HuggingFace models)
        try:
            model.config.save_pretrained(save_path)
            logger.info(f"  ✓ Saved model config to {model_config_path}")
        except Exception as e:
            # Fallback: manually save config as JSON (for Mamba models)
            logger.warning(f"  save_pretrained() failed: {e}")
            logger.info(f"  Falling back to manual config.json creation...")
            
            # Convert config to dict
            if hasattr(model.config, 'to_dict'):
                config_dict_to_save = model.config.to_dict()
            elif hasattr(model.config, '__dict__'):
                config_dict_to_save = {k: v for k, v in model.config.__dict__.items() if not k.startswith('_')}
            else:
                config_dict_to_save = {"raw_config": str(model.config)}
            
            # For Mamba models, DON'T add HuggingFace-specific fields
            # (MambaConfig doesn't accept model_type, architectures)
            # For GLA models, add these fields for HuggingFace compatibility
            if model_type == "gla":
                if 'model_type' not in config_dict_to_save:
                    config_dict_to_save['model_type'] = model_type
                if 'architectures' not in config_dict_to_save:
                    arch_name = model.__class__.__name__
                    config_dict_to_save['architectures'] = [arch_name]
            
            # Save as JSON
            with open(model_config_path, 'w') as f:
                json.dump(config_dict_to_save, f, indent=2)
            logger.info(f"  ✓ Manually saved model config to {model_config_path}")
    
    # Build quantization config based on mode
    quant_config = _build_quantization_config(quant_mode)
    
    # Add weight-only specific config
    if is_weight_only:
        quant_config["weight_only"] = True
        quant_config["requires_special_loading"] = True
        if base_model_path:
            quant_config["base_model_path"] = base_model_path
        # Extract group size from metadata or use default
        if metadata and "group_size" in metadata:
            quant_config["group_size"] = metadata["group_size"]
        else:
            quant_config["group_size"] = 128  # default
        # Extract skip_gk_proj setting from metadata
        if metadata and "skip_gk_proj" in metadata:
            quant_config["skip_gk_proj"] = metadata["skip_gk_proj"]
        else:
            quant_config["skip_gk_proj"] = True  # default (conservative)
    
    # Save config with quantization info
    config_dict = {
        "model_type": model_type,
        "model_name": model_name,
        "quantization": quant_config
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
    logger.info(f"  ✓ Saved quantization config to {config_save_path}")
    
    # Count quantized layers
    from torchao_baseline.utils_torchao import QuantizedLinear
    quantized_count = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
    
    # Also count weight-only quantized layers (for W8A16/W4A16)
    weight_only_count = _count_weight_only_quantized_layers(model)
    
    total_quantized = quantized_count + weight_only_count
    
    # Get model size
    model_size_mb = get_directory_size(save_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Model saved successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"  Location: {save_path}")
    logger.info(f"  Quantization mode: {quant_mode.upper()}")
    if is_weight_only:
        logger.info(f"  ⚠️  Weight-only quantization: Requires special loading")
    if quantized_count > 0:
        logger.info(f"  W8A8 QuantizedLinear layers: {quantized_count}")
    if weight_only_count > 0:
        logger.info(f"  Weight-only quantized layers: {weight_only_count}")
    logger.info(f"  Total quantized layers: {total_quantized}")
    logger.info(f"  Size on disk: {model_size_mb:.2f} MB")
    logger.info(f"{'='*60}\n")
    
    return save_path


def load_weight_only_quantized_model(
    model_path: str,
    device: str = "cuda",
):
    """
    Load a weight-only quantized model (W8A16 or W4A16).
    
    This function handles the special loading required for weight-only quantized models:
    1. Loads the quantization config to determine the quantization mode
    2. Loads the base model
    3. Applies quantization (to create AffineQuantizedTensor structure)
    4. Loads the quantized weights
    
    Args:
        model_path: Path to the quantized model directory
        device: Device to load the model on (default: "cuda")
    
    Returns:
        Loaded and quantized model
    """
    import json
    from pathlib import Path
    from fla.models import GLAForCausalLM
    
    # Convert to absolute path to avoid HuggingFace repo ID errors
    # Relative paths starting with './' cause issues with from_pretrained()
    model_path = Path(model_path).resolve()
    
    # Load quantization config
    config_path = model_path / "quantization_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Quantization config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    quant_config = config.get('quantization', {})
    quant_mode = None
    
    # Detect quantization mode from config
    if quant_config.get('weight_bits') == 8 and quant_config.get('activation_bits') == 16:
        quant_mode = 'w8a16'
    elif quant_config.get('weight_bits') == 4 and quant_config.get('activation_bits') == 16:
        quant_mode = 'w4a16'
    else:
        raise ValueError(f"Unsupported quantization config: {quant_config}")
    
    # Get base model path
    base_model_path = quant_config.get('base_model_path')
    if not base_model_path:
        # Try to infer base model path from quantized model path
        # E.g., delta_net_ptq-w8a16-1.3b -> delta_net-1.3b
        model_name = model_path.name if hasattr(model_path, 'name') else str(model_path).split('/')[-1]
        
        # Remove _ptq suffix and quantization mode from name
        if '_ptq-' in model_name:
            base_name = model_name.split('_ptq-')[0]
            size_suffix = model_name.split('-')[-1]  # e.g., "1.3b"
            inferred_base = f"{base_name}-{size_suffix}"
            
            # Construct full path (model_path is already absolute from resolve() above)
            parent_dir = model_path.parent if hasattr(model_path, 'parent') else '/'.join(str(model_path).split('/')[:-1])
            inferred_path = str(Path(parent_dir) / inferred_base)
            
            logger.warning(f"  ⚠️  base_model_path not in config, inferring from model name:")
            logger.warning(f"     Quantized: {model_name}")
            logger.warning(f"     Inferred base: {inferred_path}")
            
            # Check if the inferred path exists
            if Path(inferred_path).exists():
                base_model_path = str(Path(inferred_path).resolve())
                logger.info(f"  ✓ Using inferred base model path: {base_model_path}")
            else:
                logger.error(f"  ✗ Inferred path does not exist: {inferred_path}")
                raise ValueError(
                    f"base_model_path not found in quantization config and could not infer it. "
                    f"Please re-save the model with base_model_path parameter, or ensure the base model "
                    f"exists at the inferred location: {inferred_path}"
                )
        else:
            raise ValueError(
                f"base_model_path not found in quantization config and model name doesn't match expected pattern. "
                "Weight-only quantized models require the base model path to reconstruct. "
                f"Model path: {model_path}"
            )
    
    # Get group_size from config (0 means per-tensor, None means not set)
    group_size_from_meta = config.get('metadata', {}).get('group_size')
    group_size_from_quant = quant_config.get('group_size')
    
    logger.info(f"  Config group_size (metadata): {group_size_from_meta}")
    logger.info(f"  Config group_size (quantization): {group_size_from_quant}")
    
    # Use metadata first (more reliable), then quantization config
    if group_size_from_meta is not None:
        group_size = group_size_from_meta
        logger.info(f"  → Using group_size from metadata: {group_size}")
    elif group_size_from_quant is not None:
        group_size = group_size_from_quant
        logger.info(f"  → Using group_size from quantization config: {group_size}")
    else:
        # No group_size in config - try to detect from weights as fallback
        logger.warning("  ⚠️  No group_size in config, detecting from saved weights...")
        
        state_dict_path = model_path / "pytorch_model.bin"
        if not state_dict_path.exists():
            raise FileNotFoundError(f"Model weights not found at {state_dict_path}")
        
        temp_state = torch.load(state_dict_path, map_location='cpu')
        for key, value in temp_state.items():
            if 'weight' in key and 'norm' not in key and hasattr(value, 'block_size'):
                detected_gs = value.block_size[1] if len(value.block_size) > 1 else None
                # Check if per-tensor (block_size >= tensor dim)
                if detected_gs and detected_gs >= value.shape[1]:
                    group_size = 0  # per-tensor
                    logger.info(f"  ✓ Detected PER-TENSOR quantization")
                else:
                    group_size = detected_gs
                    logger.info(f"  ✓ Detected group_size={group_size}")
                del temp_state
                break
        else:
            # Ultimate fallback
            group_size = 128
            logger.error(f"  ❌ Could not detect group_size, using fallback: {group_size}")
    
    logger.info(f"\n  📌 FINAL group_size to use: {group_size}")
    
    # Get skip_gk_proj setting
    skip_gk_proj = quant_config.get('skip_gk_proj', True)  # Default to True if not specified
    logger.info(f"  skip_gk_proj: {skip_gk_proj}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading weight-only quantized model ({quant_mode.upper()})...")
    logger.info(f"{'='*60}")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Base model: {base_model_path}")
    if group_size == 0:
        logger.info(f"  Quantization: PER-TENSOR (group_size=0)")
    else:
        logger.info(f"  Group size: {group_size}")
    logger.info(f"  Skip gk_proj: {skip_gk_proj}")
    
    # Step 1: Load the base model
    logger.info("\n  Step 1: Loading base model...")
    model_type = config.get('model_type', 'gla')
    
    # Convert base_model_path to absolute path if it's relative
    # HuggingFace from_pretrained doesn't handle relative paths starting with './' well
    from pathlib import Path
    base_model_path = str(Path(base_model_path).resolve())
    logger.info(f"  Resolved base model path: {base_model_path}")
    
    if model_type == 'gla':
        model = GLAForCausalLM.from_pretrained(base_model_path).to(device)
    elif model_type in ['mamba2', 'mamba']:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        dtype = torch.float16 if device == 'cuda' else torch.float32
        model = MambaLMHeadModel.from_pretrained(base_model_path, device=device, dtype=dtype)
    elif model_type in ['delta_net', 'delta_net_ptq']:
        model = DeltaNetForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)
    elif model_type in ['retnet', 'retnet_ptq']:
        model = RetNetForCausalLM.from_pretrained(base_model_path).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info("    ✓ Base model loaded")
    
    # Step 2: Apply quantization with group_size from config
    logger.info(f"\n  Step 2: Applying {quant_mode.upper()} quantization structure...")
    if group_size == 0:
        logger.info(f"    Using group_size=0 (per-tensor quantization)")
    else:
        logger.info(f"    Using group_size={group_size} (per-channel quantization)")
    logger.info(f"    Using skip_gk_proj={skip_gk_proj}")
    
    if quant_mode == 'w8a16':
        model = convert_static_w8a16(model, group_size=group_size, skip_gk_proj=skip_gk_proj)
    elif quant_mode == 'w4a16':
        model = convert_static_w4a16(model, group_size=group_size, skip_gk_proj=skip_gk_proj)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")
    
    logger.info("    ✓ Quantization structure applied")
    
    # Step 3: Load the quantized weights
    logger.info(f"\n  Step 3: Loading quantized weights...")
    state_dict_path = model_path / "pytorch_model.bin"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Model weights not found at {state_dict_path}")
    
    saved_state_dict = torch.load(state_dict_path, map_location=device)
    
    # Load with strict=False
    missing_keys, unexpected_keys = model.load_state_dict(saved_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"    ⚠️  Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                logger.warning(f"      - {key}")
    
    if unexpected_keys:
        logger.warning(f"    ⚠️  Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                logger.warning(f"      - {key}")
    
    logger.info(f"    ✓ Quantized weights loaded successfully")
    
    del saved_state_dict  # Free memory
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Model loaded successfully!")
    logger.info(f"{'='*60}\n")
    
    return model


def _detect_quantization_mode(model: nn.Module) -> str:
    """
    Auto-detect quantization mode from model structure.
    
    Returns one of: "w8a8", "w8a16", "w4a16", "w4a8"
    """
    from torchao_baseline.utils_torchao import QuantizedLinear
    
    # Check for custom QuantizedLinear (W8A8)
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            # Has activation quantization - could be W8A8 or W4A8
            # Check target_dtype if available
            if hasattr(module, 'target_dtype'):
                if module.target_dtype == torch.int8:
                    # Check weight bit-width by inspecting qweight
                    # For now, assume W8A8 if we have QuantizedLinear
                    return "w8a8"
            return "w8a8"  # Default for QuantizedLinear
    
    # Check for weight-only quantization (torchao's built-in)
    # Look for Linear layers with quantized weights
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if weight has quantization attributes
            if hasattr(module.weight, '__class__'):
                weight_type = type(module.weight).__name__
                if 'Affine' in weight_type or 'Quantized' in weight_type:
                    # Weight-only quantization detected
                    # Try to determine bit width from weight shape/dtype
                    # For now, assume W8A16 (most common weight-only)
                    logger.debug(f"Found quantized weight in {name}: {weight_type}")
                    return "w8a16"
    
    # Default if no quantization detected
    logger.warning("Could not auto-detect quantization mode, defaulting to w8a16")
    return "w8a16"


def _build_quantization_config(quant_mode: str) -> Dict[str, Any]:
    """Build quantization config dict based on mode."""
    
    configs = {
        "w8a8": {
            "method": "torchao_w8a8_static",
            "weight_bits": 8,
            "activation_bits": 8,
            "weight_dtype": "int8",
            "activation_dtype": "uint8",
            "weight_granularity": "per_channel",
            "activation_granularity": "per_tensor",
            "weight_symmetric": True,
            "activation_symmetric": False,
            "description": "8-bit weights + 8-bit activations (static PTQ)",
        },
        "w8a16": {
            "method": "torchao_w8a16_weight_only",
            "weight_bits": 8,
            "activation_bits": 16,
            "weight_dtype": "int8",
            "activation_dtype": "float16",
            "weight_granularity": "per_channel",
            "activation_granularity": "none",
            "weight_symmetric": True,
            "activation_symmetric": None,
            "description": "8-bit weights + FP16 activations (weight-only)",
        },
        "w4a16": {
            "method": "torchao_w4a16_weight_only",
            "weight_bits": 4,
            "activation_bits": 16,
            "weight_dtype": "int4",
            "activation_dtype": "float16",
            "weight_granularity": "per_channel",
            "activation_granularity": "none",
            "weight_symmetric": True,
            "activation_symmetric": None,
            "description": "4-bit weights + FP16 activations (weight-only)",
        },
        "w4a8": {
            "method": "torchao_w4a8_static",
            "weight_bits": 4,
            "activation_bits": 8,
            "weight_dtype": "int4",
            "activation_dtype": "uint8",
            "weight_granularity": "per_channel",
            "activation_granularity": "per_tensor",
            "weight_symmetric": True,
            "activation_symmetric": False,
            "description": "4-bit weights + 8-bit activations (static PTQ)",
        },
    }
    
    return configs[quant_mode]


def _count_weight_only_quantized_layers(model: nn.Module) -> int:
    """Count layers that have weight-only quantization (not QuantizedLinear)."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Check if weight is quantized
            if hasattr(module.weight, '__class__'):
                weight_type = type(module.weight).__name__
                if 'Affine' in weight_type or 'Quantized' in weight_type:
                    count += 1
    return count


def get_directory_size(path: str) -> float:
    """Calculate total size of directory in MB."""
    from pathlib import Path
    total_size = 0
    for file in Path(path).rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size / (1024 * 1024)  # Convert to MB
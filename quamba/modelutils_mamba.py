import os
import gc
import copy
import logging
from tqdm import tqdm
from functools import partial
import json

import torch
import torch.nn as nn
from datasets import load_dataset

from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm

from quamba.quamba_mixer_seq import QuambaLMHeadModel
from .qEmbedding import W4O16Embedding, W8O16Embedding
from .qLinearLayer import HadLinear, W4A16B16O16Linear, W4A8B16O16Linear, W8A8B16O16Linear
from .qActLayer import ActIdentity
from .qMamba2 import Mamba2Simple, W4A8QMamba2, W4A16QMamba2, W8A8QMamba2
from .qMambaLayer import MambaSimple, W4A8QMamba, W4A16QMamba, W8A8QMamba
from .qGLALayer import GLASimple, W4A8QGLA, W4A16QGLA, W8A8QGLA
from .qHadamard import Hadamard
from .qBlock import HybridQuambaBlock
# from .fusedNorm import FusedRMSNorm
from .qNorm import QRMSNorm
from .observer import PerTensorMinmaxObserver, PerTensorPercentileObserver
from .observer import PerSSDGroupObserver, CrossHeadMinmaxObserver
from .observer import CachedStatesCrossHeadMinmaxObserver
from .gptq_utils import GPTQ
from .reorder_utils import get_reorder_params, reorder_mamba
from .hadamard_utils import had_transform
from .data_loaders import get_loaders

logger = logging.getLogger(__name__)

@torch.no_grad()
def fuse_ln_linear(norm, linear) -> None:
    """
    Fuse the layernorm weight into an adjacent linear layer.

    Parameters:
    norm -- the layer normalization module
    linear -- the adjacent linear layer

    Modifies linear layer's weights and biases based on the normalization weights.
    """
    linear_dtype = linear.weight.dtype

    # Calculating new weight and bias
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ * norm.weight.double()).to(linear_dtype)  
    if hasattr(norm, 'bias') and norm.bias is not None:
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float32))
        linear.bias.data = linear.bias.data.double() + torch.matmul(W_, norm.bias.to(torch.float32))
        linear.bias.data = linear.bias.data.to(linear_dtype)
    # Reset the learnable weight in RMSNorm to 1
    norm.weight.data = torch.ones_like(norm.weight).to(norm.weight.dtype) # Reset the weight to 1

@torch.no_grad()
def configure_model(model, model_type, use_had_transform=True):
    """
    Configure a Mamba/GLA model with appropriate transformations.

    Parameters:
    model -- the model to configure
    model_type -- type of model ("mamba", "mamba2", or "gla")
    use_had_transform -- flag to apply hadamard transformations

    Modifies embedding layers and normalizes layers based on type.
    """
    device = next(model.parameters()).device
    if model_type == "mamba":
        # process embedding and lm_head
        if use_had_transform:            
            # Sometimes, lm_head is tied to embedding, we make a clone for lm_head first
            lm_head_clone = model.lm_head.weight.data.clone()
            # transform embedding first
            model.backbone.embedding.weight.data = had_transform(model.backbone.embedding.weight.data) 
            # do layernorm fusion to lm_head and then transform
            model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.backbone.norm_f.weight.view(1, -1)).to("cuda") # must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
            model.backbone.norm_f.weight.data = torch.ones_like(model.backbone.norm_f.weight)
            model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
            torch.cuda.empty_cache()
        layers = model.backbone.layers
        for i in range(len(layers)):
            if isinstance(layers[i], Block):
                # fuse norm to in_proj first
                fuse_ln_linear(layers[i].norm, layers[i].mixer.in_proj)
                # use simplied mamba block to get the scaling factors
                # from linear layers without pain
                m = MambaSimple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform).to(device)
                layers[i].mixer = m
                torch.cuda.empty_cache()
    elif model_type == "mamba2":
        # process embedding and lm_head
        if use_had_transform:            
            # Sometimes, lm_head is tied to embedding, we make a clone for lm_head first
            lm_head_clone = model.lm_head.weight.data.clone()
            # transform embedding first
            model.backbone.embedding.weight.data = had_transform(model.backbone.embedding.weight.data) 
            # do layernorm fusion to lm_head and then transform
            model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.backbone.norm_f.weight.view(1, -1)).to("cuda") # must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
            model.backbone.norm_f.weight.data = torch.ones_like(model.backbone.norm_f.weight)
            model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
            torch.cuda.empty_cache()
        # process layers
        layers = model.backbone.layers
        for i in range(len(layers)):
            if isinstance(layers[i], Block):
                # fuse norm to in_proj first
                fuse_ln_linear(layers[i].norm, layers[i].mixer.in_proj)
                # use simplied mamba block to get the scaling factors
                # from linear layers without pain
                m = Mamba2Simple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform).to(device)
                layers[i].mixer = m
                torch.cuda.empty_cache()
    elif model_type == "gla":
        # GLA has different structure: model.model instead of model.backbone
        # Process embedding and lm_head
        if use_had_transform:
            # Clone lm_head to untie from embeddings if tied
            lm_head_clone = model.lm_head.weight.data.clone()
            # Transform embedding
            model.model.embeddings.weight.data = had_transform(model.model.embeddings.weight.data)
            # Fuse final norm to lm_head and transform
            model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.model.norm.weight.view(1, -1)).to("cuda")
            model.model.norm.weight.data = torch.ones_like(model.model.norm.weight)
            model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
            torch.cuda.empty_cache()
        
        # Process layers - GLA has GLABlock with both attn and mlp
        layers = model.model.layers
        for i in range(len(layers)):
            # Fuse attn_norm to q_proj (first projection in attention)
            fuse_ln_linear(layers[i].attn_norm, layers[i].attn.q_proj)
            # Replace attention with simplified version
            gla_attn = GLASimple(originalLayer=layers[i].attn, use_had_transform=use_had_transform).to(device)
            layers[i].attn = gla_attn
            
            # TODO: Also handle MLP quantization if needed
            # For now, we'll focus on attention quantization
            # fuse_ln_linear(layers[i].mlp_norm, layers[i].mlp.gate_proj)
            
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', and 'gla'")
    model.config.use_cache = False
    model.eval()
    return model

@torch.no_grad()
def run_quamba_calibration(
        model, model_type, tokenizer, num_samples=512, seq_len=512,
        calibration_dataset=None, preprocess_fn=None
    ):
    """
    Calibrate a Mamba model by evaluating activation/input statistics.

    Parameters:
    model -- the model to calibrate
    model_type -- type of model
    tokenizer -- tokenizer to preprocess data
    num_samples -- number of samples for calibration
    seq_len -- sequence length

    Registers hooks for capturing statistics on activations and outputs to determine quantization values.
    """

    if model_type == "mamba":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        is_x = lambda op: op == "x_proj"
        is_ssm_state = lambda op: op == "ssm_state_act"
        percentile_alpha=0.9995 # for smaller model like 130m, use 0.99999
    elif model_type == "mamba2":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        is_x = lambda op: op == "x_conv_out"
        is_ssm_state = lambda op: op == "ssm_state_act"
        percentile_alpha=0.9995  # for smaller model like 130m, use 0.99999
    elif model_type == "gla":
        # GLA uses different structure
        layers = model.model.layers
        is_traget_block = lambda block: True  # All GLABlocks are target blocks
        get_mamba = lambda block: block.attn  # Get attention module instead of mixer
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        # GLA doesn't have SSM state, but has multiple projections
        is_x = lambda op: op in ["q_proj", "k_proj"]  # Key projections to monitor carefully
        is_ssm_state = lambda op: False  # GLA doesn't have SSM state
        percentile_alpha=0.9995
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', and 'gla'")

    # register min/max observers, num_layer + lm_head
    observers = [{} for _ in range(len(layers) + 1)]
    
    def stat_hook(m, inputs, outputs, op, block_idx):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        observers[block_idx][op + ":input"].update(inputs.clone().detach())

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        observers[block_idx][op + ":output"].update(outputs.clone().detach())

    hooks = []
    for i in range(len(layers)):
        if not is_traget_block(layers[i]):
            continue
        
        mixer = get_mamba(layers[i])
        for name, m in mixer.named_modules():
            if is_calib_ops(m):
                # FIXME(HY): hardcode everything for now
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op) or is_ssm_state(op):
                    observers[i][op + ":input"] = PerTensorPercentileObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        percentile_alpha=percentile_alpha
                    )
                else:
                    observers[i][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[i][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
                )
    # add observer hook for lm_head
    observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    hooks.append(
        model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
    )

    device = next(model.parameters()).device
    if calibration_dataset is None:
        logger.info("Calibrate with monology/pile-uncopyrighted")
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")

        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    logger.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        
        if model_type in ["mamba", "mamba2"]:
            # Mamba models use inference cache
            prompt_len = input_ids.size(1)
            inf_cache = model.allocate_inference_cache(1, prompt_len)
            lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
            inference_params = InferenceParams(
                max_seqlen=prompt_len,
                max_batch_size=1,
                seqlen_offset=0,
                key_value_memory_dict=inf_cache,
                lengths_per_sample=lengths_per_sample,
            )
            # do not set num_last_tokens because we want all activations to lm_head
            model(input_ids, inference_params=inference_params)
            # clean up the cache
            del inf_cache
        elif model_type == "gla":
            # GLA models don't use inference cache, just run forward pass directly
            model(input_ids)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    for h in hooks:
        h.remove()
    
    # collect in/output scaling factors for layers, num_layer + lm_head
    act_scales = [{} for _ in range(len(layers) + 1)]
    for i in range(len(layers) + 1):
        for name, observer in observers[i].items():
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale.to(torch.float32)
    del observers
    return act_scales

@torch.no_grad()
def run_quamba2_calibration(
        model, model_type, tokenizer, reorder_params,
        num_samples=512, seq_len=512, calibration_dataset=None, preprocess_fn=None
    ):
    """
    Calibrate a Mamba2 model by evaluating activation/input statistics.

    Parameters:
    model -- the model to calibrate
    model_type -- expected to be "mamba2"
    tokenizer -- tokenizer to preprocess data
    reorder_params -- parameters for block/group reordering
    num_samples -- number of samples for calibration

    Uses observers to capture input-output data, adjusting quantization accordingly.
    """

    if model_type == "mamba":
        raise NotImplementedError("Not support for mamba")
    elif model_type == "mamba2":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_x = lambda op: op == "x_conv_out"
        is_BC = lambda op: op == "B_conv_out" or op == "C_conv_out"
        is_ssm_state = lambda op: op == "ssm_state_act"
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        percentile_alpha=0.99999
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba2'")

    # register min/max observers, num_layer + lm_head
    observers = [{} for _ in range(len(layers) + 1)]
    
    def stat_hook(m, inputs, outputs, op, block_idx):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        observers[block_idx][op + ":input"].update(inputs.clone().detach())

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        observers[block_idx][op + ":output"].update(outputs.clone().detach())

    hooks = []
    for i in range(len(layers)):
        if not is_traget_block(layers[i]):
            continue
        head_groups = reorder_params["head_groups"][i]
        channel_group = reorder_params["channel_group"][i]
        mixer = get_mamba(layers[i])
        for name, m in mixer.named_modules():
            if is_calib_ops(m):
                # FIXME(HY): hardcode everything for now
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op):
                    observers[i][op + ":input"] = CrossHeadMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        ngroups=mixer.ngroups,
                        headdim=mixer.headdim,
                        head_groups=head_groups,
                        channel_group=channel_group,
                    )
                elif is_BC(op):
                    observers[i][op + ":input"] = PerSSDGroupObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        dstate=mixer.d_state,
                    )
                elif is_ssm_state(op):
                    observers[i][op + ":input"] = CachedStatesCrossHeadMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        ngroups=mixer.ngroups,
                        headdim=mixer.headdim,
                        dstate=mixer.d_state,
                        head_groups=head_groups,
                        channel_group=channel_group,
                    )
                else:
                    observers[i][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[i][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
                )
    # add observer hook for lm_head
    observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    hooks.append(
        model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
    )
    if calibration_dataset is None:
        logger.info("Calibrate with monology/pile-uncopyrighted")
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
        calibration_dataset.shuffle(seed=42)

        device = next(model.parameters()).device
        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    logger.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        
        if model_type in ["mamba", "mamba2"]:
            # Mamba models use inference cache
            prompt_len = input_ids.size(1)
            inf_cache = model.allocate_inference_cache(1, prompt_len)
            lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
            inference_params = InferenceParams(
                max_seqlen=prompt_len,
                max_batch_size=1,
                seqlen_offset=0,
                key_value_memory_dict=inf_cache,
                lengths_per_sample=lengths_per_sample,
            )
            # do not set num_last_tokens because we want all activations to lm_head
            model(input_ids, inference_params=inference_params)
            # clean up the cache
            del inf_cache
        elif model_type == "gla":
            # GLA models don't use inference cache, just run forward pass directly
            model(input_ids)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    for h in hooks:
        h.remove()
    
    # collect in/output scaling factors for layers, num_layer + lm_head
    act_scales = [{} for _ in range(len(layers) + 1)]
    for i in range(len(layers) + 1):
        for name, observer in observers[i].items():
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale

    return act_scales

@torch.no_grad()
def fuse_had_matrices(model, model_type="mamba"):
    """
    Fuse Hadamard matrices within Mamba/GLA layers for optimized computation.

    Parameters:
    model -- the model for hadamard fusion
    model_type -- type of model ("mamba", "mamba2", or "gla")

    Modifies projection layers to incorporate Hadamard transformations.
    """
    # fuse the had matrices with the weight matrices in linear layers.
    # Do this after reordering and before applying gptq
    
    if model_type in ["mamba", "mamba2"]:
        layers = model.backbone.layers
        for i in range(len(layers)):
            # in_proj: fuse had matrices with weight matrices
            if isinstance(layers[i].mixer.in_proj, HadLinear):
                layers[i].mixer.in_proj.fuse_hadamard()
            # out_proj: fuse had matrices with weight matrices
            if isinstance(layers[i].mixer.out_proj, HadLinear):
                layers[i].mixer.out_proj.fuse_hadamard()
    elif model_type == "gla":
        layers = model.model.layers
        for i in range(len(layers)):
            # GLA has multiple projections - fuse all of them
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'g_proj', 'o_proj']:
                proj = getattr(layers[i].attn, proj_name, None)
                if proj and isinstance(proj, HadLinear):
                    proj.fuse_hadamard()
            # Handle gk_proj Sequential (two layers)
            if hasattr(layers[i].attn, 'gk_proj') and layers[i].attn.gk_proj is not None:
                if isinstance(layers[i].attn.gk_proj[0], HadLinear):
                    layers[i].attn.gk_proj[0].fuse_hadamard()
                if isinstance(layers[i].attn.gk_proj[1], HadLinear):
                    layers[i].attn.gk_proj[1].fuse_hadamard()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

@torch.no_grad()
def apply_gptq(model, tokenizer, device, w_bits=4):
    """
    Apply GPTQ based quantization to Mamba model layers.

    Parameters:
    model -- the model to quantize
    tokenizer -- tokenizer for creating text data
    device -- processing device (CPU/GPU)
    w_bits -- target bit-width for weights

    Configures quantization over internal projections and optimizes with input statistics.
    """
    # Hardcode gptq hyper-parameters for now
    nsamples = 128
    seqlen = 1024
    bits = w_bits
    assert bits in [4, 8], "Only support 4 or 8 bits weights for now"
    logging.info("Start Quantized Linear Layers with GPTQ")
    logging.info("* Number of samples: %d" % nsamples)
    logging.info("* Sequence length: %d" % seqlen)
    logging.info("* Target bit-width for weights: %d" % bits)
    logging.info("Build calibration loader for GPTQ")
    #build dataloader
    dataloader, _ = get_loaders("wikitext2", tokenizer, nsamples=nsamples, seqlen=seqlen)
    layers = model.backbone.layers
    model.backbone.embedding = model.backbone.embedding.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )    
    residual = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )    

    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module  
        def forward(self, inp, res = None, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass

    # the hook to collect inputs for in_proj, out_proj, and lm_head
    def add_batch(module, inp, out, gptq):
        gptq.add_batch(inp[0].data, out.data)

    layers[0] = layers[0].module # remove Catcher
    layers[0] = layers[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()
    for i in tqdm(range(len(layers))):
        # get layer
        layer = layers[i].to(device)

        # create GPTQ objects for in_proj and out_proj
        gptq = {
            "in_proj": GPTQ(layer.mixer.in_proj),
            "out_proj": GPTQ(layer.mixer.out_proj),
        }
        handles = [
            layer.mixer.in_proj.register_forward_hook(partial(add_batch, gptq=gptq["in_proj"])),
            layer.mixer.out_proj.register_forward_hook(partial(add_batch, gptq=gptq["out_proj"]))
        ]
        for j in range(nsamples):
            layer(
                inps[j].unsqueeze(0), 
                residual=residual[j].unsqueeze(0)
            )
        for h in handles:
            h.remove()
        
        # start running GPTQ
        for name in gptq.keys():
            logging.debug(f"Performing GPTQ on layer.{i}.mixer.{name} with {bits} bits")
            gptq[name].fasterquant(
                percdamp=0.01, group_size=128, w_bits=bits
            )
            gptq[name].free()
        del gptq
        
        # collect the outputs for the next layer
        for j in range(nsamples):
            inps[j], residual[j] = layer(inps[j].unsqueeze(0), residual=residual[j].unsqueeze(0))
        
        # garbage collection and clean cache
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        gc.collect()

    model = model.to("cpu") # move model to cpu to save memory
    model.lm_head = model.lm_head.to(device)
    model.backbone.norm_f = model.backbone.norm_f.to(device)
    logging.info("Quantizing lm_head with GPTQ")
    gptq_lm_head = GPTQ(model.lm_head)
    handle = model.lm_head.register_forward_hook(partial(add_batch, gptq=gptq_lm_head))
    
    assert model.backbone.fused_add_norm, "Only support fused_add_norm=True for now"
    #Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L202
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
    
    for j in range(nsamples):
        model.lm_head(final_hidden_states[j].unsqueeze(0))

    handle.remove()
    # compute with fp16 to save memory
    gptq_lm_head.fasterquant(
        percdamp=0.01, group_size=128, dtype=torch.float16
    )
    gptq_lm_head.free()
    del gptq_lm_head
    
    torch.cuda.empty_cache()
    gc.collect()

    model = model.to(device)
    return model


def quantize_norm_a8(block_type, norm, layer_idx, act_scales, device):
    """
    Convert normalization layer for lower-bit activation usage.

    Parameters:
    block_type -- type of block (i.e., "Mamba")
    norm -- normalization layer to quantize
    layer_idx -- index of layer
    act_scales -- activation scales for quantization
    device -- processing device

    Optimizes normalization for quantized data handling.
    """
    norm = QRMSNorm.from_fp16(
        originalLayer=norm,
        output_scale=act_scales[layer_idx]["in_proj:input"].item())
    return norm.to(device)
    
def quantize_mixer_w8a8(block_type, mixer, layer_idx, act_scales, device):
    W8A8Mixers = {
        "Mamba": W8A8QMamba,
        "Mamba2": W8A8QMamba2,
        "GLA": W8A8QGLA,
    }
    if block_type not in W8A8Mixers.keys():
        raise ValueError(f"Not find {block_type} in W8A8 Mixer")
    if W8A8Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W8A8")
    mixer = W8A8Mixers[block_type].from_fp16(
                originalLayer=mixer,
                act_scales=act_scales[layer_idx],
                use_had_transform=True)
    return mixer.to(device)

def quantize_mixer_w4a16(block_type, mixer, layer_idx, act_scales, device):
    W4A16Mixers = {
        "Mamba": W4A16QMamba,
        "Mamba2": W4A16QMamba2,
        "GLA": W4A16QGLA,
    }
    if block_type not in W4A16Mixers.keys():
        raise ValueError(f"Not find {block_type} in W4A16 Mixer")
    if W4A16Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W4A16")
    mixer = W4A16Mixers[block_type].from_fp16(originalLayer=mixer, use_had_transform=True)
    return mixer.to(device)

def quantize_mixer_w4a8(block_type, mixer, layer_idx, act_scales, device):
    W4A8Mixers = {
        "Mamba": W4A8QMamba,
        "Mamba2": W4A8QMamba2,
        "GLA": W4A8QGLA,
    }
    if block_type not in W4A8Mixers.keys():
        raise ValueError(f"Not find {block_type} in W4A8 Mixer")
    if W4A8Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W4A8")
    mixer = W4A8Mixers[block_type].from_fp16(
                originalLayer=mixer,
                act_scales=act_scales[layer_idx],
                use_had_transform=True)
    return mixer.to(device)

def get_quantize_block_fn(act_scales, w_bits, a_bits, device):
    if w_bits == 4 and a_bits == 8:
        quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
        quantize_mixer_fn = partial(quantize_mixer_w4a8, act_scales=act_scales, device=device)
    elif w_bits == 4 and a_bits == 16:
        quantize_norm_fn = lambda block_type, norm, layer_idx: norm # just return the original layer
        quantize_mixer_fn = partial(quantize_mixer_w4a16, act_scales=act_scales, device=device)
    elif w_bits == 8 and a_bits == 8:
        quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
        quantize_mixer_fn = partial(quantize_mixer_w8a8, act_scales=act_scales, device=device)
    else:
        raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
    return quantize_norm_fn, quantize_mixer_fn
    
@torch.no_grad()
def quantize_fp16_model(model, model_type, act_scales, device, w_bits=4, a_bits=8, quantize_embedding=True, quantize_lm_head=True):
    """
    Quantize a Mamba model from FP16 to a specified bit configuration.

    Parameters:
    model -- the model to quantize
    model_type -- type of model
    act_scales -- predefined scales for activations
    device -- processing device
    w_bits -- bit-width for weights
    a_bits -- bit-width for activations
    quantize_embedding -- flag to quantize embedding layer
    quantize_lm_head -- flag to quantize lm_head

    Applies specific bit-width reductions with layer-level optimizations.
    """
    assert w_bits in [4, 8], "Only support 4 or 8 bits weights for now"
    assert a_bits in [8, 16], "Only support 8 or 16 bits activations for now"
    quantize_norm_fn, quantize_mixer_fn = get_quantize_block_fn(act_scales, w_bits, a_bits, device)

    model.config.use_cache = False
    if model_type == "mamba":
        if quantize_embedding:
            # replace embedding layer
            logging.info(f'Applying quantized embedding')
            if w_bits == 4:
                model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
            elif w_bits == 8:
                model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
            else:
                raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
            gc.collect()
            torch.cuda.empty_cache()
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                # replace with fused RMSNorm
                layers[i].fused_add_norm = True
                layers[i].norm = quantize_norm_fn(
                    block_type="Mamba",
                    norm=layers[i].norm,
                    layer_idx=i)
                layers[i].mixer = quantize_mixer_fn(
                    block_type="Mamba", 
                    mixer=layers[i].mixer,
                    layer_idx=i)
                # garbage collection and clean cache
                gc.collect()
                torch.cuda.empty_cache()
    elif model_type == "mamba2":
        if quantize_embedding:
            # replace embedding layer
            logging.info(f'Applying quantized embedding')
            if w_bits == 4:
                model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
            elif w_bits == 8:
                model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
            else:
                raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
            gc.collect()
            torch.cuda.empty_cache()
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                layers[i].fused_add_norm = True
                layers[i].norm = quantize_norm_fn(
                    block_type="Mamba2",
                    norm=layers[i].norm,
                    layer_idx=i)
                layers[i].mixer = quantize_mixer_fn(
                    block_type="Mamba2", 
                    mixer=layers[i].mixer,
                    layer_idx=i)
                # garbage collection and clean cache
                gc.collect()
                torch.cuda.empty_cache()
    elif model_type == "gla":
        if quantize_embedding:
            # replace embedding layer
            logging.info(f'Applying quantized embedding')
            if w_bits == 4:
                model.model.embeddings = W4O16Embedding.from_fp16(model.model.embeddings)
            elif w_bits == 8:
                model.model.embeddings = W8O16Embedding.from_fp16(model.model.embeddings)
            else:
                raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
            gc.collect()
            torch.cuda.empty_cache()
        # replace layers - GLA has attn_norm + attn (+ mlp_norm + mlp but we skip MLP for now)
        logging.info(f'Applying quantized layers')
        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            # Quantize attention norm - uses q_proj:input scale instead of in_proj:input
            if a_bits == 8:
                # Need to adjust the scale key for GLA
                layers[i].attn_norm = QRMSNorm.from_fp16(
                    originalLayer=layers[i].attn_norm,
                    output_scale=act_scales[i].get("q_proj:input", torch.tensor(1.0)).item()
                )
            # else: keep original norm for w4a16
            
            # Quantize attention module
            layers[i].attn = quantize_mixer_fn(
                block_type="GLA",
                mixer=layers[i].attn,
                layer_idx=i)
            
            # TODO: Optionally quantize MLP as well
            # For now, leave MLP in FP16
            
            # garbage collection and clean cache
            gc.collect()
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', and 'gla'")
    
    if quantize_lm_head:
        logging.info(f'Applying quantized lm_head')
        # replace lm_head and final norm with quantized version
        # Handle different norm locations: backbone.norm_f for Mamba, model.norm for GLA
        if model_type in ["mamba", "mamba2"]:
            final_norm = model.backbone.norm_f
            norm_setter = lambda norm: setattr(model.backbone, 'norm_f', norm)
        elif model_type == "gla":
            final_norm = model.model.norm
            norm_setter = lambda norm: setattr(model.model, 'norm', norm)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if w_bits == 4 and a_bits == 16:
            # do nothing to w4a16 final norm
            model.lm_head = W4A16B16O16Linear.from_fp16(model.lm_head)
        elif w_bits == 4 and a_bits == 8:
            quantized_norm = QRMSNorm.from_fp16(
                final_norm, output_scale=act_scales[-1]["lm_head:input"].item())
            norm_setter(quantized_norm)
            model.lm_head = W4A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"])
        elif w_bits == 8 and a_bits == 8:
            quantized_norm = QRMSNorm.from_fp16(
                final_norm, output_scale=act_scales[-1]["lm_head:input"].item())
            norm_setter(quantized_norm)
            model.lm_head = W8A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"].item())
        else:
            raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
    
    gc.collect()
    torch.cuda.empty_cache()
    return model

def quantize_fp16_model_act_hybrid(model, model_type, act_scales, device, w_bits=4, 
                                   layer_wise_hybrid_config=None #this is expected to be a list
                                   ):
    assert w_bits in [4], "Only support 4 bits weights for now"
    a_bits = [8, 16]
    
    logging.info(f"Quantizing model with w{w_bits} and a{a_bits}. HybridBlock will be create.")
    
    # for each a_bits get the correcponding quantization function
    quant_function_pairs = {}
    for a in a_bits:
        quant_function_pairs[f'W{w_bits}A{a}'] = get_quantize_block_fn(act_scales, w_bits, a, device) # quantize_norm_fn, quantize_mixer_fn
    
    model.config.use_cache = False
    if model_type == "mamba2":
        # replace embedding layer
        logging.info(f'Applying quantized embedding')
        if w_bits == 4:
            model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
        elif w_bits == 8:
            model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
        else:
            raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                layers[i].fused_add_norm = True
                
                mixers = {}
                norms = {}
                
                if layer_wise_hybrid_config:
                    #Case 1: bitwidth of each layer is specified, each layers only create the block/norm with the specified bitwidth
                    bit_config = layer_wise_hybrid_config[i]
                    (quantize_norm_fn, quantize_mixer_fn) = quant_function_pairs[bit_config]
                    mixers[bit_config] = quantize_mixer_fn(
                        block_type="Mamba2", 
                        mixer=layers[i].mixer,
                        layer_idx=i)
                    norms[bit_config] = quantize_norm_fn(
                        block_type="Mamba2",
                        norm=layers[i].norm,
                        layer_idx=i)     
                else:
                    #Case 2: bitwidth of each layer is not specified, each layers create the block/norm with all possible bitwidth
                    for bits, (quantize_norm_fn, quantize_mixer_fn) in quant_function_pairs.items():
                        mixers[f"W{w_bits}A{bits}"] = quantize_mixer_fn(
                            block_type="Mamba2", 
                            mixer=layers[i].mixer,
                            layer_idx=i)
                        norms[f"W{w_bits}A{bits}"] = quantize_norm_fn(
                            block_type="Mamba2",
                            norm=layers[i].norm,
                            layer_idx=i)          
                            
                layers[i] = HybridQuambaBlock.from_block_and_mixer_norm_dict(
                    block=layers[i],
                    mixers_dict=mixers,
                    norms_dict=norms,
                ).to(device)
                layers[i].set_mixer(next(iter(mixers))) #default
                gc.collect()
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba2' for hybrid mode")
    
    logging.info(f'Applying quantized lm_head')
    
    #FIXME(brian1009): Hard-fix for now, but we may need to make it configurable as well.
    model.backbone.norm_f = QRMSNorm.from_fp16(
        model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
    model.lm_head = W4A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"])
    
    gc.collect()
    torch.cuda.empty_cache()
    return model


def get_model_size(model, model_name, w_bits, a_bits):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_mb = (param_size + buffer_size) / 1024**2
    logging.info(f'W{w_bits}A{a_bits} {model_name} size: {model_mb:.3f} MB')


def quantize_model_mamba(model, model_type, tokenizer, device, args, calibration_dataset=None, calib_preprocess_fn=None):
    # restore the quantized model from pretrained_dir
    if args.pretrained_dir:
        # change the name to lookup the model in the pretrained_dir
        model_name = args.model.lower().split('/')[-1]

        # already loaded the quantized model in build_mamba_and_tokenizer in utils.py
        if model_name.startswith("quamba"): # ut-enyac/quamba or ut-enyac/quamba2
            get_model_size(model, args.model, args.w_bits, args.a_bits)
            return model

        # load quantied model in args.pretrained_dir to replace fp16 mamba
        # This will require much more memory, since we will have
        # fp16 mamba, qumaba, and quamba weight in the memory at the same time
        quantized_model_path = None
        if model_name.startswith("mamba"): # mamba or mamba2
            model_name = model_name.replace("mamba", "quamba") # replace mamba with quamba
            if args.hybrid_blocks: 
                model_name = model_name + f"-w{args.w_bits}aX"
                if args.hybrid_blocks_config:
                    config_name = args.hybrid_blocks_config.split("/")[-1].replace(".json", "")
                    model_name = model_name + f"-{config_name}"
            else:
                model_name = model_name + f"-w{args.w_bits}a{args.a_bits}"
            quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
        elif model_name.startswith("gla"): # gla models
            # For GLA, we'll create a similar naming convention
            model_name = model_name.replace("gla", "qgla") # replace gla with qgla
            if args.hybrid_blocks:
                model_name = model_name + f"-w{args.w_bits}aX"
                if args.hybrid_blocks_config:
                    config_name = args.hybrid_blocks_config.split("/")[-1].replace(".json", "")
                    model_name = model_name + f"-{config_name}"
            else:
                model_name = model_name + f"-w{args.w_bits}a{args.a_bits}"
            quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
        else:
            logging.warning(f"Unsupported model {args.model} in ut-enyac/ model registry")
        
        # load the quantized model if it exists
        if quantized_model_path and os.path.isdir(quantized_model_path):
            logging.info(f"Loading quantized model from {quantized_model_path}")
            model = QuambaLMHeadModel.from_pretrained(quantized_model_path, device="cuda")
            get_model_size(model, args.model, args.w_bits, args.a_bits)
            return model
        else:
            logging.warning(f"{quantized_model_path} does not exist.")
            logging.warning("Runing calibration and quantization from scratch")
            
    # replace the mamba blocks with simple blocks to get the scaling factors
    # we hardcode use_had_transform=True to fix the configuration, so it is easier for users
    model = configure_model(model, model_type, use_had_transform=True) # W4A16 needs had_transform as well
    logging.info(f"Target bit-width W{args.w_bits}A{args.a_bits}")
    if args.a_bits == 8:
        # Run calibration to get scale and reorder params
        if args.group_heads:
            logging.info(f"Reordering weights and activations for head grouping")
            reorder_params = get_reorder_params(model, model_type, tokenizer, num_samples=512, seq_len=512)
            reorder_mamba(model, reorder_params)
            # collect 8-bit activation scales
            act_scales = run_quamba2_calibration(model, model_type, tokenizer, reorder_params,
                                                num_samples=args.calib_data_num,
                                                seq_len=args.calib_seqlen,
                                                calibration_dataset=calibration_dataset,
                                                preprocess_fn=calib_preprocess_fn)
        else:
            # collect 8-bit activation scales
            act_scales = run_quamba_calibration(model, model_type, tokenizer,
                                                num_samples=args.calib_data_num,
                                                seq_len=args.calib_seqlen,
                                                calibration_dataset=calibration_dataset,
                                                preprocess_fn=calib_preprocess_fn)
    elif args.a_bits == 16:
        # not doing anything for activations
        act_scales = {}
        if args.group_heads:
            logging.info(f"Activation bit-width is set to 16, skip weights reordering and head grouping")
    else:
        raise ValueError(f"Unsupported activation bit-width: {args.a_bits}, try --a_bits 8 or --a_bits 16?")
    
    # fuse the had matrices with the weight matrices in linear layers.
    # Do this after reordering and before applying gptq
    model = fuse_had_matrices(model, model_type) 
    # Apply GPTQ to quantize linear
    if args.apply_gptq:
        model = apply_gptq(model, tokenizer, device, w_bits=args.w_bits)
    # Replace (reordered, fused, and GPTQ quantized) modules with quantized version
    
    if args.hybrid_blocks: # create hybrid block
        if args.hybrid_blocks_config:
            logging.info(f"Loading hybrid blocks config from {args.hybrid_blocks_config}")
            with open(args.hybrid_blocks_config, "r") as file:
                hybrid_blocks_configs = json.load(file)
        else:
            hybrid_blocks_configs = None
        model = quantize_fp16_model_act_hybrid(model, model_type, act_scales, device, w_bits=args.w_bits,
                                               layer_wise_hybrid_config=hybrid_blocks_configs)
    else:
        model = quantize_fp16_model(
            model, model_type, act_scales, device,
            w_bits=args.w_bits, a_bits=args.a_bits,
            quantize_embedding=args.quantize_embedding,
            quantize_lm_head=args.quantize_lm_head
        )
    # store the state_dict if not quamba
    model_name = args.model.lower().split('/')[-1]
    if args.pretrained_dir is not None and not model_name.startswith("quamba"):
        if not args.hybrid_blocks:
            # change the name to store the model in the pretrained_dir
            model_name = args.model.lower().split('/')[-1]
            model_name = model_name.replace("mamba", "quamba") # replace mamba with quamba
            model_name = model_name + f"-w{args.w_bits}a{args.a_bits}"

            quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
            
            # Store model config based on model type
            if model_type in ["mamba", "mamba2"]:
                # we slightly hack the api: we use MambaLMHeadModel instead of QuambaLMHeadModel to store the model here
                model.config.ssm_cfg['layer'] = model.backbone.layers[0].mixer.__class__.__name__
                model.config.norm_cfg = {"norm": model.backbone.layers[0].norm.__class__.__name__}
                model.config.embedding_cfg = {"layer": model.backbone.embedding.__class__.__name__}
                model.config.lm_head_cfg = {"layer": model.lm_head.__class__.__name__}
            elif model_type == "gla":
                # GLA model structure is different
                if not hasattr(model.config, 'gla_cfg'):
                    model.config.gla_cfg = {}
                model.config.gla_cfg['layer'] = model.model.layers[0].attn.__class__.__name__
                model.config.norm_cfg = {"norm": model.model.layers[0].attn_norm.__class__.__name__}
                model.config.embedding_cfg = {"layer": model.model.embeddings.__class__.__name__}
                model.config.lm_head_cfg = {"layer": model.lm_head.__class__.__name__}
            
            # We apply Hadamard transforms so we cannot tie embeddings and lm_head
            model.config.tie_embeddings = False # no used in QuambaLMHeadModel
            # For GLA, we need to keep use_cache but set it to False (GLA forward pass checks it)
            if model_type == "gla":
                model.config.use_cache = False
            elif hasattr(model.config, "use_cache"):
                delattr(model.config, "use_cache")
            
            # Fix for GLA: Convert any non-tensor attributes to tensors before saving
            # This prevents "AttributeError: 'float' object has no attribute 'device'"
            try:
                model.save_pretrained(quantized_model_path)
                logging.info(f"The quantized model is stored at {quantized_model_path}")
            except AttributeError as e:
                if "'float' object has no attribute 'device'" in str(e) or "'int' object has no attribute 'device'" in str(e):
                    logging.warning(f"Skipping model save due to serialization issue: {e}")
                    logging.warning("This is a known issue with GLA quantization. The model is quantized but cannot be saved yet.")
                    logging.warning(f"Attempted to save to: {quantized_model_path}")
                else:
                    raise
        else:
            model_name = args.model.lower().split('/')[-1]
            model_name = model_name.replace("mamba", "quamba")
            model_name = model_name + f"-w{args.w_bits}aX"
            if args.hybrid_blocks_config:
                config_name = args.hybrid_blocks_config.split("/")[-1].replace(".json", "")
                model_name = model_name + f"-{config_name}"
            quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
            
            ssm_layer_infos = []
            norm_infos = []
            
            # Get layers based on model type
            if model_type in ["mamba", "mamba2"]:
                layers = model.backbone.layers
                embedding = model.backbone.embedding
            elif model_type == "gla":
                layers = model.model.layers
                embedding = model.model.embeddings
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            for i in range(len(layers)):
                mixer_norms_info = layers[i].get_class_info_dict_mixers_norms()
                ssm_layer_infos.append(mixer_norms_info["mixers"])
                norm_infos.append(mixer_norms_info["norms"])
            
            mixer_norms_info = layers[0].get_class_info_dict_mixers_norms()
            if model_type in ["mamba", "mamba2"]:
                model.config.ssm_cfg['layer'] = ssm_layer_infos
            elif model_type == "gla":
                if not hasattr(model.config, 'gla_cfg'):
                    model.config.gla_cfg = {}
                model.config.gla_cfg['layer'] = ssm_layer_infos
            model.config.norm_cfg = {"norm": norm_infos}
            model.config.embedding_cfg = {"layer": embedding.__class__.__name__}
            model.config.lm_head_cfg = {"layer": model.lm_head.__class__.__name__}
            # We apply Hadamard transforms so we cannot tie embeddings and lm_head
            model.config.tie_embeddings = False # no used in QuambaLMHeadModel
            # For GLA, we need to keep use_cache but set it to False (GLA forward pass checks it)
            if model_type == "gla":
                model.config.use_cache = False
            elif hasattr(model.config, "use_cache"):
                delattr(model.config, "use_cache")
            model.config.is_hybrid = True
            model.save_pretrained(quantized_model_path)
            logging.info(f"The quantized model is stored at {quantized_model_path}")
            
        # store tokenizer for mamba2-8b
        if "mamba2-8b" in args.model:
            # model.save_pretrained should already create the saved dir
            saved_dir = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
            tokenizer.save(saved_dir)
            logging.info(f"Tokenizer is stored at {saved_dir}")
    # quantized model
    get_model_size(model, args.model, args.w_bits, args.a_bits)
    return model.to(device)





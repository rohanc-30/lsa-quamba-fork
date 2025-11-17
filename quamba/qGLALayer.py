"""
Quantized GLA (Gated Linear Attention) layer implementations.
Supports W4A16, W4A8, and W8A8 quantization schemes.
"""

import torch
import torch.nn as nn
from typing import Optional

from .qLinearLayer import HadLinear, W4A16B16O16Linear, W4A8B16O16Linear, W8A8B16O16Linear
from .hadamard_utils import had_transform


class GLASimple(nn.Module):
    """
    Simplified GLA attention module that exposes linear layers for quantization.
    Similar to MambaSimple but adapted for GLA's multi-projection structure.
    """
    
    def __init__(self, originalLayer, use_had_transform: bool = True):
        """
        Initialize simplified GLA layer from original GatedLinearAttention layer.
        
        Args:
            originalLayer: Original GatedLinearAttention module from fla
            use_had_transform: Whether to apply Hadamard transformation
        """
        super().__init__()
        
        # Store configuration from original layer
        self.hidden_size = originalLayer.hidden_size if hasattr(originalLayer, 'hidden_size') else 2048
        self.num_heads = originalLayer.num_heads if hasattr(originalLayer, 'num_heads') else None
        self.use_had_transform = use_had_transform
        
        # Copy additional GLA-specific attributes needed for forward pass
        self.head_k_dim = getattr(originalLayer, 'head_k_dim', 128)
        self.head_v_dim = getattr(originalLayer, 'head_v_dim', 128)
        self.num_kv_groups = getattr(originalLayer, 'num_kv_groups', 1)
        self.gate_logit_normalizer = getattr(originalLayer, 'gate_logit_normalizer', 16)
        self.mode = getattr(originalLayer, 'mode', 'chunk')
        self.feature_map_fn = getattr(originalLayer, 'feature_map_fn', None)
        
        # Copy all projections - wrap with HadLinear if using Hadamard transform
        if use_had_transform:
            # For GLA projections: q,k,v,g need input transform, o needs both input and output transform
            self.q_proj = HadLinear(originalLayer.q_proj, input_transform=True, output_transform=False)
            self.k_proj = HadLinear(originalLayer.k_proj, input_transform=True, output_transform=False)
            self.v_proj = HadLinear(originalLayer.v_proj, input_transform=True, output_transform=False)
            self.g_proj = HadLinear(originalLayer.g_proj, input_transform=True, output_transform=False)
            self.o_proj = HadLinear(originalLayer.o_proj, input_transform=True, output_transform=True)
            
            # gk_proj is Sequential with 2 layers - need special handling
            if hasattr(originalLayer, 'gk_proj') and originalLayer.gk_proj is not None:
                gk_proj_0 = HadLinear(originalLayer.gk_proj[0], input_transform=True, output_transform=False)
                gk_proj_1 = HadLinear(originalLayer.gk_proj[1], input_transform=True, output_transform=False)
                self.gk_proj = nn.Sequential(gk_proj_0, gk_proj_1)
            else:
                self.gk_proj = None
        else:
            # Direct copy without Hadamard transform
            self.q_proj = originalLayer.q_proj
            self.k_proj = originalLayer.k_proj
            self.v_proj = originalLayer.v_proj
            self.g_proj = originalLayer.g_proj
            self.o_proj = originalLayer.o_proj
            self.gk_proj = originalLayer.gk_proj if hasattr(originalLayer, 'gk_proj') else None
        
        # Copy the gated norm layer
        self.g_norm_swish_gate = originalLayer.g_norm_swish_gate
        
        # Copy any other attributes that might be needed
        if hasattr(originalLayer, 'mode'):
            self.mode = originalLayer.mode
        if hasattr(originalLayer, 'use_short_conv'):
            self.use_short_conv = originalLayer.use_short_conv
        if hasattr(originalLayer, 'conv_size'):
            self.conv_size = originalLayer.conv_size
        if hasattr(originalLayer, 'conv1d'):
            self.conv1d = originalLayer.conv1d
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ):
        """
        Forward pass adapted from fla.layers.gla.GatedLinearAttention
        Modified to use HadLinear wrapped projections for Hadamard transforms
        """
        from einops import rearrange, repeat
        import torch.nn.functional as F
        
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len]"
            )
        
        batch_size, q_len, _ = hidden_states.shape
        
        # Use the wrapped projections (HadLinear applies Hadamard transforms)
        if self.use_short_conv and hasattr(self, 'q_conv1d'):
            # If short conv is used
            q = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False)[0]
            k = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False)[0]
            v = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False)[0]
        else:
            # Direct projections
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        
        # Gate key projection
        if self.gk_proj is not None:
            gk = self.gk_proj(hidden_states)
        else:
            gk = k  # fallback
        
        # Apply feature map if exists
        if hasattr(self, 'feature_map_fn') and self.feature_map_fn is not None:
            q = self.feature_map_fn(q)
            k = self.feature_map_fn(k)
        
        # Reshape for multi-head attention
        # Note: Using attributes from original layer if they exist
        head_k_dim = getattr(self, 'head_k_dim', 128)
        head_v_dim = getattr(self, 'head_v_dim', 128)
        num_kv_groups = getattr(self, 'num_kv_groups', 1)
        gate_logit_normalizer = getattr(self, 'gate_logit_normalizer', 16)
        
        q = rearrange(q, '... (h d) -> ... h d', d=head_k_dim)
        
        if num_kv_groups > 1:
            k = repeat(k, '... (h d) -> ... (h g) d', g=num_kv_groups, d=head_k_dim)
            gk = repeat(gk, '... (h d) -> ... (h g) d', g=num_kv_groups, d=head_k_dim)
            v = repeat(v, '... (h d) -> ... (h g) d', g=num_kv_groups, d=head_v_dim)
        else:
            k = rearrange(k, '... (h d) -> ... h d', d=head_k_dim)
            gk = rearrange(gk, '... (h d) -> ... h d', d=head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=head_v_dim)
        
        gk = F.logsigmoid(gk) / gate_logit_normalizer
        
        # Import GLA ops for actual attention computation
        try:
            from fla.ops.gla import chunk_gla, fused_recurrent_gla
            
            # Choose mode based on sequence length
            mode = getattr(self, 'mode', 'chunk')
            if q_len <= 64:
                mode = 'fused_recurrent'
            
            if mode == 'fused_recurrent':
                o, _ = fused_recurrent_gla(q=q, k=k, v=v, gk=gk, initial_state=None, output_final_state=False)
            else:
                o, _ = chunk_gla(q=q, k=k, v=v, g=gk, initial_state=None, output_final_state=False)
        except ImportError:
            # Fallback: simple attention if GLA ops not available
            # This is a simplified version and won't match GLA exactly
            attn = torch.matmul(q, k.transpose(-2, -1)) * (head_k_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            o = torch.matmul(attn, v)
        
        # Apply output gate and norm
        if hasattr(self, 'g_proj'):
            g = self.g_proj(hidden_states)
            if hasattr(self, 'g_norm_swish_gate'):
                g = rearrange(g, '... (h d) -> ... h d', d=head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, '... h d -> ... (h d)')
            else:
                o = rearrange(o, '... h d -> ... (h d)')
                if hasattr(self, 'gate_fn'):
                    o = o * self.gate_fn(g)
                else:
                    o = o * F.silu(g)  # default to SiLU
        else:
            o = rearrange(o, '... h d -> ... (h d)')
        
        # Output projection (also wrapped with HadLinear)
        o = self.o_proj(o)
        
        return o, None, None  # (output, attentions, past_key_values)


class W4A16QGLA(nn.Module):
    """
    4-bit weight, 16-bit activation quantized GLA attention module.
    """
    
    def __init__(self, originalLayer, use_had_transform: bool = True):
        super().__init__()
        
        self.hidden_size = originalLayer.hidden_size if hasattr(originalLayer, 'hidden_size') else 2048
        self.num_heads = originalLayer.num_heads if hasattr(originalLayer, 'num_heads') else None
        
        # Convert all projections to W4A16
        self.q_proj = W4A16B16O16Linear.from_fp16(originalLayer.q_proj)
        self.k_proj = W4A16B16O16Linear.from_fp16(originalLayer.k_proj)
        self.v_proj = W4A16B16O16Linear.from_fp16(originalLayer.v_proj)
        self.g_proj = W4A16B16O16Linear.from_fp16(originalLayer.g_proj)
        self.o_proj = W4A16B16O16Linear.from_fp16(originalLayer.o_proj)
        
        # Handle gk_proj Sequential
        if hasattr(originalLayer, 'gk_proj') and originalLayer.gk_proj is not None:
            gk_proj_0 = W4A16B16O16Linear.from_fp16(originalLayer.gk_proj[0])
            gk_proj_1 = W4A16B16O16Linear.from_fp16(originalLayer.gk_proj[1])
            self.gk_proj = nn.Sequential(gk_proj_0, gk_proj_1)
        else:
            self.gk_proj = None
        
        # Keep gated norm as-is (no quantization for W4A16)
        self.g_norm_swish_gate = originalLayer.g_norm_swish_gate
        
        # Copy other attributes
        if hasattr(originalLayer, 'mode'):
            self.mode = originalLayer.mode
        if hasattr(originalLayer, 'use_short_conv'):
            self.use_short_conv = originalLayer.use_short_conv
        if hasattr(originalLayer, 'conv_size'):
            self.conv_size = originalLayer.conv_size
        if hasattr(originalLayer, 'conv1d'):
            self.conv1d = originalLayer.conv1d
    
    @classmethod
    def from_fp16(cls, originalLayer, use_had_transform: bool = True):
        """Factory method to create W4A16QGLA from original layer."""
        return cls(originalLayer, use_had_transform=use_had_transform)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ):
        """Forward pass for W4A16 quantized GLA - same as GLASimple but with quantized layers."""
        from einops import rearrange, repeat
        import torch.nn.functional as F
        
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2
        
        batch_size, q_len, _ = hidden_states.shape
        
        # Use quantized projections
        if self.use_short_conv and hasattr(self, 'q_conv1d'):
            q = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False)[0]
            k = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False)[0]
            v = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False)[0]
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        
        gk = self.gk_proj(hidden_states) if self.gk_proj is not None else k
        
        if hasattr(self, 'feature_map_fn') and self.feature_map_fn is not None:
            q = self.feature_map_fn(q)
            k = self.feature_map_fn(k)
        
        head_k_dim = getattr(self, 'head_k_dim', 128)
        head_v_dim = getattr(self, 'head_v_dim', 128)
        num_kv_groups = getattr(self, 'num_kv_groups', 1)
        gate_logit_normalizer = getattr(self, 'gate_logit_normalizer', 16)
        
        q = rearrange(q, '... (h d) -> ... h d', d=head_k_dim)
        
        if num_kv_groups > 1:
            k = repeat(k, '... (h d) -> ... (h g) d', g=num_kv_groups, d=head_k_dim)
            gk = repeat(gk, '... (h d) -> ... (h g) d', g=num_kv_groups, d=head_k_dim)
            v = repeat(v, '... (h d) -> ... (h g) d', g=num_kv_groups, d=head_v_dim)
        else:
            k = rearrange(k, '... (h d) -> ... h d', d=head_k_dim)
            gk = rearrange(gk, '... (h d) -> ... h d', d=head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=head_v_dim)
        
        gk = F.logsigmoid(gk) / gate_logit_normalizer
        
        try:
            from fla.ops.gla import chunk_gla, fused_recurrent_gla
            mode = getattr(self, 'mode', 'chunk')
            if q_len <= 64:
                mode = 'fused_recurrent'
            
            if mode == 'fused_recurrent':
                o, _ = fused_recurrent_gla(q=q, k=k, v=v, gk=gk, initial_state=None, output_final_state=False)
            else:
                o, _ = chunk_gla(q=q, k=k, v=v, g=gk, initial_state=None, output_final_state=False)
        except ImportError:
            attn = torch.matmul(q, k.transpose(-2, -1)) * (head_k_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            o = torch.matmul(attn, v)
        
        if hasattr(self, 'g_proj'):
            g = self.g_proj(hidden_states)
            if hasattr(self, 'g_norm_swish_gate'):
                g = rearrange(g, '... (h d) -> ... h d', d=head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, '... h d -> ... (h d)')
            else:
                o = rearrange(o, '... h d -> ... (h d)')
                if hasattr(self, 'gate_fn'):
                    o = o * self.gate_fn(g)
                else:
                    o = o * F.silu(g)
        else:
            o = rearrange(o, '... h d -> ... (h d)')
        
        o = self.o_proj(o)
        return o, None, None


class W4A8QGLA(nn.Module):
    """
    4-bit weight, 8-bit activation quantized GLA attention module.
    """
    
    def __init__(self, originalLayer, act_scales: dict, use_had_transform: bool = True):
        super().__init__()
        
        self.hidden_size = originalLayer.hidden_size if hasattr(originalLayer, 'hidden_size') else 2048
        self.num_heads = originalLayer.num_heads if hasattr(originalLayer, 'num_heads') else None
        
        # Copy GLA-specific attributes
        self.head_k_dim = getattr(originalLayer, 'head_k_dim', 128)
        self.head_v_dim = getattr(originalLayer, 'head_v_dim', 128)
        self.num_kv_groups = getattr(originalLayer, 'num_kv_groups', 1)
        self.gate_logit_normalizer = getattr(originalLayer, 'gate_logit_normalizer', 16)
        self.mode = getattr(originalLayer, 'mode', 'chunk')
        self.feature_map_fn = getattr(originalLayer, 'feature_map_fn', None)
        self.gate_fn = getattr(originalLayer, 'gate_fn', None)
        
        # Convert all projections to W4A8 using activation scales
        # Note: You'll need to adjust the scale keys based on what calibration collects
        self.q_proj = W4A8B16O16Linear.from_fp16(
            originalLayer.q_proj, 
            act_scales.get("q_proj:input", torch.tensor(1.0))
        )
        self.k_proj = W4A8B16O16Linear.from_fp16(
            originalLayer.k_proj,
            act_scales.get("k_proj:input", torch.tensor(1.0))
        )
        self.v_proj = W4A8B16O16Linear.from_fp16(
            originalLayer.v_proj,
            act_scales.get("v_proj:input", torch.tensor(1.0))
        )
        self.g_proj = W4A8B16O16Linear.from_fp16(
            originalLayer.g_proj,
            act_scales.get("g_proj:input", torch.tensor(1.0))
        )
        self.o_proj = W4A8B16O16Linear.from_fp16(
            originalLayer.o_proj,
            act_scales.get("o_proj:input", torch.tensor(1.0))
        )
        
        # Handle gk_proj Sequential
        # Note: gk_proj[1] has bias=True but quantized layers don't support bias yet
        # We'll quantize the weight and store the bias separately to apply after
        if hasattr(originalLayer, 'gk_proj') and originalLayer.gk_proj is not None:
            gk_proj_0 = W4A8B16O16Linear.from_fp16(
                originalLayer.gk_proj[0],
                act_scales.get("gk_proj.0:input", torch.tensor(1.0))
            )
            
            # For gk_proj[1], create a quantized version without bias, then add bias separately
            # Create a copy of the layer to avoid modifying the original
            import copy
            gk_proj_1_layer = copy.deepcopy(originalLayer.gk_proj[1])
            gk_proj_1_bias = gk_proj_1_layer.bias.data.clone() if gk_proj_1_layer.bias is not None else None
            
            # Remove bias from the copy for quantization
            gk_proj_1_layer.bias = None
            
            gk_proj_1 = W4A8B16O16Linear.from_fp16(
                gk_proj_1_layer,
                act_scales.get("gk_proj.1:input", torch.tensor(1.0))
            )
            
            # Store the bias to apply separately
            if gk_proj_1_bias is not None:
                self.register_buffer('gk_proj_1_bias', gk_proj_1_bias)
            else:
                self.gk_proj_1_bias = None
            
            # Store layers separately instead of Sequential - quantized layers need special handling
            self.gk_proj_0 = gk_proj_0
            self.gk_proj_1 = gk_proj_1
            self.gk_proj = None  # Mark that we have a multi-layer gk_proj
        else:
            self.gk_proj_0 = None
            self.gk_proj_1 = None
            self.gk_proj = None
            self.gk_proj_1_bias = None
        
        # Keep gated norm as-is for now
        # TODO: May need quantized version of g_norm_swish_gate
        self.g_norm_swish_gate = originalLayer.g_norm_swish_gate
        
        # Copy other attributes
        if hasattr(originalLayer, 'mode'):
            self.mode = originalLayer.mode
        if hasattr(originalLayer, 'use_short_conv'):
            self.use_short_conv = originalLayer.use_short_conv
        if hasattr(originalLayer, 'conv_size'):
            self.conv_size = originalLayer.conv_size
        if hasattr(originalLayer, 'conv1d'):
            self.conv1d = originalLayer.conv1d
    
    @classmethod
    def from_fp16(cls, originalLayer, act_scales: dict, use_had_transform: bool = True):
        """Factory method to create W4A8QGLA from original layer."""
        return cls(originalLayer, act_scales=act_scales, use_had_transform=use_had_transform)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ):
        """Forward pass for W4A8 quantized GLA."""
        from einops import rearrange, repeat
        import torch.nn.functional as F
        
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2
        
        batch_size, q_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Apply gk_proj layers manually (can't use Sequential with quantized layers)
        if self.gk_proj_0 is not None:
            gk = self.gk_proj_0(hidden_states)
            gk = self.gk_proj_1(gk)
            # Add the bias from gk_proj[1] if it was stored
            if hasattr(self, 'gk_proj_1_bias') and self.gk_proj_1_bias is not None:
                gk = gk + self.gk_proj_1_bias
        else:
            gk = k
        
        if hasattr(self, 'feature_map_fn') and self.feature_map_fn is not None:
            q = self.feature_map_fn(q)
            k = self.feature_map_fn(k)
        
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        
        if self.num_kv_groups > 1:
            k = repeat(k, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_k_dim)
            gk = repeat(gk, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_k_dim)
            v = repeat(v, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_v_dim)
        else:
            k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
            gk = rearrange(gk, '... (h d) -> ... h d', d=self.head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        
        try:
            from fla.ops.gla import chunk_gla, fused_recurrent_gla
            mode = self.mode if q_len > 64 else 'fused_recurrent'
            
            if mode == 'fused_recurrent':
                o, _ = fused_recurrent_gla(q=q, k=k, v=v, gk=gk, initial_state=None, output_final_state=False)
            else:
                o, _ = chunk_gla(q=q, k=k, v=v, g=gk, initial_state=None, output_final_state=False)
        except ImportError:
            attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_k_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            o = torch.matmul(attn, v)
        
        if hasattr(self, 'g_proj'):
            g = self.g_proj(hidden_states)
            if hasattr(self, 'g_norm_swish_gate'):
                g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, '... h d -> ... (h d)')
            else:
                o = rearrange(o, '... h d -> ... (h d)')
                if hasattr(self, 'gate_fn') and self.gate_fn is not None:
                    o = o * self.gate_fn(g)
                else:
                    o = o * F.silu(g)
        else:
            o = rearrange(o, '... h d -> ... (h d)')
        
        o = self.o_proj(o)
        return o, None, None


class W8A8QGLA(nn.Module):
    """
    8-bit weight, 8-bit activation quantized GLA attention module.
    """
    
    def __init__(self, originalLayer, act_scales: dict, use_had_transform: bool = True):
        super().__init__()
        
        self.hidden_size = originalLayer.hidden_size if hasattr(originalLayer, 'hidden_size') else 2048
        self.num_heads = originalLayer.num_heads if hasattr(originalLayer, 'num_heads') else None
        
        # Copy GLA-specific attributes
        self.head_k_dim = getattr(originalLayer, 'head_k_dim', 128)
        self.head_v_dim = getattr(originalLayer, 'head_v_dim', 128)
        self.num_kv_groups = getattr(originalLayer, 'num_kv_groups', 1)
        self.gate_logit_normalizer = getattr(originalLayer, 'gate_logit_normalizer', 16)
        self.mode = getattr(originalLayer, 'mode', 'chunk')
        self.feature_map_fn = getattr(originalLayer, 'feature_map_fn', None)
        self.gate_fn = getattr(originalLayer, 'gate_fn', None)
        
        # Convert all projections to W8A8 using activation scales
        self.q_proj = W8A8B16O16Linear.from_fp16(
            originalLayer.q_proj,
            act_scales.get("q_proj:input", torch.tensor(1.0)).item()
        )
        self.k_proj = W8A8B16O16Linear.from_fp16(
            originalLayer.k_proj,
            act_scales.get("k_proj:input", torch.tensor(1.0)).item()
        )
        self.v_proj = W8A8B16O16Linear.from_fp16(
            originalLayer.v_proj,
            act_scales.get("v_proj:input", torch.tensor(1.0)).item()
        )
        self.g_proj = W8A8B16O16Linear.from_fp16(
            originalLayer.g_proj,
            act_scales.get("g_proj:input", torch.tensor(1.0)).item()
        )
        self.o_proj = W8A8B16O16Linear.from_fp16(
            originalLayer.o_proj,
            act_scales.get("o_proj:input", torch.tensor(1.0)).item()
        )
        
        # Handle gk_proj Sequential
        # Note: gk_proj[1] has bias=True but quantized layers don't support bias yet
        # We'll quantize the weight and store the bias separately to apply after
        if hasattr(originalLayer, 'gk_proj') and originalLayer.gk_proj is not None:
            gk_proj_0 = W8A8B16O16Linear.from_fp16(
                originalLayer.gk_proj[0],
                act_scales.get("gk_proj.0:input", torch.tensor(1.0)).item()
            )
            
            # For gk_proj[1], create a quantized version without bias, then add bias separately
            # Create a copy of the layer to avoid modifying the original
            import copy
            gk_proj_1_layer = copy.deepcopy(originalLayer.gk_proj[1])
            gk_proj_1_bias = gk_proj_1_layer.bias.data.clone() if gk_proj_1_layer.bias is not None else None
            
            # Remove bias from the copy for quantization
            gk_proj_1_layer.bias = None
            
            gk_proj_1 = W8A8B16O16Linear.from_fp16(
                gk_proj_1_layer,
                act_scales.get("gk_proj.1:input", torch.tensor(1.0)).item()
            )
            
            # Store the bias to apply separately
            if gk_proj_1_bias is not None:
                self.register_buffer('gk_proj_1_bias', gk_proj_1_bias)
            else:
                self.gk_proj_1_bias = None
            
            # Store layers separately instead of Sequential - quantized layers need special handling
            self.gk_proj_0 = gk_proj_0
            self.gk_proj_1 = gk_proj_1
            self.gk_proj = None  # Mark that we have a multi-layer gk_proj
        else:
            self.gk_proj_0 = None
            self.gk_proj_1 = None
            self.gk_proj = None
            self.gk_proj_1_bias = None
        
        # Keep gated norm as-is for now
        self.g_norm_swish_gate = originalLayer.g_norm_swish_gate
        
        # Copy other attributes
        if hasattr(originalLayer, 'mode'):
            self.mode = originalLayer.mode
        if hasattr(originalLayer, 'use_short_conv'):
            self.use_short_conv = originalLayer.use_short_conv
        if hasattr(originalLayer, 'conv_size'):
            self.conv_size = originalLayer.conv_size
        if hasattr(originalLayer, 'conv1d'):
            self.conv1d = originalLayer.conv1d
    
    @classmethod
    def from_fp16(cls, originalLayer, act_scales: dict, use_had_transform: bool = True):
        """Factory method to create W8A8QGLA from original layer."""
        return cls(originalLayer, act_scales=act_scales, use_had_transform=use_had_transform)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ):
        """Forward pass for W8A8 quantized GLA."""
        from einops import rearrange, repeat
        import torch.nn.functional as F
        
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2
        
        batch_size, q_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Apply gk_proj layers manually (can't use Sequential with quantized layers)
        if self.gk_proj_0 is not None:
            gk = self.gk_proj_0(hidden_states)
            gk = self.gk_proj_1(gk)
            # Add the bias from gk_proj[1] if it was stored
            if hasattr(self, 'gk_proj_1_bias') and self.gk_proj_1_bias is not None:
                gk = gk + self.gk_proj_1_bias
        else:
            gk = k
        
        if hasattr(self, 'feature_map_fn') and self.feature_map_fn is not None:
            q = self.feature_map_fn(q)
            k = self.feature_map_fn(k)
        
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        
        if self.num_kv_groups > 1:
            k = repeat(k, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_k_dim)
            gk = repeat(gk, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_k_dim)
            v = repeat(v, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_v_dim)
        else:
            k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
            gk = rearrange(gk, '... (h d) -> ... h d', d=self.head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        
        try:
            from fla.ops.gla import chunk_gla, fused_recurrent_gla
            mode = self.mode if q_len > 64 else 'fused_recurrent'
            
            if mode == 'fused_recurrent':
                o, _ = fused_recurrent_gla(q=q, k=k, v=v, gk=gk, initial_state=None, output_final_state=False)
            else:
                o, _ = chunk_gla(q=q, k=k, v=v, g=gk, initial_state=None, output_final_state=False)
        except ImportError:
            attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_k_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            o = torch.matmul(attn, v)
        
        if hasattr(self, 'g_proj'):
            g = self.g_proj(hidden_states)
            if hasattr(self, 'g_norm_swish_gate'):
                g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, '... h d -> ... (h d)')
            else:
                o = rearrange(o, '... h d -> ... (h d)')
                if hasattr(self, 'gate_fn') and self.gate_fn is not None:
                    o = o * self.gate_fn(g)
                else:
                    o = o * F.silu(g)
        else:
            o = rearrange(o, '... h d -> ... (h d)')
        
        o = self.o_proj(o)
        return o, None, None


# TODO: Also need to handle the MLP block quantization
class GLAMLPSimple(nn.Module):
    """Simplified GLA MLP module for quantization."""
    
    def __init__(self, originalLayer, use_had_transform: bool = True):
        super().__init__()
        
        if use_had_transform:
            self.gate_proj = HadLinear(originalLayer.gate_proj, input_transform=True, output_transform=False)
            self.up_proj = HadLinear(originalLayer.up_proj, input_transform=True, output_transform=False)
            self.down_proj = HadLinear(originalLayer.down_proj, input_transform=True, output_transform=True)
        else:
            self.gate_proj = originalLayer.gate_proj
            self.up_proj = originalLayer.up_proj
            self.down_proj = originalLayer.down_proj
        
        self.swiglu_linear = originalLayer.swiglu_linear
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("GLAMLPSimple forward not implemented")


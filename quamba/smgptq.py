"""
SM-GPTQ: JVP-based Hessian approximation for Mamba2 quantization.

Implements GPTQ with a different Hessian matrix (J^T J) computed via
Jacobian-Vector Products through the Mamba2 mixer block.
"""
import gc
import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete

logger = logging.getLogger(__name__)


def rms_norm_gated(x, weight, z=None, eps=1e-5, group_size=None, norm_before_gate=True):
    """Pure-PyTorch gated RMSNorm (replaces Triton RMSNormGated).

    Args:
        x: (..., D) input tensor
        weight: (D,) learnable scale
        z: (..., D) optional gate tensor
        eps: epsilon for numerical stability
        group_size: if not None, normalize groups of this size independently.
            None means normalize over the full last dimension.
        norm_before_gate: if True, norm(x) * silu(z); else norm(x * silu(z))
    """
    if group_size is not None and group_size != x.shape[-1]:
        # Reshape into groups for independent normalization
        orig_shape = x.shape
        x = x.reshape(*orig_shape[:-1], -1, group_size)
        if z is not None:
            z = z.reshape(*orig_shape[:-1], -1, group_size)
        weight = weight.reshape(-1, group_size)

        if norm_before_gate and z is not None:
            x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
            result = x_normed * F.silu(z)
        elif z is not None:
            x_gated = x * F.silu(z)
            result = x_gated * torch.rsqrt(x_gated.pow(2).mean(-1, keepdim=True) + eps) * weight
        else:
            result = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
        return result.reshape(orig_shape)

    # No grouping (or group_size == last dim)
    if norm_before_gate and z is not None:
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
        return x_normed * F.silu(z)
    elif z is not None:
        x_gated = x * F.silu(z)
        return x_gated * torch.rsqrt(x_gated.pow(2).mean(-1, keepdim=True) + eps) * weight
    else:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def functional_mixer_forward(u, in_proj_weight, conv_weight, conv_bias,
                              A_log, D_param, dt_bias, norm_weight,
                              headdim, ngroups, d_state, d_ssm, nheads,
                              chunk_size, d_conv, norm_eps=1e-5,
                              norm_before_gate=False, norm_group_size=None):
    """Pure-PyTorch Mamba2 mixer forward for a single sample (no batch dim).

    Replicates the Mamba2 reference path using ssd_minimal_discrete (no Triton/CUDA),
    making it compatible with torch.func transformations (jacfwd, vmap, etc.).

    Args:
        u: (seqlen, d_model) — input to the mixer block
        in_proj_weight: (d_in_proj, d_model) — in_proj weight matrix
        conv_weight: (conv_dim, 1, d_conv) — depthwise conv1d weight
        conv_bias: (conv_dim,) or None — conv1d bias
        A_log: (nheads,) — log of the A parameter
        D_param: (nheads,) or (d_ssm,) or None — D skip connection parameter
        dt_bias: (nheads,) — dt bias (before softplus)
        norm_weight: (d_ssm,) — RMSNorm learnable scale
        headdim: int — head dimension P
        ngroups: int — number of groups G
        d_state: int — SSM state dimension N
        d_ssm: int — SSM dimension (= nheads * headdim)
        nheads: int — number of heads H
        chunk_size: int — SSD chunk size
        d_conv: int — convolution kernel width
        norm_eps: float — RMSNorm epsilon
        norm_before_gate: bool — RMSNorm gate ordering
        norm_group_size: int or None — RMSNorm group size

    Returns:
        y: (seqlen, d_ssm) — pre-out_proj output (after gated norm)
    """
    seqlen = u.shape[0]
    conv_dim = d_ssm + 2 * ngroups * d_state

    # 1. in_proj (linear, no bias in Mamba2)
    zxbcdt = u @ in_proj_weight.T                                  # (T, d_in_proj)

    # 2. Split into z, xBC, dt (assumes d_mlp=0 i.e. d_ssm == d_inner)
    z = zxbcdt[:, :d_ssm]                                         # (T, d_ssm)
    xBC = zxbcdt[:, d_ssm:d_ssm + conv_dim]                       # (T, conv_dim)
    dt = zxbcdt[:, -(nheads):]                                     # (T, nheads)

    # 3. Causal depthwise conv1d + SiLU
    xBC_t = xBC.T.unsqueeze(0)                                    # (1, conv_dim, T)
    xBC_t = F.pad(xBC_t, (d_conv - 1, 0))                        # left-pad for causal
    xBC_t = F.conv1d(xBC_t, conv_weight, conv_bias,
                     groups=conv_dim)                              # (1, conv_dim, T)
    xBC_conv = F.silu(xBC_t.squeeze(0).T)                         # (T, conv_dim)

    # 4. Split x, B, C from conv output
    x = xBC_conv[:, :d_ssm]                                       # (T, d_ssm)
    B = xBC_conv[:, d_ssm:d_ssm + ngroups * d_state]              # (T, ngroups*d_state)
    C = xBC_conv[:, d_ssm + ngroups * d_state:]                   # (T, ngroups*d_state)

    # 5. Reshape for ssd_minimal_discrete
    x = x.reshape(1, seqlen, nheads, headdim)                     # (1, T, H, P)
    B = B.reshape(1, seqlen, ngroups, d_state)                    # (1, T, G, N)
    C = C.reshape(1, seqlen, ngroups, d_state)                    # (1, T, G, N)

    # 6. dt softplus + A computation (all float32 for stability)
    A = -torch.exp(A_log.float())                                  # (nheads,)
    dt = F.softplus(dt + dt_bias).unsqueeze(0)                    # (1, T, H)

    # 7. Cast SSD inputs to a common dtype (float32 for torch.func compatibility)
    ssd_dtype = torch.float32
    x_dt = (x * dt.unsqueeze(-1)).to(ssd_dtype)                   # (1, T, H, P)
    A_dt = (A * dt).to(ssd_dtype)                                  # (1, T, H)
    B = B.to(ssd_dtype)
    C = C.to(ssd_dtype)

    # 8. SSD (pure PyTorch, torch.func compatible)
    y_ssd, _ = ssd_minimal_discrete(x_dt, A_dt, B, C, chunk_size) # (1, T, H, P)

    # 9. D skip connection
    if D_param is not None:
        x_squeezed = x.squeeze(0)                                  # (T, H, P)
        y_ssd = y_ssd.squeeze(0)                                   # (T, H, P)
        if D_param.numel() == nheads:
            # D_has_hdim=False: per-head scalar, broadcast over headdim
            y_ssd = y_ssd + x_squeezed * D_param.reshape(1, nheads, 1)
        else:
            # D_has_hdim=True: per-element
            y_ssd = y_ssd + x_squeezed * D_param.reshape(1, nheads, headdim)
        y_ssd = y_ssd.reshape(seqlen, d_ssm)                      # (T, d_ssm)
    else:
        y_ssd = y_ssd.reshape(seqlen, d_ssm)

    # 10. Gated RMSNorm (pure PyTorch, replaces Triton RMSNormGated)
    y = rms_norm_gated(y_ssd, norm_weight, z=z, eps=norm_eps,
                       group_size=norm_group_size,
                       norm_before_gate=norm_before_gate)

    return y  # (T, d_ssm)


class SMGPTQ:
    def __init__(self, layer, idx=0, shard_dir="../jacobian_block_samples/130m",
                 output_dir="jacobian_jvp_error", target_layers=None):
        self.idx = idx
        self.layer = layer
        self.dev = self.layer.in_proj.weight.device
        self.gptq_dtype = torch.float16
        self.shard_dir = shard_dir
        self.output_dir = output_dir
        self.target_layers = target_layers

        self.out_dim = self.layer.out_proj.weight.shape[1]
        self.in_dim = self.layer.in_proj.weight.shape[0]

        logger.debug("headdim: %d", self.layer.headdim)
        logger.debug("ngroups: %d", self.layer.ngroups)
        logger.debug("d_state: %d", self.layer.d_state)
        logger.debug("d_inner: %d", self.layer.d_inner)
        logger.debug("nheads: %d", self.layer.nheads)
        logger.debug("expansion factor: %d", self.layer.expand)
        logger.debug("d_ssm: %d", self.layer.d_ssm)
        logger.debug("d_model: %d", self.layer.d_model)

    def capture_inputs(self, inp, out):
        self.inputs = inp

    def capture_outputs(self, x, z, out):
        logger.debug("capture_outputs: x=%s, z=%s, out=%s", x.shape, z.shape, out.shape)
        logger.debug("layer: %s", self.layer)
        inp = x * torch.nn.functional.silu(z)
        self.pre_outputs = inp.sum(dim=0).unsqueeze(0)
        self.pre_outputs = self.pre_outputs[:, :self.pre_outputs.shape[1]//2]

    def read_shards(self):
        path = os.path.join(self.shard_dir, f"layer{self.idx}")
        zpath = os.path.join(path, "z.safetensors")
        xpath = os.path.join(path, "x.safetensors")
        bpath = os.path.join(path, "b.safetensors")
        cpath = os.path.join(path, "c.safetensors")
        z_shards = load_file(zpath)
        x_shards = load_file(xpath)
        b_shards = load_file(bpath)
        c_shards = load_file(cpath)
        return z_shards, x_shards, b_shards, c_shards

    def int_keys(self, tensor_shard):
        final_shard = {}
        for key, value in tensor_shard.items():
            final_shard[int(key)] = value / self.inputs.shape[0]
            final_shard[int(key)] = final_shard[int(key)]
        return final_shard

    def convert_to_hessian(self, tensor_shard, tensor_name='None'):
        hessian_shard = {}

        def flatten_shard(tensor_shard):
            return torch.stack(list(tensor_shard.values()), dim=0)

        flattened_shard = flatten_shard(tensor_shard)

        if len(flattened_shard.shape) > 3:
            flattened_shard = flattened_shard.reshape(flattened_shard.shape[0], flattened_shard.shape[1], -1)

        hessian_tensor = flattened_shard.transpose(1, 2).bmm(flattened_shard)

        for i, key in enumerate(tensor_shard.keys()):
            logger.debug("%d %s", i, key)
            logger.debug("d_state * ngroups = %d", self.layer.d_state * self.layer.ngroups)
            logger.debug("key mod = %d", key % (self.layer.d_state * self.layer.ngroups))
            logger.debug("hessian keys: %s", hessian_shard.keys())

            if (tensor_name == 'b' or tensor_name == 'c') and key % ((self.layer.d_state * self.layer.ngroups)/2) in hessian_shard.keys():
                hessian_shard[int(key % ((self.layer.d_state * self.layer.ngroups)/2))] += hessian_tensor[i, :, :]
            else:
                if (tensor_name == 'b' or tensor_name == 'c'):
                    hessian_shard[int(key % ((self.layer.d_state * self.layer.ngroups)/2))] = hessian_tensor[i, :, :]
                else:
                    hessian_shard[key] = hessian_tensor[i, :, :]
        logger.debug("Final hessian keys: %s", hessian_shard.keys())
        if max(hessian_shard.keys()) > 3372:
            raise ValueError("Hessian keys are greater than d_inner!")
        return hessian_shard

    def estimate_hessian_jvp(self, tensor_shard, max_probes=1024, step_size=8, tensor_name='z'):
        channels = torch.tensor(list(tensor_shard.keys()))
        logger.info("Estimating Hessian via JVP for %s, channels: %s", tensor_name, channels)

        out_channels = channels

        probes = torch.randn(max_probes, self.pre_outputs.shape[1], channels.shape[0]).to(self.pre_outputs.device)

        JVP_scalars = (probes * self.pre_outputs[:, :, channels]/self.inputs.shape[0]).sum(dim=1).sum(dim=1)

        Gs = []

        if tensor_name == 'x':
            channels = channels + self.layer.d_inner
        elif tensor_name == 'b':
            channels = channels + 2*self.layer.d_inner
        elif tensor_name == 'c':
            channels = channels + 2*self.layer.d_inner + self.layer.d_state * self.layer.ngroups

        logger.debug("Derivative channels: %s", channels)

        if tensor_name == 'b' or tensor_name == 'c':
            JVP_scalars = []
            for o in out_channels:
                all_heads = torch.arange(self.layer.nheads) * (self.layer.headdim)
                out_channels_this_head = all_heads + o
                logger.debug("out_channels_this_head: %s", out_channels_this_head)
                probes = torch.randn(max_probes, self.pre_outputs.shape[1], out_channels_this_head.shape[0]).to(self.pre_outputs.device)
                JVP_scalars.append((probes * self.pre_outputs[:, :, out_channels_this_head]/self.inputs.shape[0]).sum(dim=1).sum(dim=1))
            logger.debug("JVP_scalars count: %d, shape: %s", len(JVP_scalars), JVP_scalars[0].shape)
            rel_outs = [[c, c+1] for c in channels]
            for JVP_channels, derivative_channels in zip(JVP_scalars, rel_outs):
                l = []
                for i in range(max_probes):
                    gi_full, = torch.autograd.grad(
                        outputs=JVP_channels[i], inputs=self.layer.in_proj.weight,
                        retain_graph=True,
                        create_graph=False,
                    )
                    gi = gi_full[derivative_channels, :]
                    l.append(gi)
                Gs.append(torch.stack(l, dim=0).reshape(max_probes, -1))
                logger.debug("Gs[-1] shape: %s", Gs[-1].shape)
        else:
            for i in range(max_probes):
                gi_full, = torch.autograd.grad(
                    outputs=JVP_scalars[i], inputs=self.layer.in_proj.weight,
                    retain_graph=True,
                    create_graph=False,
                )
                gi = gi_full[channels]
                Gs.append(gi)

        G = torch.stack(Gs, dim=0)
        if tensor_name == 'b' or tensor_name == 'c':
            G = G.transpose(0, 1)
        logger.debug("G shape: %s", G.shape)

        G = G.float()

        H_dict = {}

        for i in range(step_size, max_probes + 1, step_size):
            H_dict[i] = (G[:i].permute(1, 2, 0) / i).bmm(G[:i].permute(1, 0, 2))
        return H_dict

    def plot_hessian_estimations(self, true_hessians, H_dict, gptq_val=None, tensor_name=None):
        probe_counts = list(H_dict.keys())
        logger.debug("Probe counts: %s", probe_counts)
        errors = []
        gptq_errors = []
        for i, k in enumerate(true_hessians.keys()):
            true_hessian = true_hessians[k]
            error_channel = []
            for k2 in H_dict.keys():
                estimated_hessian = H_dict[k2][i, :, :]
                true_hessian = true_hessian.to(estimated_hessian.device)
                if torch.isinf(estimated_hessian).any():
                    logger.warning("Inf in estimated hessian!")
                if torch.isinf(true_hessian).any():
                    logger.warning("Inf in true hessian!")
                    continue
                if torch.linalg.norm(true_hessian).isinf():
                    logger.debug("norm(true_hessian) double: %s", torch.linalg.norm(true_hessian.double()))
                    logger.debug("norm(true_hessian/10000) double: %s", torch.linalg.norm(true_hessian.double()/10000))

                error_item = torch.nn.functional.cosine_similarity(true_hessian.flatten().double(), estimated_hessian.flatten().double(), dim=0)
                if torch.isnan(error_item):
                    error_item = torch.nn.functional.cosine_similarity(true_hessian.flatten().double()/10000, estimated_hessian.flatten().double()/10000, dim=0)
                error_channel.append(error_item)
            if len(error_channel) == 0:
                continue
            if gptq_val is not None:
                if torch.isinf(gptq_val).any():
                    logger.warning("Inf in gptq_val!")
                gptq_error_item = torch.nn.functional.cosine_similarity(gptq_val.flatten().double()/10000, true_hessian.flatten().double()/10000, dim=0)
                gptq_errors.append(gptq_error_item.item())
            error_channel = torch.stack(error_channel, dim=0)
            if torch.isnan(error_channel).any():
                logger.warning("NaN in error_channel at i=%d, k=%s", i, k)
                logger.debug("H_dict[1024]: %s", H_dict[1024][i, :, :])
                logger.debug("true_hessian: %s", true_hessian)
            logger.debug("error_channel: %s", error_channel)
            errors.append(error_channel)
        errors = torch.stack(errors, dim=0)

        raw_data_dir = os.path.join(self.output_dir, "raw_data", tensor_name)
        os.makedirs(raw_data_dir, exist_ok=True)
        np.save(os.path.join(raw_data_dir, f"{self.idx}_cosine.npy"), errors.cpu().numpy())

        errors = errors.mean(dim=0)
        logger.info("Mean errors: %s", errors)
        if gptq_val is not None:
            logger.info("GPTQ errors: %s (count=%d)", gptq_errors, len(gptq_errors))
            np.save(os.path.join(raw_data_dir, f"{self.idx}_gptq_cosine.npy"), np.array(gptq_errors))
        plt.plot(probe_counts, errors.cpu().numpy())
        plt.xlabel('Probe Count')
        plt.ylabel('Error')
        title = f'Error vs. Probe Count in Layer {self.idx}'
        if tensor_name is not None:
            title += f' for {tensor_name}'
        plt.title(title)
        if gptq_val is not None:
            gptq_error_mean = sum(gptq_errors)/len(gptq_errors)
            logger.info("GPTQ error mean: %s", gptq_error_mean)
        plt.show()
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, f"Gaussian_130m_Layer_{self.idx}_{tensor_name}_Error.png"))
        plt.clf()

    def compute_exact_hessian(self, inputs, channel_group='z', jvp_chunk_size=32):
        """Compute exact H = J^T J for a channel group of in_proj.weight.

        Uses forward-mode AD via a single vmap(jvp) call with internal chunking
        to compute the full Jacobian J, then forms H = J^T @ J.

        Args:
            inputs: (batch, seqlen, d_model) — calibration inputs to the mixer
            channel_group: 'z', 'x', 'b', or 'c'
            jvp_chunk_size: number of tangent directions to vmap simultaneously.
                Controls peak GPU memory: O(jvp_chunk_size * nheads * chunk_size^2).
                Default 32 uses ~5 GB for 130m; full Jacobian storage ~5 GB.

        Returns:
            dict mapping channel_index → (d_model, d_model) Hessian tensor
        """
        mixer = self.layer
        d_model = mixer.d_model
        d_ssm = mixer.d_ssm

        # Channel ranges in in_proj output (assumes d_mlp=0, d_ssm == d_inner)
        if channel_group == 'z':
            channels = list(range(0, d_ssm))
        elif channel_group == 'x':
            channels = list(range(d_ssm, 2 * d_ssm))
        elif channel_group == 'b':
            channels = list(range(2 * d_ssm,
                                  2 * d_ssm + mixer.ngroups * mixer.d_state))
        elif channel_group == 'c':
            channels = list(range(2 * d_ssm + mixer.ngroups * mixer.d_state,
                                  2 * d_ssm + 2 * mixer.ngroups * mixer.d_state))
        else:
            raise ValueError(f"Unknown channel_group: {channel_group}")

        # Extract all fixed parameters from the mixer module
        W = mixer.in_proj.weight.detach()
        conv_w = mixer.conv1d.weight.detach()
        conv_b = mixer.conv1d.bias.detach() if mixer.conv1d.bias is not None else None
        A_log = mixer.A_log.detach()
        D_param = mixer.D.detach() if mixer.D is not None else None
        dt_bias = mixer.dt_bias.detach()
        norm_w = mixer.norm.weight.detach()
        norm_eps = mixer.norm.eps
        norm_before_gate = mixer.norm.norm_before_gate
        norm_group_size = getattr(mixer.norm, 'group_size', None)

        basis = torch.eye(d_model, device=self.dev, dtype=W.dtype)
        hessians = {}

        for j in channels:
            H_j = torch.zeros(d_model, d_model, device=self.dev, dtype=torch.float32)

            for sample_idx in range(inputs.shape[0]):
                u = inputs[sample_idx].detach()  # (seqlen, d_model)
                w_j = W[j]

                def f_j(w_j_arg):
                    """Forward pass with row j of in_proj as free variable."""
                    W_full = torch.cat([W[:j], w_j_arg.unsqueeze(0), W[j+1:]], dim=0)
                    y = functional_mixer_forward(
                        u, W_full, conv_w, conv_b, A_log, D_param, dt_bias,
                        norm_w, mixer.headdim, mixer.ngroups, mixer.d_state,
                        mixer.d_ssm, mixer.nheads, mixer.chunk_size,
                        mixer.d_conv, norm_eps=norm_eps,
                        norm_before_gate=norm_before_gate,
                        norm_group_size=norm_group_size,
                    )
                    return y.flatten()  # (T * d_ssm,)

                def jvp_fn(tangent):
                    _, jvp_out = torch.func.jvp(f_j, (w_j,), (tangent,))
                    return jvp_out

                # Single vmap call with internal chunking (avoids PyTorch
                # multi-call vmap bug). J_T[i] = J @ e_i = column i of J,
                # so J_T = J^T and H = J_T @ J_T^T.
                J_T = torch.vmap(jvp_fn, chunk_size=jvp_chunk_size)(basis)
                H_j += (J_T.float() @ J_T.float().T)
                del J_T

            hessians[j] = H_j / inputs.shape[0]
            logger.debug("Exact Hessian computed for channel %d, shape: %s",
                         j, hessians[j].shape)

        return hessians

    def read_and_compare(self, use_exact_hessian=False, max_channels=8):
        if self.target_layers is not None and self.idx not in self.target_layers:
            logger.info("Skipping layer %d (not in target_layers)", self.idx)
            return

        logger.info("Processing layer %d", self.idx)
        logger.debug("inputs shape: %s", self.inputs.shape)
        comb_inp = self.inputs.mean(dim=0)
        if torch.isinf(comb_inp).any() or torch.isnan(comb_inp).any():
            raise ValueError(f"Layer {self.idx}: combined input has inf/nan values")
        comb_inp = comb_inp.float()
        gptq_parallel = (self.inputs.transpose(1, 2).bmm(self.inputs)).mean(dim=0)
        gptq_aggregated = comb_inp.T @ comb_inp
        logger.debug("gptq_parallel shape: %s", gptq_parallel.shape)
        logger.debug("gptq_aggregated shape: %s", gptq_aggregated.shape)
        logger.debug("gptq_parallel: %s", gptq_parallel)
        logger.debug("gptq_aggregated: %s", gptq_aggregated)

        if use_exact_hessian:
            # Compute exact Hessians on-the-fly via jacfwd (no shard files needed)
            for group_name in ['z', 'x', 'b', 'c']:
                logger.info("Computing exact Hessian for group '%s' (max_channels=%d)",
                            group_name, max_channels)
                exact_H = self.compute_exact_hessian(self.inputs,
                                                     channel_group=group_name)
                # Subsample to max_channels for feasibility
                channel_keys = list(exact_H.keys())[:max_channels]
                exact_H_subset = {k: exact_H[k] for k in channel_keys}
                jvp_H = self.estimate_hessian_jvp(exact_H_subset,
                                                   tensor_name=group_name)
                gptq_val = gptq_aggregated
                if group_name in ('b', 'c'):
                    gptq_val = torch.block_diag(gptq_aggregated, gptq_aggregated)
                self.plot_hessian_estimations(exact_H_subset, jvp_H,
                                              gptq_val=gptq_val,
                                              tensor_name=group_name)
                del exact_H, exact_H_subset, jvp_H
                self.free()
        else:
            # Existing shard-based path
            z_shards, x_shards, b_shards, c_shards = self.read_shards()
            z_shards = self.int_keys(z_shards)

            z_hessian = self.convert_to_hessian(z_shards)
            z_jvp_gradients = self.estimate_hessian_jvp(z_hessian, max_probes=1024, step_size=8, tensor_name='z')

            logger.info("True Hessian (z, channel 0): %s", z_hessian[0])
            for k in z_jvp_gradients.keys():
                if k in [8, 16, 32, 64, 128, 256, 512, 1024]:
                    logger.debug("Estimated Hessian (z, probes=%d): %s", k, z_jvp_gradients[k][0, :, :])

            self.plot_hessian_estimations(z_hessian, z_jvp_gradients, gptq_val=gptq_aggregated, tensor_name='z')

            del z_shards, z_hessian, z_jvp_gradients
            self.free()

            x_shards = self.int_keys(x_shards)
            x_hessian = self.convert_to_hessian(x_shards)
            x_jvp_gradients = self.estimate_hessian_jvp(x_hessian, max_probes=1024, step_size=8, tensor_name='x')

            logger.info("True Hessian (x, channel 0): %s", x_hessian[0])
            for k in x_jvp_gradients.keys():
                if k in [8, 16, 32, 64, 128, 256, 512, 1024]:
                    logger.debug("Estimated Hessian (x, probes=%d): %s", k, x_jvp_gradients[k][0, :, :])

            self.plot_hessian_estimations(x_hessian, x_jvp_gradients, gptq_val=gptq_aggregated, tensor_name='x')

            del x_shards, x_hessian, x_jvp_gradients
            self.free()

            b_shards = self.int_keys(b_shards)
            b_hessian = self.convert_to_hessian(b_shards, tensor_name='b')
            b_jvp_gradients = self.estimate_hessian_jvp(b_hessian, max_probes=1024, step_size=8, tensor_name='b')

            logger.info("True Hessian (b, channel 0): %s", b_hessian[0])
            for k in b_jvp_gradients.keys():
                if k in [8, 16, 32, 64, 128, 256, 512, 1024]:
                    logger.debug("Estimated Hessian (b, probes=%d): %s", k, b_jvp_gradients[k][0, :, :])

            self.plot_hessian_estimations(b_hessian, b_jvp_gradients, gptq_val=torch.block_diag(gptq_aggregated, gptq_aggregated), tensor_name='b')

            del b_shards, b_hessian, b_jvp_gradients
            self.free()

            c_shards = self.int_keys(c_shards)
            c_hessian = self.convert_to_hessian(c_shards, tensor_name='c')
            c_jvp_gradients = self.estimate_hessian_jvp(c_hessian, max_probes=1024, step_size=8, tensor_name='c')

            logger.info("True Hessian (c, channel 0): %s", c_hessian[0])
            for k in c_jvp_gradients.keys():
                if k in [8, 16, 32, 64, 128, 256, 512, 1024]:
                    logger.debug("Estimated Hessian (c, probes=%d): %s", k, c_jvp_gradients[k][0, :, :])

            self.plot_hessian_estimations(c_hessian, c_jvp_gradients, gptq_val=torch.block_diag(gptq_aggregated, gptq_aggregated), tensor_name='c')

            del c_shards, c_hessian, c_jvp_gradients
            self.free()

    def free(self):
        torch.cuda.empty_cache()
        gc.collect()

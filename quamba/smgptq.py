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
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


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

    def read_and_compare(self):
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

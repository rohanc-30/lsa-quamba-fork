"""
JVP-based Jacobian/Hessian Estimator for Mamba Models

This module provides tools for estimating Jacobians and Hessians in Mamba models
using Jacobian-Vector Product (JVP) sampling techniques. The estimator uses forward-mode
automatic differentiation to efficiently compute Jacobian approximations.

Original implementation: SMGPTQ class from quamba/gptq_utils.py
Refactored for clarity and better organization.
"""

import math
import time
import gc
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file, save_file


class MambaJacobianEstimator:
    """
    Jacobian/Hessian estimator for Mamba model layers using JVP sampling.
    
    This class captures layer inputs and outputs during forward passes and
    computes Jacobian estimates using Jacobian-Vector Products (JVPs). The
    Hessian is approximated using the Gauss-Newton approach: H ≈ 2J^T J.
    
    The estimator is specifically designed for Mamba/Mamba2 mixer layers and
    handles the fused projection structure (in_proj contains z, x, b, c, t components).
    
    Attributes:
        idx (int): Layer index for identification and selective processing
        layer: The Mamba mixer layer to analyze
        dev: Device (CPU/GPU) where computations occur
        gptq_dtype: Data type for GPTQ computations (typically float16)
        out_dim (int): Output dimension (after mixer, before out_proj)
        in_dim (int): Input dimension (after in_proj)
        inputs: Captured layer inputs
        pre_outputs: Captured layer outputs (pre-out_proj)
    
    Usage:
        estimator = MambaJacobianEstimator(mixer_layer, idx=0)
        # Register hooks to capture inputs/outputs during forward pass
        # Then call estimator.stitch_plots() to compute and visualize results
    """
    
    def __init__(self, layer, idx=0):
        """
        Initialize the Jacobian estimator for a Mamba mixer layer.
        
        Parameters:
            layer: Mamba/Mamba2 mixer layer to analyze
            idx (int): Layer index (default: 0)
        """
        self.idx = idx
        self.layer = layer
        self.dev = self.layer.in_proj.weight.device
        self.gptq_dtype = torch.float16  # Use float16 to save memory
        
        # Dimensions
        self.out_dim = self.layer.out_proj.weight.shape[1]  # channels after mixer
        self.in_dim = self.layer.in_proj.weight.shape[0]    # channels after in_proj
        
        # Print layer configuration for debugging
        print("headdim: ", self.layer.headdim)
        print("ngroups: ", self.layer.ngroups)
        print("d_state: ", self.layer.d_state)
        print("d_inner: ", self.layer.d_inner)
        print("nheads: ", self.layer.nheads)
        print("expansion factor: ", self.layer.expand)
        print("d_ssm: ", self.layer.d_ssm)
        print("d_model: ", self.layer.d_model)
        print()
    
    def capture_inputs(self, inp, out):
        """
        Capture layer inputs during forward pass.
        
        This method should be called via a forward hook on the mixer layer.
        
        Parameters:
            inp: Input tensor to the mixer
            out: Output tensor from the mixer (not used here)
        """
        self.inputs = inp
    
    def capture_outputs(self, x, z, out):
        """
        Capture layer outputs during forward pass.
        
        This method should be called via a forward hook on the norm layer
        (which is at the end of the mixer).
        
        For Mamba2, the mixer computes: output = out_proj(silu(z) * x)
        We capture x and z before the final out_proj to get the pre-output
        activations needed for Jacobian computation.
        
        Parameters:
            x: X component (from in_proj)
            z: Z component (from in_proj)  
            out: Final output from norm layer
        """
        print(x.shape)
        print(z.shape)
        print(out.shape)
        print(self.layer)
        
        # Compute pre-output: element-wise product of x and silu(z)
        # Sum over batch dimension to get aggregate statistics
        inp = x * torch.nn.functional.silu(z)
        self.pre_outputs = inp.sum(dim=0).unsqueeze(0)
        
        # For Mamba2, we only need half the channels (d_inner)
        self.pre_outputs = self.pre_outputs[:, :self.pre_outputs.shape[1]//2]
    
    def read_shards(self):
        """
        Load previously saved Jacobian shards from disk.
        
        Jacobian data is saved component-wise (z, x, b, c) in safetensors format.
        This method loads the data for the current layer index.
        
        Returns:
            tuple: (z_shards, x_shards, b_shards, c_shards) containing
                   Jacobian data for each component
        """
        path = f'../jacobian_block_samples/130m/layer{self.idx}'
        zpath = os.path.join(path, f"z.safetensors")
        xpath = os.path.join(path, f"x.safetensors")
        bpath = os.path.join(path, f"b.safetensors")
        cpath = os.path.join(path, f"c.safetensors")
        
        z_shards = load_file(zpath)
        x_shards = load_file(xpath)
        b_shards = load_file(bpath)
        c_shards = load_file(cpath)
        
        return z_shards, x_shards, b_shards, c_shards
    
    def int_keys(self, tensor_shard):
        """
        Convert shard keys from strings to integers and normalize values.
        
        Parameters:
            tensor_shard (dict): Dictionary with string keys
        
        Returns:
            dict: Dictionary with integer keys and normalized values
        """
        final_shard = {}
        for key, value in tensor_shard.items():
            # Convert key to int and normalize by batch size
            final_shard[int(key)] = value / self.inputs.shape[0]
        return final_shard
    
    def convert_to_hessian(self, tensor_shard, tensor_name='None'):
        """
        Convert Jacobian shards to Hessian approximations.
        
        Uses the formula: H ≈ J^T J (Gauss-Newton approximation)
        
        For b and c components (which have grouped structure), special handling
        is applied to aggregate across groups.
        
        Parameters:
            tensor_shard (dict): Jacobian shards keyed by channel
            tensor_name (str): Component name ('z', 'x', 'b', or 'c')
        
        Returns:
            dict: Hessian matrices keyed by channel
        """
        hessian_shard = {}
        
        def flatten_shard(tensor_shard):
            return torch.stack(list(tensor_shard.values()), dim=0)
        
        flattened_shard = flatten_shard(tensor_shard)
        
        # Flatten if needed
        if len(flattened_shard.shape) > 3:
            flattened_shard = flattened_shard.reshape(
                flattened_shard.shape[0], flattened_shard.shape[1], -1
            )
        
        # Compute Hessian: H = J^T J (batched matrix multiplication)
        hessian_tensor = flattened_shard.transpose(1, 2).bmm(flattened_shard)
        
        # Map to output dictionary
        for i, key in enumerate(tensor_shard.keys()):
            print(i, key)
            print(self.layer.d_state * self.layer.ngroups)
            print(key % (self.layer.d_state * self.layer.ngroups))
            print(hessian_shard.keys())
            print()
            
            # Special handling for b and c components (grouped structure)
            if (tensor_name == 'b' or tensor_name == 'c') and \
               key % ((self.layer.d_state * self.layer.ngroups)/2) in hessian_shard.keys():
                hessian_shard[int(key % ((self.layer.d_state * self.layer.ngroups)/2))] += \
                    hessian_tensor[i, :, :]
            else:
                if (tensor_name == 'b' or tensor_name == 'c'):
                    hessian_shard[int(key % ((self.layer.d_state * self.layer.ngroups)/2))] = \
                        hessian_tensor[i, :, :]
                else:
                    hessian_shard[key] = hessian_tensor[i, :, :]
        
        print(hessian_shard.keys())
        if max(hessian_shard.keys()) > 3372:
            raise ValueError("Hessian keys are greater than d_inner!")
        
        return hessian_shard
    
    def jvp_gradients_and_hessian_slow(self, tensor_shard, max_probes=1024, 
                                       step_size=8, tensor_name='z'):
        """
        Compute JVP-based gradient estimates and Hessian approximations.
        
        This method uses Jacobian-Vector Products to estimate the Hessian
        without explicitly computing the full Jacobian matrix. Random probe
        vectors are used to sample the Jacobian, and the Hessian is built
        incrementally from these samples.
        
        Algorithm:
        1. Generate random probe vectors
        2. Compute JVPs: v^T J for each probe v
        3. Use autodiff to get gradients: ∇(v^T J) = J^T v
        4. Accumulate: H ≈ (1/n) Σ (J^T v_i)(J^T v_i)^T
        
        Parameters:
            tensor_shard (dict): True Hessian values (for comparison)
            max_probes (int): Number of probe vectors to use (default: 1024)
            step_size (int): Increment for saving intermediate results (default: 8)
            tensor_name (str): Component name ('z', 'x', 'b', or 'c')
        
        Returns:
            dict: Hessian estimates keyed by number of probes used
        """
        channels = torch.tensor(list(tensor_shard.keys()))
        print(channels)
        
        out_channels = channels
        print(out_channels)
        
        t0 = time.time()
        times_list = []
        
        # Generate random probe vectors
        probes = torch.randn(max_probes, self.pre_outputs.shape[1], channels.shape[0])
        probes = probes.to(self.pre_outputs.device)
        
        # Compute JVP scalars: probe · pre_outputs
        JVP_scalars = (probes * self.pre_outputs[:, :, channels] / self.inputs.shape[0])
        JVP_scalars = JVP_scalars.sum(dim=1).sum(dim=1)
        
        Gs = []
        
        # Map channels to in_proj indices based on component
        if tensor_name == 'x':
            channels = channels + self.layer.d_inner
        elif tensor_name == 'b':
            channels = channels + 2 * self.layer.d_inner
        elif tensor_name == 'c':
            channels = channels + 2 * self.layer.d_inner + \
                      self.layer.d_state * self.layer.ngroups
        
        print(channels)
        
        # Special handling for b and c components (grouped across heads)
        if tensor_name == 'b' or tensor_name == 'c':
            for c in channels:
                print(c)
            
            JVP_scalars = []
            for o in out_channels:
                print(o)
                # Replicate across all heads
                all_heads = torch.arange(self.layer.nheads) * (self.layer.headdim)
                out_channels_this_head = all_heads + o
                print(out_channels_this_head)
                
                probes = torch.randn(max_probes, self.pre_outputs.shape[1], 
                                    out_channels_this_head.shape[0])
                probes = probes.to(self.pre_outputs.device)
                
                JVP_scalars.append(
                    (probes * self.pre_outputs[:, :, out_channels_this_head] / 
                     self.inputs.shape[0]).sum(dim=1).sum(dim=1)
                )
            
            print(len(JVP_scalars))
            print(JVP_scalars[0].shape)
            print(JVP_scalars)
            
            rel_outs = [[c, c+1] for c in channels]
            print(rel_outs)
            
            common_time = time.time() - t0
            print(common_time)
            
            for i in range(max_probes // step_size):
                times_list.append(common_time)
            
            # Compute gradients for each JVP scalar
            for JVP_channels, derivative_channels in zip(JVP_scalars, rel_outs):
                l = []
                t0 = time.time()
                for i in range(max_probes):
                    gi_full, = torch.autograd.grad(
                        outputs=JVP_channels[i],
                        inputs=self.layer.in_proj.weight,
                        retain_graph=True,
                        create_graph=False,
                    )
                    gi = gi_full[derivative_channels, :]
                    l.append(gi)
                    
                    if (i + 1) % step_size == 0:
                        times_list[int(i / step_size)] += time.time() - t0
                
                Gs.append(torch.stack(l, dim=0).reshape(max_probes, -1))
                print(Gs[-1].shape)
            print()
        else:
            # For z and x components
            common_time = time.time() - t0
            t0 = time.time()
            
            for i in range(max_probes):
                gi_full, = torch.autograd.grad(
                    outputs=JVP_scalars[i],
                    inputs=self.layer.in_proj.weight,
                    retain_graph=True,
                    create_graph=False,
                )
                gi = gi_full[channels]
                Gs.append(gi)
                
                if (i + 1) % step_size == 0:
                    times_list.append(time.time() - t0 + common_time)
        
        # Stack gradients
        G = torch.stack(Gs, dim=0)
        if tensor_name == 'b' or tensor_name == 'c':
            G = G.transpose(0, 1)
        print(G.shape)
        
        G = G.float()
        
        # Compute Hessian estimates for different probe counts
        H_dict = {}
        for i in range(step_size, max_probes + 1, step_size):
            t0 = time.time()
            H_dict[i] = (G[:i].permute(1, 2, 0) / i).bmm(G[:i].permute(1, 0, 2))
            times_list[int(i / step_size) - 1] += time.time() - t0
        
        # Save timing information
        np.save(f"jacobian_jvp_error/times/{tensor_name}/{self.idx}.npy", 
                np.array(times_list))
        print(times_list)
        print(len(times_list))
        
        return H_dict
    
    def plot_hessian_estimations(self, true_hessians, H_dict, gptq_val=None, 
                                 tensor_name=None):
        """
        Plot Hessian estimation errors vs. number of probes.
        
        Compares the JVP-estimated Hessians against ground truth Hessians
        computed from full Jacobians. Plots cosine similarity as the error metric.
        
        Parameters:
            true_hessians (dict): Ground truth Hessian matrices
            H_dict (dict): Estimated Hessians keyed by probe count
            gptq_val: GPTQ Hessian approximation (for comparison)
            tensor_name (str): Component name for plot labels
        """
        probe_counts = list(H_dict.keys())
        print(probe_counts)
        
        errors = []
        gptq_errors = []
        errors_bd = []
        gptq_errors_bd = []
        error_bar = []
        error_bar_bd = []
        
        for i, k in enumerate(true_hessians.keys()):
            true_hessian = true_hessians[k]
            
            # Create block diagonal version
            true_hessian_block_1 = true_hessian[:true_hessian.shape[0]//2, 
                                               :true_hessian.shape[1]//2]
            true_hessian_block_2 = true_hessian[true_hessian.shape[0]//2:, 
                                               true_hessian.shape[1]//2:]
            true_hessians_bd = torch.block_diag(true_hessian_block_1, 
                                                true_hessian_block_2)
            
            error_channel = []
            error_channel_bd = []
            error_bar_channel = []
            error_bar_channel_bd = []
            
            for k2 in H_dict.keys():
                estimated_hessian = H_dict[k2][i, :, :]
                estimated_hessian_bar = H_dict[k2].mean(dim=0)
                
                # Create block diagonal versions of estimates
                estimated_hessian_block_1 = estimated_hessian[:estimated_hessian.shape[0]//2,
                                                             :estimated_hessian.shape[1]//2]
                estimated_hessian_block_2 = estimated_hessian[estimated_hessian.shape[0]//2:,
                                                             estimated_hessian.shape[1]//2:]
                estimated_hessian_bd = torch.block_diag(estimated_hessian_block_1,
                                                       estimated_hessian_block_2)
                
                estimated_hessian_block_1_bar = estimated_hessian_bar[:estimated_hessian_bar.shape[0]//2,
                                                                     :estimated_hessian_bar.shape[1]//2]
                estimated_hessian_block_2_bar = estimated_hessian_bar[estimated_hessian_bar.shape[0]//2:,
                                                                     estimated_hessian_bar.shape[1]//2:]
                estimated_hessian_bd_bar = torch.block_diag(estimated_hessian_block_1_bar,
                                                           estimated_hessian_block_2_bar)
                
                true_hessian = true_hessian.to(estimated_hessian.device)
                true_hessians_bd = true_hessians_bd.to(estimated_hessian_bd.device)
                
                # Check for infinities
                if torch.isinf(estimated_hessian).any():
                    print("Inf in estimated hessian!")
                if torch.isinf(true_hessian).any():
                    print("Inf in true hessian!")
                    continue
                if torch.linalg.norm(true_hessian).isinf():
                    print(torch.linalg.norm(true_hessian.double()))
                    print(torch.linalg.norm(true_hessian.double()/10000))
                    print()
                
                # Compute cosine similarity as error metric
                error_item = torch.nn.functional.cosine_similarity(
                    true_hessian.flatten().double(),
                    estimated_hessian.flatten().double(),
                    dim=0
                )
                if torch.isnan(error_item):
                    error_item = torch.nn.functional.cosine_similarity(
                        true_hessian.flatten().double()/10000,
                        estimated_hessian.flatten().double()/10000,
                        dim=0
                    )
                error_channel.append(error_item)
                
                error_item_bd = torch.nn.functional.cosine_similarity(
                    true_hessians_bd.flatten().double(),
                    estimated_hessian_bd.flatten().double(),
                    dim=0
                )
                if torch.isnan(error_item_bd):
                    error_item_bd = torch.nn.functional.cosine_similarity(
                        true_hessians_bd.flatten().double()/10000,
                        estimated_hessian_bd.flatten().double()/10000,
                        dim=0
                    )
                error_channel_bd.append(error_item_bd)
                
                error_item_bar = torch.nn.functional.cosine_similarity(
                    true_hessian.flatten().double(),
                    estimated_hessian_bar.flatten().double(),
                    dim=0
                )
                if torch.isnan(error_item_bar):
                    error_item_bar = torch.nn.functional.cosine_similarity(
                        true_hessian.flatten().double()/10000,
                        estimated_hessian_bar.flatten().double()/10000,
                        dim=0
                    )
                error_bar_channel.append(error_item_bar)
                
                error_item_bar_bd = torch.nn.functional.cosine_similarity(
                    true_hessians_bd.flatten().double(),
                    estimated_hessian_bd_bar.flatten().double(),
                    dim=0
                )
                if torch.isnan(error_item_bar_bd):
                    error_item_bar_bd = torch.nn.functional.cosine_similarity(
                        true_hessians_bd.flatten().double()/10000,
                        estimated_hessian_bd_bar.flatten().double()/10000,
                        dim=0
                    )
                error_bar_channel_bd.append(error_item_bar_bd)
            
            # Skip if no valid errors
            if len(error_channel) == 0 or len(error_channel_bd) == 0 or \
               len(error_bar_channel) == 0 or len(error_bar_channel_bd) == 0:
                continue
            
            # Compare with GPTQ if provided
            if gptq_val is not None:
                if torch.isinf(gptq_val).any():
                    print("Inf in gptq_val!")
                gptq_error_item = torch.nn.functional.cosine_similarity(
                    gptq_val.flatten().double()/10000,
                    true_hessian.flatten().double()/10000,
                    dim=0
                )
                gptq_errors.append(gptq_error_item.item())
                
                gptq_error_item_bd = torch.nn.functional.cosine_similarity(
                    gptq_val.flatten().double()/10000,
                    true_hessians_bd.flatten().double()/10000,
                    dim=0
                )
                gptq_errors_bd.append(gptq_error_item_bd.item())
            
            error_channel = torch.stack(error_channel, dim=0)
            error_channel_bd = torch.stack(error_channel_bd, dim=0)
            error_bar_channel = torch.stack(error_bar_channel, dim=0)
            error_bar_channel_bd = torch.stack(error_bar_channel_bd, dim=0)
            
            if torch.isnan(error_channel).any():
                print(i, k)
                print(H_dict[1024][i, :, :])
                print(true_hessian)
                print(H_dict[1024][i, :, :].isnan().any())
                print(true_hessian.isnan().any())
                print(torch.linalg.norm(true_hessian))
                print()
            
            print(error_channel)
            errors.append(error_channel)
            errors_bd.append(error_channel_bd)
            error_bar.append(error_bar_channel)
            error_bar_bd.append(error_bar_channel_bd)
        
        # Aggregate and save results
        print(f"="*100)
        errors = torch.stack(errors, dim=0)
        np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_cosine.npy",
                errors.cpu().numpy())
        errors = errors.mean(dim=0)
        print(errors)
        
        errors_bd = torch.stack(errors_bd, dim=0)
        np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_cosine_bd.npy",
                errors_bd.cpu().numpy())
        errors_bd = errors_bd.mean(dim=0)
        print(errors_bd)
        
        error_bar = torch.stack(error_bar, dim=0)
        np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_cosine_bar.npy",
                error_bar.cpu().numpy())
        error_bar = error_bar.mean(dim=0)
        print(error_bar)
        
        error_bar_bd = torch.stack(error_bar_bd, dim=0)
        np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_cosine_bar_bd.npy",
                error_bar_bd.cpu().numpy())
        error_bar_bd = error_bar_bd.mean(dim=0)
        print(error_bar_bd)
        
        # Save GPTQ comparisons
        if gptq_val is not None:
            print(gptq_errors)
            print(gptq_errors_bd)
            print(len(gptq_errors))
            np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_gptq_cosine.npy",
                    np.array(gptq_errors))
            np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_gptq_cosine_bd.npy",
                    np.array(gptq_errors_bd))
        
        # Create plot
        plt.plot(probe_counts, errors.cpu().numpy())
        plt.xlabel('Probe Count')
        plt.ylabel('Error')
        title = f'Error vs. Probe Count in Layer {self.idx}'
        if tensor_name is not None:
            title += f' for {tensor_name}'
        plt.title(title)
        
        if gptq_val is not None:
            gptq_error_mean = sum(gptq_errors)/len(gptq_errors)
            print(gptq_error_mean)
            gptq_error_mean_bd = sum(gptq_errors_bd)/len(gptq_errors_bd)
            print(gptq_error_mean_bd)
        
        plt.show()
        plt.savefig(f"jacobian_jvp_error/Gaussian_130m_Layer_{self.idx}_{tensor_name}_Error.png")
        plt.clf()
    
    def read_and_compare(self):
        """
        Load previously computed Jacobians and compare with JVP estimates.
        
        This is the main analysis method that:
        1. Loads true Jacobian shards from disk
        2. Converts them to Hessians
        3. Computes JVP-based Hessian estimates
        4. Compares and plots the results
        
        Only processes selected layers to save computation time.
        """
        # Only process selected layers
        if self.idx not in [0, 1, 4, 7, 10, 13, 16, 19, 22, 23]:
            print("skipping layer\n")
            return
        
        print(self.inputs.shape)
        
        # Compute GPTQ-style Hessian approximation for comparison
        comb_inp = self.inputs.mean(dim=0)
        if torch.isinf(comb_inp).any() or torch.isnan(comb_inp).any():
            print("Comb inp broke!!")
            raise ValueError("ERROR!!")
        comb_inp = comb_inp.float()
        
        gptq_parallel = (self.inputs.transpose(1, 2).bmm(self.inputs)).mean(dim=0)
        gptq_aggregated = comb_inp.T @ comb_inp
        print(gptq_parallel.shape)
        print(gptq_aggregated.shape)
        print(gptq_parallel)
        print(gptq_aggregated)
        print()
        
        # Split fused W into individual weight matrices as per mamba2
        W_z = self.layer.in_proj.weight[:self.layer.d_inner, :]
        W_x = self.layer.in_proj.weight[self.layer.d_inner:2 * self.layer.d_inner, :]
        W_b = self.layer.in_proj.weight[
            2 * self.layer.d_inner:
            2 * self.layer.d_inner + self.layer.ngroups * self.layer.d_state, :
        ]
        W_c = self.layer.in_proj.weight[
            2 * self.layer.d_inner + self.layer.ngroups * self.layer.d_state:
            2 * self.layer.d_inner + 2 * self.layer.ngroups * self.layer.d_state, :
        ]
        W_t = self.layer.in_proj.weight[-self.layer.nheads:, :]
        
        # Load and process each component (z, x, b, c)
        z_shards, x_shards, b_shards, c_shards = self.read_shards()
        
        # Process z component
        z_shards = self.int_keys(z_shards)
        z_hessian = self.convert_to_hessian(z_shards)
        z_jvp_gradients = self.jvp_gradients_and_hessian_slow(
            z_hessian, max_probes=1024, step_size=8, tensor_name='z'
        )
        
        print("True Hessian:")
        print(z_hessian[0])
        print()
        
        print("Estimated Hessians:")
        for k in z_jvp_gradients.keys():
            if k not in [8, 16, 32, 64, 128, 256, 512, 1024]:
                continue
            print(k)
            print(z_jvp_gradients[k][0, :, :])
            print()
        
        self.plot_hessian_estimations(z_hessian, z_jvp_gradients, 
                                      gptq_val=gptq_aggregated, tensor_name='z')
        
        del z_shards, z_hessian, z_jvp_gradients
        self.free()
        
        # Process x component
        x_shards = self.int_keys(x_shards)
        x_hessian = self.convert_to_hessian(x_shards)
        x_jvp_gradients = self.jvp_gradients_and_hessian_slow(
            x_hessian, max_probes=1024, step_size=8, tensor_name='x'
        )
        
        print("True Hessian:")
        print(x_hessian[0])
        print()
        
        print("Estimated Hessians:")
        for k in x_jvp_gradients.keys():
            if k not in [8, 16, 32, 64, 128, 256, 512, 1024]:
                continue
            print(k)
            print(x_jvp_gradients[k][0, :, :])
            print()
        
        self.plot_hessian_estimations(x_hessian, x_jvp_gradients,
                                      gptq_val=gptq_aggregated, tensor_name='x')
        
        del x_shards, x_hessian, x_jvp_gradients
        self.free()
        
        # Process b component
        b_shards = self.int_keys(b_shards)
        b_hessian = self.convert_to_hessian(b_shards, tensor_name='b')
        b_jvp_gradients = self.jvp_gradients_and_hessian_slow(
            b_hessian, max_probes=1024, step_size=8, tensor_name='b'
        )
        
        print("True Hessian:")
        print(b_hessian[0])
        print()
        
        print("Estimated Hessians:")
        for k in b_jvp_gradients.keys():
            if k not in [8, 16, 32, 64, 128, 256, 512, 1024]:
                continue
            print(k)
            print(b_jvp_gradients[k][0, :, :])
            print()
        
        self.plot_hessian_estimations(b_hessian, b_jvp_gradients,
                                      gptq_val=torch.block_diag(gptq_aggregated, gptq_aggregated),
                                      tensor_name='b')
        
        del b_shards, b_hessian, b_jvp_gradients
        self.free()
        
        # Process c component
        c_shards = self.int_keys(c_shards)
        c_hessian = self.convert_to_hessian(c_shards, tensor_name='c')
        c_jvp_gradients = self.jvp_gradients_and_hessian_slow(
            c_hessian, max_probes=1024, step_size=8, tensor_name='c'
        )
        
        print("True Hessian:")
        print(c_hessian[0])
        print()
        
        print("Estimated Hessians:")
        for k in c_jvp_gradients.keys():
            if k not in [8, 16, 32, 64, 128, 256, 512, 1024]:
                continue
            print(k)
            print(c_jvp_gradients[k][0, :, :])
            print()
        
        self.plot_hessian_estimations(c_hessian, c_jvp_gradients,
                                      gptq_val=torch.block_diag(gptq_aggregated, gptq_aggregated),
                                      tensor_name='c')
        
        del c_shards, c_hessian, c_jvp_gradients
        self.free()
        
        return
    
    def stitch_plots(self):
        """
        Create aggregated plots across all layers.
        
        This method loads the previously saved Jacobian data from all processed
        layers and creates comprehensive visualization plots showing:
        - Error trends across layers
        - Timing information
        - Comparisons with GPTQ baseline
        
        Only runs for layer 0 (to avoid duplicate processing).
        """
        # Only run for layer 0
        if self.idx != 0:
            print("skipping layer\n")
            return
        
        # Layers that were processed
        layers = [0, 1, 4, 7, 10, 13, 16, 19, 22, 23]
        
        probe_counts = [8*i for i in range(1, 129)]
        
        # Lists to store data from all layers
        z_list = []
        x_list = []
        b_list = []
        c_list = []
        
        gptq_list_z = []
        gptq_list_x = []
        gptq_list_b = []
        gptq_list_c = []
        
        z_list_bd = []
        x_list_bd = []
        b_list_bd = []
        c_list_bd = []
        
        gptq_list_z_bd = []
        gptq_list_x_bd = []
        gptq_list_b_bd = []
        gptq_list_c_bd = []
        
        # Load timing data from all layers
        for layer in layers:
            if not np.isnan(np.load(f"jacobian_jvp_error/times/z/{layer}.npy")).any():
                z_list.append(np.load(f"jacobian_jvp_error/times/z/{layer}.npy"))
            if not np.isnan(np.load(f"jacobian_jvp_error/times/x/{layer}.npy")).any():
                x_list.append(np.load(f"jacobian_jvp_error/times/x/{layer}.npy"))
            if not np.isnan(np.load(f"jacobian_jvp_error/times/b/{layer}.npy")).any():
                b_list.append(np.load(f"jacobian_jvp_error/times/b/{layer}.npy"))
            if not np.isnan(np.load(f"jacobian_jvp_error/times/c/{layer}.npy")).any():
                c_list.append(np.load(f"jacobian_jvp_error/times/c/{layer}.npy"))
            
            # Load cosine similarity data (block diagonal)
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/z/{layer}_cosine_bd.npy")).any():
                z_list_bd.append(np.load(f"jacobian_jvp_error/raw_data/z/{layer}_cosine_bd.npy").mean(axis=0))
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/x/{layer}_cosine_bd.npy")).any():
                x_list_bd.append(np.load(f"jacobian_jvp_error/raw_data/x/{layer}_cosine_bd.npy").mean(axis=0))
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/b/{layer}_cosine_bd.npy")).any():
                b_list_bd.append(np.load(f"jacobian_jvp_error/raw_data/b/{layer}_cosine_bd.npy").mean(axis=0))
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/c/{layer}_cosine_bd.npy")).any():
                c_list_bd.append(np.load(f"jacobian_jvp_error/raw_data/c/{layer}_cosine_bd.npy").mean(axis=0))
            
            # Load GPTQ comparison data
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/z/{layer}_gptq_cosine.npy")).any():
                gptq_list_z.extend(np.load(f"jacobian_jvp_error/raw_data/z/{layer}_gptq_cosine.npy").flatten())
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/x/{layer}_gptq_cosine.npy")).any():
                gptq_list_x.extend(np.load(f"jacobian_jvp_error/raw_data/x/{layer}_gptq_cosine.npy").flatten())
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/b/{layer}_gptq_cosine.npy")).any():
                gptq_list_b.extend(np.load(f"jacobian_jvp_error/raw_data/b/{layer}_gptq_cosine.npy").flatten())
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/c/{layer}_gptq_cosine.npy")).any():
                gptq_list_c.extend(np.load(f"jacobian_jvp_error/raw_data/c/{layer}_gptq_cosine.npy").flatten())
            
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/z/{layer}_gptq_cosine_bd.npy")).any():
                gptq_list_z_bd.extend(np.load(f"jacobian_jvp_error/raw_data/z/{layer}_gptq_cosine_bd.npy").flatten())
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/x/{layer}_gptq_cosine_bd.npy")).any():
                gptq_list_x_bd.extend(np.load(f"jacobian_jvp_error/raw_data/x/{layer}_gptq_cosine_bd.npy").flatten())
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/b/{layer}_gptq_cosine_bd.npy")).any():
                gptq_list_b_bd.extend(np.load(f"jacobian_jvp_error/raw_data/b/{layer}_gptq_cosine_bd.npy").flatten())
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/c/{layer}_gptq_cosine_bd.npy")).any():
                gptq_list_c_bd.extend(np.load(f"jacobian_jvp_error/raw_data/c/{layer}_gptq_cosine_bd.npy").flatten())
        
        # Stack and average across layers
        z_list = np.stack(z_list, axis=0)
        x_list = np.stack(x_list, axis=0)
        b_list = np.stack(b_list, axis=0)
        c_list = np.stack(c_list, axis=0)
        z_list_bd = np.stack(z_list_bd, axis=0)
        x_list_bd = np.stack(x_list_bd, axis=0)
        b_list_bd = np.stack(b_list_bd, axis=0)
        c_list_bd = np.stack(c_list_bd, axis=0)
        
        print(z_list)
        
        z_list = z_list.mean(axis=0)
        x_list = x_list.mean(axis=0)
        b_list = b_list.mean(axis=0)
        c_list = c_list.mean(axis=0)
        z_list_bd = z_list_bd.mean(axis=0)
        x_list_bd = x_list_bd.mean(axis=0)
        b_list_bd = b_list_bd.mean(axis=0)
        c_list_bd = c_list_bd.mean(axis=0)
        
        print(gptq_list_z)
        for i in range(len(gptq_list_z)):
            print(gptq_list_z[i])
            print(gptq_list_z[i].shape)
            print()
        
        gptq_mean_z = np.mean(gptq_list_z)
        gptq_mean_x = np.mean(gptq_list_x)
        gptq_mean_b = np.mean(gptq_list_b)
        gptq_mean_c = np.mean(gptq_list_c)
        
        gptq_mean_z_bd = np.mean(gptq_list_z_bd)
        gptq_mean_x_bd = np.mean(gptq_list_x_bd)
        gptq_mean_b_bd = np.mean(gptq_list_b_bd)
        gptq_mean_c_bd = np.mean(gptq_list_c_bd)
        
        print(gptq_mean_z)
        print(gptq_mean_x)
        print(gptq_mean_b)
        print(gptq_mean_c)
        
        # Create comprehensive runtime plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(probe_counts, z_list, label=r'$W_z$', color='red')
        ax1.plot(probe_counts, x_list, label=r'$W_x$', color='green')
        
        ax1.set_xlabel('Probe Count (# IID Samples)')
        ax1.set_ylabel(r'Runtime for $W_z$, $W_x$ (s)')
        ax1.set_title(r'Runtime vs. Probe Count (Averaged Across 10 Layers)', pad=10)
        
        values = [gptq_mean_z, gptq_mean_x, gptq_mean_b, gptq_mean_c]
        labels = [r"GPTQ $W_z$", r"GPTQ $W_x$", r"GPTQ $W_b$", r"GPTQ $W_c$"]
        colors = ['red', 'green', 'blue', 'purple']
        
        # Create second y-axis for b and c components
        ax2 = ax1.twinx()
        ax2.plot(probe_counts, b_list, label=r'$W_b$', color='blue')
        ax2.plot(probe_counts, c_list, label=r'$W_c$', color='purple')
        ax2.set_ylabel(r'Runtime for $W_b$, $W_c$ (s)')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', 
                  bbox_to_anchor=(0.5, 1.10), ncol=8, frameon=True, fontsize='small')
        
        plt.subplots_adjust(top=0.82)
        plt.show()
        
        plt.savefig(f"jacobian_jvp_error/Gaussian_130m_Stitched_Averaged_Runtime2.png", 
                   bbox_inches='tight')
        
        plt.clf()
        
        print("Stitched plots complete!")
    
    def free(self):
        """Free GPU memory."""
        torch.cuda.empty_cache()
        gc.collect()


# Legacy alias for backward compatibility
SMGPTQ = MambaJacobianEstimator


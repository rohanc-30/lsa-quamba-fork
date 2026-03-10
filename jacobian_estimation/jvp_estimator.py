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
    
    # ========================================================================
    # Centralized Error Metric Configuration
    # ========================================================================
    # Error metrics are computed as a cartesian product of:
    #   1. Matrix representations (full vs block-diagonal)
    #   2. Aggregation methods (per-channel vs mean)
    #
    # To add a new variant:
    #   1. Add to _get_matrix_transforms() or _get_aggregation_functions()
    #   2. Add label to _get_matrix_labels() or _get_aggregation_labels()
    #   3. Everything else (saving, loading, plotting) updates automatically
    # ========================================================================
    
    def _get_matrix_transforms(self):
        """
        Get dictionary mapping matrix representation suffixes to transform functions.
        
        Returns:
            dict: {suffix: transform_function}
        """
        return {
            '': lambda x: x,                    # full matrix (identity)
            '_bd': self._make_block_diagonal    # block diagonal
        }
    
    def _get_aggregation_functions(self):
        """
        Get dictionary mapping aggregation suffixes to aggregation functions.
        
        Returns:
            dict: {suffix: aggregation_function}
        """
        return {
            '': self._per_channel_aggregation,  # per-channel
            '_bar': self._mean_aggregation      # mean across channels
        }
    
    @staticmethod
    def _get_matrix_labels():
        """Get human-readable labels for matrix representations (for plotting)."""
        return {
            '': 'Full Matrix',
            '_bd': 'Block Diagonal',
        }
    
    @staticmethod
    def _get_aggregation_labels():
        """Get human-readable labels for aggregation methods (for plotting)."""
        return {
            '': 'Per-Channel',
            '_bar': 'Mean',
        }
    
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
        base_path = f'../jacobian_block_samples/130m/layer{self.idx}'
        components = ['z', 'x', 'b', 'c']
        return tuple(load_file(os.path.join(base_path, f"{comp}.safetensors")) 
                     for comp in components)
    
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
        # Stack and flatten Jacobian shards
        flattened_shard = torch.stack(list(tensor_shard.values()), dim=0)
        if len(flattened_shard.shape) > 3:
            flattened_shard = flattened_shard.reshape(
                flattened_shard.shape[0], flattened_shard.shape[1], -1
            )
        
        # Compute Hessian: H = J^T J (batched matrix multiplication)
        hessian_tensor = flattened_shard.transpose(1, 2).bmm(flattened_shard)
        
        # Map Hessian tensors back to dictionary
        hessian_shard = {}
        is_grouped = tensor_name in ('b', 'c')
        group_size = int((self.layer.d_state * self.layer.ngroups) / 2) if is_grouped else None
        
        for i, key in enumerate(tensor_shard.keys()):
            # For b/c components, use modulo to aggregate across groups
            hessian_key = int(key % group_size) if is_grouped else key
            hessian_value = hessian_tensor[i, :, :]
            
            # Accumulate if key already exists, otherwise create new entry
            if hessian_key in hessian_shard:
                hessian_shard[hessian_key] += hessian_value
            else:
                hessian_shard[hessian_key] = hessian_value
        
        # Sanity check
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
        out_channels = torch.tensor(list(tensor_shard.keys()))
        is_grouped = tensor_name in ('b', 'c')
        
        # Map out_channels to in_proj weight indices based on component type
        channel_offsets = {'z': 0, 'x': self.layer.d_inner, 
                          'b': 2 * self.layer.d_inner,
                          'c': 2 * self.layer.d_inner + self.layer.d_state * self.layer.ngroups}
        in_proj_channels = out_channels + channel_offsets.get(tensor_name, 0)
        
        # Compute gradients using JVP
        start_time = time.time()
        times_list = []
        gradients = []
        
        if is_grouped:
            # Special handling for b and c: replicate across heads
            times_list = [time.time() - start_time] * (max_probes // step_size)
            derivative_channels_list = [[c, c+1] for c in in_proj_channels]
            
            for out_ch, deriv_channels in zip(out_channels, derivative_channels_list):
                # Compute output channels for all heads
                head_offsets = torch.arange(self.layer.nheads) * self.layer.headdim
                out_channels_all_heads = head_offsets + out_ch
                
                # Generate probes and compute JVP scalars for this channel
                probes = torch.randn(max_probes, self.pre_outputs.shape[1], 
                                    len(out_channels_all_heads), device=self.pre_outputs.device)
                jvp_scalars = (probes * self.pre_outputs[:, :, out_channels_all_heads] / 
                              self.inputs.shape[0]).sum(dim=1).sum(dim=1)
                
                # Compute gradients for each probe
                channel_gradients = []
                grad_start_time = time.time()
                for i in range(max_probes):
                    grad_full, = torch.autograd.grad(
                        outputs=jvp_scalars[i],
                        inputs=self.layer.in_proj.weight,
                        retain_graph=True,
                        create_graph=False,
                    )
                    channel_gradients.append(grad_full[deriv_channels, :])
                    
                    if (i + 1) % step_size == 0:
                        times_list[(i + 1) // step_size - 1] += time.time() - grad_start_time
                
                gradients.append(torch.stack(channel_gradients, dim=0).reshape(max_probes, -1))
        else:
            # For z and x components: standard JVP computation
            common_time = time.time() - start_time
            
            # Generate probes and compute JVP scalars
            probes = torch.randn(max_probes, self.pre_outputs.shape[1], 
                                len(out_channels), device=self.pre_outputs.device)
            jvp_scalars = (probes * self.pre_outputs[:, :, out_channels] / 
                          self.inputs.shape[0]).sum(dim=1).sum(dim=1)
            
            # Compute gradients for each probe
            grad_start_time = time.time()
            for i in range(max_probes):
                grad_full, = torch.autograd.grad(
                    outputs=jvp_scalars[i],
                    inputs=self.layer.in_proj.weight,
                    retain_graph=True,
                    create_graph=False,
                )
                gradients.append(grad_full[in_proj_channels])
                
                if (i + 1) % step_size == 0:
                    times_list.append(time.time() - grad_start_time + common_time)
        
        # Stack and transpose gradients
        G = torch.stack(gradients, dim=0)
        if is_grouped:
            G = G.transpose(0, 1)
        G = G.float()
        
        # Compute Hessian estimates for different probe counts
        H_dict = {}
        for num_probes in range(step_size, max_probes + 1, step_size):
            hessian_start = time.time()
            H_dict[num_probes] = (G[:num_probes].permute(1, 2, 0) / num_probes).bmm(
                G[:num_probes].permute(1, 0, 2))
            times_list[num_probes // step_size - 1] += time.time() - hessian_start
        
        # Save timing information
        np.save(f"jacobian_jvp_error/times/{tensor_name}/{self.idx}.npy", 
                np.array(times_list))
        
        return H_dict
    
    def _make_block_diagonal(self, tensor):
        """Create block diagonal matrix from halves of input tensor."""
        mid = tensor.shape[0] // 2
        block_1 = tensor[:mid, :mid]
        block_2 = tensor[mid:, mid:]
        return torch.block_diag(block_1, block_2)
    
    def _robust_cosine_similarity(self, tensor1, tensor2):
        """Compute cosine similarity with NaN handling and rescaling."""
        similarity = torch.nn.functional.cosine_similarity(
            tensor1.flatten().double(),
            tensor2.flatten().double(),
            dim=0
        )
        # Rescale if NaN (typically due to numerical issues)
        if torch.isnan(similarity):
            similarity = torch.nn.functional.cosine_similarity(
                tensor1.flatten().double() / 10000,
                tensor2.flatten().double() / 10000,
                dim=0
            )
        return similarity
    
    def _save_error_data(self, errors, suffix, tensor_name):
        """Save error data to disk and return mean."""
        stacked = torch.stack(errors, dim=0)
        np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_{suffix}.npy",
                stacked.cpu().numpy())
        return stacked.mean(dim=0)
    
    # Hessian aggregation functions for H_dict
    @staticmethod
    def _per_channel_aggregation(H_dict, probe_count, channel_idx):
        """Return the hessian for a specific channel (identity aggregation)."""
        return H_dict[probe_count][channel_idx, :, :]
    
    @staticmethod
    def _mean_aggregation(H_dict, probe_count, channel_idx):
        """Return the mean hessian across all channels."""
        return H_dict[probe_count].mean(dim=0)
    
    def get_all_metric_names(self):
        """
        Generate all metric names from the cartesian product.
        
        Returns:
            list: All metric names (e.g., ['cosine', 'cosine_bd', 'cosine_bar', 'cosine_bar_bd'])
        """
        return [f"cosine{mat_suffix}{agg_suffix}" 
                for mat_suffix in self._get_matrix_transforms().keys()
                for agg_suffix in self._get_aggregation_functions().keys()]
    
    def plot_hessian_estimations(self, true_hessians, H_dict, gptq_val=None, 
                                 tensor_name=None):
        """
        Plot Hessian estimation errors vs. number of probes.
        
        Compares the JVP-estimated Hessians against ground truth Hessians
        computed from full Jacobians. Plots cosine similarity as the error metric.
        
        The error computation is structured as a cartesian product of:
        1. Matrix representations (full vs block-diagonal)
        2. Aggregation functions (per-channel vs mean across channels)
        
        Parameters:
            true_hessians (dict): Ground truth Hessian matrices
            H_dict (dict): Estimated Hessians keyed by probe count
            gptq_val: GPTQ Hessian approximation (for comparison)
            tensor_name (str): Component name for plot labels
        """
        probe_counts = list(H_dict.keys())
        
        # Get transformation and aggregation functions from centralized config
        matrix_representations = self._get_matrix_transforms()
        aggregation_functions = self._get_aggregation_functions()
        
        # Generate all error keys from cartesian product
        error_keys = self.get_all_metric_names()
        error_collectors = {key: [] for key in error_keys}
        
        # GPTQ only varies by matrix representation (not by aggregation)
        gptq_error_keys = [f"cosine{mat_suffix}" for mat_suffix in matrix_representations.keys()]
        gptq_errors = {key: [] for key in gptq_error_keys}
        
        for channel_idx, channel_key in enumerate(true_hessians.keys()):
            true_hessian_full = true_hessians[channel_key]
            
            # Check for infinities and skip if found
            if torch.isinf(true_hessian_full).any():
                continue
            
            # Track errors for this channel across all probe counts
            channel_errors = {key: [] for key in error_keys}
            
            for probe_count in probe_counts:
                # Move to correct device
                device = H_dict[probe_count].device
                true_hessian_full = true_hessian_full.to(device)
                
                # Compute errors for all combinations (cartesian product)
                for mat_suffix, mat_fn in matrix_representations.items():
                    for agg_suffix, agg_fn in aggregation_functions.items():
                        # Apply matrix representation to true hessian
                        true_hessian = mat_fn(true_hessian_full)
                        
                        # Apply aggregation function to estimated hessian
                        estimated = agg_fn(H_dict, probe_count, channel_idx)
                        
                        # Apply matrix representation to estimated hessian
                        estimated = mat_fn(estimated)
                        
                        # Compute error and store
                        error_key = f"cosine{mat_suffix}{agg_suffix}"
                        error = self._robust_cosine_similarity(true_hessian, estimated)
                        channel_errors[error_key].append(error)
            
            # Skip if no valid errors for this channel
            if not all(channel_errors.values()):
                continue
            
            # Compute GPTQ errors if provided
            if gptq_val is not None:
                gptq_val = gptq_val.to(true_hessian_full.device)
                for mat_suffix, mat_fn in matrix_representations.items():
                    true_hessian = mat_fn(true_hessian_full)
                    gptq_key = f"cosine{mat_suffix}"
                    gptq_errors[gptq_key].append(
                        self._robust_cosine_similarity(gptq_val, true_hessian).item())
            
            # Stack and accumulate errors across channels
            for key in error_keys:
                error_collectors[key].append(torch.stack(channel_errors[key], dim=0))
        
        # Check if any channels were successfully processed
        if not any(error_collectors.values()):
            print(f"Warning: No valid channels processed for {tensor_name} in layer {self.idx}")
            return
        
        # Aggregate, save, and print all error types
        print("=" * 100)
        error_means = {}
        for key, errors in error_collectors.items():
            if errors:  # Only process non-empty error lists
                error_means[key] = self._save_error_data(errors, key, tensor_name)
                print(f"{key}: {error_means[key]}")
        
        # Save GPTQ comparisons
        if gptq_val is not None:
            for key, errors in gptq_errors.items():
                if errors:  # Only save if we have data
                    np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_gptq_{key}.npy",
                            np.array(errors))
                    print(f"GPTQ {key} mean: {np.mean(errors)}")
        
        # Create and save plot (using the first error key by default)
        if error_means:
            primary_error_key = list(error_means.keys())[0]
            plt.plot(probe_counts, error_means[primary_error_key].cpu().numpy())
            plt.xlabel('Probe Count')
            plt.ylabel('Error (Cosine Similarity)')
            plt.title(f'Error vs. Probe Count in Layer {self.idx}' + 
                     (f' for {tensor_name}' if tensor_name else ''))
            plt.savefig(f"jacobian_jvp_error/Gaussian_130m_Layer_{self.idx}_{tensor_name}_Error.png")
            plt.show()
            plt.clf()
    
    def _process_component(self, shard, component_name, gptq_val, max_probes=1024, step_size=8):
        """
        Process a single component (z, x, b, or c) through the full pipeline.
        
        Parameters:
            shard (dict): Raw Jacobian shard data
            component_name (str): Name of component ('z', 'x', 'b', or 'c')
            gptq_val: GPTQ Hessian approximation for comparison
            max_probes (int): Number of JVP probe vectors
            step_size (int): Increment for intermediate results
        """
        # Convert shard keys and compute Hessian
        shard = self.int_keys(shard)
        hessian = self.convert_to_hessian(shard, tensor_name=component_name)
        
        # Compute JVP-based Hessian estimates
        jvp_gradients = self.jvp_gradients_and_hessian_slow(
            hessian, max_probes=max_probes, step_size=step_size, tensor_name=component_name
        )
        
        # Plot and save results
        self.plot_hessian_estimations(hessian, jvp_gradients, 
                                      gptq_val=gptq_val, tensor_name=component_name)
        
        # Clean up memory
        del shard, hessian, jvp_gradients
        self.free()
    
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
        comb_inp = self.inputs.mean(dim=0).float()
        if torch.isinf(comb_inp).any() or torch.isnan(comb_inp).any():
            raise ValueError("Invalid values in input: inf or nan detected")
        
        gptq_aggregated = comb_inp.T @ comb_inp
        gptq_block_diag = torch.block_diag(gptq_aggregated, gptq_aggregated)
        
        # Load Jacobian shards for all components
        z_shards, x_shards, b_shards, c_shards = self.read_shards()
        
        # Process each component with appropriate GPTQ baseline
        component_specs = [
            (z_shards, 'z', gptq_aggregated),
            (x_shards, 'x', gptq_aggregated),
            (b_shards, 'b', gptq_block_diag),
            (c_shards, 'c', gptq_block_diag),
        ]
        
        for shard, name, gptq_val in component_specs:
            self._process_component(shard, name, gptq_val, max_probes=64, step_size=4)
    
    @staticmethod
    def _load_metric_across_layers(component, metric, layers, base_path="jacobian_jvp_error/raw_data"):
        """
        Load a metric for a component across multiple layers.
        
        Parameters:
            component (str): Component name ('z', 'x', 'b', or 'c')
            metric (str): Metric name (e.g., 'cosine', 'cosine_bd', 'cosine_bar')
            layers (list): List of layer indices to load
            base_path (str): Base directory for data files
        
        Returns:
            np.ndarray: Averaged metric across layers and channels
        """
        data_list = []
        for layer in layers:
            try:
                file_path = f"{base_path}/{component}/{layer}_{metric}.npy"
                data = np.load(file_path)
                if not np.isnan(data).any():
                    # Average across channels (axis 0) to get probe-count dimension
                    data_list.append(data.mean(axis=0))
            except FileNotFoundError:
                continue
        
        if data_list:
            # Stack across layers and average
            return np.stack(data_list, axis=0).mean(axis=0)
        return None
    
    @staticmethod
    def _load_timing_across_layers(component, layers, base_path="jacobian_jvp_error/times"):
        """Load timing data for a component across multiple layers."""
        timing_list = []
        for layer in layers:
            try:
                file_path = f"{base_path}/{component}/{layer}.npy"
                data = np.load(file_path)
                if not np.isnan(data).any():
                    timing_list.append(data)
            except FileNotFoundError:
                continue
        
        if timing_list:
            return np.stack(timing_list, axis=0).mean(axis=0)
        return None
    
    def plot_aggregated_errors(self, layers=None, step_size=4, max_probes=64):
        """
        Plot error curves for all components and aggregation methods.
        
        Creates one plot per aggregation method showing all components (z, x, b, c).
        
        Parameters:
            layers (list): Layer indices to aggregate over (default: [0, 1, 4, 7, 10, 13, 16, 19, 22, 23])
            step_size (int): Step size used during estimation
            max_probes (int): Maximum number of probes used
        """
        if self.idx != 23:
            return
        
        if layers is None:
            layers = [0, 1, 4, 7, 10, 13, 16, 19, 22, 23]
        
        probe_counts = [step_size * i for i in range(1, max_probes // step_size + 1)]
        components = ['z', 'x', 'b', 'c']
        component_labels = [r'$W_z$', r'$W_x$', r'$W_b$', r'$W_c$']
        component_colors = ['red', 'green', 'blue', 'purple']
        
        # Get labels for plotting
        matrix_labels = self._get_matrix_labels()
        aggregation_labels = self._get_aggregation_labels()
        
        # Create a plot for each combination of matrix representation and aggregation
        for mat_suffix in self._get_matrix_transforms().keys():
            for agg_suffix in self._get_aggregation_functions().keys():
                metric = f"cosine{mat_suffix}{agg_suffix}"
                mat_label = matrix_labels[mat_suffix]
                agg_label = aggregation_labels[agg_suffix]
                
                plt.figure(figsize=(10, 6))
                
                # Plot each component
                for comp, label, color in zip(components, component_labels, component_colors):
                    errors = self._load_metric_across_layers(comp, metric, layers)
                    if errors is not None:
                        plt.plot(probe_counts, errors, label=label, color=color, linewidth=2)
                
                plt.xlabel('Probe Count (# IID Samples)', fontsize=12)
                plt.ylabel('Cosine Similarity', fontsize=12)
                plt.title(f'JVP Hessian Estimation Error: {mat_label}, {agg_label}\n'
                         f'(Averaged Across {len(layers)} Layers)', fontsize=13)
                plt.legend(loc='best', fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                save_path = f"jacobian_jvp_error/Aggregated_Error_{metric}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved error plot: {save_path}")
    
    def plot_runtime_comparison(self, layers=None, step_size=4, max_probes=64):
        """
        Plot runtime vs probe count for all components.
        
        Creates a dual y-axis plot with z,x on one axis and b,c on another
        (since b,c typically take longer due to grouped structure).
        
        Parameters:
            layers (list): Layer indices to aggregate over
            step_size (int): Step size used during estimation
            max_probes (int): Maximum number of probes used
        """
        if self.idx != 23:
            return
        
        if layers is None:
            layers = [0, 1, 4, 7, 10, 13, 16, 19, 22, 23]
        
        probe_counts = [step_size * i for i in range(1, max_probes // step_size + 1)]
        
        # Load timing data for all components
        timing_data = {}
        for comp in ['z', 'x', 'b', 'c']:
            timing_data[comp] = self._load_timing_across_layers(comp, layers)
        
        # Create dual y-axis plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot z and x on primary axis
        if timing_data['z'] is not None:
            ax1.plot(probe_counts, timing_data['z'], label=r'$W_z$', 
                    color='red', linewidth=2)
        if timing_data['x'] is not None:
            ax1.plot(probe_counts, timing_data['x'], label=r'$W_x$', 
                    color='green', linewidth=2)
        
        ax1.set_xlabel('Probe Count (# IID Samples)', fontsize=12)
        ax1.set_ylabel(r'Runtime for $W_z$, $W_x$ (s)', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Create secondary y-axis for b and c
        ax2 = ax1.twinx()
        if timing_data['b'] is not None:
            ax2.plot(probe_counts, timing_data['b'], label=r'$W_b$', 
                    color='blue', linewidth=2)
        if timing_data['c'] is not None:
            ax2.plot(probe_counts, timing_data['c'], label=r'$W_c$', 
                    color='purple', linewidth=2)
        
        ax2.set_ylabel(r'Runtime for $W_b$, $W_c$ (s)', fontsize=12, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper left', fontsize=11, frameon=True)
        
        plt.title(f'Runtime vs. Probe Count (Averaged Across {len(layers)} Layers)', 
                 fontsize=13, pad=15)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = "jacobian_jvp_error/Aggregated_Runtime.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved runtime plot: {save_path}")
    
    def create_all_plots(self, layers=None, step_size=4, max_probes=64):
        """
        Create all aggregated plots: error curves and runtime comparison.
        
        This is the main entry point for generating plots after running
        JVP estimation across multiple layers.
        
        Parameters:
            layers (list): Layer indices to aggregate over
            step_size (int): Step size used during estimation
            max_probes (int): Maximum number of probes used
        """
        if self.idx != 23:
            print("Skipping plot generation (only runs on layer 23)")
            return
        
        print("Generating aggregated error plots...")
        self.plot_aggregated_errors(layers, step_size, max_probes)
        
        print("Generating runtime comparison plot...")
        self.plot_runtime_comparison(layers, step_size, max_probes)
        
        print("All plots generated successfully!")
    
    def free(self):
        """Free GPU memory."""
        torch.cuda.empty_cache()
        gc.collect()


# Legacy alias for backward compatibility
SMGPTQ = MambaJacobianEstimator


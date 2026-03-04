"""
This file is a modified version of the original file from the GPTQ repo.
https://github.com/IST-DASLab/gptq
"""
import math
# from tarfile import _Bz2ReadableFileobj
import time
import gc
from unittest import BaseTestSuite

import matplotlib.pyplot as plt

import os

import torch
import torch.nn as nn
import transformers
from torch.autograd.functional import jacobian
import numpy as np

from safetensors.torch import load_file, save_file

from quamba.qLinearLayer import HadLinear
from quamba.datatype_utils import fake_quantize_with_type, get_datatypes, get_type_scales, get_quant_value_from_dtype_lists

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# class Quantizer_GPTQ(nn.Module):
#     def __init__(self, shape=1):
#         super(Quantizer_GPTQ, self).__init__()

#     def configure(self, bits, sym=True, clip_ratio=1.0, data_type='int'):
#         self.bits = bits
#         self.datatype_list = get_datatypes(data_type, bits)
#         quant_value_list = get_quant_value_from_dtype_lists(self.datatype_list)
#         quant_value_tensor_list = [torch.tensor(quant_value) for quant_value in quant_value_list]
#         self.quant_values_tensor = torch.stack(quant_value_tensor_list)  # Shape: (num_qsets, num_qvalues)
#         #logging.info(f"Set Quantization values in GPTQ_quantizer: {self.quant_values_tensor}")
#         # HY: Asymmetric quantization has not been tested yet
#         self.sym = sym
#         self.clip_ratio = clip_ratio

#     @torch.no_grad()
#     def find_params(self, x, dtype):
#         dev = x.device
#         self.quant_values_tensor = self.quant_values_tensor.to(dev)

#         # Reshape x to 2D tensor (num_rows, num_elements)
#         original_shape = x.shape
#         x_flat = x.reshape(-1, x.shape[-1])  # Shape: (num_rows, num_elements)
#         num_rows = x_flat.shape[0]

#         # Compute scales and zero points for each quantization set
#         scale, zero = get_type_scales(x_flat, self.quant_values_tensor, self.sym)  # Shape: (3, num_rows, 1)

#         # Quantize and dequantize x with each quantization set
#         x_dq = fake_quantize_with_type(x_flat, scale, zero, self.quant_values_tensor, dtype=dtype)  # Shape: (3, num_rows, num_elements)

#         # Compute MSE between original and quantized x with each quantization set
#         mse = ((x_dq - x_flat.unsqueeze(0)) ** 2).mean(dim=2)  # Shape: (3, num_rows)
#         # Select the quantization set with minimal MSE for each row
#         _, best_qset_indices = mse.min(dim=0)  # Shape: (num_rows)
        
#         # Gather the best scales, zero points, and quantization values for each row
#         best_scale = scale[best_qset_indices, torch.arange(num_rows)]  # Shape: (num_rows, 1)
#         best_zero = zero[best_qset_indices, torch.arange(num_rows)]    # Shape: (num_rows, 1)

#         # Store the parameters
#         scale = best_scale.view(*original_shape[:-1], 1)  # Reshape to match x
#         zero = best_zero.view(*original_shape[:-1], 1)
#         torch.cuda.empty_cache()
#         return scale, zero, best_qset_indices

@torch.no_grad()
def get_per_channel_scale(w, num_bits=4):
    # HY: use the simple quant in qqq_quantize_weights
    max_q_val = 2**num_bits - 1 # 15
    # Compute scale for each output channel
    s = torch.max(torch.abs(w), 1, keepdim=True)[0] # w: [Dout, Din] 
    s *= 2 / max_q_val  # 2 => symmetric, 2 / 15 # s: [Dout, 1] 
    return s

@torch.no_grad()
def quant(w, s, num_bits=4):
    # HY: use the simple quant in qqq_quantize_weights
    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2
    # w: [Dout, Din], s: [Dout, 1]
    q_w = torch.round(w / s).int() # round([-7.5, 7.5]) -> [-8, 8], .int() will replace NaN with 0
    q_w += half_q_val # [0, 16]
    q_w = torch.clamp(q_w, 0, max_q_val) #[0, 15]
    # Compute ref (dequantized)
    w_ref = (q_w - half_q_val) * s
    return w_ref.half()

class GPTQ:
    def __init__(self, layer):

        self.layer = layer
        self.dev = self.layer.weight.device
        # HY: To save memory, we use float16 to compute distances for non-uniform quantization
        #     see datatype_utils.py for def fake_quantize_with_type 
        self.gptq_dtype = torch.float16
        W = layer.weight.data.clone()

        if not isinstance(self.layer, (nn.Linear, HadLinear)):
            raise NotImplementedError("Only HadLinear and nn.Linear is supported for now")
        
        self.out_dim = W.shape[0] # out dim, row
        self.in_dim = W.shape[1] # in dim, columns
        self.H = torch.zeros((self.in_dim, self.in_dim), device=self.dev) 
        self.nsamples = 0 
        del W

    ### Need to modify this hook for SM-GPTQ -- have to either calculate J online or redo to calculate J offline
    ### J is the derivative of the mixer output with respect to the activations/inputs
    ### 2J^TJ is the new Hessian of the mixer output with respect to the activations/inputs; H after being calculated is used the same as in GPTQ
    ### Math work to do: verify 2J^TJ is is actually H, reading Gauss-Newton if you have to
    ### Engineering work to do: determine if hook can materialize J online
    ### Side quests: work on understanding activation rearrangement, extend quantization to OPT
    ### Questions: does SM-GPTQ look like normal GPTQ for out_proj?
    def add_batch(self, inp, out, name='in_proj'):
        '''
        print(torch.cuda.get_device_name())
        print("Total VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)

        print()
        print(inp.shape)
        print(self.layer.weight.shape)
        print(out.shape)
        print()
        '''
        self.name = name


        # Out-channel 0 jacobian

        # raise ValueError
        t0_hessian = time.time()
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0] 
        if isinstance(self.layer, (nn.Linear, HadLinear)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t() 
        else:
            raise NotImplementedError("Only HadLinear and nn.Linear is supported for now")
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        print(f"Time taken to compute Hessian for {self.name}: {time.time() - t0_hessian} seconds")
    
    def fasterquant(self, group_size=128, percdamp=.01, w_bits=4, dtype=torch.float32):
        t0_fasterquant = time.time()
        bits = w_bits # 4-bit quantization
        W = self.layer.weight.data.clone().to(dtype)
        device = W.device
        
        # preprocess H
        H = self.H.clone().to(torch.float32)
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0 
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.in_dim, device=self.dev)
        H[diag, diag] += damp 
        # cholesky must be torch.float32
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True).to(dtype)
        Hinv = H
        
        # init Losses and Q
        assert group_size <= self.in_dim
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        n_groups = math.ceil(self.in_dim / group_size)
        group_scale = torch.zeros(n_groups, self.out_dim, dtype=torch.float32, device=device)   # QQQ requires [n_groups, out_dim]
        for i1 in range(0, self.in_dim, group_size):
            gidx = i1 // group_size
            i2 = min(i1 + group_size, self.in_dim)
            count = i2 - i1
            # get weight group
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            # Get per-group scale and zero
            per_group_scale = get_per_channel_scale(W1, num_bits=bits)
            group_scale[gidx] = per_group_scale.squeeze()
            for i in range(count):
                w = W1[:, i].clone() # [Dout]
                d = Hinv1[i, i]
                q = quant(w.unsqueeze(1), per_group_scale, num_bits=bits).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                err1 = (w - q) / d       
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        torch.cuda.synchronize()

        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype) # fake quantized weight
        self.layer.weight.data = Q.contiguous() # [Dout, Din]
        self.layer.apply_gptq = True
        self.layer.bits = bits
        self.layer.group_size = group_size
        self.layer.group_scale = group_scale    # QQQ requires [n_groups, out_dim]

        del Losses
        del H
        del W
        torch.cuda.empty_cache()
        print(f"Time taken to fasterquant for {self.name}: {time.time() - t0_fasterquant} seconds")

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
        gc.collect()


# implement GPTQ with a different J matrix
class SMGPTQ():
    def __init__(self, layer, idx=0):
        self.idx = idx

        # print(self.idx)

        # print(layer)
        self.layer = layer
        self.dev = self.layer.in_proj.weight.device
        # HY: To save memory, we use float16 to compute distances for non-uniform quantization
        #     see datatype_utils.py for def fake_quantize_with_type 
        self.gptq_dtype = torch.float16
        
        # print(self.layer.in_proj.weight.shape)
        # print(self.layer.out_proj.weight.shape)

        self.out_dim = self.layer.out_proj.weight.shape[1] # channels after mixer, before out_proj
        self.in_dim = self.layer.in_proj.weight.shape[0] # channels after in_proj
        # print(self.out_dim)
        # print(self.in_dim)

        print("headdim: ", self.layer.headdim)
        print("ngroups: ", self.layer.ngroups)
        print("d_state: ", self.layer.d_state)
        print("d_inner: ", self.layer.d_inner)
        print("nheads: ", self.layer.nheads)
        print("expansion factor: ", self.layer.expand)
        print("d_ssm: ", self.layer.d_ssm)
        print("d_model: ", self.layer.d_model)
        print()

    # How do we take derivative of out with respect to the WEIGHT of the layer (self.layer.weight)?
    # Store this derivative in J
    # That is, J = d(out)/d(self.layer.weight)
    def capture_inputs(self, inp, out):
        # print(inp.shape)
        # print(out.shape)
        # print(self.layer)
        # print()
        self.inputs = inp

    def capture_outputs(self, x, z, out):
        print(x.shape)
        print(z.shape)
        print(out.shape)
        print(self.layer)
        # print(self.layer.__dict__)
        # print()
        # sum over batch dimension
        inp = x * torch.nn.functional.silu(z)
        self.pre_outputs = inp.sum(dim=0).unsqueeze(0)

        # Cut to half the size
        self.pre_outputs = self.pre_outputs[:, :self.pre_outputs.shape[1]//2]
    
    def compute_jacobian_sampled(self):
        if self.idx != 13:
        # if self.idx != 23:
            print("skipping layer\n")
            return
        # print(self.inputs.shape)
        # print(self.pre_outputs.shape)

        # Split fused W into individual weight matrices as per mamba2
        W_z = self.layer.in_proj.weight[:self.layer.d_inner, :]
        W_x = self.layer.in_proj.weight[self.layer.d_inner:2 * self.layer.d_inner, :]
        W_b = self.layer.in_proj.weight[2 * self.layer.d_inner:2 * self.layer.d_inner + self.layer.ngroups * self.layer.d_state, :]
        W_c = self.layer.in_proj.weight[2 * self.layer.d_inner + self.layer.ngroups * self.layer.d_state:2 * self.layer.d_inner + 2 * self.layer.ngroups * self.layer.d_state, :]
        W_t = self.layer.in_proj.weight[-self.layer.nheads:, :]

        # print(W_z.shape, W_x.shape, W_b.shape, W_c.shape, W_t.shape)
        # print()

        sample_channel_count = 16
        sample_temporal_count = self.pre_outputs.shape[1]

        sample_param_count = 4
        sample_input_count = 4

        # Randomly sample all dimensions
        rand_channel = np.arange(sample_channel_count)
        rand_temporal = np.arange(sample_temporal_count)

        # rand_channel += 64

        rand_param = np.arange(sample_param_count) + 8192
        rand_input = np.arange(sample_input_count)

        # Code to compute entire Jacobian for sampled outputs.

        # print(self.pre_outputs.shape)
        #print(self.pre_outputs[0, :sample_temporal_count, :sample_channel_count])

        rand_channel = np.array([0, 4, 8, 12, 16, 20, 24, 28, 35, 39, 43, 47, 51, 55, 59, 63])
        rand_channel = np.array([0, 9, 18, 27, 36, 45, 54, 63])
        heads = self.layer.headdim*np.arange(self.layer.nheads)
        print(heads)
        rand_channel = np.add.outer(heads, rand_channel).transpose().flatten()
        print(rand_channel)
        
        t0 = time.time()
        z_shards = {}
        x_shards = {}
        b_shards = {}
        c_shards = {}
        J_z = []
        J_x = []
        J_b = []
        J_c = []
        J = []
        t0 = time.time()
        for i in range(len(rand_channel)):
            # iterate over L dimension
            z_channel = rand_channel[i]
            x_channel = rand_channel[i] + W_z.shape[0]

            # Create lists for query-key channels. Hardcode expansion factor of 2 for now.
            expansion_factor = self.layer.d_state/self.layer.headdim
            head_location = rand_channel[i] % (self.layer.headdim)
            b_channel_start = W_z.shape[0] + W_x.shape[0] + head_location
            c_channel_start = W_z.shape[0] + W_x.shape[0] + W_b.shape[0] + head_location
            b_channel = np.arange(b_channel_start, b_channel_start + expansion_factor)
            c_channel = np.arange(c_channel_start, c_channel_start + expansion_factor)

            print(rand_channel[i])
            print(z_channel)
            print(x_channel)
            print(b_channel)
            print(c_channel)
            # sprint()

            J_z.append([])
            J_x.append([])
            J_b.append([])
            J_c.append([])

            t0 = time.time()

            for l in range(len(rand_temporal)):
                # tensor_key = f"L{self.idx}_C{rand_channel[i]}_T{rand_temporal[l]}"
                # print(tensor_key)

                gW, = torch.autograd.grad(self.pre_outputs[0, rand_temporal[l], rand_channel[i]], self.layer.in_proj.weight, retain_graph=True)
                # shard_data[tensor_key] = gW.detach().cpu()

                # print(gW.shape)
                # J.append(gW.detach().cpu()[rand_param, :][:, rand_input])
                relevant_activations = gW.nonzero()[:, 0].unique()

                J_z[-1].append(gW.detach().cpu()[z_channel, :])
                J_x[-1].append(gW.detach().cpu()[x_channel, :])
                J_b[-1].append(gW.detach().cpu()[b_channel, :])
                J_c[-1].append(gW.detach().cpu()[c_channel, :])
                '''
                b = False
                if relevant_activations.shape[0] != 259:
                    b = True
                    print(f"Channel {rand_channel[i]}, Temporal {rand_temporal[l]}")
                    print("Size mismatch: ", relevant_activations.shape[0], "!= 259")
                first_element = relevant_activations[0].item()
                expected_first_element = rand_channel[i]
                if first_element != expected_first_element:
                    if not b:
                        print(f"Channel {rand_channel[i]}, Temporal {rand_temporal[l]}")
                    print(f"First element mismatch: {first_element} != {expected_first_element}")
                    b = True
                second_element = relevant_activations[1].item()
                expected_second_element = 4096 + rand_channel[i]
                if second_element != expected_second_element:
                    if not b:
                        print(f"Channel {rand_channel[i]}, Temporal {rand_temporal[l]}")
                    print(f"First element mismatch: {second_element} != {expected_second_element}")
                    b = True
                last_element = relevant_activations[-1].item()
                expected_last_element = 8448 + (rand_channel[i] // 64)
                if last_element != expected_last_element:
                    if not b:
                        print(f"Channel {rand_channel[i]}, Temporal {rand_temporal[l]}")
                    print(f"Last element mismatch: {last_element} != {expected_last_element}")
                    b = True
                third_element = relevant_activations[2].item()
                expected_third_element_minimum = 8192
                if third_element < expected_third_element_minimum:
                    if not b:
                        print(f"Channel {rand_channel[i]}, Temporal {rand_temporal[l]}")
                    print(f"Second element mismatch: {third_element} < {expected_third_element_minimum}")
                    b = True
                if b:
                    print()
                '''

                
            # print("Channel {rand_channel[i]} time: ", time.time() - t0)
            J_z[-1] = torch.stack(J_z[-1])
            J_x[-1] = torch.stack(J_x[-1])
            J_b[-1] = torch.stack(J_b[-1])
            J_c[-1] = torch.stack(J_c[-1])
            # print(J_z[-1].shape)
            # print(J_x[-1].shape)
            # print(J_b[-1].shape)
            # print(J_c[-1].shape)
            print(time.time() - t0)
            print()

            z_shards[str(rand_channel[i])] = J_z[-1]
            x_shards[str(rand_channel[i])] = J_x[-1]
            b_shards[str(rand_channel[i])] = J_b[-1]
            c_shards[str(rand_channel[i])] = J_c[-1]

            '''
            if rand_channel[i] > self.layer.nheads * 2:
                print("z_shards:")
                for key, value in z_shards.items():
                    print(key, value.shape)
                print()
                print("x_shards:")
                for key, value in x_shards.items():
                    print(key, value.shape)
                print()
                print("b_shards:")
                for key, value in b_shards.items():
                    print(key, value.shape)
                print()
                print("c_shards:")
                for key, value in c_shards.items():
                    print(key, value.shape)
                print()
                raise ValueError("Stop!!")
            '''
        
        path = f'../jacobian_block_samples/130m/layer{self.idx}'
        os.makedirs(path, exist_ok=True)
        zpath = os.path.join(path, f"z.safetensors")
        save_file(z_shards, zpath)
        xpath = os.path.join(path, f"x.safetensors")
        save_file(x_shards, xpath)
        bpath = os.path.join(path, f"b.safetensors")
        save_file(b_shards, bpath)
        cpath = os.path.join(path, f"c.safetensors")
        save_file(c_shards, cpath)

        raise ValueError(f"Saved tensors in layer {self.idx}!")
        
        J = torch.stack(J)
        print(J.shape)

        #J_filt = J[:, rand_param, :][:, :, rand_input]
        #J_filt = J_filt.reshape(J_filt.shape[0], -1)
        #print(J_filt.shape)

        J = J.reshape(J.shape[0], -1)

        print(J)

        H_raw = J.t() @ J
        print(H_raw.shape)
        print(H_raw)


        # Build probes for JVP -- they should be dotted with outputs, so dimension is same as self.pre_outputs
        max_probes = 1024
        print(self.pre_outputs.shape)
        probes = torch.randn(max_probes, self.pre_outputs.shape[1], self.pre_outputs.shape[2], device=self.dev)
        probes_rademacher = (torch.randint(0, 2, (max_probes, self.pre_outputs.shape[1], self.pre_outputs.shape[2]), device=self.dev)) * 2 - 1

        # probes = probes_rademacher

        # dot product of probes and J_filt, batched across probes
        JVP_scalars = (probes.reshape(max_probes, -1) * self.pre_outputs.reshape(-1)).sum(dim=1)
        print(JVP_scalars.shape)

        t0 = time.time()
        # get gradients of JVP_scalars with respect to self.layer.in_proj.weight
        Gs = []
        for i in range(max_probes):
            gi_full, = torch.autograd.grad(
                outputs=JVP_scalars[i], inputs=self.layer.in_proj.weight,
                retain_graph=True,
                create_graph=False,
            )
            Gs.append(gi_full[rand_param, :][:, rand_input])

        G = torch.stack(Gs, dim=0)
        print(G.shape) 

        G = G.reshape(G.shape[0], -1).double()
        print(time.time() - t0)

        '''
        H_individual = torch.bmm(G.transpose(1, 2), G)
        print(H_individual.shape)
        '''

        print(G[:8].t())
        print(G[:8])

        H_est_8 = (G[:8].t()/8 @ G[:8])
        H_est_16 = (G[:16].t()/16 @ G[:16])
        H_est_32 = (G[:32].t()/32 @ G[:32])
        H_est_64 = (G[:64].t()/64 @ G[:64])
        H_est_128 = (G[:128].t()/128 @ G[:128])
        H_est_256 = (G[:256].t()/256 @ G[:256])
        H_est_512 = (G[:512].t()/512 @ G[:512])
        H_est_full = (G.t()/max_probes @ G)

        H_estimate_list = [H_est_8, H_est_16, H_est_32, H_est_64, H_est_128, H_est_256, H_est_512, H_est_full, H_raw]

        fro_norm_error_percents = []
        diagonal_error_percents = []
        fro_norm_error_percents_ref = []
        diagonal_error_percents_ref = []

        probe_counts = [8, 16, 32, 64, 128, 256, 512, max_probes]
        log_probe_counts = [math.log2(x) for x in probe_counts]

        '''
        for H in H_estimate_list: 
            # print(H)
            print(((H.to(H_raw.device) - H_raw).abs()).mean()) 
        '''   

        for i, H in enumerate(H_estimate_list):  
            if i == len(H_estimate_list) - 1:
                break
            samples_used = 2 ** (i+3)
            if i == 0:
                print(f"Using {samples_used} samples")
                # print(H)
            else:
                print(f"Using {samples_used} samples")
                # print(H)
                error = (H.to(H_estimate_list[i-1].device) - H_estimate_list[i-1]).abs()
                error_ref = (H.to(H_estimate_list[-2].device) - H_estimate_list[-2]).abs()
                fro_norm_prior = torch.linalg.norm(H_estimate_list[i-1])
                fro_norm_current = torch.linalg.norm(H)
                fro_norm_error = torch.linalg.norm(error)
                fro_norm_error_ref = torch.linalg.norm(error_ref)
                fro_norm_ref = torch.linalg.norm(H_estimate_list[-2])

                diagonal_error_norm = torch.linalg.norm(error.diag())
                diagonal_prior_norm = torch.linalg.norm(H_estimate_list[i-1].diag())
                diagonal_current_norm = torch.linalg.norm(H.diag())
                diagonal_ref_norm = torch.linalg.norm(H_estimate_list[-2].diag())
                diagonal_error_norm_ref = torch.linalg.norm(error_ref.diag())


                print(100*fro_norm_error/fro_norm_prior)
                print(100*diagonal_error_norm/diagonal_prior_norm)
                print(100*fro_norm_error_ref/fro_norm_ref)
                print(100*diagonal_error_norm_ref/diagonal_ref_norm)

                fro_norm_error_percents.append((100*fro_norm_error/fro_norm_prior).item())
                diagonal_error_percents.append((100*diagonal_error_norm/diagonal_prior_norm).item())
                fro_norm_error_percents_ref.append((100*fro_norm_error_ref/fro_norm_ref).item())
                diagonal_error_percents_ref.append((100*diagonal_error_norm_ref/diagonal_ref_norm).item())
                # print(100*(((H.to(H_estimate_list[i-1].device) - H_estimate_list[i-1]).abs())/H_estimate_list[i-1]))
                # print(100*(((H.to(H_estimate_list[i-1].device) - H_estimate_list[i-1]).abs())/H_estimate_list[i-1]).median())
            print()
            print()    

        # pass

        # Create plots
        plt.plot(log_probe_counts[1:], fro_norm_error_percents, label='||H(P) - H(P/2)||/||H(P/2)||')
        plt.plot(log_probe_counts[1:], diagonal_error_percents, label='||Diag(H(P) - H(P/2))||/||Diag(H(P/2))||')
        plt.plot(log_probe_counts[1:], fro_norm_error_percents_ref, label='||H(P) - H(1024)||/||H(1024)||')
        plt.plot(log_probe_counts[1:], diagonal_error_percents_ref, label='||Diag(H(P) - H(P/2))||/||Diag(H(1024))||')
        plt.xlabel('Log2(Probe Count P)')
        plt.ylabel('Error Percentage')
        plt.title('Error vs. Log Probe Count P')
        plt.legend()
        plt.show()
        plt.savefig(f"jacobian_jvp_error/Gaussian_1.3b_Layer_{self.idx}_Log.png")
        # clear plot
        plt.clf()

        plt.plot(probe_counts[1:], fro_norm_error_percents, label='||H(P) - H(P/2)||/||H(P/2)||')
        plt.plot(probe_counts[1:], diagonal_error_percents, label='||Diag(H(P) - H(P/2))||/||Diag(H(P/2))||')
        plt.plot(probe_counts[1:], fro_norm_error_percents_ref, label='||H(P) - H(1024)||/||H(1024)||')
        plt.plot(probe_counts[1:], diagonal_error_percents_ref, label='||Diag(H(P) - H(P/2))||/||Diag(H(1024))||')
        plt.xlabel('Probe Count P')
        plt.ylabel('Error Percentage')
        plt.title('Error vs. Probe Count P')
        plt.legend()
        plt.show()
        plt.savefig(f"jacobian_jvp_error/Gaussian_130m_Layer_{self.idx}_Identity.png")


        raise ValueError(f"Layer {self.idx} over!")
        
        '''
        file_path = os.path.join("jacobian_samples", f"Layer_{self.idx}.safetensors")
        save_file(shard_data, file_path)
        print(f"Saved shard data for layer {self.idx} to {file_path}")
        print(f"Used temporal dimensions: {rand_temporal}")
        print(f"Used channel dimensions: {rand_channel}")
        print()
        '''
        raise ValueError(f"Layer {self.idx} over!")
        del shard_data
    
    def read_shards(self):
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
        final_shard = {}
        for key, value in tensor_shard.items():
            final_shard[int(key)] = value / self.inputs.shape[0] # divide by batch size to get average instead of sum
            final_shard[int(key)] = final_shard[int(key)]
        return final_shard
    
    def convert_to_hessian_slow(self, tensor_shard):
        hessian_shard = {}
        for key, value in tensor_shard.items():
            # print(value)
            if len(value.shape) > 2:
                value = value.reshape(value.shape[0], -1)
            # print(value)
            hessian_shard[key] = value.t() @ value
            print(key, hessian_shard[key].shape)
        print()
        return hessian_shard

    def convert_to_hessian(self, tensor_shard, tensor_name='None'):
        hessian_shard = {}

        def flatten_shard(tensor_shard):
            return torch.stack(list(tensor_shard.values()), dim=0)

        def convert_to_hessian_1d(tensor):
            return tensor.t() @ tensor
        
        flattened_shard = flatten_shard(tensor_shard)

        if len(flattened_shard.shape) > 3:
            flattened_shard = flattened_shard.reshape(flattened_shard.shape[0], flattened_shard.shape[1], -1)

        hessian_tensor = flattened_shard.transpose(1, 2).bmm(flattened_shard)
        
        # torch.func.vmap(convert_to_hessian_helper)(torch.tensor(list(tensor_shard.keys())))
        for i, key in enumerate(tensor_shard.keys()):
            print(i, key)
            print(self.layer.d_state * self.layer.ngroups)
            print(key % (self.layer.d_state * self.layer.ngroups))
            print(hessian_shard.keys())
            print()
            if (tensor_name == 'b' or tensor_name == 'c') and key % ((self.layer.d_state * self.layer.ngroups)/2) in hessian_shard.keys():
                hessian_shard[int(key % ((self.layer.d_state * self.layer.ngroups)/2))] += hessian_tensor[i, :, :]
            else:
                if (tensor_name == 'b' or tensor_name == 'c'):
                    hessian_shard[int(key % ((self.layer.d_state * self.layer.ngroups)/2))] = hessian_tensor[i, :, :]
                else:
                    hessian_shard[key] = hessian_tensor[i, :, :]
        print(hessian_shard.keys())
        if max(hessian_shard.keys()) > 3372:
            raise ValueError("Hessian keys are greater than d_inner!")
        return hessian_shard
    
    def jvp_gradients_and_hessian_slow(self, tensor_shard, max_probes=1024, step_size=8, tensor_name='z'):
        # implement JVPs
        # JVP gradients are the gradients of the JVP with respect to the tensor shard
        # JVP is the Jacobian-Vector Product
        # JVP = J * v
        # J is the Jacobian of the tensor shard
        # v is the vector
        # JVP gradients are the gradients of the JVP with respect to the tensor shard
        # JVP gradients are the gradients of the JVP with respect to the tensor shard
        # operate naively
        channels = torch.tensor(list(tensor_shard.keys()))
        print(channels)

        out_channels = channels
        
        print(out_channels)

        t0 = time.time()
        times_list = []

        probes = torch.randn(max_probes, self.pre_outputs.shape[1], channels.shape[0]).to(self.pre_outputs.device)

        JVP_scalars = (probes * self.pre_outputs[:, :, channels]/self.inputs.shape[0]).sum(dim=1).sum(dim=1) # divide by batch size to get average instead of sum

        Gs = []

        if tensor_name == 'x':
            channels = channels + self.layer.d_inner
        elif tensor_name == 'b':
            # channels = 2 * channels
            # channels = torch.cat([channels, channels + 1], dim=0)
            channels = channels + 2*self.layer.d_inner
        elif tensor_name == 'c':
            # channels = 2 * channels
            # channels = torch.cat([channels, channels + 1], dim=0)
            channels = channels + 2*self.layer.d_inner + self.layer.d_state * self.layer.ngroups

        print(channels)

        if tensor_name == 'b' or tensor_name == 'c':
            for c in channels:
                print(c)
            JVP_scalars = []
            for o in out_channels:
                print(o)
                all_heads = torch.arange(self.layer.nheads) * (self.layer.headdim)
                out_channels_this_head = all_heads + o
                print(out_channels_this_head)
                probes = torch.randn(max_probes, self.pre_outputs.shape[1], out_channels_this_head.shape[0]).to(self.pre_outputs.device)
                JVP_scalars.append((probes * self.pre_outputs[:, :, out_channels_this_head]/self.inputs.shape[0]).sum(dim=1).sum(dim=1))
            print(len(JVP_scalars))
            print(JVP_scalars[0].shape)
            print(JVP_scalars)
            rel_outs = [[c, c+1] for c in channels]
            print(rel_outs)
            common_time = time.time() - t0
            print(common_time)
            for i in range(max_probes//step_size):
                times_list.append(common_time)
            for JVP_channels, derivative_channels in zip(JVP_scalars, rel_outs):
                l = []
                t0 = time.time()
                for i in range(max_probes):
                    gi_full, = torch.autograd.grad(
                        outputs=JVP_channels[i], inputs=self.layer.in_proj.weight,
                        retain_graph=True,
                        create_graph=False,
                    )
                    # print(gi_full.shape)
                    gi = gi_full[derivative_channels, :]
                    # print(gi.shape)
                    l.append(gi)
                    if (i + 1) % step_size == 0:
                        times_list[int(i/step_size)] += time.time() - t0 
                Gs.append(torch.stack(l, dim=0).reshape(max_probes, -1))
                print(Gs[-1].shape)
            print()
        else:
            common_time = time.time() - t0
            t0 = time.time()
            for i in range(max_probes):
                gi_full, = torch.autograd.grad(
                    outputs=JVP_scalars[i], inputs=self.layer.in_proj.weight,
                    retain_graph=True,
                    create_graph=False,
                )
                gi = gi_full[channels]
                Gs.append(gi)
                if (i + 1) % step_size == 0:
                    times_list.append(time.time() - t0 + common_time)

        G = torch.stack(Gs, dim=0)
        if tensor_name == 'b' or tensor_name == 'c':
            G = G.transpose(0, 1)
        print(G.shape)

        G = G.float()

        H_dict = {}

        for i in range(step_size, max_probes + 1, step_size):
            # print(i)
            # print(step_size)
            t0 = time.time()
            H_dict[i] = (G[:i].permute(1, 2, 0) / i).bmm(G[:i].permute(1, 0, 2))
            times_list[int(i/step_size) - 1] += time.time() - t0
        np.save(f"jacobian_jvp_error/times/{tensor_name}/{self.idx}.npy", np.array(times_list))
        print(times_list)
        print(len(times_list))
        return H_dict


    
    def jvp_gradients_fast(self, tensor_shard, max_probes=1024):
        # implement JVPs
        # JVP gradients are the gradients of the JVP with respect to the tensor shard
        # JVP is the Jacobian-Vector Product
        # JVP = J * v
        # J is the Jacobian of the tensor shard
        # v is the vector
        # JVP gradients are the gradients of the JVP with respect to the tensor shard
        # JVP gradients are the gradients of the JVP with respect to the tensor shard
        # use torch.func.jvp and torch.func.vmap

        params = dict(self.layer.named_parameters())
        buffers = dict(self.layer.named_buffers())

        W = params['in_proj.weight']

        params_const = {k: v for k, v in params.items() if k != 'in_proj.weight'}

        def funcforward(W, x):
            p = dict(params_const)
            p['in_proj.weight'] = W
            return torch.func.functional_call(self.layer, (p, buffers), (x,))

        self.layer.use_mem_eff_path = True
        print(self.layer.use_mem_eff_path)
        
        y, pullback = torch.func.vjp(funcforward, W, self.inputs)

        print('in function!!')
        print(y.shape)

        def one_vjp(r, channels=None):
            dW, _ = pullback(r)
            if channels == None:
                return dW
            else:
                return dW[channels]

        cotangent = torch.randn_like(y)
        JVP_scalars, _ = pullback(cotangent)
        print(JVP_scalars.shape)

        cotangent_matrix = torch.randn(max_probes, *y.shape)
        JVP_scalars_matrix = torch.func.vmap(one_vjp)(cotangent_matrix)
        print(JVP_scalars_matrix.shape)

        raise ValueError("Stop!!")

    def jvp_hessian_estimations(self, gradients, num_probes):
        pass

    def plot_hessian_estimations(self, true_hessians, H_dict, gptq_val=None, tensor_name=None):
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
            true_hessian_block_1 = true_hessian[:true_hessian.shape[0]//2, :true_hessian.shape[1]//2]
            true_hessian_block_2 = true_hessian[true_hessian.shape[0]//2:, true_hessian.shape[1]//2:]
            true_hessians_bd = torch.block_diag(true_hessian_block_1, true_hessian_block_2)
            error_channel = []
            error_channel_bd = []
            error_bar_channel = []
            error_bar_channel_bd = []
            for k2 in H_dict.keys():
                estimated_hessian = H_dict[k2][i, :, :]
                estimated_hessian_bar = H_dict[k2].mean(dim=0)

                estimated_hessian_block_1 = estimated_hessian[:estimated_hessian.shape[0]//2, :estimated_hessian.shape[1]//2]
                estimated_hessian_block_2 = estimated_hessian[estimated_hessian.shape[0]//2:, estimated_hessian.shape[1]//2:]
                estimated_hessian_bd = torch.block_diag(estimated_hessian_block_1, estimated_hessian_block_2)

                estimated_hessian_block_1_bar = estimated_hessian_bar[:estimated_hessian_bar.shape[0]//2, :estimated_hessian_bar.shape[1]//2]
                estimated_hessian_block_2_bar = estimated_hessian_bar[estimated_hessian_bar.shape[0]//2:, estimated_hessian_bar.shape[1]//2:]
                estimated_hessian_bd_bar = torch.block_diag(estimated_hessian_block_1_bar, estimated_hessian_block_2_bar)

                true_hessian = true_hessian.to(estimated_hessian.device)
                true_hessians_bd = true_hessians_bd.to(estimated_hessian_bd.device)
                # print(true_hessian - estimated_hessian)
                # print(torch.linalg.norm(true_hessian - estimated_hessian))
                # print(torch.linalg.norm(true_hessian))
                if torch.isinf(estimated_hessian).any():
                    print("Inf in estimated hessian!")
                if torch.isinf(true_hessian).any():
                    print("Inf in true hessian!")
                    continue
                if torch.linalg.norm(true_hessian).isinf():
                    print(torch.linalg.norm(true_hessian.double()))
                    print(torch.linalg.norm(true_hessian.double()/10000))
                    print()
                # error_item = torch.linalg.norm(true_hessian - estimated_hessian)/torch.linalg.norm(true_hessian.double())

                error_item = torch.nn.functional.cosine_similarity(true_hessian.flatten().double(), estimated_hessian.flatten().double(), dim=0)
                if torch.isnan(error_item):
                    # error_item = torch.linalg.norm(true_hessian/10000 - estimated_hessian/10000)/torch.linalg.norm(true_hessian.double()/10000)
                    error_item = torch.nn.functional.cosine_similarity(true_hessian.flatten().double()/10000, estimated_hessian.flatten().double()/10000, dim=0)
                error_channel.append(error_item)

                error_item_bd = torch.nn.functional.cosine_similarity(true_hessians_bd.flatten().double(), estimated_hessian_bd.flatten().double(), dim=0)
                if torch.isnan(error_item_bd):
                    error_item_bd = torch.nn.functional.cosine_similarity(true_hessians_bd.flatten().double()/10000, estimated_hessian_bd.flatten().double()/10000, dim=0)
                error_channel_bd.append(error_item_bd)

                error_item_bar = torch.nn.functional.cosine_similarity(true_hessian.flatten().double(), estimated_hessian_bar.flatten().double(), dim=0)
                if torch.isnan(error_item_bar):
                    error_item_bar = torch.nn.functional.cosine_similarity(true_hessian.flatten().double()/10000, estimated_hessian_bar.flatten().double()/10000, dim=0)
                error_bar_channel.append(error_item_bar)

                error_item_bar_bd = torch.nn.functional.cosine_similarity(true_hessians_bd.flatten().double(), estimated_hessian_bd_bar.flatten().double(), dim=0)
                if torch.isnan(error_item_bar_bd):
                    error_item_bar_bd = torch.nn.functional.cosine_similarity(true_hessians_bd.flatten().double()/10000, estimated_hessian_bd_bar.flatten().double()/10000, dim=0)
                error_bar_channel_bd.append(error_item_bar_bd)

            if len(error_channel) == 0:
                continue
            if len(error_channel_bd) == 0:
                continue
            if len(error_bar_channel) == 0:
                continue
            if len(error_bar_channel_bd) == 0:
                continue
            if gptq_val is not None:
                # gptq_error_item = torch.linalg.norm((gptq_val/10000 - true_hessian/10000).double())/torch.linalg.norm(true_hessian.double()/10000)
                if torch.isinf(gptq_val).any():
                    print("Inf in gptq_val!")
                gptq_error_item = torch.nn.functional.cosine_similarity(gptq_val.flatten().double()/10000, true_hessian.flatten().double()/10000, dim=0)
                gptq_errors.append(gptq_error_item.item())
                gptq_error_item_bd = torch.nn.functional.cosine_similarity(gptq_val.flatten().double()/10000, true_hessians_bd.flatten().double()/10000, dim=0)
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
        
        print(f"="*100)
        errors = torch.stack(errors, dim=0)
        np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_cosine.npy", errors.cpu().numpy())
        errors = errors.mean(dim=0)
        print(errors)

        errors_bd = torch.stack(errors_bd, dim=0)
        np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_cosine_bd.npy", errors_bd.cpu().numpy())
        errors_bd = errors_bd.mean(dim=0)
        print(errors_bd)

        error_bar = torch.stack(error_bar, dim=0)
        np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_cosine_bar.npy", error_bar.cpu().numpy())
        error_bar = error_bar.mean(dim=0)
        print(error_bar)

        error_bar_bd = torch.stack(error_bar_bd, dim=0)
        np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_cosine_bar_bd.npy", error_bar_bd.cpu().numpy())
        error_bar_bd = error_bar_bd.mean(dim=0)
        print(error_bar_bd)

        if gptq_val is not None:
            print(gptq_errors)
            print(gptq_errors_bd)
            print(len(gptq_errors))
            np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_gptq_cosine.npy", np.array(gptq_errors))
            np.save(f"jacobian_jvp_error/raw_data/{tensor_name}/{self.idx}_gptq_cosine_bd.npy", np.array(gptq_errors_bd))
        plt.plot(probe_counts, errors.cpu().numpy())
        plt.xlabel('Probe Count')
        plt.ylabel('Error')
        title = f'Error vs. Probe Count in Layer {self.idx}'
        if tensor_name is not None:
            title += f' for {tensor_name}'
        plt.title(title)
        if gptq_val is not None:
            # print horizontal line for gptq_error_mean
            gptq_error_mean = sum(gptq_errors)/len(gptq_errors)
            print(gptq_error_mean)
            gptq_error_mean_bd = sum(gptq_errors_bd)/len(gptq_errors_bd)
            print(gptq_error_mean_bd)
            # plt.axhline(gptq_error_mean, color='red', linestyle='--')
        plt.show()
        plt.savefig(f"jacobian_jvp_error/Gaussian_130m_Layer_{self.idx}_{tensor_name}_Error.png")
        plt.clf()


    def read_and_compare(self):
        if self.idx not in [0, 1, 4, 7, 10, 13, 16, 19, 22, 23]:
        # if self.idx != 0:
            print("skipping layer\n")
            return
        # print(self.inputs.shape)
        # print(self.pre_outputs.shape)

        print(self.inputs.shape)
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
        W_b = self.layer.in_proj.weight[2 * self.layer.d_inner:2 * self.layer.d_inner + self.layer.ngroups * self.layer.d_state, :]
        W_c = self.layer.in_proj.weight[2 * self.layer.d_inner + self.layer.ngroups * self.layer.d_state:2 * self.layer.d_inner + 2 * self.layer.ngroups * self.layer.d_state, :]
        W_t = self.layer.in_proj.weight[-self.layer.nheads:, :]

        z_shards, x_shards, b_shards, c_shards = self.read_shards()
        z_shards = self.int_keys(z_shards)


        
        z_hessian = self.convert_to_hessian(z_shards)
        z_jvp_gradients = self.jvp_gradients_and_hessian_slow(z_hessian, max_probes=1024, step_size=8, tensor_name='z')

        print("True Hessian:")
        print(z_hessian[0])
        print()

        print("Estimated Hessians:")
        for k in z_jvp_gradients.keys():
            if k not in [8, 16, 32, 64, 128, 256, 512, 1024]:
                continue
            print(k)
            # print(z_jvp_gradients[k].shape)
            print(z_jvp_gradients[k][0, :, :])
            print()
        
        self.plot_hessian_estimations(z_hessian, z_jvp_gradients, gptq_val=gptq_aggregated, tensor_name='z')
        
        del z_shards, z_hessian, z_jvp_gradients
        self.free()
        
        x_shards = self.int_keys(x_shards)
        x_hessian = self.convert_to_hessian(x_shards)
        x_jvp_gradients = self.jvp_gradients_and_hessian_slow(x_hessian, max_probes=1024, step_size=8, tensor_name='x')
        
        print("True Hessian:")
        print(x_hessian[0])
        print()

        print("Estimated Hessians:")
        for k in x_jvp_gradients.keys():
            if k not in [8, 16, 32, 64, 128, 256, 512, 1024]:
                continue
            print(k)
            # print(z_jvp_gradients[k].shape)
            print(x_jvp_gradients[k][0, :, :])
            print()
        
        self.plot_hessian_estimations(x_hessian, x_jvp_gradients, gptq_val=gptq_aggregated, tensor_name='x')

        del x_shards, x_hessian, x_jvp_gradients
        self.free()
        
        b_shards = self.int_keys(b_shards)
        b_hessian = self.convert_to_hessian(b_shards, tensor_name='b')
        b_jvp_gradients = self.jvp_gradients_and_hessian_slow(b_hessian, max_probes=1024, step_size=8, tensor_name='b')

        print("True Hessian:")
        print(b_hessian[0])
        print()

        print("Estimated Hessians:")
        for k in b_jvp_gradients.keys():
            if k not in [8, 16, 32, 64, 128, 256, 512, 1024]:
                continue
            print(k)
            # print(z_jvp_gradients[k].shape)
            print(b_jvp_gradients[k][0, :, :])
            print()
        
        self.plot_hessian_estimations(b_hessian, b_jvp_gradients, gptq_val=torch.block_diag(gptq_aggregated, gptq_aggregated), tensor_name='b')

        del b_shards, b_hessian, b_jvp_gradients
        self.free()
        
        c_shards = self.int_keys(c_shards)
        c_hessian = self.convert_to_hessian(c_shards, tensor_name='c')
        c_jvp_gradients = self.jvp_gradients_and_hessian_slow(c_hessian, max_probes=1024, step_size=8, tensor_name='c')

        print("True Hessian:")
        print(c_hessian[0])
        print()

        print("Estimated Hessians:")
        for k in c_jvp_gradients.keys():
            if k not in [8, 16, 32, 64, 128, 256, 512, 1024]:
                continue
            print(k)
            # print(z_jvp_gradients[k].shape)
            print(c_jvp_gradients[k][0, :, :])
            print()
        
        self.plot_hessian_estimations(c_hessian, c_jvp_gradients, gptq_val=torch.block_diag(gptq_aggregated, gptq_aggregated), tensor_name='c')

        del c_shards, c_hessian, c_jvp_gradients
        self.free()

        return

        raise ValueError("Stop!!")
        
        


        raise ValueError("Stop!!")


        x_hessian = self.convert_to_hessian(x_shards)
        b_hessian = self.convert_to_hessian(b_shards)
        c_hessian = self.convert_to_hessian(c_shards)

        rand_channel = np.array([0, 4, 8, 12, 16, 20, 24, 28, 35, 39, 43, 47, 51, 55, 59, 63])
        rand_channel = np.array([0, 9, 18, 27, 36, 45, 54, 63])
        heads = self.layer.headdim*np.arange(self.layer.nheads)
        rand_channel = np.add.outer(heads, rand_channel).transpose().flatten()

        '''


        # Build probes for JVP -- they should be dotted with outputs, so dimension is same as self.pre_outputs
        max_probes = 1024
        print(self.pre_outputs.shape)
        probes = torch.randn(max_probes, self.pre_outputs.shape[1], self.pre_outputs.shape[2], device=self.dev)
        probes_rademacher = (torch.randint(0, 2, (max_probes, self.pre_outputs.shape[1], self.pre_outputs.shape[2]), device=self.dev)) * 2 - 1

        # probes = probes_rademacher

        # dot product of probes and J_filt, batched across probes
        JVP_scalars = (probes.reshape(max_probes, -1) * self.pre_outputs.reshape(-1)).sum(dim=1)
        print(JVP_scalars.shape)

        t0 = time.time()
        # get gradients of JVP_scalars with respect to self.layer.in_proj.weight
        Gs = []
        for i in range(max_probes):
            gi_full, = torch.autograd.grad(
                outputs=JVP_scalars[i], inputs=self.layer.in_proj.weight,
                retain_graph=True,
                create_graph=False,
            )
            Gs.append(gi_full[rand_param, :][:, rand_input])

        G = torch.stack(Gs, dim=0)
        print(G.shape) 

        G = G.reshape(G.shape[0], -1).double()
        print(time.time() - t0)

        '''
        
        '''
        H_individual = torch.bmm(G.transpose(1, 2), G)
        print(H_individual.shape)
        '''

        '''

        print(G[:8].t())
        print(G[:8])

        H_est_8 = (G[:8].t()/8 @ G[:8])
        H_est_16 = (G[:16].t()/16 @ G[:16])
        H_est_32 = (G[:32].t()/32 @ G[:32])
        H_est_64 = (G[:64].t()/64 @ G[:64])
        H_est_128 = (G[:128].t()/128 @ G[:128])
        H_est_256 = (G[:256].t()/256 @ G[:256])
        H_est_512 = (G[:512].t()/512 @ G[:512])
        H_est_full = (G.t()/max_probes @ G)

        H_estimate_list = [H_est_8, H_est_16, H_est_32, H_est_64, H_est_128, H_est_256, H_est_512, H_est_full, H_raw]

        fro_norm_error_percents = []
        diagonal_error_percents = []
        fro_norm_error_percents_ref = []
        diagonal_error_percents_ref = []

        probe_counts = [8, 16, 32, 64, 128, 256, 512, max_probes]
        log_probe_counts = [math.log2(x) for x in probe_counts]

        '''

        '''
        for H in H_estimate_list: 
            # print(H)
            print(((H.to(H_raw.device) - H_raw).abs()).mean()) 

        '''

        '''   

        for i, H in enumerate(H_estimate_list):  
            if i == len(H_estimate_list) - 1:
                break
            samples_used = 2 ** (i+3)
            if i == 0:
                print(f"Using {samples_used} samples")
                # print(H)
            else:
                print(f"Using {samples_used} samples")
                # print(H)
                error = (H.to(H_estimate_list[i-1].device) - H_estimate_list[i-1]).abs()
                error_ref = (H.to(H_estimate_list[-2].device) - H_estimate_list[-2]).abs()
                fro_norm_prior = torch.linalg.norm(H_estimate_list[i-1])
                fro_norm_current = torch.linalg.norm(H)
                fro_norm_error = torch.linalg.norm(error)
                fro_norm_error_ref = torch.linalg.norm(error_ref)
                fro_norm_ref = torch.linalg.norm(H_estimate_list[-2])

                diagonal_error_norm = torch.linalg.norm(error.diag())
                diagonal_prior_norm = torch.linalg.norm(H_estimate_list[i-1].diag())
                diagonal_current_norm = torch.linalg.norm(H.diag())
                diagonal_ref_norm = torch.linalg.norm(H_estimate_list[-2].diag())
                diagonal_error_norm_ref = torch.linalg.norm(error_ref.diag())


                print(100*fro_norm_error/fro_norm_prior)
                print(100*diagonal_error_norm/diagonal_prior_norm)
                print(100*fro_norm_error_ref/fro_norm_ref)
                print(100*diagonal_error_norm_ref/diagonal_ref_norm)

                fro_norm_error_percents.append((100*fro_norm_error/fro_norm_prior).item())
                diagonal_error_percents.append((100*diagonal_error_norm/diagonal_prior_norm).item())
                fro_norm_error_percents_ref.append((100*fro_norm_error_ref/fro_norm_ref).item())
                diagonal_error_percents_ref.append((100*diagonal_error_norm_ref/diagonal_ref_norm).item())
                # print(100*(((H.to(H_estimate_list[i-1].device) - H_estimate_list[i-1]).abs())/H_estimate_list[i-1]))
                # print(100*(((H.to(H_estimate_list[i-1].device) - H_estimate_list[i-1]).abs())/H_estimate_list[i-1]).median())
            print()
            print()    

        # pass

        # Create plots
        plt.plot(log_probe_counts[1:], fro_norm_error_percents, label='||H(P) - H(P/2)||/||H(P/2)||')
        plt.plot(log_probe_counts[1:], diagonal_error_percents, label='||Diag(H(P) - H(P/2))||/||Diag(H(P/2))||')
        plt.plot(log_probe_counts[1:], fro_norm_error_percents_ref, label='||H(P) - H(1024)||/||H(1024)||')
        plt.plot(log_probe_counts[1:], diagonal_error_percents_ref, label='||Diag(H(P) - H(P/2))||/||Diag(H(1024))||')
        plt.xlabel('Log2(Probe Count P)')
        plt.ylabel('Error Percentage')
        plt.title('Error vs. Log Probe Count P')
        plt.legend()
        plt.show()
        plt.savefig(f"jacobian_jvp_error/Gaussian_1.3b_Layer_{self.idx}_Log.png")
        # clear plot
        plt.clf()

        plt.plot(probe_counts[1:], fro_norm_error_percents, label='||H(P) - H(P/2)||/||H(P/2)||')
        plt.plot(probe_counts[1:], diagonal_error_percents, label='||Diag(H(P) - H(P/2))||/||Diag(H(P/2))||')
        plt.plot(probe_counts[1:], fro_norm_error_percents_ref, label='||H(P) - H(1024)||/||H(1024)||')
        plt.plot(probe_counts[1:], diagonal_error_percents_ref, label='||Diag(H(P) - H(P/2))||/||Diag(H(1024))||')
        plt.xlabel('Probe Count P')
        plt.ylabel('Error Percentage')
        plt.title('Error vs. Probe Count P')
        plt.legend()
        plt.show()
        plt.savefig(f"jacobian_jvp_error/Gaussian_130m_Layer_{self.idx}_Identity.png")


        raise ValueError(f"Layer {self.idx} over!")
        '''
        
        raise ValueError(f"Layer {self.idx} over!")
    
    def stitch_plots(self):
        if self.idx != 0:
            print("skipping layer\n")
        
        layers = [0, 1, 4, 7, 10, 13, 16, 19, 22, 23]


        probe_counts = [8*i for i in range(1, 129)]
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

        for layer in layers:
            if not np.isnan(np.load(f"jacobian_jvp_error/times/z/{layer}.npy")).any():
                z_list.append(np.load(f"jacobian_jvp_error/times/z/{layer}.npy"))
            if not np.isnan(np.load(f"jacobian_jvp_error/times/x/{layer}.npy")).any():
                x_list.append(np.load(f"jacobian_jvp_error/times/x/{layer}.npy"))
            if not np.isnan(np.load(f"jacobian_jvp_error/times/b/{layer}.npy")).any():
                b_list.append(np.load(f"jacobian_jvp_error/times/b/{layer}.npy"))
            if not np.isnan(np.load(f"jacobian_jvp_error/times/c/{layer}.npy")).any():
                c_list.append(np.load(f"jacobian_jvp_error/times/c/{layer}.npy"))

            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/z/{layer}_cosine_bd.npy")).any():
                z_list_bd.append(np.load(f"jacobian_jvp_error/raw_data/z/{layer}_cosine_bd.npy").mean(axis=0))
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/x/{layer}_cosine_bd.npy")).any():
                x_list_bd.append(np.load(f"jacobian_jvp_error/raw_data/x/{layer}_cosine_bd.npy").mean(axis=0))
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/b/{layer}_cosine_bd.npy")).any():
                b_list_bd.append(np.load(f"jacobian_jvp_error/raw_data/b/{layer}_cosine_bd.npy").mean(axis=0))
            if not np.isnan(np.load(f"jacobian_jvp_error/raw_data/c/{layer}_cosine_bd.npy")).any():
                c_list_bd.append(np.load(f"jacobian_jvp_error/raw_data/c/{layer}_cosine_bd.npy").mean(axis=0))

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

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(probe_counts, z_list, label=r'$W_z$', color='red')
        ax1.plot(probe_counts, x_list, label=r'$W_x$', color='green')
        # ax1.plot(probe_counts, b_list, label=r'$W_b$', color='blue')
        # ax1.plot(probe_counts, c_list, label=r'$W_c$', color='purple')

        ax1.set_xlabel('Probe Count (# IID Samples)')
        ax1.set_ylabel(r'Runtime for $W_z$, $W_x$ (s)')
        ax1.set_title(r'Runtime vs. Probe Count (Averaged Across 10 Layers)', pad=10)



        values = [gptq_mean_z, gptq_mean_x, gptq_mean_b, gptq_mean_c]
        labels = [r"GPTQ $W_z$", r"GPTQ $W_x$", r"GPTQ $W_b$", r"GPTQ $W_c$"]
        colors = ['red', 'green', 'blue', 'purple']

        # for value, label, color in zip(values, labels, colors):
        #     ax1.axhline(y=value, label=label, color=color, linestyle='--')

        ax2 = ax1.twinx()
        ax2.plot(probe_counts, b_list, label=r'$W_b$', color='blue')
        ax2.plot(probe_counts, c_list, label=r'$W_c$', color='purple')
        ax2.set_ylabel(r'Runtime for $W_b$, $W_c$ (s)')

        # for value, label, color in zip(values, labels, colors):
        #     ax1.axhline(y=value, label=label, color=color, linestyle='--')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, 1.10), ncol=8, frameon=True, fontsize='small')

        plt.subplots_adjust(top=0.82)
        plt.show()

        plt.savefig(f"jacobian_jvp_error/Gaussian_130m_Stitched_Averaged_Runtime2.png", bbox_inches='tight')

        plt.clf()
        
        '''
        plt.show()
        plt.savefig(f"jacobian_jvp_error/Gaussian_130m_Stitched.png")
        plt.clf()
        '''
        raise ValueError("Stop!!")
    
    def free(self):
        torch.cuda.empty_cache()
        gc.collect()

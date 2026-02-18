from math import e
import torch
import numpy as np
import os
from safetensors.torch import load_file
import gc

def load_shards(layer_idx):
    file_path = os.path.join("../jacobian_samples", f"Layer_{layer_idx}.safetensors")
    shard_data = load_file(file_path)
    return shard_data

def basic_summary(shard_data):
    for key in shard_data.keys():
        print(key)
        print("Dtype: ",shard_data[key].dtype)
        print("Mean: ",shard_data[key].mean())
        print("Std: ",shard_data[key].std())
        print("Min: ",shard_data[key].min())
        print("0.01%: ",np.percentile(shard_data[key], 0.01))
        print("0.1%: ",np.percentile(shard_data[key], 0.1))
        print("1%: ",np.percentile(shard_data[key], 1))
        print("5%: ",np.percentile(shard_data[key], 5))
        print("10%: ",np.percentile(shard_data[key], 10))
        print("25%: ",np.percentile(shard_data[key], 25))
        print("50%: ",np.percentile(shard_data[key], 50))
        print("75%: ",np.percentile(shard_data[key], 75))
        print("90%: ",np.percentile(shard_data[key], 90))
        print("95%: ",np.percentile(shard_data[key], 95))
        print("99%: ",np.percentile(shard_data[key], 99))
        print("99.9%: ",np.percentile(shard_data[key], 99.9))
        print("99.99%: ",np.percentile(shard_data[key], 99.99))
        print("Max: ",shard_data[key].max())
        print("-"*80)
    
    return shard_data

def build_gradient_matrix(shard_data):
    gradient_matrix = []
    for key in shard_data.keys():
        gradient_matrix.append(shard_data[key])
    gradient_matrix = torch.stack(gradient_matrix)
    return gradient_matrix.float()

def agg_summary(gradient_matrix):
    print("Dtype: ",gradient_matrix.dtype)
    print("Mean: ",gradient_matrix.mean())
    print("Std: ",gradient_matrix.std())
    print("Min: ",gradient_matrix.min())
    print("Max: ",gradient_matrix.max())
    print("0.01%: ",np.percentile(gradient_matrix, 0.01))
    print("0.1%: ",np.percentile(gradient_matrix, 0.1))
    print("1%: ",np.percentile(gradient_matrix, 1))
    print("5%: ",np.percentile(gradient_matrix, 5))
    print("10%: ",np.percentile(gradient_matrix, 10))
    print("25%: ",np.percentile(gradient_matrix, 25))
    print("50%: ",np.percentile(gradient_matrix, 50))
    print("75%: ",np.percentile(gradient_matrix, 75))
    print("90%: ",np.percentile(gradient_matrix, 90))
    print("95%: ",np.percentile(gradient_matrix, 95))
    print("99%: ",np.percentile(gradient_matrix, 99))
    print("99.9%: ",np.percentile(gradient_matrix, 99.9))
    print("99.99%: ",np.percentile(gradient_matrix, 99.99))
    print("Max: ",gradient_matrix.max())
    print("-"*80)
    return gradient_matrix

def flatten_gradient_matrix(gradient_matrix):
    return gradient_matrix.reshape(gradient_matrix.shape[0], -1)

def normalized_flattened_gradient_matrix(gradient_matrix):
    return gradient_matrix / gradient_matrix.norm(dim=1, keepdim=True)

def get_eigenvalues(gradient_matrix_flat):
    if gradient_matrix_flat.device.type == "cpu":
        raise ValueError("Gradient matrix must be on GPU")
    symmetric_matrix = gradient_matrix_flat @ gradient_matrix_flat.T
    # add a small diagonal to the symmetric matrix to make it positive definite
    symmetric_matrix = symmetric_matrix + torch.eye(symmetric_matrix.shape[0]).to(symmetric_matrix.device) * 1e-9

    symmetric_matrix = (symmetric_matrix + symmetric_matrix.T) / 2

    eigenvalues = torch.linalg.eigvalsh(symmetric_matrix.float())
    return eigenvalues.real.clone().detach()

def eigenvalue_density_test(eigenvalues):
    eigenvalues_sorted, _ = torch.sort(eigenvalues, descending=True)
    eigenvalue_densities = torch.cumsum(eigenvalues_sorted, dim=0) / eigenvalues_sorted.sum()
    return eigenvalue_densities

def stable_rank(eigenvalues):
    eigenvalues_sorted, _ = torch.sort(eigenvalues, descending=True)
    return torch.sum(eigenvalues_sorted) ** 2 / torch.sum(eigenvalues_sorted ** 2)


layer = 0
for layer_seed in range(12):
    layer = 2 * layer_seed
    print(f"Layer {layer}")
    shard_data = load_shards(layer)
    # basic_summary(shard_data)

    gradient_matrix = build_gradient_matrix(shard_data)

    if gradient_matrix.device.type == "cpu":
        gradient_matrix = gradient_matrix.to(torch.device("cuda"))
    print(gradient_matrix.device)


    # agg_summary(gradient_matrix)

    gradient_matrix_flat = flatten_gradient_matrix(gradient_matrix)
    normalized_gradient_matrix_flat = normalized_flattened_gradient_matrix(gradient_matrix_flat)


    eigenvalues = get_eigenvalues(gradient_matrix_flat)
    normalized_eigenvalues = get_eigenvalues(normalized_gradient_matrix_flat)

    eigenvalue_densities = eigenvalue_density_test(eigenvalues)
    normalized_eigenvalue_densities = eigenvalue_density_test(normalized_eigenvalues)

    print("Eigenvalue densities:")
    print(eigenvalue_densities.shape)
    print(eigenvalue_densities.numel() - (eigenvalue_densities > 0.9).sum())
    print(eigenvalue_densities.numel() - (eigenvalue_densities > 0.95).sum())
    print(eigenvalue_densities.numel() - (eigenvalue_densities  > 0.99).sum())
    print(eigenvalue_densities.numel() - (eigenvalue_densities > 0.999).sum())
    print()
    print("Normalized eigenvalue densities:")
    print(normalized_eigenvalue_densities.shape)
    print(normalized_eigenvalue_densities.numel() - (normalized_eigenvalue_densities > 0.9).sum())
    print(normalized_eigenvalue_densities.numel() - (normalized_eigenvalue_densities > 0.95).sum())
    print(normalized_eigenvalue_densities.numel() - (normalized_eigenvalue_densities  > 0.99).sum())
    print(normalized_eigenvalue_densities.numel() - (normalized_eigenvalue_densities > 0.999).sum())
    print()
    print("Stable rank: ",stable_rank(eigenvalues))
    print("Normalized stable rank: ",stable_rank(normalized_eigenvalues))
    print("-"*80)
    print()
    torch.cuda.empty_cache()
    gc.collect()


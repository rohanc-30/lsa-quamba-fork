"""Collect exact Hessian (J^T J) for Mamba2-130m using forward-mode AD.

Usage:
    python scripts/collect_exact_hessian.py \
        --model state-spaces/mamba2-130m \
        --nsamples 4 --seqlen 256 \
        --max_channels 8 --target_layers 0 1 \
        --output_dir exact_hessians/130m
"""
import argparse
import gc
import logging
import os
import sys
import time
from functools import partial

import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import smgptq directly to avoid CUDA extension dependency in quamba/__init__.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "smgptq", os.path.join(os.path.dirname(__file__), "..", "quamba", "smgptq.py"))
_smgptq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smgptq)
SMGPTQ = _smgptq.SMGPTQ

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba2-130m")
    parser.add_argument("--nsamples", type=int, default=4,
                        help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--max_channels", type=int, default=8,
                        help="Number of channels per group to compute")
    parser.add_argument("--target_layers", type=int, nargs="+", default=[0],
                        help="Which layers to compute Hessians for")
    parser.add_argument("--groups", type=str, nargs="+", default=["z"],
                        choices=["z", "x", "b", "c"],
                        help="Which channel groups to compute")
    parser.add_argument("--jvp_chunk_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="exact_hessians/130m")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    logger.info("Loading model: %s", args.model)
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    model = MambaLMHeadModel.from_pretrained(args.model, device="cpu", dtype=torch.float32)
    model.eval()
    logger.info("Model loaded: %d layers, d_model=%d",
                len(model.backbone.layers), model.config.d_model)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Build calibration data
    logger.info("Building calibration data: %d samples, seqlen=%d", args.nsamples, args.seqlen)
    _du_spec = importlib.util.spec_from_file_location(
        "data_loaders", os.path.join(os.path.dirname(__file__), "..", "quamba", "data_loaders.py"))
    _du = importlib.util.module_from_spec(_du_spec)
    _du_spec.loader.exec_module(_du)
    get_loaders = _du.get_loaders
    dataloader, _ = get_loaders("wikitext2", tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)

    layers = model.backbone.layers
    model.backbone.embedding = model.backbone.embedding.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype

    nsamples = args.nsamples
    seqlen = args.seqlen
    d_model = model.config.d_model

    inps = torch.zeros((nsamples, seqlen, d_model), dtype=dtype, device=device)
    residual = torch.zeros((nsamples, seqlen, d_model), dtype=dtype, device=device)

    # Capture inputs to the first layer
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, res=None, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()

    # Hook to capture mixer inputs
    def capture_mixer_input(module, inp, out, storage):
        storage["inputs"] = inp[0].detach()

    # Process each layer
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(len(layers)):
        layer = layers[i].to(device)

        if i in args.target_layers:
            logger.info("=== Layer %d ===", i)

            # Capture mixer inputs via hook
            storage = {}
            handle = layer.mixer.register_forward_hook(
                partial(capture_mixer_input, storage=storage))

            with torch.no_grad():
                layer(inps, residual=residual)
            handle.remove()

            mixer_inputs = storage["inputs"]  # (nsamples, seqlen, d_model)
            logger.info("Mixer inputs shape: %s", mixer_inputs.shape)

            # Create SMGPTQ and compute exact Hessians
            smgptq = SMGPTQ(layer.mixer, idx=i)

            layer_dir = os.path.join(args.output_dir, f"layer{i}")
            os.makedirs(layer_dir, exist_ok=True)

            for group_name in args.groups:
                channel_range = smgptq._channel_range(group_name)
                channels = channel_range[:args.max_channels]
                logger.info("Computing exact Hessian for group '%s', channels %s",
                            group_name, channels)

                t0 = time.time()
                hessians = smgptq.compute_exact_hessian(
                    mixer_inputs, channel_group=group_name,
                    channels=channels, jvp_chunk_size=args.jvp_chunk_size)
                elapsed = time.time() - t0

                logger.info("Group '%s': %d channels x %d samples in %.1fs (%.2fs/channel/sample)",
                            group_name, len(channels), nsamples, elapsed,
                            elapsed / len(channels) / nsamples)

                # Save Hessians
                save_dict = {str(k): v.cpu() for k, v in hessians.items()}
                save_path = os.path.join(layer_dir, f"{group_name}_exact_hessian.safetensors")
                save_file(save_dict, save_path)
                logger.info("Saved to %s", save_path)

                # Log diagnostics
                for j, H in hessians.items():
                    eigs = torch.linalg.eigvalsh(H)
                    logger.info("  ch %d: sym=%s, min_eig=%.4e, max_eig=%.4e, cond=%.4e",
                                j, torch.allclose(H, H.T, atol=1e-3),
                                eigs.min().item(), eigs.max().item(),
                                eigs.max().item() / max(eigs.min().abs().item(), 1e-10))

                del hessians
                smgptq.free()

            del smgptq, storage, mixer_inputs

        # Propagate inputs to next layer
        with torch.no_grad():
            inps, residual = layer(inps, residual=residual)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Done!")


if __name__ == "__main__":
    main()

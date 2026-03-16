"""Compare all three quantization methods at a given bit width.

Methods:
  1. Vanilla GPTQ (baseline)
  2. Gramian-rotated GPTQ (rotate x/B/C rows by Gramian eigenvectors)
  3. K-FAC 2D Kronecker GPTQ (BoA-style solver with rank-truncated factors)

Usage:
    python scripts/quantize_compare.py --w_bits 2
    python scripts/quantize_compare.py --w_bits 3
"""
import argparse
import gc
import importlib.util
import logging
import math
import os
import sys
import time

import torch
import torch.nn as nn

_repo_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _repo_root)


def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lr = _import_module("lr_kfac", os.path.join(_repo_root, "quamba", "lr_kfac.py"))
_gu = _import_module("gptq_utils", os.path.join(_repo_root, "quamba", "gptq_utils.py"))
_kg = _import_module("kfac_gptq", os.path.join(_repo_root, "quamba", "kfac_gptq.py"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Import shared utilities from quantize_kfac.py
_qk = _import_module("quantize_kfac", os.path.join(_repo_root, "scripts", "quantize_kfac.py"))
gptq_block = _qk.gptq_block
compute_gramian_rotations = _qk.compute_gramian_rotations
quantize_layer_rotated = _qk.quantize_layer_rotated
quantize_layer_baseline = _qk.quantize_layer_baseline
eval_ppl = _qk.eval_ppl
capture_inputs = _qk.capture_inputs


def quantize_layer_kfac(mixer, mixer_inputs, slices, nsamples_factors, group_size, w_bits, device):
    """Quantize with K-FAC 2D Kronecker solver (precore factors + rank truncation)."""
    factors = _lr.compute_precore_factors(mixer, mixer_inputs, nsamples=nsamples_factors)
    q = _kg.KFACQuantizer(mixer.in_proj)
    q.set_factors(factors, percdamp=0.01)
    q.fasterquant(group_size=group_size, w_bits=w_bits)
    q.free()
    del factors


def run_method(model, inps, residual, args, device, method):
    """Quantize all layers with specified method."""
    layers = model.backbone.layers
    nlayers = len(layers)
    t0_total = time.time()

    for i in range(nlayers):
        layer = layers[i].to(device)
        storage = {}
        def capture(module, inp, out, s=storage):
            s["inputs"] = inp[0].detach()
        handle = layer.mixer.register_forward_hook(capture)
        with torch.no_grad():
            layer(inps, residual=residual)
        handle.remove()
        mixer_inputs = storage["inputs"]
        mixer = layer.mixer
        slices = _lr.get_inproj_slices(mixer)

        if method == 'gptq':
            quantize_layer_baseline(mixer, mixer_inputs, args.group_size, args.w_bits)
        elif method == 'gramian_rotate':
            A = _lr.compute_input_factor(mixer_inputs)
            rotations = compute_gramian_rotations(
                mixer, mixer_inputs, nsamples=min(args.nsamples_factors, args.nsamples),
                device=device, cond_threshold=args.cond_threshold)
            quantize_layer_rotated(mixer, mixer_inputs, rotations, A, slices,
                                    args.group_size, args.w_bits)
        elif method == 'kfac_2d':
            quantize_layer_kfac(mixer, mixer_inputs, slices,
                                 min(args.nsamples_factors, args.nsamples),
                                 args.group_size, args.w_bits, device)

        if i % 6 == 0:
            logger.info("  [%s] L%d (%.0fs)", method, i, time.time() - t0_total)

        del mixer_inputs, storage
        gc.collect(); torch.cuda.empty_cache()

        with torch.no_grad():
            inps, residual = layer(inps, residual=residual)
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()

    total = time.time() - t0_total
    logger.info("[%s] %.1fs total (%.1fs/layer)", method, total, total / nlayers)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba2-130m")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--w_bits", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--nsamples_factors", type=int, default=32)
    parser.add_argument("--cond_threshold", type=float, default=1e5)
    parser.add_argument("--methods", type=str, default="gptq,gramian_rotate,kfac_2d",
                        help="Comma-separated methods to run")
    args = parser.parse_args()
    methods = args.methods.split(",")

    device = torch.device("cuda")
    logger.info("Model: %s | %d-bit | seqlen=%d | nsamples=%d | methods=%s",
                args.model, args.w_bits, args.seqlen, args.nsamples, methods)

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    _du = _import_module("data_loaders", os.path.join(_repo_root, "quamba", "data_loaders.py"))
    dataloader, _ = _du.get_loaders("wikitext2", tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)

    # FP baseline
    logger.info("\n=== FP BASELINE ===")
    model_fp = MambaLMHeadModel.from_pretrained(args.model, device="cpu", dtype=torch.float32)
    model_fp.eval(); model_fp = model_fp.to(device)
    ppl_fp = eval_ppl(model_fp, tokenizer, args.seqlen, device)
    logger.info("FP perplexity: %.2f", ppl_fp)
    model_fp = model_fp.cpu(); torch.cuda.empty_cache()

    results = {}
    for method in methods:
        logger.info("\n=== %s ===", method.upper())
        model = MambaLMHeadModel.from_pretrained(args.model, device="cpu", dtype=torch.float32)
        model.eval()
        inps, residual = capture_inputs(model, dataloader, device, args.nsamples, args.seqlen)
        t = run_method(model, inps, residual, args, device, method)
        model = model.to(device)
        ppl = eval_ppl(model, tokenizer, args.seqlen, device)
        results[method] = {'ppl': ppl, 'time': t}
        logger.info("[%s] PPL=%.2f (%.1fs)", method, ppl, t)
        del model, inps, residual; gc.collect(); torch.cuda.empty_cache()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (%s, %d-bit, seqlen=%d, nsamples=%d)",
                args.model, args.w_bits, args.seqlen, args.nsamples)
    logger.info("=" * 70)
    logger.info("  %-25s  PPL=%.2f", "FP baseline", ppl_fp)
    for method in methods:
        r = results[method]
        logger.info("  %-25s  PPL=%.2f  (%.1fs, +%.2f vs FP)",
                    method, r['ppl'], r['time'], r['ppl'] - ppl_fp)


if __name__ == "__main__":
    main()

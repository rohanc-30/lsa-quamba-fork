"""Gramian-rotated GPTQ vs vanilla GPTQ comparison for Mamba-2 in_proj.

Rotates x/B/C rows into the reverse-Gramian eigenbasis before quantization,
then de-rotates. The rotation aligns quantization-sensitive directions with
the per-row quantization grid.

Usage:
    python scripts/quantize_kfac.py --model state-spaces/mamba2-130m --w_bits 4
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def gptq_block(W_block, A, group_size=128, w_bits=4):
    """Run GPTQ on a weight sub-block with a given column Hessian."""
    dev = W_block.device
    out_dim, in_dim = W_block.shape
    W = W_block.clone().float()
    H = A.to(dev).float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0
    damp = 0.01 * torch.mean(torch.diag(H))
    if damp == 0:
        damp = torch.tensor(0.01, device=dev)
    diag = torch.arange(in_dim, device=dev)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)

    Q = torch.zeros_like(W)
    for i1 in range(0, in_dim, group_size):
        i2 = min(i1 + group_size, in_dim)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        pgs = _gu.get_per_channel_scale(W1, num_bits=w_bits)
        for i in range(i2 - i1):
            w = W1[:, i].clone()
            d = Hinv1[i, i]
            q = _gu.quant(w.unsqueeze(1), pgs, num_bits=w_bits).flatten()
            Q[:, i1 + i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return Q


def compute_gramian_rotations(mixer, mixer_inputs, nsamples, device, cond_threshold=1e5):
    """Compute per-block Gramian eigenvector rotations.

    Returns dict with rotation matrices V_x (H, P, P), V_B (N, N), V_C (N, N).
    Skips rotation (returns identity) for blocks where the Gramian conditioning
    exceeds cond_threshold.
    """
    H = mixer.nheads
    N = mixer.d_state
    P = mixer.headdim
    D_param = mixer.D.detach().float() if mixer.D is not None else None

    F_B_accum = torch.zeros(H, N, N, device=device)
    F_C_accum = torch.zeros(H, N, N, device=device)
    F_x_accum = torch.zeros(H, P, P, device=device)

    for s in range(nsamples):
        cache = _lr.extract_ssd_cache(mixer_inputs[s], mixer)
        M = _lr.reverse_gramian_scan(cache['C'], cache['gamma'], mixer.ngroups, H)
        gssd = _lr.compute_gssd_band(M, cache['B'], cache['gamma'],
                                      cache['g_of_h'], mixer.d_conv)
        F_x_accum += _lr.compute_front_factor_x(cache, gssd, D_param=D_param)
        F_B_accum += (cache['x_dt'].pow(2).sum(-1).unsqueeze(-1).unsqueeze(-1) * M).mean(0)
        _, StS, _ = _lr.forward_state_scan_with_StS(
            cache['x_dt'], cache['B'], cache['gamma'], mixer.ngroups, H)
        F_C_accum += StS.mean(0)
        del cache, M, gssd, StS

    F_x = F_x_accum / nsamples
    F_B_g = (F_B_accum / nsamples).sum(0)  # (N, N) group-level
    F_C_g = (F_C_accum / nsamples).sum(0)

    # B rotation
    F_B_g = 0.5 * (F_B_g + F_B_g.T)
    eigs_B, V_B = torch.linalg.eigh(F_B_g)
    cond_B = eigs_B.max() / eigs_B.clamp(min=1e-30).min()
    if cond_B > cond_threshold:
        logger.info("  B Gramian cond=%.1e > threshold, skipping rotation", cond_B.item())
        V_B = torch.eye(N, device=device)
    else:
        V_B = V_B.flip(-1)

    # C rotation
    F_C_g = 0.5 * (F_C_g + F_C_g.T)
    eigs_C, V_C = torch.linalg.eigh(F_C_g)
    cond_C = eigs_C.max() / eigs_C.clamp(min=1e-30).min()
    if cond_C > cond_threshold:
        logger.info("  C Gramian cond=%.1e > threshold, skipping rotation", cond_C.item())
        V_C = torch.eye(N, device=device)
    else:
        V_C = V_C.flip(-1)

    # x rotation: per head
    V_x = torch.zeros(H, P, P, device=device)
    n_skipped = 0
    for h in range(H):
        Fh = 0.5 * (F_x[h] + F_x[h].T)
        eigs_h, Vh = torch.linalg.eigh(Fh)
        cond_h = eigs_h.max() / eigs_h.clamp(min=1e-30).min()
        if cond_h > cond_threshold:
            V_x[h] = torch.eye(P, device=device)
            n_skipped += 1
        else:
            V_x[h] = Vh.flip(-1)
    if n_skipped > 0:
        logger.info("  x: skipped %d/%d heads (cond > %.0e)", n_skipped, H, cond_threshold)

    return {'V_x': V_x, 'V_B': V_B, 'V_C': V_C}


def quantize_layer_rotated(mixer, mixer_inputs, rotations, A, slices, group_size, w_bits):
    """Quantize in_proj with Gramian-rotated x/B/C rows."""
    W = mixer.in_proj.weight.data.clone().float()
    Q_full = W.clone()
    V_x, V_B, V_C = rotations['V_x'], rotations['V_B'], rotations['V_C']
    H, P = mixer.nheads, mixer.headdim

    # z, dt: standard GPTQ
    for bn in ['z', 'dt']:
        Q_full[slices[bn]] = gptq_block(W[slices[bn]], A, group_size, w_bits)

    # B: rotate, quantize, de-rotate
    sl_B = slices['B']
    Q_full[sl_B] = V_B @ gptq_block(V_B.T @ W[sl_B], A, group_size, w_bits)

    # C: rotate, quantize, de-rotate
    sl_C = slices['C']
    Q_full[sl_C] = V_C @ gptq_block(V_C.T @ W[sl_C], A, group_size, w_bits)

    # x: per-head rotate, quantize, de-rotate
    sl_x = slices['x']
    W_x = W[sl_x].reshape(H, P, -1)
    Q_x = torch.zeros_like(W_x)
    for h in range(H):
        W_rot = V_x[h].T @ W_x[h]
        Q_rot = gptq_block(W_rot, A, group_size, w_bits)
        Q_x[h] = V_x[h] @ Q_rot
    Q_full[sl_x] = Q_x.reshape(-1, W.shape[1])

    mixer.in_proj.weight.data = Q_full.to(mixer.in_proj.weight.dtype)


def quantize_layer_baseline(mixer, mixer_inputs, group_size, w_bits):
    """Quantize in_proj with vanilla GPTQ."""
    gptq = _gu.GPTQ(mixer.in_proj)
    for s in range(mixer_inputs.shape[0]):
        gptq.add_batch(mixer_inputs[s], None)
    gptq.fasterquant(group_size=group_size, w_bits=w_bits)
    gptq.free()


def eval_ppl(model, tokenizer, seqlen, device):
    """Evaluate perplexity on wikitext2 test set."""
    _du = _import_module("data_loaders", os.path.join(_repo_root, "quamba", "data_loaders.py"))
    _, testenc = _du.get_loaders("wikitext2", tokenizer, nsamples=1, seqlen=seqlen)
    test_ids = testenc.input_ids.to(device)
    nsamples_test = test_ids.numel() // seqlen

    nlls = []
    for i in range(nsamples_test):
        batch = test_ids[:, i * seqlen:(i + 1) * seqlen]
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction='sum')
        nlls.append(loss.item())

    n_tokens = nsamples_test * (seqlen - 1)
    return torch.exp(torch.tensor(sum(nlls) / n_tokens)).item()


def capture_inputs(model, dataloader, device, nsamples, seqlen):
    """Capture inputs to the first layer."""
    layers = model.backbone.layers
    model.backbone.embedding = model.backbone.embedding.to(device)
    layers[0] = layers[0].to(device)
    d_model = model.config.d_model
    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((nsamples, seqlen, d_model), dtype=dtype, device=device)
    residual = torch.zeros_like(inps)
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

    return inps, residual


def run_quantization(model, inps, residual, args, device, method):
    """Quantize all layers with specified method."""
    layers = model.backbone.layers
    nlayers = len(layers)
    t0_total = time.time()

    for i in range(nlayers):
        layer = layers[i].to(device)

        # Capture mixer inputs
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

        if method == 'baseline':
            quantize_layer_baseline(mixer, mixer_inputs, args.group_size, args.w_bits)
        elif method == 'gramian_rotate':
            A = _lr.compute_input_factor(mixer_inputs)
            rotations = compute_gramian_rotations(
                mixer, mixer_inputs, nsamples=min(args.nsamples_factors, args.nsamples),
                device=device, cond_threshold=args.cond_threshold)
            quantize_layer_rotated(mixer, mixer_inputs, rotations, A, slices,
                                    args.group_size, args.w_bits)

        if i % 6 == 0:
            logger.info("  L%d done (%.0fs)", i, time.time() - t0_total)

        del mixer_inputs, storage
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            inps, residual = layer(inps, residual=residual)
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()

    total = time.time() - t0_total
    logger.info("%s: %.1fs total (%.1fs/layer)", method, total, total / nlayers)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba2-130m")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--nsamples_factors", type=int, default=32)
    parser.add_argument("--cond_threshold", type=float, default=1e5)
    args = parser.parse_args()

    device = torch.device("cuda")
    logger.info("Model: %s | %d-bit | seqlen=%d | nsamples=%d | nfactors=%d | cond_thresh=%.0e",
                args.model, args.w_bits, args.seqlen, args.nsamples,
                args.nsamples_factors, args.cond_threshold)

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    _du = _import_module("data_loaders", os.path.join(_repo_root, "quamba", "data_loaders.py"))
    dataloader, _ = _du.get_loaders("wikitext2", tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)

    # FP baseline
    logger.info("\n=== FP BASELINE ===")
    model_fp = MambaLMHeadModel.from_pretrained(args.model, device="cpu", dtype=torch.float32)
    model_fp.eval()
    model_fp = model_fp.to(device)
    ppl_fp = eval_ppl(model_fp, tokenizer, args.seqlen, device)
    logger.info("FP perplexity: %.2f", ppl_fp)
    model_fp = model_fp.cpu()
    torch.cuda.empty_cache()

    # Vanilla GPTQ
    logger.info("\n=== VANILLA GPTQ ===")
    model_gptq = MambaLMHeadModel.from_pretrained(args.model, device="cpu", dtype=torch.float32)
    model_gptq.eval()
    inps, residual = capture_inputs(model_gptq, dataloader, device, args.nsamples, args.seqlen)
    time_gptq = run_quantization(model_gptq, inps, residual, args, device, 'baseline')
    model_gptq = model_gptq.to(device)
    ppl_gptq = eval_ppl(model_gptq, tokenizer, args.seqlen, device)
    logger.info("GPTQ perplexity: %.2f (%.1fs)", ppl_gptq, time_gptq)
    del model_gptq, inps, residual
    gc.collect(); torch.cuda.empty_cache()

    # Gramian-rotated GPTQ
    logger.info("\n=== GRAMIAN-ROTATED GPTQ ===")
    model_gram = MambaLMHeadModel.from_pretrained(args.model, device="cpu", dtype=torch.float32)
    model_gram.eval()
    inps, residual = capture_inputs(model_gram, dataloader, device, args.nsamples, args.seqlen)
    time_gram = run_quantization(model_gram, inps, residual, args, device, 'gramian_rotate')
    model_gram = model_gram.to(device)
    ppl_gram = eval_ppl(model_gram, tokenizer, args.seqlen, device)
    logger.info("Gramian perplexity: %.2f (%.1fs)", ppl_gram, time_gram)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY (%s, %d-bit, seqlen=%d, nsamples=%d)",
                args.model, args.w_bits, args.seqlen, args.nsamples)
    logger.info("=" * 60)
    logger.info("  %-25s  PPL=%.2f", "FP baseline", ppl_fp)
    logger.info("  %-25s  PPL=%.2f  (%.1fs, +%.2f)", "Vanilla GPTQ", ppl_gptq, time_gptq, ppl_gptq - ppl_fp)
    logger.info("  %-25s  PPL=%.2f  (%.1fs, +%.2f)", "Gramian-rotated GPTQ", ppl_gram, time_gram, ppl_gram - ppl_fp)
    logger.info("  Gramian vs GPTQ: %+.2f PPL, %.1fx time", ppl_gram - ppl_gptq, time_gram / max(time_gptq, 0.1))


if __name__ == "__main__":
    main()

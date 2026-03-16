"""Gramian-aware activation quantization for Mamba-2 SSD interface features.

Quantizes B, C, x activations after conv+SiLU but before SSD, using the
reverse Gramian to determine per-token sensitivity weights.

Usage:
    python scripts/activation_quant.py --a_bits 8
    python scripts/activation_quant.py --a_bits 4
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
import torch.nn.functional as F

_repo_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _repo_root)


def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lr = _import_module("lr_kfac", os.path.join(_repo_root, "quamba", "lr_kfac.py"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Quantization primitives
# ============================================================

def quantize_symmetric(x, bits):
    """Per-tensor symmetric quantization."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max() / qmax
    if scale == 0:
        return x
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def quantize_per_token(x, bits):
    """Per-token symmetric quantization. x: (T, D)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-10)
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def quantize_per_channel(x, bits):
    """Per-channel symmetric quantization. x: (T, D)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=0, keepdim=True) / qmax
    scale = scale.clamp(min=1e-10)
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def quantize_gramian_weighted(x, bits, sensitivity):
    """Gramian-weighted per-token quantization.

    Key insight: SENSITIVE tokens must keep full range (no clipping).
    INSENSITIVE tokens can be clipped more aggressively — their outliers
    don't matter downstream, so we trade clipping error (doesn't matter)
    for finer rounding grid (helps the non-outlier values).

    Args:
        x: (T, D) activation tensor
        bits: quantization bit width
        sensitivity: (T,) per-token sensitivity weight from Gramian
    """
    qmax = 2 ** (bits - 1) - 1
    T, D = x.shape

    # Normalize sensitivity: 0 = least sensitive, 1 = most sensitive
    s_min, s_max = sensitivity.min(), sensitivity.max()
    if s_max == s_min:
        return quantize_per_token(x, bits)
    s_norm = (sensitivity - s_min) / (s_max - s_min)

    # Clip percentile: sensitive tokens use 100% (full max, no clip),
    # insensitive tokens use ~95% (clip top 5% outliers for finer grid)
    clip_frac = 1.0 - 0.05 * (1.0 - s_norm)  # sensitive=1.0, insensitive=0.95

    sorted_abs, _ = x.abs().sort(dim=-1, descending=True)
    clip_idx = (clip_frac * D).long().clamp(min=1, max=D) - 1
    scale_val = sorted_abs[torch.arange(T, device=x.device), clip_idx]

    scale = (scale_val / qmax).clamp(min=1e-10).unsqueeze(-1)
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def quantize_gramian_per_channel(x, bits, channel_sensitivity):
    """Gramian-weighted per-channel quantization.

    SENSITIVE channels keep full range (no clipping).
    INSENSITIVE channels get clipped more aggressively for a finer grid.

    Args:
        x: (T, D)
        bits: int
        channel_sensitivity: (D,) per-channel sensitivity from Gramian
    """
    qmax = 2 ** (bits - 1) - 1
    T, D = x.shape

    s = channel_sensitivity
    s_min, s_max = s.min(), s.max()
    if s_max == s_min:
        return quantize_per_channel(x, bits)
    s_norm = (s - s_min) / (s_max - s_min)

    # Sensitive channels: keep full max (clip_frac=1.0)
    # Insensitive channels: clip at 95th percentile (clip_frac=0.95)
    clip_frac = 1.0 - 0.05 * (1.0 - s_norm)

    sorted_abs, _ = x.abs().sort(dim=0, descending=True)
    clip_idx = (clip_frac * T).long().clamp(min=1, max=T) - 1
    scale_val = sorted_abs[clip_idx, torch.arange(D, device=x.device)]

    scale = (scale_val / qmax).clamp(min=1e-10).unsqueeze(0)
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


# ============================================================
# Modified mixer forward with activation quantization
# ============================================================

def mixer_forward_with_act_quant(u, mixer, quant_fn_B=None, quant_fn_C=None, quant_fn_x=None):
    """Run mixer forward, quantizing B/C/x after conv+SiLU but before SSD.

    Args:
        u: (T, d_model) single sample input
        mixer: Mamba2 mixer module
        quant_fn_B: callable(B_flat) -> quantized B_flat, or None for FP
        quant_fn_C: callable(C_flat) -> quantized C_flat, or None for FP
        quant_fn_x: callable(x_flat) -> quantized x_flat, or None for FP

    Returns:
        y: (T, d_ssm) pre-out_proj output
    """
    seqlen = u.shape[0]
    d_ssm = mixer.d_ssm
    nheads = mixer.nheads
    ngroups = mixer.ngroups
    d_state = mixer.d_state
    headdim = mixer.headdim
    d_conv = mixer.d_conv
    chunk_size = mixer.chunk_size
    conv_dim = d_ssm + 2 * ngroups * d_state

    W = mixer.in_proj.weight.detach()
    conv_w = mixer.conv1d.weight.detach()
    conv_b = mixer.conv1d.bias.detach() if mixer.conv1d.bias is not None else None
    A_log = mixer.A_log.detach()
    dt_bias = mixer.dt_bias.detach()
    D_param = mixer.D.detach() if mixer.D is not None else None
    norm_w = mixer.norm.weight.detach()

    # 1-3: in_proj -> split -> conv+SiLU
    zxbcdt = u @ W.T
    z = zxbcdt[:, :d_ssm]
    xBC = zxbcdt[:, d_ssm:d_ssm + conv_dim]
    dt = zxbcdt[:, -nheads:]

    xBC_t = F.pad(xBC.T.unsqueeze(0), (d_conv - 1, 0))
    xBC_t = F.conv1d(xBC_t, conv_w, conv_b, groups=conv_dim)
    xBC_conv = F.silu(xBC_t.squeeze(0).T)

    # 4: Split into x, B, C — THIS is where we quantize
    x_flat = xBC_conv[:, :d_ssm]
    B_flat = xBC_conv[:, d_ssm:d_ssm + ngroups * d_state]
    C_flat = xBC_conv[:, d_ssm + ngroups * d_state:]

    # Apply activation quantization
    if quant_fn_x is not None:
        x_flat = quant_fn_x(x_flat)
    if quant_fn_B is not None:
        B_flat = quant_fn_B(B_flat)
    if quant_fn_C is not None:
        C_flat = quant_fn_C(C_flat)

    # 5-8: reshape, dt, SSD
    x = x_flat.reshape(1, seqlen, nheads, headdim)
    B = B_flat.reshape(1, seqlen, ngroups, d_state).float()
    C = C_flat.reshape(1, seqlen, ngroups, d_state).float()

    A = -torch.exp(A_log.float())
    dt_val = F.softplus(dt + dt_bias).unsqueeze(0)
    x_dt = (x * dt_val.unsqueeze(-1)).float()
    A_dt = (A * dt_val).float()

    from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete
    y_ssd, _ = ssd_minimal_discrete(x_dt, A_dt, B, C, chunk_size)

    # 9: D-skip
    if D_param is not None:
        x_sq = x.squeeze(0)
        y_ssd = y_ssd.squeeze(0)
        if D_param.numel() == nheads:
            y_ssd = y_ssd + x_sq * D_param.reshape(1, nheads, 1)
        else:
            y_ssd = y_ssd + x_sq * D_param.reshape(1, nheads, headdim)
        y_ssd = y_ssd.reshape(seqlen, d_ssm)
    else:
        y_ssd = y_ssd.reshape(seqlen, d_ssm)

    # 10: gated RMSNorm
    y = _lr.rms_norm_gated(y_ssd, norm_w, z=z.float(),
                            eps=mixer.norm.eps,
                            group_size=getattr(mixer.norm, 'group_size', None),
                            norm_before_gate=mixer.norm.norm_before_gate)
    return y


# ============================================================
# Compute Gramian sensitivity weights
# ============================================================

def compute_activation_sensitivity(mixer, mixer_inputs, nsamples):
    """Compute per-token, per-channel sensitivity for B/C/x activations.

    Returns dict with:
        w_B: (T, G*N) per-element B sensitivity
        w_C: (T, G*N) per-element C sensitivity
        w_x: (T, d_ssm) per-element x sensitivity
        w_B_token: (T,) per-token B sensitivity (summed over channels)
        w_C_token: (T,) per-token C sensitivity
        w_x_token: (T,) per-token x sensitivity
        w_B_channel: (G*N,) per-channel B sensitivity (averaged over tokens)
        w_C_channel: (G*N,) per-channel C sensitivity
        w_x_channel: (d_ssm,) per-channel x sensitivity
    """
    H = mixer.nheads
    N = mixer.d_state
    P = mixer.headdim
    G = mixer.ngroups
    device = mixer_inputs.device

    cache0 = _lr.extract_ssd_cache(mixer_inputs[0], mixer)
    T = cache0['gamma'].shape[0]
    del cache0

    w_B_accum = torch.zeros(T, device=device)          # per-token
    w_C_accum = torch.zeros(T, device=device)
    w_x_accum = torch.zeros(T, device=device)
    w_B_chan = torch.zeros(G * N, device=device)        # per-channel
    w_C_chan = torch.zeros(G * N, device=device)
    w_x_chan = torch.zeros(H * P, device=device)

    g_of_h = torch.arange(H, device=device) // (H // G)

    for s in range(nsamples):
        cache = _lr.extract_ssd_cache(mixer_inputs[s], mixer)
        gamma = cache['gamma']      # (T, H)
        B_s = cache['B']            # (T, G, N)
        x_dt = cache['x_dt']       # (T, H, P)

        # Reverse Gramian
        M = _lr.reverse_gramian_scan(cache['C'], gamma, G, H)  # (T, H, N, N)

        # Forward state for C sensitivity
        S = torch.zeros(H, P, N, device=device, dtype=torch.float32)

        for t in range(T):
            gamma_t = gamma[t].unsqueeze(-1).unsqueeze(-1)
            v_t = x_dt[t]
            k_t = B_s[t, g_of_h]
            S = gamma_t * S + torch.einsum('hp,hn->hpn', v_t, k_t)

            # C sensitivity: ||S_t[:, n]||^2 per (h, n)
            # S_t is (H, P, N), so ||S_t[h, :, n]||^2 = sum_p S_t[h,p,n]^2
            StS_diag = S.pow(2).sum(dim=1)  # (H, N) = diag of S^T S per head
            # Sum across heads in each group
            for g in range(G):
                hmask = (g_of_h == g)
                w_C_chan[g * N:(g + 1) * N] += StS_diag[hmask].sum(0)
            w_C_accum[t] += StS_diag.sum().item()

        # B sensitivity: ||x_dt_t||^2 * M_t[n,n] per (h, n)
        xdt_sq = x_dt.pow(2).sum(dim=-1)  # (T, H)
        M_diag = M.diagonal(dim1=-2, dim2=-1)  # (T, H, N)
        # Per-token: sum over h and n
        w_B_accum += (xdt_sq.unsqueeze(-1) * M_diag).sum(dim=(1, 2))
        # Per-channel: sum over t and relevant heads
        for g in range(G):
            hmask = (g_of_h == g)
            w_B_chan[g * N:(g + 1) * N] += (xdt_sq[:, hmask].unsqueeze(-1) * M_diag[:, hmask]).sum(dim=(0, 1))

        # x sensitivity: k_t^T M_t k_t per (t, h)
        k_t = B_s[:, g_of_h]  # (T, H, N)
        Mk = torch.einsum('thnm,thm->thn', M, k_t)
        kMk = torch.einsum('thn,thn->th', k_t, Mk)  # (T, H)
        w_x_accum += kMk.sum(dim=-1)
        # Per-channel for x: kMk is per (t, h), broadcast to (t, h, p)
        for h in range(H):
            w_x_chan[h * P:(h + 1) * P] += kMk[:, h].sum()  # same for all p in head

        del cache, M, Mk, kMk

    # Average
    w_B_accum /= nsamples
    w_C_accum /= nsamples
    w_x_accum /= nsamples
    w_B_chan /= nsamples
    w_C_chan /= nsamples
    w_x_chan /= nsamples

    logger.info("Sensitivity stats:")
    logger.info("  B: token ratio=%.0f, chan ratio=%.0f",
                w_B_accum.max() / w_B_accum.clamp(min=1e-30).min(),
                w_B_chan.max() / w_B_chan.clamp(min=1e-30).min())
    logger.info("  C: token ratio=%.0f, chan ratio=%.0f",
                w_C_accum.max() / w_C_accum.clamp(min=1e-30).min(),
                w_C_chan.max() / w_C_chan.clamp(min=1e-30).min())
    logger.info("  x: token ratio=%.0f, chan ratio=%.0f",
                w_x_accum.max() / w_x_accum.clamp(min=1e-30).min(),
                w_x_chan.max() / w_x_chan.clamp(min=1e-30).min())

    # Sanity check: B sensitivity should decrease with token position
    # (early tokens have larger M_t due to more future dependents)
    first_quarter = w_B_accum[:T // 4].mean()
    last_quarter = w_B_accum[-T // 4:].mean()
    logger.info("  B sanity: first_quarter=%.2e, last_quarter=%.2e, ratio=%.1f (expect > 1)",
                first_quarter, last_quarter, first_quarter / max(last_quarter, 1e-30))

    # C sensitivity should increase with token position
    # (later tokens have larger accumulated state S_t)
    first_q_C = w_C_accum[:T // 4].mean()
    last_q_C = w_C_accum[-T // 4:].mean()
    logger.info("  C sanity: first_quarter=%.2e, last_quarter=%.2e, ratio=%.1f (expect < 1)",
                first_q_C, last_q_C, first_q_C / max(last_q_C, 1e-30))

    return {
        'w_B_token': w_B_accum, 'w_C_token': w_C_accum, 'w_x_token': w_x_accum,
        'w_B_channel': w_B_chan, 'w_C_channel': w_C_chan, 'w_x_channel': w_x_chan,
    }


# ============================================================
# Main: compare activation quantization methods
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba2-130m")
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument("--nsamples_sensitivity", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda")
    logger.info("Model: %s | A%d activation quant | seqlen=%d | nsamples=%d",
                args.model, args.a_bits, args.seqlen, args.nsamples)

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    _du = _import_module("data_loaders", os.path.join(_repo_root, "quamba", "data_loaders.py"))
    dataloader, _ = _du.get_loaders("wikitext2", tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)

    model = MambaLMHeadModel.from_pretrained(args.model, device="cpu", dtype=torch.float32)
    model.eval()

    layers = model.backbone.layers
    model.backbone.embedding = model.backbone.embedding.to(device)
    layers[0] = layers[0].to(device)
    d_model = model.config.d_model

    inps = torch.zeros((args.nsamples, args.seqlen, d_model), dtype=torch.float32, device=device)
    residual = torch.zeros_like(inps)
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, inp, res=None, **kw): inps[cache["i"]] = inp; cache["i"] += 1; raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try: model(batch[0].to(device))
        except ValueError: pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()

    # Test on a few layers
    TEST_LAYERS = [0, 6, 12, 18, 23]
    bits = args.a_bits

    methods = ['fp', 'per_tensor', 'per_token', 'per_channel',
               'gramian_per_token', 'gramian_per_channel']
    results = {m: {} for m in methods}

    for layer_idx in TEST_LAYERS:
        logger.info("\n=== Layer %d ===", layer_idx)

        # Propagate
        cur_inps, cur_res = inps.clone(), residual.clone()
        for i in range(layer_idx):
            layers[i] = layers[i].to(device)
            with torch.no_grad():
                cur_inps, cur_res = layers[i](cur_inps, residual=cur_res)
            layers[i] = layers[i].cpu()
            torch.cuda.empty_cache()

        layer = layers[layer_idx].to(device)
        mixer = layer.mixer

        # Capture mixer inputs
        storage = {}
        def cap(m, i, o, s=storage): s["inputs"] = i[0].detach()
        h = mixer.register_forward_hook(cap)
        with torch.no_grad(): layer(cur_inps, residual=cur_res)
        h.remove()
        mixer_inputs = storage["inputs"]

        # FP reference output (full layer)
        with torch.no_grad():
            out_fp, _ = layer(cur_inps, residual=cur_res)

        # Compute Gramian sensitivity
        sens = compute_activation_sensitivity(
            mixer, mixer_inputs, min(args.nsamples_sensitivity, args.nsamples))

        W_out = mixer.out_proj.weight.detach().float()

        for method in methods:
            # Build quant functions
            if method == 'fp':
                qB = qC = qx = None
            elif method == 'per_tensor':
                qB = lambda x: quantize_symmetric(x, bits)
                qC = lambda x: quantize_symmetric(x, bits)
                qx = lambda x: quantize_symmetric(x, bits)
            elif method == 'per_token':
                qB = lambda x: quantize_per_token(x, bits)
                qC = lambda x: quantize_per_token(x, bits)
                qx = lambda x: quantize_per_token(x, bits)
            elif method == 'per_channel':
                qB = lambda x: quantize_per_channel(x, bits)
                qC = lambda x: quantize_per_channel(x, bits)
                qx = lambda x: quantize_per_channel(x, bits)
            elif method == 'gramian_per_token':
                wB, wC, wx = sens['w_B_token'], sens['w_C_token'], sens['w_x_token']
                qB = lambda x, w=wB: quantize_gramian_weighted(x, bits, w[:x.shape[0]])
                qC = lambda x, w=wC: quantize_gramian_weighted(x, bits, w[:x.shape[0]])
                qx = lambda x, w=wx: quantize_gramian_weighted(x, bits, w[:x.shape[0]])
            elif method == 'gramian_per_channel':
                wB, wC, wx = sens['w_B_channel'], sens['w_C_channel'], sens['w_x_channel']
                qB = lambda x, w=wB: quantize_gramian_per_channel(x, bits, w)
                qC = lambda x, w=wC: quantize_gramian_per_channel(x, bits, w)
                qx = lambda x, w=wx: quantize_gramian_per_channel(x, bits, w)

            # Run quantized forward for each sample, measure layer output error
            err_total = 0.0
            norm_total = 0.0

            for s in range(args.nsamples):
                with torch.no_grad():
                    y_quant = mixer_forward_with_act_quant(
                        mixer_inputs[s], mixer, qB, qC, qx)
                    y_quant_out = y_quant @ W_out.T  # (T, d_model)

                    y_fp = mixer_forward_with_act_quant(
                        mixer_inputs[s], mixer, None, None, None)
                    y_fp_out = y_fp @ W_out.T

                err_total += (y_fp_out - y_quant_out).pow(2).sum().item()
                norm_total += y_fp_out.pow(2).sum().item()

            rel_err = err_total / max(norm_total, 1e-20)
            results[method][layer_idx] = rel_err

        # Log results for this layer
        fp_err = results['fp'][layer_idx]
        for method in methods:
            if method == 'fp':
                logger.info("  %-25s  rel_err=%.6e", method, results[method][layer_idx])
            else:
                logger.info("  %-25s  rel_err=%.6e", method, results[method][layer_idx])

        layers[layer_idx] = layer.cpu()
        del mixer_inputs, cur_inps, cur_res
        gc.collect(); torch.cuda.empty_cache()

    # Summary table
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY (A%d activation quantization of B/C/x at SSD interface)", bits)
    logger.info("=" * 100)
    header = f"{'Layer':>6}"
    for m in methods:
        header += f"  {m:>18}"
    logger.info(header)
    logger.info("-" * 100)

    for li in TEST_LAYERS:
        row = f"{li:>6}"
        base = results['per_token'].get(li, 1e-20)
        for m in methods:
            val = results[m][li]
            row += f"  {val:>18.4e}"
        logger.info(row)

    # Ratio vs per_token
    logger.info("\nRatio vs per_token (< 1.0 means better):")
    for li in TEST_LAYERS:
        base = results['per_token'][li]
        row = f"{li:>6}"
        for m in methods:
            val = results[m][li]
            row += f"  {val/base:>18.4f}"
        logger.info(row)


if __name__ == "__main__":
    main()

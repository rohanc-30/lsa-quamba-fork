"""Gauss-Newton approximation taxonomy for Mamba-2 K-FAC PTQ.

Full approximation hierarchy study covering all 5 levels:

  Level 0->1: Cross-token shared-weight coupling
  Level 1->2: Single-Kronecker collapse quality
  Level 3a:   Cross-weight (x/B/C) coupling
  Level 3b:   Inter-head coupling within x-block
  Level 3c:   Intra-head row coupling

Decision tree:
  small 0->1 -> safe to drop cross-token terms
    small 1->2 -> single Kronecker A (x) G_bar is sufficient
      small 3a -> x/B/C blocks are independent
        small 3b -> heads within x are independent
          small 3c -> channelwise OK
          large 3c -> need full row covariance (Kronecker G_h (x) A)

Usage:
    python scripts/gn_taxonomy.py
"""
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


_lr_kfac = _import_module("lr_kfac", os.path.join(_repo_root, "quamba", "lr_kfac.py"))
functional_mixer_forward = _lr_kfac.functional_mixer_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================

# Shared
NSAMPLES_SCAN = 4           # probe scans (cheap)
NSAMPLES_EXACT = 2          # exact GN blocks (expensive)
SEQLEN = 256
TARGET_LAYERS = [0, 23]
BLOCKS = ['x', 'B', 'C']
JACFWD_CHUNK = 128

# Level 0->1: Cross-token
SUBSPACE_DIM = 32
TOKEN_PAIRS_NEAR = [(0, 1), (0, 2), (0, 3)]
TOKEN_PAIRS_FAR = [(0, 64), (0, 128), (0, 200)]
REPRESENTATIVE_TOKENS = [0, 32, 64, 128, 200]

# Level 1->2: Single-Kronecker
N_OUTPUT_PROBES = 8

# Level 3a: Cross-weight
N_PROBES = 16
CHUNK_ROWS = 8

# Level 3b: Inter-head
HEADS_EXACT = [0]
HEAD_PAIRS_EXACT = [(0, 12)]

# Level 3c: Intra-head
OUTPUT_CHUNK = 32768


# ============================================================
# Helpers
# ============================================================

def get_inproj_slices(mixer):
    """Return dict of row slices for each sub-weight block of in_proj."""
    d = mixer.d_ssm
    ng_ns = mixer.ngroups * mixer.d_state
    nh = mixer.nheads
    return {
        'z':  slice(0, d),
        'x':  slice(d, 2 * d),
        'B':  slice(2 * d, 2 * d + ng_ns),
        'C':  slice(2 * d + ng_ns, 2 * d + 2 * ng_ns),
        'dt': slice(2 * d + 2 * ng_ns, 2 * d + 2 * ng_ns + nh),
    }


def _mixer_params(mixer):
    """Extract detached mixer parameters dict."""
    return dict(
        conv_w=mixer.conv1d.weight.detach(),
        conv_b=mixer.conv1d.bias.detach() if mixer.conv1d.bias is not None else None,
        A_log=mixer.A_log.detach(),
        D_param=mixer.D.detach() if mixer.D is not None else None,
        dt_bias=mixer.dt_bias.detach(),
        norm_w=mixer.norm.weight.detach(),
        norm_eps=mixer.norm.eps,
        nbg=mixer.norm.norm_before_gate,
        ngs=getattr(mixer.norm, 'group_size', None),
        W_out=mixer.out_proj.weight.detach().float(),
    )


def _full_mixer_output(u_s, w_in, mixer, mp):
    """Full mixer forward + out_proj. Returns (T, d_model)."""
    y_pre = functional_mixer_forward(
        u_s, w_in, mp['conv_w'], mp['conv_b'], mp['A_log'], mp['D_param'],
        mp['dt_bias'], mp['norm_w'], mixer.headdim, mixer.ngroups,
        mixer.d_state, mixer.d_ssm, mixer.nheads,
        mixer.chunk_size, mixer.d_conv,
        norm_eps=mp['norm_eps'], norm_before_gate=mp['nbg'],
        norm_group_size=mp['ngs'])
    return y_pre @ mp['W_out'].T


# ============================================================
# JVP primitives
# ============================================================

def _block_jvp_batch(mixer, mp, u_s, block_slice, tangents):
    """Batched JVP for a weight block. tangents: (R, n_rows, d_model) -> (R, T*d_model)."""
    W_full = mixer.in_proj.weight.detach()
    W_block = W_full[block_slice].clone()
    W_before = W_full[:block_slice.start]
    W_after = W_full[block_slice.stop:]

    def _jvp_one(tangent, _u=u_s, _wb=W_block, _wbf=W_before, _wa=W_after):
        def f(wb):
            w_in = torch.cat([_wbf, wb, _wa], dim=0)
            return _full_mixer_output(_u, w_in, mixer, mp).reshape(-1)
        _, out = torch.func.jvp(f, (_wb,), (tangent,))
        return out.float()

    return torch.vmap(_jvp_one, chunk_size=4)(tangents)


def compute_full_jvp_per_direction(mixer, mp, u_s, block_slice, tangent):
    """Single JVP returning full (T, d_model) output perturbation."""
    W_full = mixer.in_proj.weight.detach()
    W_block = W_full[block_slice].clone()
    W_before = W_full[:block_slice.start]
    W_after = W_full[block_slice.stop:]

    def f(wb):
        w_in = torch.cat([W_before, wb, W_after], dim=0)
        return _full_mixer_output(u_s, w_in, mixer, mp)

    _, jvp_out = torch.func.jvp(f, (W_block,), (tangent,))
    return jvp_out.float()


# ============================================================
# Exact GN builders (for Level 3)
# ============================================================

def _compute_exact_gn_subblock_general(mixer, mixer_inputs, rows_a_abs, rows_b_abs, nsamples):
    """Exact GN subblock using absolute in_proj row indices.
    Returns: (len(rows_a), len(rows_b), d_model, d_model) float64.
    """
    mp = _mixer_params(mixer)
    W_full = mixer.in_proj.weight.detach()
    d_model = W_full.shape[1]
    device = W_full.device
    na, nb = len(rows_a_abs), len(rows_b_abs)
    all_rows = sorted(set(rows_a_abs) | set(rows_b_abs))
    G_blocks = torch.zeros(na, nb, d_model, d_model, dtype=torch.float64)
    t0 = time.time()

    for s in range(nsamples):
        u_s = mixer_inputs[s].detach()
        J_cache = {}
        for row_idx in all_rows:
            w_row = W_full[row_idx]

            def f(w_row_vec, _idx=row_idx, _u=u_s):
                W_new = W_full.clone()
                W_new[_idx] = w_row_vec
                return _full_mixer_output(_u, W_new, mixer, mp).reshape(-1)

            basis = torch.eye(d_model, device=device, dtype=w_row.dtype)

            def _jvp_one(tangent, _f=f, _w=w_row):
                _, out = torch.func.jvp(_f, (_w,), (tangent,))
                return out

            J_p = torch.vmap(_jvp_one, chunk_size=JACFWD_CHUNK)(basis).T
            J_cache[row_idx] = J_p

        for i, ri in enumerate(rows_a_abs):
            for j, rj in enumerate(rows_b_abs):
                G_blocks[i, j] += (J_cache[ri].double().T @ J_cache[rj].double()).cpu() / nsamples

        del J_cache
        torch.cuda.empty_cache()

    logger.info("    exact subblock %d x %d: %.0fs", na, nb, time.time() - t0)
    return G_blocks


def compute_exact_gn_subblock(mixer, mixer_inputs, rows_a, rows_b, nsamples=None):
    """Exact GN cross-block for x-block-relative row indices.
    Returns: (len(rows_a), len(rows_b), d_model, d_model) float64.
    Auto-selects memory strategy (GPU vs output chunking at 64-row threshold).
    """
    if nsamples is None:
        nsamples = mixer_inputs.shape[0]

    slices = get_inproj_slices(mixer)
    sl_x = slices['x']
    mp = _mixer_params(mixer)
    W_full = mixer.in_proj.weight.detach()
    W_block = W_full[sl_x].clone()
    W_before = W_full[:sl_x.start]
    W_after = W_full[sl_x.stop:]
    d_model = W_full.shape[1]
    device = W_full.device
    na, nb = len(rows_a), len(rows_b)
    all_rows_set = sorted(set(rows_a) | set(rows_b))
    use_chunking = (na + nb) > 64
    G_blocks = torch.zeros(na, nb, d_model, d_model, dtype=torch.float64)
    t0 = time.time()

    if not use_chunking:
        for s in range(nsamples):
            u_s = mixer_inputs[s].detach()
            J_cache = {}
            for row_idx in all_rows_set:
                w_row = W_block[row_idx]

                def f(w_row_vec, _idx=row_idx, _u=u_s):
                    wb_new = torch.cat([W_block[:_idx], w_row_vec.unsqueeze(0),
                                        W_block[_idx + 1:]], dim=0)
                    w_in = torch.cat([W_before, wb_new, W_after], dim=0)
                    return _full_mixer_output(_u, w_in, mixer, mp).reshape(-1)

                basis = torch.eye(d_model, device=device, dtype=w_row.dtype)

                def _jvp_one(tangent, _f=f, _w=w_row):
                    _, out = torch.func.jvp(_f, (_w,), (tangent,))
                    return out

                J_cache[row_idx] = torch.vmap(_jvp_one, chunk_size=JACFWD_CHUNK)(basis).T

            for i, ri in enumerate(rows_a):
                for j, rj in enumerate(rows_b):
                    G_blocks[i, j] += (J_cache[ri].double().T @ J_cache[rj].double()).cpu() / nsamples

            del J_cache
            torch.cuda.empty_cache()
    else:
        with torch.no_grad():
            w_in = torch.cat([W_before, W_block, W_after], dim=0)
            out_dim = _full_mixer_output(mixer_inputs[0].detach(), w_in, mixer, mp).reshape(-1).shape[0]

        for s in range(nsamples):
            u_s = mixer_inputs[s].detach()
            for chunk_start in range(0, out_dim, OUTPUT_CHUNK):
                chunk_end = min(chunk_start + OUTPUT_CHUNK, out_dim)
                chunk_size = chunk_end - chunk_start
                J_a = torch.zeros(na, d_model, chunk_size, device=device)
                J_b = torch.zeros(nb, d_model, chunk_size, device=device)

                for label, rows, J_out in [('a', rows_a, J_a), ('b', rows_b, J_b)]:
                    for li, row_idx in enumerate(rows):
                        w_row = W_block[row_idx]

                        def f(w_row_vec, _idx=row_idx, _u=u_s, _cs=chunk_start, _ce=chunk_end):
                            wb_new = torch.cat([W_block[:_idx], w_row_vec.unsqueeze(0),
                                                W_block[_idx + 1:]], dim=0)
                            w_in = torch.cat([W_before, wb_new, W_after], dim=0)
                            return _full_mixer_output(_u, w_in, mixer, mp).reshape(-1)[_cs:_ce]

                        basis = torch.eye(d_model, device=device, dtype=w_row.dtype)

                        def _jvp_one(tangent, _f=f, _w=w_row):
                            _, out = torch.func.jvp(_f, (_w,), (tangent,))
                            return out

                        J_out[li] = torch.vmap(_jvp_one, chunk_size=JACFWD_CHUNK)(basis).T.T

                for i in range(na):
                    for j in range(nb):
                        G_blocks[i, j] += (J_a[i].double() @ J_b[j].double().T).cpu() / nsamples

                del J_a, J_b
                torch.cuda.empty_cache()

    logger.info("  exact_gn_subblock: %d x %d rows, %d samples, %.0fs",
                na, nb, nsamples, time.time() - t0)
    return G_blocks


# ============================================================
# Metrics suite (Section 7.5)
# ============================================================

def compute_approximation_metrics(H_exact, H_approx):
    """Relative Frobenius, trace retained, relative spectral error."""
    diff = H_exact - H_approx
    norm_exact = torch.norm(H_exact, p='fro').item()
    rel_frob = torch.norm(diff, p='fro').item() / max(norm_exact, 1e-30)
    trace_retained = torch.trace(H_approx).item() / max(abs(torch.trace(H_exact).item()), 1e-30)
    sv_exact = torch.linalg.svdvals(H_exact)
    sv_diff = torch.linalg.svdvals(diff)
    rel_spectral = sv_diff[0].item() / max(sv_exact[0].item(), 1e-30)
    return {'rel_frob': rel_frob, 'trace_retained': trace_retained, 'rel_spectral': rel_spectral}


def compute_coupling_score(H, block_sizes):
    """Normalized coupling Gamma = A^{-1/2} B C^{-1/2} for 2-block partition."""
    n1, n2 = block_sizes
    A_blk = H[:n1, :n1].double()
    B_blk = H[:n1, n1:n1 + n2].double()
    C_blk = H[n1:n1 + n2, n1:n1 + n2].double()
    eps = 1e-8 * max(torch.trace(A_blk).item() / n1, 1e-10)
    A_blk += eps * torch.eye(n1, device=H.device, dtype=torch.float64)
    C_blk += eps * torch.eye(n2, device=H.device, dtype=torch.float64)
    eigA, VA = torch.linalg.eigh(A_blk)
    A_inv_sqrt = VA @ torch.diag(eigA.clamp(min=1e-12).pow(-0.5)) @ VA.T
    eigC, VC = torch.linalg.eigh(C_blk)
    C_inv_sqrt = VC @ torch.diag(eigC.clamp(min=1e-12).pow(-0.5)) @ VC.T
    Gamma = A_inv_sqrt @ B_blk @ C_inv_sqrt
    return torch.norm(Gamma, p='fro').item() / math.sqrt(n1 * n2)


def compute_quadratic_distortion(H_exact, H_approx, n_directions=50):
    """Quadratic-form distortion u^T P(H) u / u^T H u for random + eigen directions."""
    m = H_exact.shape[0]
    H_ex, H_ap = H_exact.double(), H_approx.double()
    torch.manual_seed(42)
    U_rand = torch.randn(m, n_directions, device=H_exact.device, dtype=torch.float64)
    U_rand = U_rand / U_rand.norm(dim=0, keepdim=True)
    ratios_rand = (U_rand.T @ H_ap @ U_rand).diagonal() / (U_rand.T @ H_ex @ U_rand).diagonal().clamp(min=1e-30)
    n_eig = min(10, m)
    _, eigvecs = torch.linalg.eigh(H_ex)
    top = eigvecs[:, -n_eig:]
    ratios_eig = (top.T @ H_ap @ top).diagonal() / (top.T @ H_ex @ top).diagonal().clamp(min=1e-30)
    all_r = torch.cat([ratios_rand, ratios_eig])
    return {'stats': {'mean': all_r.mean().item(), 'median': all_r.median().item(),
                      'min': all_r.min().item(), 'max': all_r.max().item(),
                      'p10': all_r.quantile(0.1).item(), 'p90': all_r.quantile(0.9).item()}}


def summarize_exact_head_local_gn(G_blocks):
    """Summarize intra-head GN coupling from (P, P, d_model, d_model) tensor."""
    P = G_blocks.shape[0]
    E_pq = G_blocks.pow(2).sum(dim=(2, 3))
    E_pp = E_pq.diagonal()
    rho_off_row = 1.0 - E_pp.sum().item() / max(E_pq.sum().item(), 1e-20)
    kappa_pq = E_pq / (torch.sqrt(E_pp.unsqueeze(1) * E_pp.unsqueeze(0)) + 1e-30)
    mask = ~torch.eye(P, dtype=torch.bool)
    off = kappa_pq[mask]
    return {'rho_off_row': rho_off_row, 'kappa_mean': off.mean().item(),
            'kappa_median': off.median().item(), 'kappa_p95': off.quantile(0.95).item(),
            'kappa_max': off.max().item()}


# ============================================================
# Level 0->1: Cross-token coupling
# ============================================================

def _make_subspace(n_rows, d_model, m, device, seed=0):
    """Random m-dimensional orthonormal subspace of (n_rows, d_model) parameter space."""
    torch.manual_seed(seed)
    raw = torch.randn(m, n_rows * d_model, device=device)
    Q, _ = torch.linalg.qr(raw.T)
    return Q.T.reshape(m, n_rows, d_model)


def compute_token_resolved_hessian(mixer, u_s, block_slice, subspace_U, token_indices):
    """Per-token Jacobian projections J_t U via m full-output JVPs.
    Returns: dict mapping t -> (d_model, m) Jacobian matrix.
    """
    mp = _mixer_params(mixer)
    m = subspace_U.shape[0]
    jvp_all = []
    for i in range(m):
        jvp_all.append(compute_full_jvp_per_direction(mixer, mp, u_s, block_slice, subspace_U[i]))
        if (i + 1) % 8 == 0:
            logger.info("      JVP direction %d/%d", i + 1, m)
    jvp_stack = torch.stack(jvp_all, dim=0)  # (m, T, d_model)
    J_tokens = {t: jvp_stack[:, t, :].T.double() for t in token_indices if t < jvp_stack.shape[1]}
    del jvp_all, jvp_stack
    torch.cuda.empty_cache()
    return J_tokens


def study_cross_token_coupling(mixer, mixer_inputs, block_name, block_slice, nsamples):
    """Level 0->1: Measure cross-token shared-weight coupling for a weight block."""
    n_rows = block_slice.stop - block_slice.start
    d_model = mixer.in_proj.weight.shape[1]
    device = mixer.in_proj.weight.device
    T = mixer_inputs.shape[1]
    m = SUBSPACE_DIM
    subspace_U = _make_subspace(n_rows, d_model, m, device, seed=hash(block_name) % (2**31))
    all_pairs = TOKEN_PAIRS_NEAR + TOKEN_PAIRS_FAR
    all_tokens = sorted(set(REPRESENTATIVE_TOKENS) | set(t for p in all_pairs for t in p))
    all_tokens = [t for t in all_tokens if t < T]

    logger.info("    cross-token: block=%s, m=%d, %d tokens, %d samples",
                block_name, m, len(all_tokens), nsamples)

    H_diag_accum = torch.zeros(m, m, dtype=torch.float64)
    H_full_accum = torch.zeros(m, m, dtype=torch.float64)
    cross_energy = {}
    t0 = time.time()

    for s_idx in range(nsamples):
        J_tokens = compute_token_resolved_hessian(
            mixer, mixer_inputs[s_idx].detach(), block_slice, subspace_U, all_tokens)

        for t in all_tokens:
            if t in J_tokens:
                H_diag_accum += J_tokens[t].T @ J_tokens[t]

        J_sum = sum(J_tokens[t] for t in all_tokens if t in J_tokens)
        H_full_accum += J_sum.T @ J_sum

        for t1, t2 in all_pairs:
            if t1 in J_tokens and t2 in J_tokens:
                H_ts = J_tokens[t1].T @ J_tokens[t2]
                cross_energy.setdefault((t1, t2), 0.0)
                cross_energy[(t1, t2)] += H_ts.pow(2).sum().item()

        del J_tokens
        torch.cuda.empty_cache()
        logger.info("    sample %d/%d (%.0fs)", s_idx + 1, nsamples, time.time() - t0)

    H_diag, H_full = H_diag_accum / nsamples, H_full_accum / nsamples
    H_cross = H_full - H_diag
    norm_diag = torch.norm(H_diag, p='fro').item() ** 2
    norm_cross = torch.norm(H_cross, p='fro').item() ** 2
    norm_total = torch.norm(H_full, p='fro').item() ** 2
    rho = norm_cross / max(norm_total, 1e-30)
    metrics = compute_approximation_metrics(H_full.float().to(device), H_diag.float().to(device))
    quad = compute_quadratic_distortion(H_full.float().to(device), H_diag.float().to(device))

    return {'rho_cross_token': rho, 'norm_diag_sq': norm_diag, 'norm_cross_sq': norm_cross,
            'pair_energies': {k: v / nsamples for k, v in cross_energy.items()},
            'approx_metrics': metrics, 'quad_distortion': quad, 'elapsed': time.time() - t0}


# ============================================================
# Level 1->2: Single-Kronecker collapse
# ============================================================

def measure_kronecker_collapse(mixer, mixer_inputs, block_name, block_slice,
                                nsamples, n_output_probes=N_OUTPUT_PROBES):
    """Level 1->2: Measure single-Kronecker collapse quality using structured probes."""
    n_rows = block_slice.stop - block_slice.start
    d_model = mixer.in_proj.weight.shape[1]
    device = mixer.in_proj.weight.device
    mp = _mixer_params(mixer)

    logger.info("    kronecker collapse: block=%s, n_rows=%d, probes=%d",
                block_name, n_rows, n_output_probes)

    torch.manual_seed(hash(('kron_g', block_name)) % (2**31))
    g_probes = torch.randn(n_output_probes, n_rows, device=device)
    g_probes = g_probes / g_probes.norm(dim=1, keepdim=True)

    n_input_probes = min(8, d_model)
    torch.manual_seed(hash(('kron_a', block_name)) % (2**31))
    a_probes = torch.randn(n_input_probes, d_model, device=device)
    a_probes = a_probes / a_probes.norm(dim=1, keepdim=True)

    ng, na = n_output_probes, n_input_probes
    dim = ng * na
    H_exact_accum = torch.zeros(dim, dim, dtype=torch.float64)
    A_input_accum = torch.zeros(d_model, d_model, dtype=torch.float64, device=device)
    t0 = time.time()

    for s_idx in range(nsamples):
        u_s = mixer_inputs[s_idx].detach()
        jvp_cache = {}
        for gi in range(ng):
            for aj in range(na):
                tangent = g_probes[gi].unsqueeze(1) * a_probes[aj].unsqueeze(0)
                jvp_cache[(gi, aj)] = compute_full_jvp_per_direction(
                    mixer, mp, u_s, block_slice, tangent).double()
            if (gi + 1) % 4 == 0:
                logger.info("      sample %d/%d, probe %d/%d (%.0fs)",
                            s_idx + 1, nsamples, gi + 1, ng, time.time() - t0)

        for gi in range(ng):
            for aj in range(na):
                idx1 = gi * na + aj
                for gi2 in range(gi, ng):
                    aj_start = aj if gi2 == gi else 0
                    for aj2 in range(aj_start, na):
                        idx2 = gi2 * na + aj2
                        val = (jvp_cache[(gi, aj)] * jvp_cache[(gi2, aj2)]).sum().item()
                        H_exact_accum[idx1, idx2] += val / nsamples
                        if idx1 != idx2:
                            H_exact_accum[idx2, idx1] += val / nsamples

        A_input_accum += u_s.double().T @ u_s.double() / u_s.shape[0]
        del jvp_cache
        torch.cuda.empty_cache()

    A_input = (A_input_accum / nsamples).cpu()
    A_probe = torch.zeros(na, na, dtype=torch.float64)
    for j1 in range(na):
        for j2 in range(na):
            A_probe[j1, j2] = (a_probes[j1].double().cpu() @ A_input @ a_probes[j2].double().cpu()).item()

    G_bar_r = torch.zeros(ng, ng, dtype=torch.float64)
    for gi in range(ng):
        for gi2 in range(ng):
            vals = []
            for aj in range(na):
                aAa = A_probe[aj, aj]
                if abs(aAa) > 1e-20:
                    vals.append(H_exact_accum[gi * na + aj, gi2 * na + aj].item() / aAa)
            if vals:
                G_bar_r[gi, gi2] = sum(vals) / len(vals)

    H_kfac = torch.zeros(dim, dim, dtype=torch.float64)
    for gi in range(ng):
        for aj in range(na):
            for gi2 in range(ng):
                for aj2 in range(na):
                    H_kfac[gi * na + aj, gi2 * na + aj2] = A_probe[aj, aj2] * G_bar_r[gi, gi2]

    H_ex_f, H_kf_f = H_exact_accum.float().to(device), H_kfac.float().to(device)
    metrics = compute_approximation_metrics(H_ex_f, H_kf_f)
    quad = compute_quadratic_distortion(H_ex_f, H_kf_f)
    coupling = compute_coupling_score(H_ex_f, (ng * na // 2, dim - ng * na // 2)) if ng >= 2 else float('nan')

    return {'approx_metrics': metrics, 'quad_distortion': quad,
            'coupling_exact': coupling, 'elapsed': time.time() - t0}


# ============================================================
# Level 3a: Cross-weight coupling (x/B/C independence)
# ============================================================

def estimate_cross_weight_probe_energy(mixer, mixer_inputs, blocks, n_probes, nsamples):
    """Bilinear probe estimator for cross-weight GN coupling."""
    slices = get_inproj_slices(mixer)
    device = mixer.in_proj.weight.device
    d_model = mixer.in_proj.weight.shape[1]
    mp = _mixer_params(mixer)
    nb = len(blocks)

    probes_L, probes_R = {}, {}
    for bname in blocks:
        sl = slices[bname]
        n_rows = sl.stop - sl.start
        torch.manual_seed(hash(('L', bname)) % (2**31))
        probes_L[bname] = torch.randn(n_probes, n_rows, d_model, device=device)
        torch.manual_seed(hash(('R', bname)) % (2**31))
        probes_R[bname] = torch.randn(n_probes, n_rows, d_model, device=device)

    YL, YR = {b: None for b in blocks}, {b: None for b in blocks}
    for s in range(nsamples):
        u_s = mixer_inputs[s].detach()
        for bname in blocks:
            sl = slices[bname]
            yl = _block_jvp_batch(mixer, mp, u_s, sl, probes_L[bname])
            yr = _block_jvp_batch(mixer, mp, u_s, sl, probes_R[bname])
            YL[bname] = yl if YL[bname] is None else YL[bname] + yl
            YR[bname] = yr if YR[bname] is None else YR[bname] + yr
        logger.info("  3a probe: sample %d/%d", s + 1, nsamples)

    for b in blocks:
        YL[b] /= nsamples
        YR[b] /= nsamples

    energy = torch.zeros(nb, nb)
    for i, bi in enumerate(blocks):
        for j, bj in enumerate(blocks):
            energy[i, j] = (YL[bi] @ YR[bj].T).pow(2).mean().item()

    diag_e = energy.diagonal()
    kappa = torch.zeros(nb, nb)
    for i in range(nb):
        for j in range(nb):
            kappa[i, j] = energy[i, j] / (math.sqrt(diag_e[i] * diag_e[j]) + 1e-30)
    rho = 1.0 - diag_e.sum().item() / max(energy.sum().item(), 1e-20)

    return {'energy': energy, 'rho_cross_weight': rho, 'kappa': kappa, 'block_names': blocks}


def sample_exact_cross_weight_blocks(mixer, mixer_inputs, blocks, chunk_rows, nsamples):
    """Exact GN subblocks for sampled rows from each block pair."""
    slices = get_inproj_slices(mixer)
    sampled_rows = {}
    for bname in blocks:
        sl = slices[bname]
        n_rows = sl.stop - sl.start
        torch.manual_seed(hash(('sample', bname)) % (2**31))
        sampled_rows[bname] = sorted(torch.randperm(n_rows)[:chunk_rows].tolist())

    results = {}
    for i, bi in enumerate(blocks):
        for j, bj in enumerate(blocks):
            if j < i:
                continue
            rows_a = [slices[bi].start + r for r in sampled_rows[bi]]
            rows_b = [slices[bj].start + r for r in sampled_rows[bj]]
            logger.info("  3a exact: %s(%d) x %s(%d)", bi, len(rows_a), bj, len(rows_b))
            G_ab = _compute_exact_gn_subblock_general(mixer, mixer_inputs, rows_a, rows_b, nsamples)
            E_ab = G_ab.pow(2).sum().item()

            if i == j:
                results[(bi, bj)] = {'E': E_ab}
            else:
                if (bi, bi) not in results:
                    G_aa = _compute_exact_gn_subblock_general(mixer, mixer_inputs, rows_a, rows_a, nsamples)
                    results[(bi, bi)] = {'E': G_aa.pow(2).sum().item()}
                if (bj, bj) not in results:
                    G_bb = _compute_exact_gn_subblock_general(mixer, mixer_inputs, rows_b, rows_b, nsamples)
                    results[(bj, bj)] = {'E': G_bb.pow(2).sum().item()}
                kappa = E_ab / (math.sqrt(results[(bi, bi)]['E'] * results[(bj, bj)]['E']) + 1e-30)
                results[(bi, bj)] = {'E': E_ab, 'kappa': kappa}
                results[(bj, bi)] = {'E': E_ab, 'kappa': kappa}
    return results


# ============================================================
# Level 3b: Inter-head coupling within x-block
# ============================================================

def estimate_inter_head_probe_energy_x(mixer, mixer_inputs, n_probes, nsamples):
    """Bilinear probe estimator for inter-head GN coupling within x-block."""
    slices = get_inproj_slices(mixer)
    sl_x = slices['x']
    device = mixer.in_proj.weight.device
    d_model = mixer.in_proj.weight.shape[1]
    mp = _mixer_params(mixer)
    P = mixer.headdim
    nheads = mixer.nheads
    d_ssm = mixer.d_ssm

    YL, YR = [None] * nheads, [None] * nheads
    for s in range(nsamples):
        u_s = mixer_inputs[s].detach()
        for h in range(nheads):
            off = h * P
            torch.manual_seed(hash(('BL', h)) % (2**31))
            tL = torch.zeros(n_probes, d_ssm, d_model, device=device)
            tL[:, off:off + P, :] = torch.randn(n_probes, P, d_model, device=device)
            torch.manual_seed(hash(('BR', h)) % (2**31))
            tR = torch.zeros(n_probes, d_ssm, d_model, device=device)
            tR[:, off:off + P, :] = torch.randn(n_probes, P, d_model, device=device)

            yl = _block_jvp_batch(mixer, mp, u_s, sl_x, tL)
            yr = _block_jvp_batch(mixer, mp, u_s, sl_x, tR)
            YL[h] = yl if YL[h] is None else YL[h] + yl
            YR[h] = yr if YR[h] is None else YR[h] + yr
        logger.info("  3b probe: sample %d/%d", s + 1, nsamples)

    for h in range(nheads):
        YL[h] /= nsamples
        YR[h] /= nsamples

    energy = torch.zeros(nheads, nheads)
    for i in range(nheads):
        for j in range(nheads):
            energy[i, j] = (YL[i] @ YR[j].T).pow(2).mean().item()

    diag_e = energy.diagonal()
    kappa = torch.zeros(nheads, nheads)
    for i in range(nheads):
        for j in range(nheads):
            kappa[i, j] = energy[i, j] / (math.sqrt(diag_e[i] * diag_e[j]) + 1e-30)
    rho = 1.0 - diag_e.sum().item() / max(energy.sum().item(), 1e-20)

    off_diag = sorted([(kappa[i, j].item(), i, j) for i in range(nheads)
                        for j in range(i + 1, nheads)], reverse=True)[:5]

    return {'energy': energy, 'rho_inter_head': rho, 'kappa': kappa, 'top_pairs': off_diag}


def sample_exact_inter_head_blocks_x(mixer, mixer_inputs, head_pairs, nsamples):
    """Exact GN subblocks for head pairs within x-block."""
    P = mixer.headdim
    results = {}
    for h1, h2 in head_pairs:
        logger.info("  3b exact: heads (%d, %d)", h1, h2)
        r1, r2 = list(range(h1 * P, (h1 + 1) * P)), list(range(h2 * P, (h2 + 1) * P))
        G_cross = compute_exact_gn_subblock(mixer, mixer_inputs, r1, r2, nsamples)
        G_h1 = compute_exact_gn_subblock(mixer, mixer_inputs, r1, r1, nsamples)
        G_h2 = compute_exact_gn_subblock(mixer, mixer_inputs, r2, r2, nsamples)
        E_c, E_1, E_2 = G_cross.pow(2).sum().item(), G_h1.pow(2).sum().item(), G_h2.pow(2).sum().item()
        results[(h1, h2)] = {'E_cross': E_c, 'E_h1h1': E_1, 'E_h2h2': E_2,
                              'kappa': E_c / (math.sqrt(E_1 * E_2) + 1e-30)}
        del G_cross, G_h1, G_h2
        gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Level 3c: Intra-head coupling
# ============================================================

def compute_exact_head_local_gn(mixer, mixer_inputs, head, nsamples):
    """Exact intra-head GN: (P, P, d_model, d_model) for one head's x-rows."""
    P = mixer.headdim
    rows = list(range(head * P, (head + 1) * P))
    logger.info("  3c exact: head %d (%d rows)", head, P)
    return compute_exact_gn_subblock(mixer, mixer_inputs, rows, rows, nsamples)


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda")

    logger.info("Loading model...")
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    model = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba2-130m", device="cpu", dtype=torch.float32)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    _du = _import_module(
        "data_loaders", os.path.join(_repo_root, "quamba", "data_loaders.py"))
    nsamples_total = max(NSAMPLES_SCAN, NSAMPLES_EXACT)
    dataloader, _ = _du.get_loaders("wikitext2", tokenizer,
                                     nsamples=nsamples_total, seqlen=SEQLEN)

    # Capture first-layer inputs
    layers = model.backbone.layers
    model.backbone.embedding = model.backbone.embedding.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype
    d_model = model.config.d_model

    inps = torch.zeros((nsamples_total, SEQLEN, d_model), dtype=dtype, device=device)
    residual = torch.zeros((nsamples_total, SEQLEN, d_model), dtype=dtype, device=device)

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

    all_results = {}

    for layer_idx in TARGET_LAYERS:
        logger.info("")
        logger.info("=" * 80)
        logger.info("LAYER %d", layer_idx)
        logger.info("=" * 80)

        cur_inps, cur_res = inps.clone(), residual.clone()
        for i in range(layer_idx):
            layers[i] = layers[i].to(device)
            with torch.no_grad():
                cur_inps, cur_res = layers[i](cur_inps, residual=cur_res)
            layers[i] = layers[i].cpu()
            torch.cuda.empty_cache()

        layer = layers[layer_idx].to(device)
        mixer_storage = {}
        def capture(module, inp, out, s=mixer_storage):
            s["inputs"] = inp[0].detach()
        handle = layer.mixer.register_forward_hook(capture)
        with torch.no_grad():
            layer(cur_inps, residual=cur_res)
        handle.remove()
        mixer_inputs = mixer_storage["inputs"]
        mixer = layer.mixer

        sl = get_inproj_slices(mixer)
        lr = {}

        # --- Level 0->1: Cross-token ---
        logger.info("")
        logger.info("--- Level 0->1: Cross-token coupling ---")
        lr['L01'] = {}
        for bname in BLOCKS:
            logger.info("  Block: %s", bname)
            r = study_cross_token_coupling(mixer, mixer_inputs[:NSAMPLES_SCAN], bname, sl[bname], NSAMPLES_SCAN)
            logger.info("    rho_cross_token = %.6f  (%.0fs)", r['rho_cross_token'], r['elapsed'])
            m = r['approx_metrics']
            logger.info("    rel_frob=%.4f  trace_ret=%.4f  rel_spec=%.4f",
                        m['rel_frob'], m['trace_retained'], m['rel_spectral'])
            lr['L01'][bname] = r

        # --- Level 1->2: Single-Kronecker ---
        logger.info("")
        logger.info("--- Level 1->2: Single-Kronecker collapse ---")
        lr['L12'] = {}
        for bname in BLOCKS:
            logger.info("  Block: %s", bname)
            r = measure_kronecker_collapse(mixer, mixer_inputs[:NSAMPLES_SCAN], bname, sl[bname], NSAMPLES_SCAN)
            m = r['approx_metrics']
            logger.info("    rel_frob=%.4f  trace_ret=%.4f  rel_spec=%.4f  (%.0fs)",
                        m['rel_frob'], m['trace_retained'], m['rel_spectral'], r['elapsed'])
            lr['L12'][bname] = r

        # --- Level 3a: Cross-weight ---
        logger.info("")
        logger.info("--- Level 3a: Cross-weight coupling ---")
        lr['L3a_probe'] = estimate_cross_weight_probe_energy(
            mixer, mixer_inputs[:NSAMPLES_SCAN], BLOCKS, N_PROBES, NSAMPLES_SCAN)
        logger.info("  rho_cross_weight = %.6f", lr['L3a_probe']['rho_cross_weight'])
        for i, bi in enumerate(BLOCKS):
            vals = " ".join(f"{lr['L3a_probe']['kappa'][i,j]:.4f}" for j in range(len(BLOCKS)))
            logger.info("    kappa %s: %s", bi, vals)

        lr['L3a_exact'] = sample_exact_cross_weight_blocks(
            mixer, mixer_inputs[:NSAMPLES_EXACT], BLOCKS, CHUNK_ROWS, NSAMPLES_EXACT)

        # --- Level 3b: Inter-head ---
        logger.info("")
        logger.info("--- Level 3b: Inter-head coupling (x-block) ---")
        lr['L3b_probe'] = estimate_inter_head_probe_energy_x(
            mixer, mixer_inputs[:NSAMPLES_SCAN], N_PROBES, NSAMPLES_SCAN)
        logger.info("  rho_inter_head = %.6f", lr['L3b_probe']['rho_inter_head'])
        for kval, hi, hj in lr['L3b_probe']['top_pairs']:
            logger.info("    top pair (%d,%d): kappa=%.6f", hi, hj, kval)

        lr['L3b_exact'] = sample_exact_inter_head_blocks_x(
            mixer, mixer_inputs[:NSAMPLES_EXACT], HEAD_PAIRS_EXACT, NSAMPLES_EXACT)
        for (h1, h2), v in lr['L3b_exact'].items():
            logger.info("  exact (%d,%d): kappa=%.6f", h1, h2, v['kappa'])

        # --- Level 3c: Intra-head ---
        logger.info("")
        logger.info("--- Level 3c: Intra-head coupling ---")
        lr['L3c'] = {}
        for head in HEADS_EXACT:
            G = compute_exact_head_local_gn(mixer, mixer_inputs[:NSAMPLES_EXACT], head, NSAMPLES_EXACT)
            cs = summarize_exact_head_local_gn(G)
            logger.info("  head %d: rho_off_row=%.6f  kappa mean=%.4f median=%.4f p95=%.4f max=%.4f",
                        head, cs['rho_off_row'], cs['kappa_mean'], cs['kappa_median'],
                        cs['kappa_p95'], cs['kappa_max'])
            lr['L3c'][head] = cs
            del G; gc.collect(); torch.cuda.empty_cache()

        all_results[layer_idx] = lr
        layer = layer.cpu()
        del mixer_inputs; gc.collect(); torch.cuda.empty_cache()

    # === Summary ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("DECISION TREE")
    logger.info("=" * 80)
    for layer_idx in TARGET_LAYERS:
        r = all_results[layer_idx]
        logger.info("")
        logger.info("Layer %d:", layer_idx)
        for bname in BLOCKS:
            rho01 = r['L01'][bname]['rho_cross_token']
            frob12 = r['L12'][bname]['approx_metrics']['rel_frob']
            logger.info("  %s: L0->1 rho=%.4f %s | L1->2 frob=%.4f %s",
                        bname, rho01,
                        "OK" if rho01 < 0.1 else "COUPLED",
                        frob12,
                        "OK" if frob12 < 0.3 else "VARIES")
        logger.info("  L3a rho_cross_weight = %.4f  %s",
                    r['L3a_probe']['rho_cross_weight'],
                    "OK" if r['L3a_probe']['rho_cross_weight'] < 0.1 else "COUPLED")
        logger.info("  L3b rho_inter_head   = %.4f  %s",
                    r['L3b_probe']['rho_inter_head'],
                    "OK" if r['L3b_probe']['rho_inter_head'] < 0.1 else "COUPLED")
        for head, cs in r['L3c'].items():
            logger.info("  L3c head %d rho_off_row = %.4f  (kappa_max=%.4f)  %s",
                        head, cs['rho_off_row'], cs['kappa_max'],
                        "OK" if cs['rho_off_row'] < 0.1 else "ROW-COUPLED")


if __name__ == "__main__":
    main()

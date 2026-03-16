"""K-FAC Gauss-Newton factors for Mamba-2 in_proj quantization.

Pure-PyTorch mixer forward (torch.func compatible), SSD intermediate extraction,
reverse Gramian, input covariance, tail metric, and K-FAC factor computation.

Notation mapping (plan -> Mamba-2):
    q -> C   (state-to-output, d_state per group)
    k -> B   (input-to-state, d_state per group)
    v -> x_dt (dt-scaled SSM input, headdim per head)
    alpha -> gamma = exp(A*dt) (transition decay per head)
    S -> hidden state (P x N per head, P=headdim, N=d_state)
    M -> reverse Gramian (N x N per head)

Contains:
  Primitives:
    rms_norm_gated, functional_mixer_forward, extract_ssd_cache
  Recurrences:
    reverse_gramian_scan (M_t), forward_state_scan (G_Q accumulation)
  Factors:
    compute_input_factor (A), compute_tail_factor (Omega_bar)
    compute_precore_factors (G_Q, G_K, G_V_scalar)
    compute_full_factors (G_Q_full, G_K_full, G_V_full with shared Omega_bar)
  Utility:
    get_inproj_slices
"""
import logging

import torch
import torch.nn.functional as F

from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-PyTorch Mamba2 forward (torch.func compatible)
# ---------------------------------------------------------------------------

def rms_norm_gated(x, weight, z=None, eps=1e-5, group_size=None, norm_before_gate=True):
    """Pure-PyTorch gated RMSNorm (replaces Triton RMSNormGated).

    Args:
        x: (..., D) input tensor
        weight: (D,) learnable scale
        z: (..., D) optional gate tensor
        eps: epsilon for numerical stability
        group_size: if not None, normalize groups of this size independently.
            None means normalize over the full last dimension.
        norm_before_gate: if True, norm(x) * silu(z); else norm(x * silu(z))
    """
    if group_size is not None and group_size != x.shape[-1]:
        # Reshape into groups for independent normalization
        orig_shape = x.shape
        x = x.reshape(*orig_shape[:-1], -1, group_size)
        if z is not None:
            z = z.reshape(*orig_shape[:-1], -1, group_size)
        weight = weight.reshape(-1, group_size)

        if norm_before_gate and z is not None:
            x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
            result = x_normed * F.silu(z)
        elif z is not None:
            x_gated = x * F.silu(z)
            result = x_gated * torch.rsqrt(x_gated.pow(2).mean(-1, keepdim=True) + eps) * weight
        else:
            result = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
        return result.reshape(orig_shape)

    # No grouping (or group_size == last dim)
    if norm_before_gate and z is not None:
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
        return x_normed * F.silu(z)
    elif z is not None:
        x_gated = x * F.silu(z)
        return x_gated * torch.rsqrt(x_gated.pow(2).mean(-1, keepdim=True) + eps) * weight
    else:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def functional_mixer_forward(u, in_proj_weight, conv_weight, conv_bias,
                              A_log, D_param, dt_bias, norm_weight,
                              headdim, ngroups, d_state, d_ssm, nheads,
                              chunk_size, d_conv, norm_eps=1e-5,
                              norm_before_gate=False, norm_group_size=None):
    """Pure-PyTorch Mamba2 mixer forward for a single sample (no batch dim).

    Replicates the Mamba2 reference path using ssd_minimal_discrete (no Triton/CUDA),
    making it compatible with torch.func transformations (jacfwd, vmap, etc.).

    Args:
        u: (seqlen, d_model) — input to the mixer block
        in_proj_weight: (d_in_proj, d_model) — in_proj weight matrix
        conv_weight: (conv_dim, 1, d_conv) — depthwise conv1d weight
        conv_bias: (conv_dim,) or None — conv1d bias
        A_log: (nheads,) — log of the A parameter
        D_param: (nheads,) or (d_ssm,) or None — D skip connection parameter
        dt_bias: (nheads,) — dt bias (before softplus)
        norm_weight: (d_ssm,) — RMSNorm learnable scale
        headdim: int — head dimension P
        ngroups: int — number of groups G
        d_state: int — SSM state dimension N
        d_ssm: int — SSM dimension (= nheads * headdim)
        nheads: int — number of heads H
        chunk_size: int — SSD chunk size
        d_conv: int — convolution kernel width
        norm_eps: float — RMSNorm epsilon
        norm_before_gate: bool — RMSNorm gate ordering
        norm_group_size: int or None — RMSNorm group size

    Returns:
        y: (seqlen, d_ssm) — pre-out_proj output (after gated norm)
    """
    seqlen = u.shape[0]
    conv_dim = d_ssm + 2 * ngroups * d_state

    # 1. in_proj (linear, no bias in Mamba2)
    zxbcdt = u @ in_proj_weight.T                                  # (T, d_in_proj)

    # 2. Split into z, xBC, dt (assumes d_mlp=0 i.e. d_ssm == d_inner)
    z = zxbcdt[:, :d_ssm]                                         # (T, d_ssm)
    xBC = zxbcdt[:, d_ssm:d_ssm + conv_dim]                       # (T, conv_dim)
    dt = zxbcdt[:, -(nheads):]                                     # (T, nheads)

    # 3. Causal depthwise conv1d + SiLU
    xBC_t = xBC.T.unsqueeze(0)                                    # (1, conv_dim, T)
    xBC_t = F.pad(xBC_t, (d_conv - 1, 0))                        # left-pad for causal
    xBC_t = F.conv1d(xBC_t, conv_weight, conv_bias,
                     groups=conv_dim)                              # (1, conv_dim, T)
    xBC_conv = F.silu(xBC_t.squeeze(0).T)                         # (T, conv_dim)

    # 4. Split x, B, C from conv output
    x = xBC_conv[:, :d_ssm]                                       # (T, d_ssm)
    B = xBC_conv[:, d_ssm:d_ssm + ngroups * d_state]              # (T, ngroups*d_state)
    C = xBC_conv[:, d_ssm + ngroups * d_state:]                   # (T, ngroups*d_state)

    # 5. Reshape for ssd_minimal_discrete
    x = x.reshape(1, seqlen, nheads, headdim)                     # (1, T, H, P)
    B = B.reshape(1, seqlen, ngroups, d_state)                    # (1, T, G, N)
    C = C.reshape(1, seqlen, ngroups, d_state)                    # (1, T, G, N)

    # 6. dt softplus + A computation (all float32 for stability)
    A = -torch.exp(A_log.float())                                  # (nheads,)
    dt = F.softplus(dt + dt_bias).unsqueeze(0)                    # (1, T, H)

    # 7. Cast SSD inputs to a common dtype (float32 for torch.func compatibility)
    ssd_dtype = torch.float32
    x_dt = (x * dt.unsqueeze(-1)).to(ssd_dtype)                   # (1, T, H, P)
    A_dt = (A * dt).to(ssd_dtype)                                  # (1, T, H)
    B = B.to(ssd_dtype)
    C = C.to(ssd_dtype)

    # 8. SSD (pure PyTorch, torch.func compatible)
    y_ssd, _ = ssd_minimal_discrete(x_dt, A_dt, B, C, chunk_size) # (1, T, H, P)

    # 9. D skip connection
    if D_param is not None:
        x_squeezed = x.squeeze(0)                                  # (T, H, P)
        y_ssd = y_ssd.squeeze(0)                                   # (T, H, P)
        if D_param.numel() == nheads:
            # D_has_hdim=False: per-head scalar, broadcast over headdim
            y_ssd = y_ssd + x_squeezed * D_param.reshape(1, nheads, 1)
        else:
            # D_has_hdim=True: per-element
            y_ssd = y_ssd + x_squeezed * D_param.reshape(1, nheads, headdim)
        y_ssd = y_ssd.reshape(seqlen, d_ssm)                      # (T, d_ssm)
    else:
        y_ssd = y_ssd.reshape(seqlen, d_ssm)

    # 10. Gated RMSNorm (pure PyTorch, replaces Triton RMSNormGated)
    y = rms_norm_gated(y_ssd, norm_weight, z=z, eps=norm_eps,
                       group_size=norm_group_size,
                       norm_before_gate=norm_before_gate)

    return y  # (T, d_ssm)


# ---------------------------------------------------------------------------
# Extract SSD intermediates from mixer forward
# ---------------------------------------------------------------------------

def extract_ssd_cache(u, mixer):
    """Run mixer forward up to SSD boundary, return all intermediates.

    Args:
        u: (T, d_model) single sample input (no batch dim)
        mixer: Mamba2 mixer module

    Returns dict:
        gamma: (T, H) - exp(A * dt), transition coefficients in (0, 1)
        B: (T, G, N) - input-to-state projection (after conv+silu)
        C: (T, G, N) - state-to-output projection (after conv+silu)
        x_dt: (T, H, P) - dt-scaled SSM input
        x: (T, H, P) - SSM input before dt scaling
        u: (T, d_model) - original mixer input (for A factor)
        z: (T, d_ssm) - gate output from in_proj
        dt: (T, H) - softplus(raw_dt + dt_bias)
        A: (H,) - -exp(A_log)
        A_dt: (T, H) - A * dt (passed directly to ssd_minimal_discrete)
        g_of_h: (H,) int - group index for each head
    """
    seqlen = u.shape[0]
    d_ssm = mixer.d_ssm
    nheads = mixer.nheads
    ngroups = mixer.ngroups
    d_state = mixer.d_state
    headdim = mixer.headdim
    d_conv = mixer.d_conv
    conv_dim = d_ssm + 2 * ngroups * d_state

    # Extract parameters from mixer
    W = mixer.in_proj.weight.detach()
    conv_w = mixer.conv1d.weight.detach()
    conv_b = mixer.conv1d.bias.detach() if mixer.conv1d.bias is not None else None
    A_log = mixer.A_log.detach()
    dt_bias_val = mixer.dt_bias.detach()

    # 1. in_proj (no bias in Mamba2)
    zxbcdt = u @ W.T                                          # (T, d_in_proj)

    # 2. Split into z, xBC, dt
    z = zxbcdt[:, :d_ssm]                                     # (T, d_ssm)
    xBC = zxbcdt[:, d_ssm:d_ssm + conv_dim]                   # (T, conv_dim)
    dt_raw = zxbcdt[:, -nheads:]                               # (T, nheads)

    # 3. Causal depthwise conv1d + SiLU
    xBC_t = xBC.T.unsqueeze(0)                                 # (1, conv_dim, T)
    xBC_t = F.pad(xBC_t, (d_conv - 1, 0))                     # left-pad for causal
    xBC_t = F.conv1d(xBC_t, conv_w, conv_b, groups=conv_dim)  # (1, conv_dim, T)
    xBC_pre_silu = xBC_t.squeeze(0).T.float()                  # (T, conv_dim)
    xBC_conv = F.silu(xBC_pre_silu)                             # (T, conv_dim)

    # 4. Split x, B, C from conv output
    x_flat = xBC_conv[:, :d_ssm]                               # (T, d_ssm)
    B = xBC_conv[:, d_ssm:d_ssm + ngroups * d_state]           # (T, G*N)
    C = xBC_conv[:, d_ssm + ngroups * d_state:]                # (T, G*N)

    # 5. Reshape
    x = x_flat.reshape(seqlen, nheads, headdim).float()        # (T, H, P)
    B = B.reshape(seqlen, ngroups, d_state).float()             # (T, G, N)
    C = C.reshape(seqlen, ngroups, d_state).float()             # (T, G, N)

    # 6. dt softplus + A computation (float32 for stability)
    A = -torch.exp(A_log.float())                               # (H,)
    dt = F.softplus(dt_raw + dt_bias_val).float()               # (T, H)

    # 7. Derived quantities
    A_dt = A.unsqueeze(0) * dt                                  # (T, H) — negative
    gamma = torch.exp(A_dt)                                     # (T, H) — in (0, 1)
    x_dt = x * dt.unsqueeze(-1)                                 # (T, H, P)

    # Group-to-head mapping: g_of_h[h] = h // (nheads // ngroups)
    g_of_h = torch.arange(nheads, device=u.device) // (nheads // ngroups)

    return {
        'gamma': gamma,
        'B': B,
        'C': C,
        'x_dt': x_dt,
        'x': x,
        'u': u.float(),
        'z': z.float(),
        'dt': dt,
        'A': A,
        'A_dt': A_dt,
        'g_of_h': g_of_h,
        'xBC_pre_silu': xBC_pre_silu,
        'conv_w': conv_w,          # (conv_dim, 1, d_conv)
        'conv_b': conv_b,
        'd_conv': d_conv,
        'd_ssm': d_ssm,
        'nheads': nheads,
        'ngroups': ngroups,
        'd_state': d_state,
        'headdim': headdim,
    }


# ---------------------------------------------------------------------------
# Reverse Gramian backward recursion (= plan's M_t)
# ---------------------------------------------------------------------------

def reverse_gramian_scan(C, gamma, ngroups, nheads):
    """Reverse Gramian recursion M_{t,h}.

    M_{T} = 0  (implicit — we start from T-1)
    M_{t,h} = C_{t,g} C_{t,g}^T + gamma_{t+1,h}^2 * M_{t+1,h}

    Heads within a group share the same C (via g_of_h) but have different
    gamma values, so they produce different M matrices.

    Args:
        C: (T, G, N) — state-to-output matrix (= q_t per group)
        gamma: (T, H) — per-step transition coefficients (= alpha_t)
        ngroups: int (G)
        nheads: int (H)

    Returns:
        M: (T, H, N, N) float32 — reverse Gramian at each timestep
    """
    T, G, N = C.shape
    device = C.device

    g_of_h = torch.arange(nheads, device=device) // (nheads // ngroups)

    # Precompute Q[t,g] = C[t,g] @ C[t,g]^T : (T, G, N, N)
    Q = torch.einsum('tgn,tgm->tgnm', C, C)

    M = torch.zeros(T, nheads, N, N, device=device, dtype=torch.float32)

    # Last timestep: M[T-1, h] = Q[T-1, g_of_h[h]]
    M[T - 1] = Q[T - 1, g_of_h]  # (H, N, N)

    # Backward scan: t = T-2, ..., 0
    for t in range(T - 2, -1, -1):
        gamma_sq = (gamma[t + 1] ** 2).unsqueeze(-1).unsqueeze(-1)  # (H, 1, 1)
        M[t] = Q[t, g_of_h] + gamma_sq * M[t + 1]

    return M


# ---------------------------------------------------------------------------
# Input factor A = E[u u^T] (= plan's input covariance)
# ---------------------------------------------------------------------------

def compute_input_factor(inputs):
    """Compute input factor A = (1/NT) sum_n sum_t u_{n,t} u_{n,t}^T.

    Same computation as GPTQ.add_batch() in gptq_utils.py.

    Args:
        inputs: (N_samples, T, d_model)

    Returns:
        (d_model, d_model) float32
    """
    N_samp, T, d = inputs.shape
    X = inputs.reshape(-1, d).float()  # (N*T, d)
    A = (X.T @ X) / (N_samp * T)
    return A


# ---------------------------------------------------------------------------
# Banded SSD cross-time correlation (for x-row front factor)
# ---------------------------------------------------------------------------

def compute_gssd_band(P, B, gamma, g_of_h, d_conv):
    """Banded SSD Gauss-Newton correlation for x-row perturbations.

    G_SSD[s1, s2, h] = phi(s1,s2) * B[s1,g]^T @ P[max(s1,s2), h] @ B[s2,g]

    Only the d_conv-width band is needed (conv1d has width d_conv).

    Args:
        P: (T, H, N, N) reverse Gramian
        B: (T, G, N) input-to-state
        gamma: (T, H) transition coefficients
        g_of_h: (H,) int — group index for each head
        d_conv: int — conv1d kernel width

    Returns:
        band: (T, H, d_conv, d_conv) float32
    """
    T, H, N, _ = P.shape
    device = P.device

    B_exp = B[:, g_of_h]  # (T, H, N)
    PB = torch.einsum('thnm,thm->thn', P, B_exp)  # (T, H, N)

    phi_offsets = torch.ones(T, H, d_conv, device=device, dtype=torch.float32)
    for k in range(1, d_conv):
        valid = T - k
        if valid <= 0:
            break
        phi_offsets[:valid, :, k] = phi_offsets[:valid, :, k - 1] * gamma[k:k + valid]

    band = torch.zeros(T, H, d_conv, d_conv, device=device, dtype=torch.float32)

    for k1 in range(d_conv):
        for k2 in range(k1, d_conv):
            valid = T - k2
            if valid <= 0:
                continue
            offset = k2 - k1
            bp = torch.einsum('thn,thn->th',
                              B_exp[k1:k1 + valid],
                              PB[k2:k2 + valid])
            phi = phi_offsets[k1:k1 + valid, :, offset]
            band[:valid, :, k1, k2] = phi * bp
            if k1 != k2:
                band[:valid, :, k2, k1] = phi * bp

    return band


def compute_front_factor_x(cache, gssd_band, D_param=None):
    """Per-head front factor for x-rows at the in_proj level.

    Chains the SSD-level sensitivity (from gssd_band / reverse Gramian)
    with the conv+SiLU+dt Jacobian to get the actual per-head factor
    at the in_proj output level.

    F_front_x[h, p1, p2] = (1/T) sum_t beta^T F_full beta

    where beta[t,j,p] = conv_w[p,j] * silu'(v_{t+j}[p]) is the linearized
    conv+SiLU gain, and F_full encodes SSD cross-time correlation + dt + D-skip.

    Args:
        cache: dict from extract_ssd_cache
        gssd_band: (T, H, d_conv, d_conv) from compute_gssd_band
        D_param: (H,) or None — D skip connection parameter

    Returns:
        F_front_x: (H, P, P) float32 — per-head front factor for x-rows
    """
    T = cache['u'].shape[0]
    d_ssm = cache['d_ssm']
    nheads = cache['nheads']
    ngroups = cache['ngroups']
    d_state = cache['d_state']
    headdim = cache['headdim']
    d_conv = cache['d_conv']
    g_of_h = cache['g_of_h']
    device = cache['u'].device

    dt = cache['dt']
    xBC_pre_silu = cache['xBC_pre_silu']
    conv_w = cache['conv_w']
    gamma = cache['gamma']
    B = cache['B']
    C = cache['C']

    # SiLU derivative for x-channels (first d_ssm of conv_dim)
    v_x = xBC_pre_silu[:, :d_ssm].float()
    sig_x = torch.sigmoid(v_x)
    silu_d_x = sig_x * (1.0 + v_x * (1.0 - sig_x))

    # Conv weights for x-channels: (d_ssm, d_conv)
    cw_x = conv_w[:d_ssm, 0, :].float()

    B_exp = B[:, g_of_h]
    C_exp = C[:, g_of_h]

    phi_offsets = torch.ones(T, nheads, d_conv, device=device, dtype=torch.float32)
    for k in range(1, d_conv):
        valid = T - k
        if valid <= 0:
            break
        phi_offsets[:valid, :, k] = phi_offsets[:valid, :, k - 1] * gamma[k:k + valid]

    F_front = torch.zeros(nheads, headdim, headdim, device=device, dtype=torch.float32)

    for h in range(nheads):
        p_start = h * headdim
        p_end = p_start + headdim

        cw_h = cw_x[p_start:p_end]
        sd_h = silu_d_x[:, p_start:p_end]
        dt_h = dt[:, h]
        D_h = D_param[h].item() if D_param is not None else 0.0

        # beta[t, j, p] = conv_w[p,j] * silu'(v_{t+j}[p])
        beta = torch.zeros(T, d_conv, headdim, device=device, dtype=torch.float32)
        for j in range(d_conv):
            valid = T - j
            if valid <= 0:
                break
            beta[:valid, j, :] = cw_h[:, j].unsqueeze(0) * sd_h[j:j + valid]

        # Build F_full[t, j1, j2] from gssd_band + dt + D-skip
        gssd_h = gssd_band[:, h]
        F_full = torch.zeros(T, d_conv, d_conv, device=device, dtype=torch.float32)

        for j1 in range(d_conv):
            for j2 in range(j1, d_conv):
                valid = T - j2
                if valid <= 0:
                    continue
                dt1 = dt_h[j1:j1 + valid]
                dt2 = dt_h[j2:j2 + valid]

                entry = dt1 * dt2 * gssd_h[:valid, j1, j2]

                if D_h != 0:
                    if j1 == j2:
                        gs = torch.einsum('tn,tn->t',
                                          C_exp[j1:j1 + valid, h],
                                          B_exp[j1:j1 + valid, h])
                        entry = entry + 2 * D_h * dt1 * gs + D_h ** 2
                    else:
                        phi_val = phi_offsets[j1:j1 + valid, h, j2 - j1]
                        gs = (torch.einsum('tn,tn->t',
                                           C_exp[j2:j2 + valid, h],
                                           B_exp[j1:j1 + valid, h])
                              * phi_val)
                        entry = entry + D_h * dt1 * gs

                F_full[:valid, j1, j2] = entry
                if j1 != j2:
                    F_full[:valid, j2, j1] = entry

        # F_front_h = (1/T) sum_t beta^T F_full beta
        F_beta = torch.einsum('tjl,tlp->tjp', F_full, beta)
        F_front[h] = torch.einsum('tjp,tjq->pq', beta, F_beta) / T

    return F_front


# ---------------------------------------------------------------------------
# Front factors for B-rows and C-rows at the in_proj level
# ---------------------------------------------------------------------------

def compute_front_factor_B(cache, P, D_param=None):
    """Per-head front factor for B-rows at the in_proj level.

    Chains the SSD-level B-sensitivity with the conv+SiLU Jacobian.

    SSD sensitivity for B perturbations:
        G_B_SSD[s1, s2, h, n1, n2] = phi(s1,s2) * P[s2][n1,n2] * (x_dt_s1 . x_dt_s2)_h

    Chained with conv+SiLU:
        F_B[h, n1, n2] = (1/T) sum_t sum_{j1,j2} beta_B[t,j1,n1] * beta_B[t,j2,n2]
                          * phi(t+j1, t+j2) * P[t+j2][n1,n2] * (x_dt_{t+j1} . x_dt_{t+j2})_h

    Args:
        cache: dict from extract_ssd_cache
        P: (T, H, N, N) reverse Gramian
        D_param: unused, for API consistency

    Returns:
        F_front_B: (H, N, N) float32 — per-head front factor for B-rows
    """
    T = cache['u'].shape[0]
    d_ssm = cache['d_ssm']
    nheads = cache['nheads']
    ngroups = cache['ngroups']
    d_state = cache['d_state']
    d_conv = cache['d_conv']
    device = cache['u'].device

    x_dt = cache['x_dt']             # (T, H, P)
    gamma = cache['gamma']           # (T, H)
    xBC_pre_silu = cache['xBC_pre_silu']
    conv_w = cache['conv_w']

    # SiLU derivative for B-channels
    b_start = d_ssm
    b_end = d_ssm + ngroups * d_state
    v_B = xBC_pre_silu[:, b_start:b_end].float()  # (T, G*N)
    sig_B = torch.sigmoid(v_B)
    silu_d_B = sig_B * (1.0 + v_B * (1.0 - sig_B))  # (T, G*N)

    # Conv weights for B-channels: (G*N, d_conv)
    cw_B = conv_w[b_start:b_end, 0, :].float()  # (G*N, d_conv)

    # beta_B[t, j, n] = conv_w[n, j] * silu'(v_{t+j}[n])
    beta_B = torch.zeros(T, d_conv, ngroups * d_state, device=device, dtype=torch.float32)
    for j in range(d_conv):
        valid = T - j
        if valid <= 0:
            break
        beta_B[:valid, j, :] = cw_B[:, j].unsqueeze(0) * silu_d_B[j:j + valid]

    # phi offsets
    phi_offsets = torch.ones(T, nheads, d_conv, device=device, dtype=torch.float32)
    for k in range(1, d_conv):
        valid = T - k
        if valid <= 0:
            break
        phi_offsets[:valid, :, k] = phi_offsets[:valid, :, k - 1] * gamma[k:k + valid]

    # Accumulate F_front_B: (H, N, N)
    # For ngroups=1, all N channels are in group 0
    F_B = torch.zeros(nheads, d_state, d_state, device=device, dtype=torch.float64)

    for j1 in range(d_conv):
        for j2 in range(j1, d_conv):
            valid = T - j2
            if valid <= 0:
                continue
            offset = j2 - j1

            # phi(t+j1, t+j2): (valid, H)
            phi = phi_offsets[j1:j1 + valid, :, offset]

            # x_dt inner product: (valid, H)
            xdot = torch.einsum('thp,thp->th', x_dt[j1:j1 + valid], x_dt[j2:j2 + valid])

            # coeff = phi * xdot: (valid, H)
            coeff = phi * xdot

            # P at max timestep: P[j2:j2+valid] = (valid, H, N, N)
            P_block = P[j2:j2 + valid]

            # beta products: (valid, G*N) for j1 and j2
            b1 = beta_B[:valid, j1, :]  # (valid, G*N)
            b2 = beta_B[:valid, j2, :]  # (valid, G*N)

            # For each group g, accumulate:
            # F_B[h, n1, n2] += sum_t coeff[t,h] * b1[t,n1] * b2[t,n2] * P[t,h,n1,n2]
            # Since ngroups typically = 1, all n1,n2 are in group 0
            # and all heads see the same B-channels
            g_of_h = cache['g_of_h']
            for g in range(ngroups):
                n_start = g * d_state
                n_end = n_start + d_state
                heads_in_g = (g_of_h == g).nonzero(as_tuple=True)[0]

                b1_g = b1[:, n_start:n_end]  # (valid, N)
                b2_g = b2[:, n_start:n_end]  # (valid, N)

                for h_idx in heads_in_g:
                    h = h_idx.item()
                    # weighted_P[t,n,m] = coeff[t,h] * P[t,h,n,m]
                    weighted_P = coeff[:valid, h].unsqueeze(-1).unsqueeze(-1) * P_block[:valid, h]
                    # contrib[n,m] = sum_t b1[t,n] * weighted_P[t,n,m] * b2[t,m]
                    contrib = torch.einsum('tn,tnm,tm->nm',
                                           b1_g.double(), weighted_P.double(), b2_g.double())
                    F_B[h] += contrib
                    if j1 != j2:
                        F_B[h] += contrib.T  # symmetric in j1, j2

    F_B = (F_B / T).float()
    return F_B


def compute_front_factor_C(cache, forward_StS):
    """Per-head front factor for C-rows at the in_proj level.

    C-row SSD sensitivity is diagonal in time (perturbing C_s only affects y_s):
        G_C_SSD[s, h, n1, n2] = [S_s^T S_s]_{h, n1, n2}

    Chained with conv+SiLU (only same-time j1=j2 terms survive):
        F_C[h, n1, n2] = (1/T) sum_s sum_j beta_C[s-j, j, n1] * beta_C[s-j, j, n2]
                          * [S_s^T S_s]_{h, n1, n2}

    Args:
        cache: dict from extract_ssd_cache
        forward_StS: (T, H, N, N) per-timestep S_t^T S_t from forward scan

    Returns:
        F_front_C: (H, N, N) float32 — per-head front factor for C-rows
    """
    T = cache['u'].shape[0]
    d_ssm = cache['d_ssm']
    nheads = cache['nheads']
    ngroups = cache['ngroups']
    d_state = cache['d_state']
    d_conv = cache['d_conv']
    device = cache['u'].device

    xBC_pre_silu = cache['xBC_pre_silu']
    conv_w = cache['conv_w']

    # SiLU derivative for C-channels
    c_start = d_ssm + ngroups * d_state
    c_end = c_start + ngroups * d_state
    v_C = xBC_pre_silu[:, c_start:c_end].float()  # (T, G*N)
    sig_C = torch.sigmoid(v_C)
    silu_d_C = sig_C * (1.0 + v_C * (1.0 - sig_C))  # (T, G*N)

    # Conv weights for C-channels: (G*N, d_conv)
    cw_C = conv_w[c_start:c_end, 0, :].float()

    # For C, only j1=j2 terms survive (SSD is diagonal in time).
    # F_C[h, n1, n2] = (1/T) sum_s sum_j [cw[n1,j]*silu'_s[n1]] * [cw[n2,j]*silu'_s[n2]] * StS[s,h,n1,n2]
    # = (1/T) sum_s [sum_j cw[n1,j]*cw[n2,j]] * silu'_s[n1]*silu'_s[n2] * StS[s,h,n1,n2]
    # But careful: beta[t,j,n] uses silu' at t+j, and we need s=t+j, so for different j
    # the silu' is at the same s but the "source" t differs. The beta formula is:
    # beta[t,j,n] = cw[n,j] * silu'_{t+j}[n], and we sum over j at fixed s=t+j.

    # Accumulate: for each output timestep s, sum over j such that t=s-j >= 0
    F_C = torch.zeros(nheads, d_state, d_state, device=device, dtype=torch.float64)

    for j in range(d_conv):
        # Source timestep t = s - j, so s ranges from j to T-1
        valid = T - j
        if valid <= 0:
            break
        # beta at (t=s-j, j, n) = cw[n, j] * silu'_s[n]  (since t+j = s)
        # For all valid s: silu' is at s = j..T-1
        silu_s = silu_d_C[j:j + valid]  # (valid, G*N)
        cw_j = cw_C[:, j]  # (G*N,)
        beta_j = cw_j.unsqueeze(0) * silu_s  # (valid, G*N)

        # StS at timestep s = j..T-1
        StS_s = forward_StS[j:j + valid]  # (valid, H, N, N)

        # For each group
        g_of_h = cache['g_of_h']
        for g in range(ngroups):
            n_start = g * d_state
            n_end = n_start + d_state
            b_g = beta_j[:, n_start:n_end]  # (valid, N)

            heads_in_g = (g_of_h == g).nonzero(as_tuple=True)[0]
            for h_idx in heads_in_g:
                h = h_idx.item()
                # F_C[h, n1, n2] += sum_s b_g[s,n1] * b_g[s,n2] * StS[s,h,n1,n2]
                StS_h = StS_s[:, h]  # (valid, N, N)
                F_C[h] += torch.einsum('tn,tnm,tm->nm',
                                       b_g.double(), StS_h.double(), b_g.double())

    F_C = (F_C / T).float()
    return F_C


# ---------------------------------------------------------------------------
# Modified forward state scan: also returns per-timestep S^T S for C-rows
# ---------------------------------------------------------------------------

def forward_state_scan_with_StS(x_dt, B, gamma, ngroups, nheads):
    """Forward state recurrence returning G_Q and per-timestep S_t^T S_t.

    Same as forward_state_scan but also stores S_t^T S_t at each timestep
    (needed by compute_front_factor_C).

    Returns:
        G_Q: (H, N, N) float32
        StS: (T, H, N, N) float32 — per-timestep S_t^T S_t
        S_last: (H, P, N) float32
    """
    T, H, P = x_dt.shape
    _, G, N = B.shape
    device = x_dt.device

    g_of_h = torch.arange(H, device=device) // (H // ngroups)

    S = torch.zeros(H, P, N, device=device, dtype=torch.float32)
    G_Q_accum = torch.zeros(H, N, N, device=device, dtype=torch.float64)
    StS = torch.zeros(T, H, N, N, device=device, dtype=torch.float32)

    for t in range(T):
        gamma_t = gamma[t].unsqueeze(-1).unsqueeze(-1)
        v_t = x_dt[t]
        k_t = B[t, g_of_h]

        S = gamma_t * S + torch.einsum('hp,hn->hpn', v_t, k_t)

        StS_t = torch.einsum('hpn,hpm->hnm', S, S)
        StS[t] = StS_t
        G_Q_accum += StS_t.double()

    G_Q = (G_Q_accum / T).float()
    return G_Q, StS, S.clone()


# ---------------------------------------------------------------------------
# Tail factor: time-averaged Omega_bar (= plan's shared post metric)
# ---------------------------------------------------------------------------

def compute_tail_factor(mixer, inputs):
    """Time-averaged tail Jacobian metric Omega_bar_h per head.

    Omega_bar_h = (1/(T*S)) sum_{s,t} D_{t,h}^T D_{t,h}

    where D_{t,h} = d(output_t)/d(y_ssd_h) is the Jacobian of the full tail:
        y_ssd_h -> gate(silu(z)) -> RMSNorm -> out_proj -> output

    Uses closed-form M_{t,h} = D_{t,h}^T D_{t,h} to avoid materializing
    the (d_model, P) Jacobian. With D_{t,h} = W_h diag(u) - a v^T:

        M_{t,h} = diag(u) S_h diag(u) - diag(u) b v^T - v b^T diag(u) + ||a||^2 v v^T

    where S_h = W_h^T W_h, b = W_h^T a (both precomputable per head).

    Args:
        mixer: Mamba2 mixer module
        inputs: (N_samples, T, d_model) calibration data

    Returns:
        Omega_bar: (H, P, P) float32 — time-averaged per-head tail metric
    """
    if getattr(mixer.norm, 'norm_before_gate', False):
        raise NotImplementedError(
            "compute_tail_factor only supports norm_before_gate=False")

    W_out = mixer.out_proj.weight.detach().float()    # (d_model, d_ssm)
    norm_w = mixer.norm.weight.detach().float()        # (d_ssm,)
    norm_eps = mixer.norm.eps
    nheads = mixer.nheads
    headdim = mixer.headdim
    d_ssm = mixer.d_ssm
    chunk_size = mixer.chunk_size
    device = inputs.device

    D_param = mixer.D.detach().float() if mixer.D is not None else None

    # Precompute S_h = W_h^T W_h for each head: (H, P, P)
    W_out_heads = W_out.reshape(W_out.shape[0], nheads, headdim)  # (d_model, H, P)
    S_h = torch.einsum('dhp,dhq->hpq', W_out_heads, W_out_heads)  # (H, P, P)

    N_samples, T_seq = inputs.shape[0], inputs.shape[1]
    M_accum = torch.zeros(nheads, headdim, headdim, device=device, dtype=torch.float32)

    for s in range(N_samples):
        cache = extract_ssd_cache(inputs[s].detach(), mixer)

        # Run SSD to get y_ssd: (T, H, P)
        x_dt = cache['x_dt']    # (T, H, P)
        A_dt = cache['A_dt']    # (T, H)
        B = cache['B']          # (T, G, N)
        C = cache['C']          # (T, G, N)

        y_ssd, _ = ssd_minimal_discrete(
            x_dt.unsqueeze(0), A_dt.unsqueeze(0),
            B.unsqueeze(0), C.unsqueeze(0), chunk_size)
        y_ssd = y_ssd.squeeze(0)  # (T, H, P)

        # D-skip
        if D_param is not None:
            if D_param.numel() == nheads:
                y_ssd = y_ssd + cache['x'] * D_param.reshape(1, nheads, 1)
            else:
                y_ssd = y_ssd + cache['x'] * D_param.reshape(1, nheads, headdim)

        T = y_ssd.shape[0]
        z = cache['z']  # (T, d_ssm)

        # Gate and RMSNorm intermediates
        silu_z = F.silu(z)                                    # (T, d_ssm)
        y_flat = y_ssd.reshape(T, d_ssm)                     # (T, d_ssm)
        x_gated = y_flat * silu_z                             # (T, d_ssm)
        rms_sq = x_gated.pow(2).mean(dim=-1) + norm_eps       # (T,)
        rms_inv = torch.rsqrt(rms_sq)                         # (T,)

        # a_t = rms_inv * W_out @ (norm_w * x_gated):  (T, d_model)
        a = (x_gated * norm_w.unsqueeze(0)) @ W_out.T         # (T, d_model)
        a = a * rms_inv.unsqueeze(-1)                          # (T, d_model)
        a_sq = (a * a).sum(dim=-1)                             # (T,) = ||a_t||^2

        for h in range(nheads):
            p_s = h * headdim
            p_e = p_s + headdim
            w_h = norm_w[p_s:p_e]                              # (P,)

            g_h = silu_z[:, p_s:p_e]                           # (T, P)
            x_gated_h = x_gated[:, p_s:p_e]                   # (T, P)

            # u_{t,h} = rms_inv_t * w_h * g_{t,h}:  (T, P)
            u_h = rms_inv.unsqueeze(-1) * w_h.unsqueeze(0) * g_h  # (T, P)

            # v_{t,h} = x_gated_h * g_h / (D * rms_sq):  (T, P)
            v_h = x_gated_h * g_h / (d_ssm * rms_sq.unsqueeze(-1))  # (T, P)

            # b_{t,h} = W_h^T a_t:  (T, P)
            b_h = a[:, :] @ W_out_heads[:, h, :]               # (T, P)

            # Term 1: (u^T u) . S_h
            uTu = torch.einsum('tp,tq->pq', u_h, u_h)         # (P, P)
            M_accum[h] += uTu * S_h[h]

            # Term 2+3: -(u*b) v^T - v (u*b)^T
            ub = u_h * b_h                                     # (T, P)
            cross = torch.einsum('tp,tq->pq', ub, v_h)        # (P, P)
            M_accum[h] -= cross + cross.T

            # Term 4: ||a||^2 v v^T
            vTv = torch.einsum('t,tp,tq->pq', a_sq, v_h, v_h)  # (P, P)
            M_accum[h] += vTv

        del cache, y_ssd, x_gated, silu_z, a, rms_inv, rms_sq

    Omega_bar = M_accum / (N_samples * T_seq)
    return Omega_bar


# ---------------------------------------------------------------------------
# Utility: in_proj row slices
# ---------------------------------------------------------------------------

def get_inproj_slices(mixer):
    """Return dict of row slices for each sub-weight block of in_proj.

    in_proj layout: [z (d_ssm) | x (d_ssm) | B (ngroups*d_state) |
                     C (ngroups*d_state) | dt (nheads)]

    Args:
        mixer: Mamba2 mixer module

    Returns:
        dict with keys 'z', 'x', 'B', 'C', 'dt', each mapping to a slice
        over the output (row) dimension of in_proj.weight.
    """
    d_ssm = mixer.d_ssm
    ngroups = mixer.ngroups
    d_state = mixer.d_state
    nheads = mixer.nheads

    ng_ns = ngroups * d_state
    offset = 0
    slices = {}

    slices['z'] = slice(offset, offset + d_ssm)
    offset += d_ssm

    slices['x'] = slice(offset, offset + d_ssm)
    offset += d_ssm

    slices['B'] = slice(offset, offset + ng_ns)
    offset += ng_ns

    slices['C'] = slice(offset, offset + ng_ns)
    offset += ng_ns

    slices['dt'] = slice(offset, offset + nheads)

    return slices


# ---------------------------------------------------------------------------
# Forward state scan with on-the-fly G_Q accumulation
# ---------------------------------------------------------------------------

def forward_state_scan(x_dt, B, gamma, ngroups, nheads):
    """Forward state recurrence with on-the-fly G_Q accumulation.

    Computes the forward state recursion:
        S_t = gamma_t * S_{t-1} + v_t (x) k_t^T

    where v_t = x_dt[t, h] (headdim,) and k_t = B[t, g_of_h[h]] (d_state,).

    Accumulates G_Q = (1/T) sum_t S_t^T @ S_t without storing all S_t.

    Args:
        x_dt: (T, H, P) -- dt-scaled SSM input (v_t per head)
        B: (T, G, N) -- input-to-state projection (k_t per group)
        gamma: (T, H) -- transition decay per head

    Returns:
        G_Q: (H, N, N) float32 -- query (C) factor, (1/T) sum_t S_t^T S_t
        S_last: (H, P, N) float32 -- final state S_T (for diagnostics)
    """
    T, H, P = x_dt.shape
    _, G, N = B.shape
    device = x_dt.device

    g_of_h = torch.arange(H, device=device) // (H // ngroups)

    # State: (H, P, N) in float32, accumulator in float64
    S = torch.zeros(H, P, N, device=device, dtype=torch.float32)
    G_Q_accum = torch.zeros(H, N, N, device=device, dtype=torch.float64)

    for t in range(T):
        gamma_t = gamma[t].unsqueeze(-1).unsqueeze(-1)  # (H, 1, 1)
        v_t = x_dt[t]           # (H, P)
        k_t = B[t, g_of_h]     # (H, N)

        # S_t = gamma_t * S_{t-1} + v_t (x) k_t^T
        S = gamma_t * S + torch.einsum('hp,hn->hpn', v_t, k_t)

        # Accumulate S_t^T @ S_t = (N, P) @ (P, N) -> (H, N, N)
        G_Q_accum += torch.einsum('hpn,hpm->hnm', S.double(), S.double())

    G_Q = (G_Q_accum / T).float()
    S_last = S.clone()

    return G_Q, S_last


# ---------------------------------------------------------------------------
# Gramian-weighted column Hessians (per-block weighted input covariance)
# ---------------------------------------------------------------------------

def compute_weighted_hessians(mixer, inputs, nsamples=None):
    """Compute per-block weighted column Hessians using reverse Gramian sensitivity.

    Standard GPTQ uses A = (1/NT) sum_t a_t a_t^T (uniform token weighting).
    This function computes per-block weighted versions:

        A_x = (1/Z) sum_t w_x(t) * a_t a_t^T
        A_B = (1/Z) sum_t w_B(t) * a_t a_t^T
        A_C = (1/Z) sum_t w_C(t) * a_t a_t^T

    where the per-token weights come from the reverse Gramian:
        w_x(t) = sum_h k_t^T M_t k_t          (B-Gramian quadratic form)
        w_B(t) = sum_h ||x_dt_t||^2 * tr(M_t)  (value norm * Gramian trace)
        w_C(t) = sum_h ||S_t||_F^2             (state energy)

    Tokens where the recurrence is more sensitive get higher weight,
    so the GPTQ column solver focuses on minimizing error at those tokens.

    Args:
        mixer: Mamba2 mixer module
        inputs: (N_samples, T, d_model) calibration data
        nsamples: int or None

    Returns dict with:
        A_uniform: (d_model, d_model) — standard GPTQ Hessian (for z/dt rows)
        A_x: (d_model, d_model) — weighted Hessian for x-rows
        A_B: (d_model, d_model) — weighted Hessian for B-rows
        A_C: (d_model, d_model) — weighted Hessian for C-rows
        nheads, ngroups, headdim, d_state, d_ssm — layout metadata
    """
    N_total = inputs.shape[0]
    N = nsamples if nsamples is not None else N_total
    N = min(N, N_total)

    nheads = mixer.nheads
    ngroups = mixer.ngroups
    d_state = mixer.d_state
    headdim = mixer.headdim
    d_model = inputs.shape[2]
    device = inputs.device

    A_uniform_accum = torch.zeros(d_model, d_model, device=device, dtype=torch.float64)
    A_x_accum = torch.zeros(d_model, d_model, device=device, dtype=torch.float64)
    A_B_accum = torch.zeros(d_model, d_model, device=device, dtype=torch.float64)
    A_C_accum = torch.zeros(d_model, d_model, device=device, dtype=torch.float64)

    w_x_total = 0.0
    w_B_total = 0.0
    w_C_total = 0.0
    n_tokens_total = 0

    logger.info("Computing Gramian-weighted column Hessians over %d samples", N)

    g_of_h = torch.arange(nheads, device=device) // (nheads // ngroups)

    for s in range(N):
        u = inputs[s].detach()  # (T, d_model)
        cache = extract_ssd_cache(u, mixer)

        T_seq = cache['gamma'].shape[0]
        gamma = cache['gamma']     # (T, H)
        B_s = cache['B']           # (T, G, N)
        C_s = cache['C']           # (T, G, N)
        x_dt = cache['x_dt']      # (T, H, P)

        # Reverse Gramian: M_t (T, H, N, N)
        M = reverse_gramian_scan(C_s, gamma, ngroups, nheads)

        # Forward state scan for ||S_t||_F^2
        S = torch.zeros(nheads, headdim, d_state, device=device, dtype=torch.float32)
        w_C_per_t = torch.zeros(T_seq, device=device)

        # Per-token weights
        w_x_per_t = torch.zeros(T_seq, device=device)
        w_B_per_t = torch.zeros(T_seq, device=device)

        # w_B(t) = sum_h ||x_dt_t,h||^2 * trace(M_t,h)
        xdt_sq = x_dt.pow(2).sum(dim=-1)  # (T, H)
        M_trace = M.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (T, H)
        w_B_per_t = (xdt_sq * M_trace).sum(dim=-1)  # (T,)

        # w_x(t) = sum_h k_t^T M_t k_t
        k_t = B_s[:, g_of_h]  # (T, H, N)
        Mk = torch.einsum('thnm,thm->thn', M, k_t)
        kMk = torch.einsum('thn,thn->th', k_t, Mk)
        w_x_per_t = kMk.sum(dim=-1)  # (T,)

        # w_C(t) = sum_h ||S_t||_F^2 — need forward scan
        for t in range(T_seq):
            gamma_t = gamma[t].unsqueeze(-1).unsqueeze(-1)
            v_t = x_dt[t]
            kt = B_s[t, g_of_h]
            S = gamma_t * S + torch.einsum('hp,hn->hpn', v_t, kt)
            w_C_per_t[t] = S.pow(2).sum().item()

        # Accumulate weighted Hessians
        a = u.float().double()  # (T, d_model)
        # Uniform
        A_uniform_accum += a.T @ a

        # Weighted: A_block = sum_t w(t) * a_t a_t^T
        # Efficient: (a * sqrt(w))^T @ (a * sqrt(w)) but w can be negative in theory
        # Safer: a^T @ diag(w) @ a = (a * w.unsqueeze(-1))^T @ a
        A_x_accum += (a * w_x_per_t.double().unsqueeze(-1)).T @ a
        A_B_accum += (a * w_B_per_t.double().unsqueeze(-1)).T @ a
        A_C_accum += (a * w_C_per_t.double().unsqueeze(-1)).T @ a

        w_x_total += w_x_per_t.sum().item()
        w_B_total += w_B_per_t.sum().item()
        w_C_total += w_C_per_t.sum().item()
        n_tokens_total += T_seq

        del cache, M, S, Mk, kMk

    # Normalize
    A_uniform = (A_uniform_accum / (N * T_seq)).float()
    A_x = (A_x_accum / max(w_x_total, 1e-30)).float()
    A_B = (A_B_accum / max(w_B_total, 1e-30)).float()
    A_C = (A_C_accum / max(w_C_total, 1e-30)).float()

    logger.info("Weighted Hessians computed: w_x_total=%.2e, w_B_total=%.2e, w_C_total=%.2e",
                w_x_total, w_B_total, w_C_total)

    # Log how different the weighted Hessians are from uniform
    for name, Aw in [('A_x', A_x), ('A_B', A_B), ('A_C', A_C)]:
        diff = torch.norm(Aw - A_uniform).item() / torch.norm(A_uniform).item()
        logger.info("  %s vs A_uniform: rel_diff=%.4f", name, diff)

    return {
        'A_uniform': A_uniform,
        'A_x': A_x,
        'A_B': A_B,
        'A_C': A_C,
        'nheads': nheads,
        'ngroups': ngroups,
        'headdim': headdim,
        'd_state': d_state,
        'd_ssm': mixer.d_ssm,
    }


# ---------------------------------------------------------------------------
# Pre+core factor computation (plan Sections 8.2-8.3)
# ---------------------------------------------------------------------------

def compute_precore_factors(mixer, inputs, nsamples=None):
    """Compute pre+core K-FAC factors for Mamba-2 xBC quantization.

    For each calibration sample:
        1. extract_ssd_cache -> gamma, B, C, x_dt
        2. reverse_gramian_scan -> M_t (T, H, N, N)
        3. forward_state_scan -> G_Q (H, N, N)
        4. G_K = (1/T) sum_t ||v_t||^2 * M_t
        5. G_V_scalar = (1/T) sum_t k_t^T M_t k_t
    Then average across samples.

    Args:
        mixer: Mamba2 mixer module
        inputs: (N_samples, T, d_model) calibration data
        nsamples: int or None -- number of samples to use (None = all)

    Returns:
        dict with:
            G_Q: (H, N, N) -- query (C) factor per head
            G_K: (H, N, N) -- key (B) factor per head
            G_V_scalar: (H,) -- value (x) factor per head (scalar, G_V = scalar * I_P)
            G_Q_group: (G, N, N) -- C-factor aggregated per group
            G_K_group: (G, N, N) -- B-factor aggregated per group
            A: (d_model, d_model) -- input covariance
            metadata: dict with nheads, ngroups, headdim, d_state
    """
    N_total = inputs.shape[0]
    N = nsamples if nsamples is not None else N_total
    N = min(N, N_total)

    nheads = mixer.nheads
    ngroups = mixer.ngroups
    headdim = mixer.headdim
    d_state = mixer.d_state
    device = inputs.device

    g_of_h = torch.arange(nheads, device=device) // (nheads // ngroups)

    G_Q_accum = torch.zeros(nheads, d_state, d_state, device=device, dtype=torch.float32)
    G_K_accum = torch.zeros(nheads, d_state, d_state, device=device, dtype=torch.float32)
    G_V_scalar_accum = torch.zeros(nheads, device=device, dtype=torch.float32)

    logger.info("Computing pre+core factors over %d samples (H=%d, G=%d, P=%d, N=%d)",
                N, nheads, ngroups, headdim, d_state)

    for s in range(N):
        u = inputs[s].detach()
        cache = extract_ssd_cache(u, mixer)

        gamma = cache['gamma']   # (T, H)
        B_s = cache['B']         # (T, G, N_state)
        C_s = cache['C']         # (T, G, N_state)
        x_dt = cache['x_dt']    # (T, H, P)

        T_seq = gamma.shape[0]

        # (2) Reverse Gramian: M_t (T, H, N, N)
        M = reverse_gramian_scan(C_s, gamma, ngroups, nheads)

        # (3) Forward state scan: G_Q (H, N, N)
        G_Q_s, _ = forward_state_scan(x_dt, B_s, gamma, ngroups, nheads)
        G_Q_accum += G_Q_s

        # (4) G_K = (1/T) sum_t ||v_t||^2 * M_t
        v_norm_sq = x_dt.pow(2).sum(dim=-1)  # (T, H)
        G_K_s = (v_norm_sq.unsqueeze(-1).unsqueeze(-1) * M).sum(dim=0) / T_seq
        G_K_accum += G_K_s

        # (5) G_V_scalar = (1/T) sum_t k_t^T M_t k_t
        k_t = B_s[:, g_of_h]  # (T, H, N)
        Mk = torch.einsum('thnm,thm->thn', M, k_t)  # (T, H, N)
        kMk = torch.einsum('thn,thn->th', k_t, Mk)   # (T, H)
        G_V_scalar_s = kMk.sum(dim=0) / T_seq         # (H,)
        G_V_scalar_accum += G_V_scalar_s

        del cache, M, Mk, kMk, G_Q_s, G_K_s

    # Average across samples
    G_Q = G_Q_accum / N
    G_K = G_K_accum / N
    G_V_scalar = G_V_scalar_accum / N

    # Per-group aggregated factors (for B/C rows shared across heads in a group)
    heads_per_group = nheads // ngroups
    G_Q_group = torch.zeros(ngroups, d_state, d_state, device=device, dtype=torch.float32)
    G_K_group = torch.zeros(ngroups, d_state, d_state, device=device, dtype=torch.float32)

    for g in range(ngroups):
        h_start = g * heads_per_group
        h_end = h_start + heads_per_group
        G_Q_group[g] = G_Q[h_start:h_end].sum(dim=0)
        G_K_group[g] = G_K[h_start:h_end].sum(dim=0)

    # Input covariance
    A = compute_input_factor(inputs[:N])

    logger.info("Pre+core factors computed: G_Q %s, G_K %s, G_V_scalar %s",
                G_Q.shape, G_K.shape, G_V_scalar.shape)

    return {
        'G_Q': G_Q,
        'G_K': G_K,
        'G_V_scalar': G_V_scalar,
        'G_Q_group': G_Q_group,
        'G_K_group': G_K_group,
        'A': A,
        'nheads': nheads,
        'ngroups': ngroups,
        'headdim': headdim,
        'd_state': d_state,
        'd_ssm': mixer.d_ssm,
    }


# ---------------------------------------------------------------------------
# Full-cut factor computation (shared-Omega approximation, plan Section 8.5)
# ---------------------------------------------------------------------------

def compute_full_factors(mixer, inputs, nsamples=None):
    """Full-cut K-FAC factors using the shared-Omega approximation.

    Extends the pre+core factors with tail-aware versions:
        G_Q_full[h] = (1/T) sum_t S_t^T @ Omega_bar_h @ S_t
        G_K_full[h] = (1/T) sum_t (v_t^T Omega_bar_h v_t) * M_t
        G_V_full[h] = (1/T) sum_t (k_t^T M_t k_t) * Omega_bar_h

    Args:
        mixer: Mamba2 mixer module
        inputs: (N_samples, T, d_model) calibration data
        nsamples: int or None -- number of samples to use (None = all)

    Returns:
        dict extending compute_precore_factors with:
            G_Q_full: (H, N, N), G_K_full: (H, N, N), G_V_full: (H, P, P),
            F_tail: (H, P, P)
    """
    N_total = inputs.shape[0]
    N = nsamples if nsamples is not None else N_total
    N = min(N, N_total)

    nheads = mixer.nheads
    ngroups = mixer.ngroups
    headdim = mixer.headdim
    d_state = mixer.d_state
    device = inputs.device

    g_of_h = torch.arange(nheads, device=device) // (nheads // ngroups)

    logger.info("Computing tail factor (Omega_bar) for full-cut factors...")
    F_tail = compute_tail_factor(mixer, inputs[:N])  # (H, P, P)

    precore = compute_precore_factors(mixer, inputs, nsamples=nsamples)

    G_Q_full_accum = torch.zeros(nheads, d_state, d_state, device=device, dtype=torch.float32)
    G_K_full_accum = torch.zeros(nheads, d_state, d_state, device=device, dtype=torch.float32)
    G_V_full_accum = torch.zeros(nheads, headdim, headdim, device=device, dtype=torch.float32)

    logger.info("Computing full-cut factors over %d samples...", N)

    for s in range(N):
        u = inputs[s].detach()
        cache = extract_ssd_cache(u, mixer)

        gamma = cache['gamma']
        B_s = cache['B']
        C_s = cache['C']
        x_dt = cache['x_dt']

        T_seq = gamma.shape[0]

        M = reverse_gramian_scan(C_s, gamma, ngroups, nheads)

        # G_Q_full: forward scan with Omega_bar inserted
        S = torch.zeros(nheads, headdim, d_state, device=device, dtype=torch.float32)
        G_Q_full_s = torch.zeros(nheads, d_state, d_state, device=device, dtype=torch.float64)

        for t in range(T_seq):
            gamma_t = gamma[t].unsqueeze(-1).unsqueeze(-1)
            v_t = x_dt[t]
            k_t = B_s[t, g_of_h]
            S = gamma_t * S + torch.einsum('hp,hn->hpn', v_t, k_t)
            Omega_S = torch.einsum('hpq,hqn->hpn', F_tail, S)
            G_Q_full_s += torch.einsum('hpn,hpm->hnm', S.double(), Omega_S.double())

        G_Q_full_accum += (G_Q_full_s / T_seq).float()

        # G_K_full: (v^T Omega_bar v) * M_t
        Omega_v = torch.einsum('hpq,thq->thp', F_tail, x_dt)
        v_Omega_v = torch.einsum('thp,thp->th', x_dt, Omega_v)
        G_K_full_accum += (v_Omega_v.unsqueeze(-1).unsqueeze(-1) * M).sum(dim=0) / T_seq

        # G_V_full: (k^T M k) * Omega_bar
        k_t_all = B_s[:, g_of_h]
        Mk = torch.einsum('thnm,thm->thn', M, k_t_all)
        kMk = torch.einsum('thn,thn->th', k_t_all, Mk)
        G_V_full_accum += kMk.mean(dim=0).unsqueeze(-1).unsqueeze(-1) * F_tail

        del cache, M, S, Mk, kMk, Omega_v, v_Omega_v, G_Q_full_s

    G_Q_full = G_Q_full_accum / N
    G_K_full = G_K_full_accum / N
    G_V_full = G_V_full_accum / N

    logger.info("Full-cut factors computed: G_Q_full %s, G_K_full %s, G_V_full %s",
                G_Q_full.shape, G_K_full.shape, G_V_full.shape)

    result = dict(precore)
    result.update({
        'G_Q_full': G_Q_full,
        'G_K_full': G_K_full,
        'G_V_full': G_V_full,
        'F_tail': F_tail,
    })
    return result

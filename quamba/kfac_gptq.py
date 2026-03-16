"""KFACQuantizer: Generalized Kronecker-aware GPTQ for Mamba-2 in_proj.

Quantizes each sub-weight block of in_proj using the appropriate Kronecker
Hessian structure:

  - z rows: standard column GPTQ (H = A)
  - x rows: scalar-reweighted column GPTQ (G_V is scalar x I per head),
             or full 2D Kronecker GPTQ with dense G_V (optional)
  - B rows: 2D Kronecker GPTQ with per-group G_K (N x N)
  - C rows: 2D Kronecker GPTQ with per-group G_Q (N x N)
  - dt rows: standard column GPTQ (H = A)

The matrix objective for a Kronecker block is:
  L(Q) = tr(G * E * A * E^T)
where E = W - Q.

The 2D solver (following LRGPTQ / BoA):
  1. Cholesky decompose G = L_G L_G^T, then U_row = chol(G^{-1}, upper=True)
  2. Reshape block weights: W_block -> (n_units, M, d_in)
  3. Outer loop over row index p = 0..M-1:
     a. Column GPTQ on row p across all units with Hinv
     b. Row propagation: for p' > p, adjust via U_row scale factors
"""

import gc
import logging
import math

import torch
import torch.nn as nn

from quamba.gptq_utils import get_per_channel_scale, quant

logger = logging.getLogger(__name__)


def get_inproj_slices(mixer):
    """Return dict of row slices for each sub-weight block of in_proj.

    The in_proj layout is:
        [z (d_ssm) | x (d_ssm) | B (ngroups*d_state) | C (ngroups*d_state) | dt (nheads)]

    Args:
        mixer: Mamba2 mixer module with d_ssm, ngroups, d_state, nheads attrs

    Returns:
        dict mapping block name -> slice into the output (row) dimension
    """
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


def _compute_column_hinv(A, in_dim, device, percdamp=0.01):
    """Compute upper-Cholesky of the damped inverse input covariance.

    This is the column factor shared by all blocks, identical to the
    Hessian preprocessing in GPTQ.fasterquant().

    Args:
        A: (d_model, d_model) input covariance matrix
        in_dim: int, expected dimension (= d_model)
        device: torch device
        percdamp: damping as fraction of mean diagonal

    Returns:
        Hinv: (d_model, d_model) upper-triangular Cholesky of (A + damp*I)^{-1}
        dead: (d_model,) bool mask of dead (zero-variance) input channels
    """
    H = A.to(device).float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0

    damp = percdamp * torch.mean(torch.diag(H))
    if damp == 0:
        damp = torch.tensor(percdamp, device=device)
    diag_idx = torch.arange(in_dim, device=device)
    H[diag_idx, diag_idx] += damp

    # cholesky -> inverse -> cholesky of inverse (upper triangular)
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    return H, dead


def _compute_row_U(G, n_units, unit_dim, device, reg_eps=1e-6, trace_frac=0.99):
    """Compute upper-Cholesky of G^{-1} per unit with rank truncation.

    Uses truncated eigendecomposition: only inverts eigenvalues that
    cumulatively explain trace_frac of the total trace. Directions in
    the null/noise space get identity (no row propagation), preventing
    the solver from amplifying noise in rank-deficient factors.

    Args:
        G: (n_units, unit_dim, unit_dim) PSD row-side Gramian per unit
        n_units: int, number of heads or groups
        unit_dim: int, head dimension or state dimension
        device: torch device
        reg_eps: minimum eigenvalue floor as fraction of max eigenvalue
        trace_frac: fraction of trace to retain (default 0.99)

    Returns:
        U_row: (n_units, unit_dim, unit_dim) upper-triangular factor
    """
    U_row = torch.zeros(n_units, unit_dim, unit_dim,
                         device=device, dtype=torch.float32)
    for i in range(n_units):
        G_i = G[i].clone()
        G_i = 0.5 * (G_i + G_i.T)
        trace_val = torch.trace(G_i)
        if trace_val <= 0:
            logger.warning("Row factor unit %d has non-positive trace (%.2e); "
                           "using identity", i, trace_val.item())
            U_row[i] = torch.eye(unit_dim, device=device)
            continue

        eigvals, eigvecs = torch.linalg.eigh(G_i)

        # Truncate: keep top-k eigenvalues explaining trace_frac of the trace
        sorted_eigs, sort_idx = eigvals.sort(descending=True)
        cumsum = sorted_eigs.cumsum(0)
        total = sorted_eigs.sum()
        k = (cumsum < trace_frac * total).sum().item() + 1
        k = max(k, 1)
        threshold = sorted_eigs[min(k, len(sorted_eigs) - 1)]

        # Floor: eigenvalues below threshold get floored to
        # reg_eps * max_eigenvalue (so they contribute ~identity, not amplified noise)
        eig_max = eigvals.max()
        eig_floor = max(reg_eps * eig_max, threshold)
        eigvals_reg = eigvals.clamp(min=eig_floor)

        logger.debug("Row factor unit %d: rank=%d/%d (%.1f%% trace), "
                     "cond=%.1e -> %.1e", i, k, unit_dim,
                     100 * cumsum[min(k-1, len(cumsum)-1)] / total,
                     (eig_max / eigvals.clamp(min=1e-30).min()).item(),
                     (eig_max / eig_floor).item())

        # Build U_row via eigendecomposition + QR
        L = eigvecs @ torch.diag(eigvals_reg.pow(-0.5))
        Q_qr, R = torch.linalg.qr(L.T)
        signs = R.diagonal().sign()
        U_row[i] = R * signs.unsqueeze(0)
    return U_row


class KFACQuantizer:
    """Generalized Kronecker-aware GPTQ for Mamba-2 in_proj.

    Quantizes each weight block using the appropriate Kronecker Hessian:
      - z rows: standard column GPTQ (H = A)
      - x rows: Kronecker GPTQ with per-head G_V (scalar x I or dense P x P)
      - B rows: Kronecker GPTQ with per-group G_K (N x N)
      - C rows: Kronecker GPTQ with per-group G_Q (N x N)
      - dt rows: standard column GPTQ (H = A)

    The matrix objective for a Kronecker block is:
      L(Q) = tr(G * E * A * E^T)
    where E = W - Q.

    The solver uses the BoA/LRGPTQ approach:
      1. Cholesky decompose G = L_G L_G^T, then U_row = chol(G^{-1}, upper=True)
      2. Reshape block weights: W_block -> (n_units, M, d_in)
      3. Outer loop over row index p = 0..M-1:
         a. Column GPTQ on row p across all units with Hinv
         b. Row propagation: for p' > p, adjust via U_row scale factors

    Usage:
        quantizer = KFACQuantizer(mixer.in_proj)
        quantizer.set_factors(kfac_factors, percdamp=0.01)
        quantizer.fasterquant(group_size=128, w_bits=4)
        quantizer.free()
    """

    def __init__(self, layer):
        """Initialize the quantizer for a given linear layer.

        Args:
            layer: nn.Linear (the in_proj weight to quantize)
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"KFACQuantizer expects nn.Linear, got {type(layer).__name__}")
        self.layer = layer
        self.dev = layer.weight.device
        self.out_dim = layer.weight.shape[0]
        self.in_dim = layer.weight.shape[1]
        self._factors_set = False

    def set_factors(self, kfac_factors, percdamp=0.01, reg_eps=1e-6):
        """Configure Kronecker factors for each block of in_proj.

        Args:
            kfac_factors: dict with keys:
                A: (d_model, d_model) input covariance
                nheads: int
                ngroups: int
                headdim: int (P)
                d_state: int (N)
                d_ssm: int

                For x-rows (at least one of):
                  G_V_scalar: (H,) scalar per head  (standard GPTQ, no row prop)
                  G_V: (H, P, P) dense per-head factor (full 2D Kronecker)

                For B-rows (optional):
                  G_K: (H, N, N) per-head, or
                  G_K_group: (G, N, N) per-group

                For C-rows (optional):
                  G_Q: (H, N, N) per-head, or
                  G_Q_group: (G, N, N) per-group

            percdamp: damping for column Hessian inverse (fraction of mean diag)
            reg_eps: regularization for row-factor Cholesky stability
        """
        # Unpack layout dimensions
        self.nheads = kfac_factors['nheads']
        self.ngroups = kfac_factors['ngroups']
        self.headdim = kfac_factors['headdim']
        self.d_state = kfac_factors['d_state']
        self.d_ssm = kfac_factors['d_ssm']

        # Compute in_proj sub-block slices
        d = self.d_ssm
        ng_ns = self.ngroups * self.d_state
        nh = self.nheads
        self.slices = {
            'z':  slice(0, d),
            'x':  slice(d, 2 * d),
            'B':  slice(2 * d, 2 * d + ng_ns),
            'C':  slice(2 * d + ng_ns, 2 * d + 2 * ng_ns),
            'dt': slice(2 * d + 2 * ng_ns, 2 * d + 2 * ng_ns + nh),
        }

        # --- Column factor (shared across all blocks) ---
        A = kfac_factors['A']
        self.Hinv, self.dead = _compute_column_hinv(
            A, self.in_dim, self.dev, percdamp=percdamp)
        logger.info("Column factor Hinv computed (d_model=%d, percdamp=%.4f)",
                    self.in_dim, percdamp)

        # --- x-rows row factor ---
        self.use_dense_x_factor = False
        self.U_row_x = None

        if 'G_V' in kfac_factors and kfac_factors['G_V'] is not None:
            # Dense per-head factor -> full 2D Kronecker for x-rows
            G_V = kfac_factors['G_V'].to(self.dev).float()
            assert G_V.shape == (self.nheads, self.headdim, self.headdim), \
                f"G_V shape mismatch: {G_V.shape}"
            self.U_row_x = _compute_row_U(
                G_V, self.nheads, self.headdim, self.dev, reg_eps=reg_eps)
            self.use_dense_x_factor = True
            logger.info("x-rows: using dense G_V factor (2D Kronecker, P=%d)",
                        self.headdim)
        elif 'G_V_scalar' in kfac_factors and kfac_factors['G_V_scalar'] is not None:
            # Scalar per head -> standard column GPTQ (no row propagation)
            self.G_V_scalar = kfac_factors['G_V_scalar'].to(self.dev).float()
            logger.info("x-rows: using scalar G_V (standard column GPTQ)")
        else:
            logger.info("x-rows: no G_V factor provided, using standard column GPTQ")

        # --- B-rows row factor ---
        self.U_row_B = None
        G_K = None

        if 'G_K_group' in kfac_factors and kfac_factors['G_K_group'] is not None:
            G_K = kfac_factors['G_K_group'].to(self.dev).float()
            assert G_K.shape == (self.ngroups, self.d_state, self.d_state), \
                f"G_K_group shape mismatch: {G_K.shape}"
            self.U_row_B = _compute_row_U(
                G_K, self.ngroups, self.d_state, self.dev, reg_eps=reg_eps)
            logger.info("B-rows: using per-group G_K factor (2D Kronecker, N=%d, G=%d)",
                        self.d_state, self.ngroups)
        elif 'G_K' in kfac_factors and kfac_factors['G_K'] is not None:
            # Per-head factor: average within groups to get per-group
            G_K_per_head = kfac_factors['G_K'].to(self.dev).float()
            assert G_K_per_head.shape == (self.nheads, self.d_state, self.d_state), \
                f"G_K shape mismatch: {G_K_per_head.shape}"
            heads_per_group = self.nheads // self.ngroups
            G_K = G_K_per_head.reshape(
                self.ngroups, heads_per_group, self.d_state, self.d_state
            ).mean(dim=1)
            self.U_row_B = _compute_row_U(
                G_K, self.ngroups, self.d_state, self.dev, reg_eps=reg_eps)
            logger.info("B-rows: using per-head G_K averaged to groups "
                        "(2D Kronecker, N=%d, G=%d)", self.d_state, self.ngroups)
        else:
            logger.info("B-rows: no G_K factor provided, using standard column GPTQ")

        # --- C-rows row factor ---
        self.U_row_C = None
        G_Q = None

        if 'G_Q_group' in kfac_factors and kfac_factors['G_Q_group'] is not None:
            G_Q = kfac_factors['G_Q_group'].to(self.dev).float()
            assert G_Q.shape == (self.ngroups, self.d_state, self.d_state), \
                f"G_Q_group shape mismatch: {G_Q.shape}"
            self.U_row_C = _compute_row_U(
                G_Q, self.ngroups, self.d_state, self.dev, reg_eps=reg_eps)
            logger.info("C-rows: using per-group G_Q factor (2D Kronecker, N=%d, G=%d)",
                        self.d_state, self.ngroups)
        elif 'G_Q' in kfac_factors and kfac_factors['G_Q'] is not None:
            G_Q_per_head = kfac_factors['G_Q'].to(self.dev).float()
            assert G_Q_per_head.shape == (self.nheads, self.d_state, self.d_state), \
                f"G_Q shape mismatch: {G_Q_per_head.shape}"
            heads_per_group = self.nheads // self.ngroups
            G_Q = G_Q_per_head.reshape(
                self.ngroups, heads_per_group, self.d_state, self.d_state
            ).mean(dim=1)
            self.U_row_C = _compute_row_U(
                G_Q, self.ngroups, self.d_state, self.dev, reg_eps=reg_eps)
            logger.info("C-rows: using per-head G_Q averaged to groups "
                        "(2D Kronecker, N=%d, G=%d)", self.d_state, self.ngroups)
        else:
            logger.info("C-rows: no G_Q factor provided, using standard column GPTQ")

        self._factors_set = True
        logger.info("KFACQuantizer factors set: d_ssm=%d, nheads=%d, headdim=%d, "
                    "ngroups=%d, d_state=%d", self.d_ssm, self.nheads,
                    self.headdim, self.ngroups, self.d_state)

    def _column_gptq_group(self, W_rows, Hinv, i1, i2, bits):
        """Run standard column GPTQ on one column group.

        Processes columns i1..i2-1 with the standard GPTQ inner loop: quantize
        each column, compute error, and propagate within the group.

        Args:
            W_rows: (n_rows, d_in) weight sub-matrix, modified in-place for
                     columns >= i1
            Hinv: (d_in, d_in) upper-triangular Cholesky of A^{-1}
            i1: int, start column index
            i2: int, end column index (exclusive)
            bits: int, quantization bit width

        Returns:
            Q1: (n_rows, i2 - i1) quantized weights for this group
            Err1: (n_rows, i2 - i1) normalized errors (w - q) / d
            per_group_scale: (n_rows, 1) per-channel scale for this group
        """
        count = i2 - i1
        W1 = W_rows[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        per_group_scale = get_per_channel_scale(W1, num_bits=bits)

        for i in range(count):
            w = W1[:, i].clone()
            d = Hinv1[i, i]
            q = quant(w.unsqueeze(1), per_group_scale, num_bits=bits).flatten()
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        # Update W_rows in-place: propagate error to remaining columns
        W_rows[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        return Q1, Err1, per_group_scale

    def _quantize_standard_rows(self, W, row_indices, Hinv, group_size, bits,
                                 Q, group_scale):
        """Standard column GPTQ for rows without Kronecker structure (z, dt).

        Extracts the specified rows, runs column GPTQ across all column groups,
        and writes results into Q and group_scale.

        Args:
            W: (out_dim, d_in) full weight matrix (not modified)
            row_indices: 1D tensor of row indices to quantize
            Hinv: (d_in, d_in) upper-triangular Cholesky of A^{-1}
            group_size: int, column group size
            bits: int, quantization bit width
            Q: (out_dim, d_in) output quantized weight, modified in-place
            group_scale: (n_groups, out_dim) output scale, modified in-place
        """
        if len(row_indices) == 0:
            return

        W_sub = W[row_indices].clone()

        for i1 in range(0, self.in_dim, group_size):
            gidx = i1 // group_size
            i2 = min(i1 + group_size, self.in_dim)

            Q1, _, per_group_scale = self._column_gptq_group(
                W_sub, Hinv, i1, i2, bits)

            Q[row_indices, i1:i2] = Q1
            for local_idx in range(len(row_indices)):
                group_scale[gidx, row_indices[local_idx]] = per_group_scale[local_idx, 0]

    def _quantize_kronecker_block(self, W, block_slice, n_units, unit_dim,
                                   U_row, Hinv, group_size, bits, Q,
                                   group_scale):
        """2D Kronecker GPTQ for a block with dense row factor.

        For each row index p in 0..unit_dim-1:
          1. Extract row p from all units: shape (n_units, d_in)
          2. Run column GPTQ (inner loop over column groups)
          3. Row propagation: adjust rows p+1..unit_dim-1 using U_row

        Args:
            W: (out_dim, d_in) full weight matrix, NOT modified
            block_slice: slice into the row dimension for this block
            n_units: int, number of heads (x) or groups (B, C)
            unit_dim: int, headdim (x) or d_state (B, C)
            U_row: (n_units, unit_dim, unit_dim) upper Cholesky of G^{-1}
            Hinv: (d_in, d_in) upper-triangular Cholesky of A^{-1}
            group_size: int, column group size
            bits: int, quantization bit width
            Q: (out_dim, d_in) output quantized weight, modified in-place
            group_scale: (n_groups, out_dim) output scale, modified in-place
        """
        block_start = block_slice.start
        block_size = block_slice.stop - block_slice.start
        assert block_size == n_units * unit_dim, \
            f"Block size {block_size} != n_units*unit_dim {n_units}*{unit_dim}"

        # Reshape block weights: (n_units, unit_dim, d_in)
        W_block = W[block_slice].clone().reshape(n_units, unit_dim, self.in_dim)
        Q_block = torch.zeros_like(W_block)

        for p in range(unit_dim):
            # Row p from all units: (n_units, d_in)
            W_p = W_block[:, p, :].clone()
            Q_p = torch.zeros(n_units, self.in_dim,
                              device=self.dev, dtype=torch.float32)
            err_full = torch.zeros(n_units, self.in_dim,
                                   device=self.dev, dtype=torch.float32)

            for i1 in range(0, self.in_dim, group_size):
                gidx = i1 // group_size
                i2 = min(i1 + group_size, self.in_dim)

                Q1, Err1, per_group_scale = self._column_gptq_group(
                    W_p, Hinv, i1, i2, bits)

                Q_p[:, i1:i2] = Q1
                err_full[:, i1:i2] = Err1

                # Write per-group scales for all units at row p
                for u_idx in range(n_units):
                    global_row = block_start + u_idx * unit_dim + p
                    group_scale[gidx, global_row] = per_group_scale[u_idx, 0]

            Q_block[:, p, :] = Q_p

            # Row propagation: adjust rows p+1..unit_dim-1
            if p < unit_dim - 1:
                # E_p = err_full @ Hinv: full error projected through column factor
                E_p = err_full @ Hinv  # (n_units, d_in)

                # scale[u, p'] = U_row[u, p, p'] / U_row[u, p, p]
                # U_row shape: (n_units, unit_dim, unit_dim)
                diag_p = U_row[:, p, p].unsqueeze(-1)  # (n_units, 1)
                scale = U_row[:, p, p + 1:] / diag_p   # (n_units, unit_dim - p - 1)

                # W_block[:, p+1:, :] -= scale[:, :, None] * E_p[:, None, :]
                W_block[:, p + 1:, :] -= scale.unsqueeze(-1) * E_p.unsqueeze(1)

        # Write back quantized block
        Q[block_slice] = Q_block.reshape(block_size, self.in_dim)

    @torch.no_grad()
    def fasterquant(self, group_size=128, w_bits=4):
        """Kronecker-aware GPTQ quantization for full in_proj.

        Dispatches each block to the appropriate solver:
          - z, dt: standard column GPTQ
          - x: standard or 2D Kronecker depending on use_dense_x_factor
          - B: 2D Kronecker if G_K provided, else standard
          - C: 2D Kronecker if G_Q provided, else standard

        Args:
            group_size: int, column group size for quantization
            w_bits: int, quantization bit width (default 4)
        """
        if not self._factors_set:
            raise RuntimeError("Call set_factors() before fasterquant()")

        bits = w_bits
        W = self.layer.weight.data.clone().float()
        W[:, self.dead] = 0

        Hinv = self.Hinv
        n_groups = math.ceil(self.in_dim / group_size)
        group_scale = torch.zeros(
            n_groups, self.out_dim, dtype=torch.float32, device=self.dev)
        Q = torch.zeros_like(W)

        # ---- z-rows: standard column GPTQ ----
        z_slice = self.slices['z']
        z_rows = torch.arange(z_slice.start, z_slice.stop, device=self.dev)
        logger.info("Quantizing z-rows [%d:%d] with standard GPTQ",
                    z_slice.start, z_slice.stop)
        self._quantize_standard_rows(W, z_rows, Hinv, group_size, bits,
                                      Q, group_scale)

        # ---- x-rows ----
        x_slice = self.slices['x']
        if self.use_dense_x_factor and self.U_row_x is not None:
            logger.info("Quantizing x-rows [%d:%d] with 2D Kronecker GPTQ "
                        "(dense G_V, P=%d, H=%d)",
                        x_slice.start, x_slice.stop,
                        self.headdim, self.nheads)
            self._quantize_kronecker_block(
                W, x_slice, self.nheads, self.headdim,
                self.U_row_x, Hinv, group_size, bits, Q, group_scale)
        else:
            # Scalar G_V or no factor -> standard column GPTQ
            x_rows = torch.arange(x_slice.start, x_slice.stop, device=self.dev)
            logger.info("Quantizing x-rows [%d:%d] with standard column GPTQ "
                        "(scalar G_V)", x_slice.start, x_slice.stop)
            self._quantize_standard_rows(W, x_rows, Hinv, group_size, bits,
                                          Q, group_scale)

        # ---- B-rows ----
        b_slice = self.slices['B']
        if self.U_row_B is not None:
            logger.info("Quantizing B-rows [%d:%d] with 2D Kronecker GPTQ "
                        "(G_K, N=%d, G=%d)",
                        b_slice.start, b_slice.stop,
                        self.d_state, self.ngroups)
            self._quantize_kronecker_block(
                W, b_slice, self.ngroups, self.d_state,
                self.U_row_B, Hinv, group_size, bits, Q, group_scale)
        else:
            b_rows = torch.arange(b_slice.start, b_slice.stop, device=self.dev)
            logger.info("Quantizing B-rows [%d:%d] with standard column GPTQ",
                        b_slice.start, b_slice.stop)
            self._quantize_standard_rows(W, b_rows, Hinv, group_size, bits,
                                          Q, group_scale)

        # ---- C-rows ----
        c_slice = self.slices['C']
        if self.U_row_C is not None:
            logger.info("Quantizing C-rows [%d:%d] with 2D Kronecker GPTQ "
                        "(G_Q, N=%d, G=%d)",
                        c_slice.start, c_slice.stop,
                        self.d_state, self.ngroups)
            self._quantize_kronecker_block(
                W, c_slice, self.ngroups, self.d_state,
                self.U_row_C, Hinv, group_size, bits, Q, group_scale)
        else:
            c_rows = torch.arange(c_slice.start, c_slice.stop, device=self.dev)
            logger.info("Quantizing C-rows [%d:%d] with standard column GPTQ",
                        c_slice.start, c_slice.stop)
            self._quantize_standard_rows(W, c_rows, Hinv, group_size, bits,
                                          Q, group_scale)

        # ---- dt-rows: standard column GPTQ ----
        dt_slice = self.slices['dt']
        dt_rows = torch.arange(dt_slice.start, dt_slice.stop, device=self.dev)
        logger.info("Quantizing dt-rows [%d:%d] with standard GPTQ",
                    dt_slice.start, dt_slice.stop)
        self._quantize_standard_rows(W, dt_rows, Hinv, group_size, bits,
                                      Q, group_scale)

        # ---- Finalize ----
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.weight.data = Q.contiguous()
        self.layer.apply_gptq = True
        self.layer.bits = bits
        self.layer.group_size = group_size
        self.layer.group_scale = group_scale

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("KFACQuantizer: quantization complete (%d-bit, group_size=%d)",
                    bits, group_size)

    def free(self):
        """Release GPU memory held by factors."""
        self.Hinv = None
        self.dead = None
        self.U_row_x = None
        self.U_row_B = None
        self.U_row_C = None
        self._factors_set = False
        torch.cuda.empty_cache()
        gc.collect()

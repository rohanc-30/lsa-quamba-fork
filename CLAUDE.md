# K-FAC PTQ for Mamba-2

High-level codebase organization for the K-FAC post-training quantization pipeline.
For math derivations, analysis notes, and experiment logs, see `docs/`.

## What this is

Module-aware PTQ for Mamba-2 that quantizes the **xBC rows of in_proj** using a
K-FAC Gauss-Newton approximation H ≈ A ⊗ G, while leaving all other weights
on the standard GPTQ path.

- A = input covariance (standard GPTQ)
- G = per-tensor-type output-side factor from the SSD recurrence:
  - G_Q (for C rows): from forward state accumulation S_t^T S_t
  - G_K (for B rows): ||v_t||^2 * M_t (reverse Gramian weighted by value norm)
  - G_V (for x rows): k_t^T M_t k_t (scalar per head, or dense via full-cut Omega_bar)

Notation mapping (plan -> Mamba-2):
  q -> C, k -> B, v -> x_dt, alpha -> gamma = exp(A*dt), M -> reverse Gramian

## File organization

### Core pipeline (our code)
- `quamba/lr_kfac.py` — All primitives and factors in one file:
  - Pure-PyTorch mixer forward (`functional_mixer_forward`, `rms_norm_gated`)
  - SSD intermediate extraction (`extract_ssd_cache`)
  - Reverse Gramian (`reverse_gramian_scan` = plan's M_t)
  - Input covariance (`compute_input_factor` = plan's A)
  - Tail metric (`compute_tail_factor` = plan's Omega_bar)
  - Forward state scan (`forward_state_scan` = G_Q accumulation)
  - Pre+core factors (`compute_precore_factors` -> G_Q, G_K, G_V_scalar)
  - Full-cut factors (`compute_full_factors` -> G_Q_full, G_K_full, G_V_full with shared Omega_bar)
  - Utility (`get_inproj_slices`)
- `quamba/kfac_gptq.py` — KFACQuantizer: Kronecker-aware GPTQ solver (BoA-style 2D coordinate descent)
  - z/dt rows: standard column GPTQ
  - x rows: standard (scalar G_V) or 2D Kronecker (dense G_V)
  - B rows: 2D Kronecker with per-group G_K (N x N)
  - C rows: 2D Kronecker with per-group G_Q (N x N)
  - Objective: tr(G * E * A * E^T) where E = W - Q

### Diagnostic script (our code)
- `scripts/gn_taxonomy.py` — Part I approximation taxonomy study (all 5 levels):
  - Level 0->1: Cross-token shared-weight coupling
  - Level 1->2: Single-Kronecker collapse quality
  - Level 3a: Cross-weight (x/B/C) coupling
  - Level 3b: Inter-head coupling within x-block
  - Level 3c: Intra-head row coupling

### Upstream (from Quamba, mostly unmodified)
- `quamba/gptq_utils.py` — Base GPTQ class, `get_per_channel_scale`, `quant`
- `quamba/modelutils_mamba.py` — Model loading, calibration data, `apply_gptq()` integration
- `quamba/qMamba2.py`, `quamba/qLinearLayer.py` — Quantized layer implementations

### Documentation
- `docs/` — Math derivations, analysis notes, experiment logs, and related reference material
- `implementation.md` — Full implementation plan for the K-FAC PTQ pipeline

## in_proj layout

`[z (d_ssm) | x (d_ssm) | B (ngroups*d_state) | C (ngroups*d_state) | dt (nheads)]`

- z: gate (silu, does NOT go through SSM) — vanilla GPTQ
- x: SSM input path — K-FAC with G_V (scalar or dense)
- B: input-to-state — K-FAC with G_K (N x N per group)
- C: state-to-output — K-FAC with G_Q (N x N per group)
- dt: discretization step — vanilla GPTQ

## Model dimensions (mamba2-130m)

d_model=768, d_ssm=1536, nheads=24, headdim=64 (P), d_state=128 (N),
ngroups=1, d_conv=4, chunk_size=256, expand=2

## Conventions

- Scripts import quamba modules via `importlib.util` to avoid CUDA extension deps in `quamba/__init__.py`
- `gptq_utils.py` is upstream code — don't modify
- Factor names: G_Q, G_K, G_V_scalar, Omega_bar, M_t
- Class name: KFACQuantizer
- Function names: compute_precore_factors, compute_full_factors, compute_tail_factor

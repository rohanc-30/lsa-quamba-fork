# K-FAC Weight Quantization for Mamba-2: Findings

## Summary

We attempted to improve post-training weight quantization (PTQ) for Mamba-2's
`in_proj` weight by using a module-aware Hessian derived from the SSD recurrence
via a reverse Gramian. The approach was inspired by BoA (Kim et al., 2024), which
achieves significant improvements over GPTQ for attention layers in Transformers
by exploiting the Kronecker structure H = A ⊗ G of the attention-aware Hessian.

**Main result:** The approach does not improve over vanilla GPTQ for Mamba-2 weight
quantization at 3-bit and 4-bit precision. At 2-bit, a Gramian-based weight rotation
gives a modest improvement (-2.2 PPL). The core K-FAC Kronecker solver (BoA-style
2D coordinate descent) is strictly worse than GPTQ at all bit widths.

## What we built

### Reverse Gramian factor extraction (`lr_kfac.py`)

Exact closed-form output-side factors for the SSD recurrence:

- **G_Q** (for C/query rows): `(1/T) Σ_t S_t^T S_t` via forward state scan
- **G_K** (for B/key rows): `(1/T) Σ_t ||v_t||² M_t` using reverse Gramian
- **G_V** (for x/value rows): `(1/T) Σ_t k_t^T M_t k_t` (scalar × I per head)

Where `M_t = C_t C_t^T + γ²_{t+1} M_{t+1}` is the reverse Gramian (backward
recursion, O(T) per sample).

Also implemented the full-cut extension with shared Ω̄ (tail metric through
gate + RMSNorm + out_proj), and conv+SiLU-chained front factors (F_front_x,
F_front_B, F_front_C) that account for the pre-SSD nonlinearity.

### Kronecker-aware GPTQ solver (`kfac_gptq.py`)

BoA-style 2D coordinate descent: outer loop over row index p, inner column GPTQ,
row error propagation via `U_row = chol(G^{-1})`. Verified correct against the
BoA paper (Proposition 3.1, Appendix C equation 19) — identity row factor
reproduces vanilla GPTQ exactly.

### Approximation taxonomy study (`gn_taxonomy.py`)

JVP-based diagnostics covering all 5 levels of the approximation hierarchy:
Levels 0→1 (cross-token), 1→2 (single-Kronecker), 3a (cross-weight),
3b (inter-head), 3c (intra-head).

## What we found

### 1. The Kronecker factorization is ~99% wrong at the in_proj level

Direct measurement via JVPs on layer 0 (4 B-rows, 4 column directions, 2 samples):

| Approximation | Relative Frobenius error | Trace ratio |
|---|---|---|
| K-FAC (A ⊗ G) | 0.994 | 0.009 |
| GPTQ (A ⊗ I) | 0.996 | 0.012 |

Both capture only ~1% of the true Hessian's structure. K-FAC is marginally
less wrong, but both are terrible approximations.

### 2. The K-FAC 2D solver hurts quantization

Layer reconstruction error `||Y_fp - Y_quant||² / ||Y_fp||²` on layer 0:

| Method | rel_err | vs GPTQ |
|---|---|---|
| GPTQ (identity row factor) | 8.78e-3 | baseline |
| K-FAC identity (sanity check) | 8.78e-3 | matches ✓ |
| K-FAC with real factors | 1.08e-2 | +22% worse |
| K-FAC with rank truncation | 9.95e-3 | +13% worse |

The solver is correct (identity sanity check passes). The factors are the
problem — they actively misdirect the row error propagation.

### 3. Root cause: conv + SiLU between weight and recurrence

In attention (where BoA works): `Q = W_Q X` is purely linear. No nonlinearity
between the weight and the attention mechanism. The Kronecker factorization is
exact for linear maps.

In Mamba-2: the chain is `in_proj → split → conv1d → SiLU → (x, B, C) → SSD`.
The conv+SiLU between in_proj and the SSD interface breaks the Kronecker
separability. The reverse Gramian captures exact sensitivity at the SSD interface,
but this structure gets diffused when pulled back through the nonlinearity.

### 4. The single-Kronecker collapse is a major error source

The exact token-diagonal Hessian is a **sum of T different Kronecker products**:

    H^(1) = Σ_t (a_t a_t^T) ⊗ G_t

K-FAC collapses this to a single Kronecker: `H^(2) = A ⊗ Ḡ`.

This collapse requires G_t ≈ constant across tokens. For Mamba-2, G_t involves
the reverse Gramian M_t which has exponential structure — M_t is large at the
start of the sequence and decays to near-zero at the end. The token-to-token
variation in G_t is orders of magnitude, making the averaging destructive.

### 5. Weighted column Hessian gives negligible improvement

Using per-token Gramian weights to compute block-specific column Hessians:

    A_B = (1/Z) Σ_t w_B(t) · a_t a_t^T

where `w_B(t) = Σ_h ||x_dt_t||² · trace(M_t)`. Despite w_B(t) varying by
~2500:1 across tokens, the weighted A_B differs from uniform A by only 4%
(eigenvalue correlation 0.9999). The input covariance structure is dominated
by the input distribution, not the sensitivity weighting.

### 6. Gramian-based rotation helps modestly at 2-bit

Rotating x/B/C rows into the Gramian eigenbasis before quantization, then
de-rotating after:

| Method | 2-bit PPL | 3-bit PPL | 4-bit PPL |
|---|---|---|---|
| FP baseline | 20.04 | 20.04 | 20.04 |
| Vanilla GPTQ | 86.73 | 24.51 | 20.78 |
| Gramian rotation | **84.54** (-2.2) | 24.61 (+0.1) | 20.80 (+0.01) |
| K-FAC 2D solver | 107.36 (+20.6) | 24.95 (+0.4) | 21.01 (+0.2) |

The rotation smooths weight distributions across rows (similar to Hadamard
rotation in QuaRot), and the Gramian-informed rotation slightly outperforms
random rotation. But the effect is only meaningful at 2-bit where GPTQ
degrades heavily (+66 PPL from FP).

### 7. B/C factor conditioning is problematic

The group-level Gramian factors are severely ill-conditioned:

- F_B_group: condition number ~10^11
- F_C_group: condition number ~10^14

Even with rank truncation (keeping eigenvalues explaining 99% of trace),
the inverted factors amplify noise in the row propagation step.

## What the reverse Gramian IS good for

The reverse Gramian provides exact, structured, cheap (O(T) scan) sensitivity
information about the SSD recurrence. It correctly captures:

- Which tokens are most sensitive to B/C perturbations (early > late)
- Which state dimensions couple through the recurrence (via M_t off-diagonals)
- Per-head sensitivity variation (different heads have different decay rates)

This information is correct at the **SSD interface level** (post-conv features).
The problem is bridging from the SSD interface to the in_proj weight level.

## Recommended next directions

1. **Activation quantization**: The Gramian is exact at the SSD interface. For
   quantizing post-conv B/C/x features, the per-token per-channel sensitivity
   `G_t[n,n]` can be used directly (no Kronecker, no averaging). Tokens with
   large M_t need finer quantization grids. This is the natural regime for
   the reverse Gramian.

2. **Mixed-precision weight allocation**: Use the Gramian to decide which
   in_proj rows get more/fewer bits. Not a solver change, just bit allocation.

3. **Different architecture**: For SSMs without pre-recurrence nonlinearity
   (direct linear projection into the recurrence), the full K-FAC pipeline
   would work as designed.

4. **Sum-of-Kroneckers solver**: Instead of collapsing to A ⊗ Ḡ, develop a
   GPTQ-like solver that works with the per-token structure Σ_t (a_t a_t^T) ⊗ G_t
   directly. This avoids the destructive averaging.

## Model and settings

All experiments on `state-spaces/mamba2-130m`:
d_model=768, d_ssm=1536, nheads=24, headdim=64, d_state=128, ngroups=1

Calibration: 128 samples, seqlen=2048, wikitext2 train split.
Evaluation: wikitext2 test split, full perplexity.

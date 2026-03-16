# Activation Quantization with Reverse Gramian: Findings

## Summary

After finding that the reverse Gramian fails to improve weight quantization
(see `findings_weight_quant.md`), we explored using it for activation
quantization of B/C/x features at the SSD interface (after conv+SiLU,
before the SSD recurrence).

**Main result:** For uniform-bit-width per-channel activation quantization,
the reverse Gramian does not improve over the standard approach. The original
B/C channel basis (d_state dimensions) is already well-aligned with both the
sensitivity structure and the quantization grid. Multiple strategies were tested
(sensitivity-weighted clipping, SmoothQuant-style rescaling, eigenrotation into
the Gramian basis) — none consistently beat standard per-channel quantization.

## What we tested

### Setup

Mamba-2 130m, 4-bit activation quantization of B, C, x features at the SSD
interface. Mixer output reconstruction error measured per-layer. 32 calibration
samples, seqlen=2048.

### Quantization granularity comparison

| Method | Layer 0 | Layer 6 | Layer 12 |
|---|---|---|---|
| per_tensor | 6.38e-1 | 7.09e-1 | 6.07e-1 |
| per_token | 7.09e-1 | 5.00e-1 | 4.64e-1 |
| **per_channel** | **6.07e-3** | **4.85e-2** | **7.08e-2** |

Per-channel is 10-100x better than per-token or per-tensor. This is the
dominant effect — quantization granularity matters far more than any
sensitivity-based refinement.

### Gramian sensitivity properties

- **Per-token sensitivity** is nearly flat across the sequence (first/last
  quarter ratio ≈ 1.0 for most layers). The expected M_t decay pattern
  is washed out when averaged over calibration samples and summed over heads.
  Exception: layer 6 shows B first/last ratio of 5.4x.

- **Per-channel sensitivity** has meaningful variation: 100-1000x ratio
  across channels for B and C features.

- **Sensitivity is uncorrelated with channel magnitude**: B corr = 0.28,
  C corr = -0.08. This means the Gramian provides genuinely orthogonal
  information to what per-channel scaling already captures.

### Approach 1: Sensitivity-weighted clipping

Idea: clip insensitive channels/tokens more aggressively for a finer
quantization grid; keep sensitive channels/tokens at full range.

Result: **consistently worse** across all layers. Even with corrected logic
(clip insensitive, protect sensitive), the clipping introduces errors that
outweigh the finer grid benefit.

### Approach 2: SmoothQuant-style rescaling

Idea: rescale channels by sqrt(sensitivity) before per-token quantization,
de-scale after. Sensitive channels get larger → occupy more of the per-token
grid → get more precision.

Result: **marginal improvement** over per-token (0.50 → 0.56 on layer 6),
but still 10x worse than per-channel. The fundamental limitation: per-token
quantization with one scale per token can't match per-channel no matter how
you rescale.

### Approach 3: Gramian eigenrotation

Idea: rotate B/C activations into the Gramian eigenbasis before per-channel
quantization. In the rotated space, the Gramian-weighted error is diagonal:

    (u_hat - u)^T G (u_hat - u) = sum_i lambda_i (z_hat_i - z_i)^2

So per-channel quantization in the rotated space is optimally aligned with
the sensitivity metric.

Result: **mixed**. Better on some layers (layer 12: 0.91x), worse on others
(layer 0: 1.5x). Random rotation sometimes beats Gramian rotation (layers 6,
12). The original channel basis is already close to optimal for per-channel
quantization because the d_state dimensions naturally align with the SSM's
state space.

| Layer | per_channel | Gramian rotated | Random rotated |
|---|---|---|---|
| 0 | 6.07e-3 | 9.11e-3 (1.50x) | 1.36e-1 (22x) |
| 6 | 4.85e-2 | 4.71e-2 (0.97x) | 4.56e-2 (0.94x) |
| 12 | 7.08e-2 | 6.43e-2 (0.91x) | 6.14e-2 (0.87x) |
| 18 | 2.30e-2 | 2.37e-2 (1.03x) | 2.40e-2 (1.04x) |

## Why the Gramian doesn't help here

### 1. Per-channel quantization already handles the dominant error source

The largest source of activation quantization error is channel magnitude
variation (some channels have 100x larger range than others). Per-channel
scaling adapts to each channel's range independently. The Gramian's
sensitivity information (which channels matter more downstream) is secondary
to this — you can't reduce a channel's rounding error without more bits.

### 2. The original basis is already well-aligned

B/C features are in the d_state basis, which is the SSM's natural state
space. The Gramian's sensitivity (M_t for B, S_t^T S_t for C) operates in
this same state space. Rotating away from this basis mixes channels and can
introduce cross-channel quantization error without clear benefit.

### 3. Per-token sensitivity is too flat

The expected M_t exponential decay pattern (early tokens >> late tokens) is
averaged out over calibration samples and heads. The per-token sensitivity
ratio is typically 5-20x, which is not enough for the clipping/rescaling
approaches to make a meaningful difference.

### 4. Uniform bit width limits what sensitivity can do

With uniform N-bit quantization per channel, each channel independently gets
its own optimal scale. The Gramian tells you "this channel matters more"
but you can't act on that information without changing the bit width for
that channel. The per-element rounding error is a fixed fraction of the
channel's scale, regardless of sensitivity.

## Where the Gramian COULD help

### Mixed-precision bit allocation

Given a total bit budget (e.g., average 4 bits across all channels), allocate
more bits to Gramian-sensitive channels and fewer to insensitive ones. The
Gramian's orthogonality to magnitude (corr=0.28) means this would capture
structure that per-channel scaling misses. This requires a mixed-precision
quantization framework.

### Cross-tensor budget allocation

The Gramian tells you the relative sensitivity of B vs C vs x at each token.
Given a fixed total budget, the optimal split might be (6-bit B, 2-bit C) at
one token and (3-bit B, 4-bit C) at another. No other method provides this
cross-tensor sensitivity information.

### SSM hidden state quantization

The reverse Gramian applies DIRECTLY to the hidden state h_t (no nonlinearity,
no factorization). Quantizing the state during inference for memory-efficient
long-context generation would benefit from Gramian-weighted precision allocation.

### Quantization-aware fine-tuning

Using the Gramian-weighted error as the training loss (instead of MSE) during
quantization-aware training or fine-tuning would focus gradient updates on the
sensitivity directions. This doesn't require changing the quantization scheme
— just the loss function.

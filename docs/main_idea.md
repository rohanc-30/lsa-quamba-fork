# Implementation Plan for Mamba-2 K-FAC PTQ on `xBC / xxBC`

## High-level goal

Implement a **module-aware PTQ pipeline for Mamba-2** that quantizes only the **`xBC / xxBC` pre-featurizer slice** using a **K-FAC-style local Gauss–Newton approximation**, while leaving **all other weights on the existing GPTQ path**.

Before implementing the quantizer itself, first run a **granularity / approximation taxonomy study** on the **true local Hessian / Gauss–Newton** to determine which approximations are actually justified on a real model.

---

# 1. Background and framing

We are studying **post-training quantization (PTQ)** for **sequence mixers** by quantizing the **pre-featurizer weights** that generate interface features like \(q_t, k_t, v_t\), and using a **reconstruction loss on the mixer output activations** rather than a purely layer-local loss.

This is the same high-level move as BoA for attention:

- do **not** reconstruct only the output of a local projection layer,
- instead reconstruct the output of a **larger subgraph / module**,
- so the local curvature sees the downstream dependencies of the actual sequence mixer.

For a squared reconstruction objective evaluated at the floating-point (FP) operating point, the exact local Hessian is exactly the **local Gauss–Newton**
\[
H = J^\top J.
\]

That Hessian is too large in parameter space, so we want a **K-FAC-style approximation**
\[
H_W \approx A \otimes G,
\]
where:

- \(A\) is an **input-side covariance**
- \(G = B^\top B\) is an **output-side factor** for the perturbed interface feature

The reason this is especially attractive for **linear recurrence sequence mixers** is that the downstream sensitivity can often be propagated by **compact backward recurrences / reverse Gramians**, so \(G\) can often be computed much more exactly and cheaply than in attention.

---

# 2. Subgraph decomposition

We split the sequence mixer as

\[
\text{pre} \;\to\; \text{core} \;\to\; \text{post}.
\]

## Meaning of each piece

- **pre** = projections / short conv / feature generation that produce interface features such as
  \[
  q_t,\; k_t,\; v_t,\; \alpha_t,\; \beta_t,\dots
  \]

- **core** = the actual sequence-mixing recurrence / scan

- **post** = output gate / normalization / out projection

## Cuts we care about

Because the quantized weight lives in **pre**, the natural cuts are:

1. `pre`
2. `pre + core`
3. `pre + core + post = full`

The key conclusion so far is:

- **`pre + core`** admits especially clean exact structured formulas
- **`full`** is still valid, but becomes harder because the post metric is token-dependent

So the implementation priority is:

1. **exact `pre + core`**
2. optionally **`full` with shared / frozen \(\bar\Omega\)**

---

# 3. Model scope

## Model family
- **Mamba-2 only** for now

## Weight scope
- apply the custom curvature/PTQ path **only** to the fused **`xBC / xxBC` rows**
- leave **all remaining rows / weights** on the existing **GPTQ** path

## Important notation note

In the math below, we refer to conceptual interface features as \(q_t, k_t, v_t\) because that makes the recurrence formulas readable.

In code, keep the **model-native tensor names and offsets** and carefully map the selected `xBC / xxBC` slice into conceptual **q/k/v-like groups** that feed the frozen-feature scan.

Do **not** assume literal checkpoint tensors are named `q`, `k`, `v`; that is only conceptual math notation.

---

# 4. Exact local Hessian / Gauss–Newton wrt a shared pre weight

Let the selected shared pre weight block be \(W\), and let the token-local input activation into that block be \(a_t\).

If the token-local interface feature is
\[
x_t = W a_t,
\]
and the chosen subgraph output over the whole sequence is \(Y\), define
\[
B_t := \frac{\partial Y}{\partial x_t}.
\]

Then the exact local Hessian / GN wrt the **shared** weight block is
\[
H_W = J_W^\top J_W
     = \sum_{t=1}^T \sum_{s=1}^T (a_t a_s^\top)\otimes G_{t,s},
\qquad
G_{t,s} := B_t^\top B_s.
\]

This decomposition is important because it makes the approximation hierarchy explicit.

---

# 5. Approximation hierarchy

There are **three broad granularity axes** we care about:

1. coupling across tensors (`Q/K/V`)
2. coupling across heads
3. coupling across intra-head channels

But there are actually **two approximation steps before those**.

So the full hierarchy is:

## Level 0 — exact local Hessian / GN
\[
H^{(0)} = \sum_{t,s} (a_t a_s^\top)\otimes G_{t,s}.
\]

## Level 1 — drop cross-token shared-weight coupling
Keep only the \(t=s\) terms:
\[
H^{(1)} = \sum_t (a_t a_t^\top)\otimes G_t,
\qquad
G_t := G_{t,t}.
\]

This drops the cross-token terms induced by the fact that the same weight is reused across tokens.

## Level 2 — collapse a sum of Kroneckers to a single Kronecker
Even after Level 1, the exact token-local object is still
\[
\sum_t (a_t a_t^\top)\otimes G_t.
\]

K-FAC makes the additional approximation
\[
\sum_t (a_t a_t^\top)\otimes G_t
\;\approx\;
A \otimes \bar G,
\]
with
\[
A = \frac{1}{N}\sum_{(b,t)} a_{b,t} a_{b,t}^\top,
\qquad
\bar G = \frac{1}{N}\sum_{(b,t)} G_{b,t}.
\]

This is the actual **single-Kronecker K-FAC step**.

## Level 3 — choose block structure of \(\bar G\)

### 3a. Tensor coupling
Should \(\bar G\) keep:
- full cross-tensor coupling across conceptual `Q/K/V` groups?
- or separate them?

### 3b. Head coupling
Should \(\bar G\) keep:
- cross-head coupling?
- or be block-diagonal across heads?

### 3c. Intra-head channel coupling
Inside a head/tensor block, should we keep:
- dense \(d\times d\)
- diagonal
- scalar times identity
- low-rank + diagonal

---

# 6. What Part I must decide

Part I should determine:

1. **whether cross-token shared-weight coupling can be dropped**
2. **whether the sum-of-Kroneckers can be collapsed to a single Kronecker**
3. **what block structure to use for \(\bar G\)** across:
   - tensor groups
   - heads
   - intra-head channels

So Part I does **not** merely decide "per-head vs per-channel". It decides the whole approximation ladder.

---

# 7. Part I — Approximation taxonomy exploration on the true local Hessian / GN

## Objective

Before implementing the quantizer, analyze the **true local Hessian / GN** on the real model and measure what happens when each approximation level is introduced.

Main questions:

- how much curvature mass is in cross-token terms?
- how much does the single-Kronecker collapse distort things?
- where does meaningful output-side coupling live?
  - across Q/K/V?
  - across heads?
  - within channels?

---

## 7.1 Two-regime study on a real model

Do both regimes on a **real pretrained Mamba-2 checkpoint** with **real calibration data**.

### Regime A — exact restricted Hessian / GN slices on a real model
Compute **exact restricted sub-blocks / subspaces** of the Hessian on the real model.

### Regime B — scalable probe-based study on a real model
Use JVP/VJP probe estimators to study the same couplings at larger scale.

---

## 7.2 Regime A — exact restricted Hessian / GN subspaces

Do **not** materialize the full Hessian.

Instead, choose a structured low-dimensional subspace
\[
U \in \mathbb{R}^{P\times m}
\]
inside the selected `xBC / xxBC` parameter block, and compute the exact restricted Hessian
\[
H_U = U^\top H^{(0)} U = (J U)^\top (J U).
\]

This is the exact Hessian / GN restricted to that chosen slice.

### How to compute \(H_U\)

For each basis vector \(u_i\) of the subspace \(U\):

1. freeze the model
2. define the chosen reconstruction output \(Y\) for the cut being studied
3. compute the JVP
   \[
   J u_i
   \]
4. form
   \[
   (H_U)_{ij} = \langle J u_i,\; J u_j\rangle
   \]

This gives the exact Hessian / GN on the selected slice of the real model.

---

## 7.3 Also compute token-resolved restricted blocks

To study cross-token coupling properly, also compute token-pair pieces
\[
H_U^{t,s} = (J_t U)^\top (J_s U),
\]
where \(J_t\) is the Jacobian of the token-\(t\) subgraph output wrt the selected parameter slice.

Then:
- token-diagonal mass = \(\sum_t H_U^{t,t}\)
- cross-token mass = \(\sum_{t\neq s} H_U^{t,s}\)

This directly measures what is lost when Level 1 is applied.

---

## 7.4 Structured slices to study in Regime A

Use **structured slices**, not arbitrary coordinates.

### A. Cross-token study
Same parameter slice, but compare:
- \(H_U^{t,t}\)
- \(H_U^{t,s}\) for nearby \(t,s\)
- \(H_U^{t,s}\) for distant \(t,s\)

This measures Level 1.

### B. Single-Kronecker study
Even after dropping cross-token terms, the exact object is
\[
\sum_t (a_t a_t^\top)\otimes G_t.
\]

We need to measure how well this is approximated by
\[
A \otimes \bar G.
\]

So in Regime A, compute the exact token-local restricted Hessian slice and compare it to its single-Kronecker approximation on the same slice.

This measures Level 2.

### C. Cross-tensor study
Within one head, compute exact restricted blocks for the conceptual q/k/v-like groups:
- Q vs K
- Q vs V
- K vs V

This measures Level 3a.

### D. Cross-head study
Compute exact restricted blocks across head pairs:
- same tensor type across different heads
- mixed tensor/head pairs

This measures Level 3b.

### E. Intra-head channel study
Pick one head/tensor block and compute an exact dense channel patch, e.g. 16–64 channels.

This measures Level 3c.

### Suggested subspace sizes
Start with:
- \(m = 32\) or \(64\) for most studies
- maybe \(m = 128\) for a small number of richer slices

---

## 7.5 Metrics to report for each approximation jump

Given an exact restricted block \(H_U\) and a projected/coarsened version \(P(H_U)\), report:

### 1. Relative Frobenius error
\[
\frac{\|H_U - P(H_U)\|_F}{\|H_U\|_F}
\]

### 2. Trace retained
\[
\frac{\operatorname{tr}(P(H_U))}{\operatorname{tr}(H_U)}
\]

### 3. Relative spectral error
\[
\frac{\|H_U - P(H_U)\|_2}{\|H_U\|_2}
\]

### 4. Normalized coupling score
For a two-block partition
\[
\begin{bmatrix}
A & B\\
B^\top & C
\end{bmatrix},
\]
compute
\[
\Gamma = A^{-1/2} B C^{-1/2}
\]
and report \(\|\Gamma\|_2\) and \(\|\Gamma\|_F\).

This measures off-block coupling relative to the diagonal block scales.

### 5. Quadratic-form distortion
For test directions \(u\), compute
\[
r(u) = \frac{u^\top P(H_U)u}{u^\top H_U u}.
\]

Use:
- random Gaussian directions
- top eigendirections of \(H_U\)
- actual quantization-error directions from a first-pass GPTQ / rounding run

---

## 7.6 Regime B — scalable probe estimators

For realistic layers / longer sequences / more heads, do not compute restricted matrices explicitly.

Use the bilinear identity
\[
u^\top H v = \langle J u,\; J v\rangle
\]
and estimate block couplings with structured random probes.

### Probe families
Use probe families aligned with the same approximation levels:
- cross-token
- single-Kronecker quality
- cross-tensor
- cross-head
- intra-head channel structure

### What to estimate
Estimate:
- off-block Frobenius mass
- normalized coupling strength
- quadratic-form distortion under block projections
- variability of \(G_t\) across tokens, to assess the single-Kronecker collapse

---

## 7.7 Deliverables from Part I

For each studied Mamba-2 layer:

1. exact restricted-block summaries from Regime A
2. scalable probe summaries from Regime B
3. a recommendation for:
   - whether to drop cross-token coupling
   - whether to use a single-Kronecker collapse
   - what output-side block granularity to use

### Likely default hypothesis for Mamba-2 `pre + core`
Expected, but must be verified:
- dropping cross-token coupling may be acceptable
- cross-head coupling may be tiny or zero for `pre + core`
- intra-head channel coupling likely matters and should stay dense
- main open question: whether to keep cross-tensor Q/K/V coupling within a head

Likely candidates after Part I:
1. **per-head fused QKV block**
2. **per-head separate Q/K/V blocks**

Do **not** hard-code this choice until Part I is done.

---

# 8. Part II — K-FAC factorization and PTQ implementation

## Objective

Implement the actual K-FAC curvature builder and Kronecker-aware PTQ pipeline for the **Mamba-2 `xBC / xxBC` rows only**.

Everything else remains on the existing GPTQ path.

---

## 8.1 Mamba-2 frozen-feature core

Use the frozen-feature core
\[
S_t = \alpha_t S_{t-1} + v_t k_t^\top,
\qquad
o_t = S_t q_t.
\]

Here:
- \(q_t, k_t, v_t\) are the conceptual interface features extracted from the selected `xBC / xxBC` slice
- \(\alpha_t\) is treated as frozen for this custom PTQ path

---

## 8.2 Exact `pre + core` backward Gramian

Define
\[
M_{T+1}=0,
\qquad
M_t = q_t q_t^\top + \alpha_{t+1}^2 M_{t+1}.
\]

This is the reverse Gramian / backward metric for the frozen-feature core.

---

## 8.3 Exact token-local output-side factors for Mamba-2 `pre + core`

These are the main formulas to implement first.

### Q
\[
G^{pc}_{Q,t} = S_t^\top S_t
\]

### K
\[
G^{pc}_{K,t} = \|v_t\|^2\, M_t
\]

### V
\[
G^{pc}_{V,t} = (k_t^\top M_t k_t)\, I
\]

These are exact token-locally for the `pre + core` cut.

This is the key structural win: exact structured output-side factors without paying an extra factor of \(T\) in sequence-length scaling.

---

## 8.4 Full exact with actual post

For the actual full graph, define
\[
\Omega_t = J_{\text{post},t}^\top J_{\text{post},t}.
\]

Then:

### Q
\[
G^{full}_{Q,t} = S_t^\top \Omega_t S_t
\]

Let
\[
\rho_{t,j} = \prod_{i=t+1}^{j}\alpha_i,
\qquad
\rho_{t,t}=1.
\]

Then:

### K
\[
G^{full}_{K,t}
=
\sum_{j=t}^T
\rho_{t,j}^2\,
(v_t^\top \Omega_j v_t)\,
q_j q_j^\top
\]

### V
\[
G^{full}_{V,t}
=
\sum_{j=t}^T
\rho_{t,j}^2\,
(k_t^\top q_j)^2\,
\Omega_j
\]

These are exact, but in general no longer admit the clean linear-time collapse because \(\Omega_j\) varies with token \(j\).

---

## 8.5 Shared-\(\bar\Omega\) approximation for linear-time full `K/V`

If we later want a more faithful full-graph approximation without paying the full exact cost, define
\[
\bar\Omega
=
\frac{1}{N}\sum_{(b,t)\in\text{calib}}\Omega_{b,t}.
\]

Use this:
- per layer
- preferably per head

Then:
- keep **Q exact**
- approximate **K/V** only

### Q exact
\[
G_{Q,t} = S_t^\top \Omega_t S_t
\]

### K approximate
\[
G_{K,t} \approx (v_t^\top \bar\Omega v_t)\, M_t
\]

### V approximate
\[
G_{V,t} \approx (k_t^\top M_t k_t)\, \bar\Omega
\]

This is the natural second-stage extension after exact `pre + core`.

---

# 9. Input-side K-FAC factor

For a linear pre-map
\[
x_t = W a_t,
\]
once cross-token shared-weight coupling has been dropped, the token-local K-FAC approximation is
\[
H_W \approx A \otimes \bar G,
\]
with
\[
A = \frac{1}{N}\sum_{(b,t)} a_{b,t} a_{b,t}^\top,
\qquad
\bar G = \frac{1}{N}\sum_{(b,t)} G_{b,t}.
\]

For the Mamba-2 `xBC / xxBC` slice:
- \(A\) is the covariance of the input activation feeding that fused projection slice
- \(\bar G\) is the averaged output-side block chosen after Part I

---

# 10. How to compute \(\Omega_t\) and \(\bar\Omega\) if needed

Only needed for the optional `full` path.

If post is of the form
\[
y_t = W_O\,\mathrm{RMSNorm}(g_t \odot o_t),
\]
let
\[
u_t = g_t \odot o_t,
\qquad
D(g_t)=\operatorname{diag}(g_t).
\]

Then
\[
J_{\text{post},t}
=
W_O\,J_{\mathrm{RMS}}(u_t)\,D(g_t),
\]
so
\[
\Omega_t
=
D(g_t)\,
J_{\mathrm{RMS}}(u_t)^\top
W_O^\top W_O
J_{\mathrm{RMS}}(u_t)\,
D(g_t).
\]

Then average over calibration tokens:
\[
\bar\Omega = \frac{1}{N}\sum_{(b,t)} \Omega_{b,t}.
\]

Recommended numerics:
- use per layer, preferably per head
- symmetrize:
  \[
  \bar\Omega \leftarrow \tfrac12(\bar\Omega+\bar\Omega^\top)
  \]
- add small damping if needed

---

# 11. Mamba-2 implementation order

## Phase 1 — exact `pre + core` factor extractor

For one Mamba-2 layer and one calibration batch, collect:
- the input activations \(a_t\) feeding the `xBC / xxBC` slice
- the per-token conceptual \(q_t, k_t, v_t\) features extracted from that slice
- the recurrence scalar \(\alpha_t\)

### Steps
1. run the frozen layer forward
2. cache the input activations to the selected `xBC / xxBC` slice
3. extract/cache the conceptual q/k/v-like features from the same fused slice
4. run the forward recurrence to get all \(S_t\)
5. run the backward Gramian recursion to get all \(M_t\)
6. compute token-local
   \[
   G^{pc}_{Q,t},\; G^{pc}_{K,t},\; G^{pc}_{V,t}
   \]
7. average over batch/time to get output-side blocks
8. compute \(A\) from the input activations

### Deliverables
- factor builder returning \(A\) and the selected output-side blocks
- per-token diagnostics for debugging
- sanity checks:
  - symmetry
  - PSD
  - trace / norm sanity
  - agreement with small exact restricted JVP-based slices from Part I

---

## Phase 2 — optional `full` shared-\(\bar\Omega\) path

After Phase 1 works:
1. compute token-local \(\Omega_t\) on calibration data
2. average to \(\bar\Omega\)
3. keep **Q exact**
4. approximate **K/V** with shared \(\bar\Omega\)

Do **not** start here. Start with exact `pre + core`.

---

# 12. Quantization backend

## 12.1 Solver shape

Once we decide to use a single Kronecker Hessian
\[
H_{\text{blk}} \approx A \otimes G_{\text{blk}},
\]
the downstream quantization pipeline should be a **Kronecker-aware GPTQ / BoA-style solver**.

That means:
- use \(A\) and \(G_{\text{blk}}\) as the two Hessian factors
- avoid materializing the full \(A \otimes G_{\text{blk}}\)
- use Kronecker identities to perform the solver efficiently

---

## 12.2 Matrix objective

For a chosen weight block \(W_{\text{blk}}\in \mathbb{R}^{m\times d_{in}}\), with quantized version \(\hat W_{\text{blk}}\), define
\[
E = W_{\text{blk}} - \hat W_{\text{blk}}.
\]

Under the Kronecker approximation, the quadratic loss is
\[
\mathcal L(\hat W_{\text{blk}})
=
\operatorname{tr}\!\left(
G_{\text{blk}}\, E\, A\, E^\top
\right).
\]

Equivalently,
\[
\mathcal L(\hat W_{\text{blk}})
=
\mathrm{vec}(E)^\top (A\otimes G_{\text{blk}})\,\mathrm{vec}(E).
\]

This is the objective the custom solver should minimize.

---

## 12.3 Which block granularity to use in the solver

This depends on Part I.

The solver block could be any of:

- full selected `xBC / xxBC` block
- per-head fused QKV block
- per-head separate Q/K/V blocks
- dense or diagonalized within those

Part I determines which one is justified.

---

## 12.4 Recommended v1 solver strategy

Implement a **generic Kronecker-aware GPTQ / BoA-style backend** that takes:
- \(A\)
- \(G_{\text{blk}}\)
- a selected weight block \(W_{\text{blk}}\)

and performs quantization under the matrix objective above.

### Recommended v1 simplification
For v1, keep the solver simple:
1. evaluate the matrix objective for a proposed quantized block
2. implement a basic row/groupwise coordinate descent or blockwise refinement scheme
3. reuse as much of the existing GPTQ machinery as possible on the input-side factor \(A\)

Only after the factor construction is validated should the solver be made fancy.

---

# 13. Validation plan

## A. Curvature correctness
For small structured slices:
- compare K-FAC block predictions against exact restricted Hessian / GN slices from Part I

## B. Local reconstruction objective
For the selected `xBC / xxBC` block:
- compare GPTQ quantization vs custom K-FAC quantization on the chosen subgraph reconstruction loss

## C. End-to-end layer/module behavior
Quantize only the `xBC / xxBC` rows:
- leave everything else FP
- compare layer output reconstruction
- compare whole-model perplexity / eval degradation

## D. Ablations
Compare:
1. GPTQ baseline on `xBC / xxBC`
2. exact `pre + core` K-FAC
3. optional `full` shared-\(\bar\Omega\) K-FAC

---

# 14. Concrete task list

## Part I — taxonomy study
1. isolate the Mamba-2 `xBC / xxBC` rows and map them to conceptual q/k/v-like groups
2. implement exact restricted Hessian / GN subspace computation with JVPs
3. implement token-resolved restricted Hessian pieces \(H_U^{t,s}\)
4. implement the approximation projections:
   - drop cross-token
   - collapse sum-of-Kroneckers to a single Kronecker
   - tensor/head/channel block projections
5. implement metrics:
   - Frobenius error
   - trace retained
   - spectral error
   - normalized coupling
   - quadratic-form distortion
6. implement scalable probe estimators for the same quantities
7. run on real pretrained Mamba-2 layers and summarize which approximation levels are acceptable

## Part II — Mamba-2 K-FAC PTQ
1. implement exact `pre + core` factor extraction for the `xBC / xxBC` slice:
   - cache input activations
   - extract q/k/v-like features
   - compute \(S_t\)
   - compute \(M_t\)
   - build token-local \(G_Q, G_K, G_V\)
   - average to block factors
2. compute input covariance \(A\)
3. choose block granularity based on Part I
4. implement a generic Kronecker-aware GPTQ / BoA-style backend using
   \[
   \operatorname{tr}(G E A E^\top)
   \]
5. compare against GPTQ on the same `xBC / xxBC` slice
6. optionally implement the `full` shared-\(\bar\Omega\) extension afterward

---

# 15. Non-goals / things not to do yet

- do **not** implement GDN yet
- do **not** try to quantize all Mamba-2 weights with this custom path
- do **not** assume cross-token coupling is negligible without measuring it
- do **not** assume the single-Kronecker collapse is good without measuring it
- do **not** hard-code per-head vs per-tensor vs full-block before Part I
- do **not** start with the full shared-\(\bar\Omega\) path; start with exact `pre + core`

---

# 16. Expected end state

If everything works, the final claim should be:

- for **Mamba-2 `pre + core`**, we can use a **richer module-aware local curvature proxy** than local GPTQ
- we can compute it using exact structured recurrence formulas
- and, if Part I supports the approximations, we can do so **without paying an extra factor of \(T\)** in sequence-length scaling

That is the main research/computational win to validate.
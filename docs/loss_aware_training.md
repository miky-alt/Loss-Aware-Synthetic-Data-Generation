# Loss-Aware Training

This document describes the loss-aware generator: the part of the project that
moves utility and privacy constraints from post-hoc evaluation into the
training objective itself. It covers the design rationale, the exact
mathematical form of each penalty term, the implementation choices we made,
the problems we hit along the way, and what the first experiments showed.

The implementation lives in `src/generators/loss_aware.py` and is registered
as `tvae_loss_aware` in `src/generators/registry.py`.

---

## 1. Motivation

The evaluation framework (see [metrics/utility.md](metrics/utility.md) and
[metrics/privacy.md](metrics/privacy.md)) tells us *after* generation how much
utility was lost and how much privacy risk was introduced. That is useful, but
it is passive: the generator was never told about either constraint, so the
only lever left is to reject a bad synthetic dataset and try again with
different hyperparameters.

The loss-aware approach makes the constraints part of what the generator
optimizes. Every gradient step, the model is penalized for producing samples
whose distribution diverges from the real one, whose feature correlations are
distorted, or which sit too close to individual real records. The trade-off
between these goals is expressed through explicit weights, which can be set
deliberately rather than emerging as an accident of architecture.

---

## 2. Base model: TVAE

We build on the Tabular Variational Autoencoder (TVAE) from the `ctgan`
package (version 0.11.1), which is the model behind `sdv`'s `TVAESynthesizer`.

A VAE learns an encoder $q_\phi(z \mid x)$ that maps a data row $x$ to a
latent distribution and a decoder $p_\theta(x \mid z)$ that maps a latent
vector back to data space. Training minimizes the negative evidence lower
bound:

$$
\mathcal{L}_{\text{VAE}} \;=\; \underbrace{-\,\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x \mid z)\big]}_{\mathcal{L}_{\text{recon}}} \;+\; \underbrace{D_{\mathrm{KL}}\big(q_\phi(z \mid x)\,\|\,\mathcal{N}(0, I)\big)}_{\mathcal{L}_{\text{KL}}}
$$

TVAE adds a tabular-specific data transformer:

- **Continuous columns** are encoded with *mode-specific normalization*: a
  Bayesian Gaussian mixture is fit per column, each value is expressed as a
  scalar $\alpha \in [-1, 1]$ (its position within the chosen mode) plus a
  one-hot vector indicating which mode was selected.
- **Discrete columns** are one-hot encoded.

The decoder outputs raw logits; `tanh` is applied to the $\alpha$ scalars and
`softmax` to every one-hot span. The reconstruction loss uses a Gaussian
likelihood on the scalars and cross-entropy on the one-hot spans.

We chose TVAE over CTGAN for one practical reason: its training loop is about
sixty lines with no adversarial component, so it can be subclassed and
extended without rewriting conditional sampling or discriminator logic.

### Why subclass rather than reimplement

`LossAwareTVAE` subclasses `ctgan.synthesizers.tvae.TVAE` and overrides only
`fit()`. The override copies the original training loop verbatim and inserts
one block between the loss computation and `loss.backward()`. Two things
follow from this:

1. **With all penalty weights set to zero, the class is exactly the stock
   TVAE.** This is the fair baseline for every experiment: same architecture,
   same transformer, same optimizer, same code path. The only difference
   between "baseline" and "loss-aware" is the value of three scalars.
2. We reuse `ctgan`'s `DataTransformer` unchanged, so categorical handling and
   mode-specific normalization are identical to the baseline.

The cost is a dependency on `ctgan` internals (`Encoder`, `Decoder`,
`_loss_function`, `output_info_list`). The version is pinned in
`pyproject.toml`.

---

## 3. The loss-aware objective

The full training objective is

$$
\mathcal{L} \;=\; \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}
\;+\; \lambda_{\text{mmd}}\,\mathcal{L}_{\text{mmd}}
\;+\; \lambda_{\text{corr}}\,\mathcal{L}_{\text{corr}}
\;+\; \lambda_{\text{priv}}\,\mathcal{L}_{\text{priv}}
$$

where $\lambda_{\text{mmd}}, \lambda_{\text{corr}}, \lambda_{\text{priv}} \geq 0$
are user-set weights.

### 3.1 Penalties act on generated samples, not reconstructions

A design decision that matters more than it first appears. The VAE loop
naturally produces $\hat{x} = \text{decoder}(\text{encoder}(x))$, a
*reconstruction* of the current real batch. It would be easy to compute the
penalties on $\hat{x}$. We do not.

Utility and privacy are properties of what the generator *produces at sample
time*, which is $\text{decoder}(z)$ for $z \sim \mathcal{N}(0, I)$. A
reconstruction is conditioned on a real row and is not representative of that
distribution. So at every step we additionally draw a fresh batch from the
prior:

$$
z_i \sim \mathcal{N}(0, I), \qquad \tilde{x}_i = \sigma\big(\text{decoder}(z_i)\big), \qquad i = 1, \dots, B
$$

where $B$ is the batch size and $\sigma(\cdot)$ is the activation described in
§3.5. All three penalties compare $\tilde{X} = \{\tilde{x}_i\}$ against the
current real batch $X = \{x_i\}$. This costs one extra decoder forward pass
per step.

### 3.2 Distribution penalty: Maximum Mean Discrepancy

$\mathcal{L}_{\text{mmd}}$ is the biased empirical estimate of squared MMD
with an RBF kernel, the differentiable counterpart of `compute_mmd` in the
evaluation module:

$$
\mathcal{L}_{\text{mmd}}(\tilde{X}, X) \;=\;
\frac{1}{B^2}\sum_{i,j} k(\tilde{x}_i, \tilde{x}_j)
\;+\; \frac{1}{B^2}\sum_{i,j} k(x_i, x_j)
\;-\; \frac{2}{B^2}\sum_{i,j} k(\tilde{x}_i, x_j),
\qquad
k(a, b) = \exp\!\big(-\gamma \,\|a - b\|_2^2\big)
$$

The RBF kernel is *characteristic*: $\text{MMD} = 0$ if and only if the two
distributions are identical, so minimizing this term is a well-posed way to
pull the generated distribution toward the real one. $\gamma$ defaults to
$1.0$, which is reasonable because all coordinates in the transformed space
are bounded ($\alpha \in [-1, 1]$, one-hot entries in $[0, 1]$).

The term is fully differentiable with respect to $\tilde{X}$ and hence with
respect to the decoder parameters; the real-real kernel block contributes no
gradient but keeps the estimate unbiased in expectation.

### 3.3 Correlation penalty

$\mathcal{L}_{\text{corr}}$ is the Frobenius norm of the difference between
the Pearson correlation matrices of the fake and real batches — the
differentiable counterpart of `compute_correlation_distance`:

$$
\mathcal{L}_{\text{corr}}(\tilde{X}, X) \;=\; \big\|\,\mathrm{Corr}(\tilde{X}) - \mathrm{Corr}(X)\,\big\|_F,
\qquad
\mathrm{Corr}(Y) = \frac{1}{B}\,\hat{Y}^\top \hat{Y},
\quad
\hat{Y}_{\cdot j} = \frac{Y_{\cdot j} - \bar{Y}_j}{\hat{\sigma}_j + \varepsilon}
$$

where columns are centred and divided by their (biased) standard deviation.
The $\varepsilon = 10^{-8}$ guard means a constant column produces a row and
column of zeros in the correlation matrix rather than NaN.

MMD already matches the joint distribution in principle, so why a separate
correlation term? Because with a fixed bandwidth and finite batches, MMD is
most sensitive to marginal mismatches and only weakly sensitive to second-order
structure. An explicit correlation penalty puts gradient pressure directly on
the feature relationships that downstream models depend on.

### 3.4 Privacy penalty: DCR hinge

$\mathcal{L}_{\text{priv}}$ turns the Distance-to-Closest-Record metric into a
gradient. For each generated row, find the nearest real row in the batch and
penalize it if it is closer than a margin $m$:

$$
d_i \;=\; \min_{j}\,\big\|\tilde{x}^{\text{hard}}_i - x_j\big\|_2,
\qquad
\mathcal{L}_{\text{priv}}(\tilde{X}, X) \;=\; \frac{1}{B}\sum_{i=1}^{B} \max\big(0,\; m - d_i\big)
$$

The hinge is zero whenever every generated row keeps its distance and grows
linearly as a row approaches a real record. Its gradient pushes offending
rows directly away from their nearest real neighbour.

Two implementation details were essential to make this term work at all; both
are covered in §4.

### 3.5 Activation of generated rows

The decoder emits raw logits. To compare against real rows in the
transformer's space we apply, span by span,

$$
\sigma(\text{raw})_{\text{span}} =
\begin{cases}
\tanh(\text{raw}_{\text{span}}) & \text{continuous scalar } \alpha \\[4pt]
\text{softmax}(\text{raw}_{\text{span}}) & \text{one-hot span, soft mode} \\[4pt]
\text{GumbelSoftmax}_{\text{ST}}(\text{raw}_{\text{span}}) & \text{one-hot span, hard mode}
\end{cases}
$$

Soft mode is used for $\mathcal{L}_{\text{mmd}}$ and $\mathcal{L}_{\text{corr}}$.
Hard mode is used for $\mathcal{L}_{\text{priv}}$. See §4.2 for why.

---

## 4. Problems encountered and how they were resolved

The first version of the loss-aware model produced a privacy term that was
identically zero throughout training. Diagnosing this took two iterations and
the fixes are part of the contribution, so we document them.

### 4.1 The margin has no natural scale

The first implementation took $m$ as an absolute distance in the transformed
space with a default of $1.0$. That space has as many dimensions as the
transformer produces — one $\alpha$ plus a mode one-hot per continuous column,
plus a one-hot per discrete column — and every one-hot span can contribute up
to $\sqrt{2}$ to a Euclidean distance. On Heart Disease (8 discrete columns,
5 continuous) typical nearest-neighbour distances are around $2$; on a wider
dataset they would be larger. A fixed $m = 1$ therefore never bites on Heart,
and any fixed value would need re-tuning per dataset.

**Fix: express the margin relative to the data's own scale.** At the start of
`fit()` we compute the median distance from each real row to its nearest
*other* real row in the transformed space,

$$
\tilde{d}_{\text{real}} \;=\; \operatorname{median}_i \;\min_{j \neq i}\, \|x_i - x_j\|_2,
$$

and set $m = \mu \cdot \tilde{d}_{\text{real}}$, where $\mu$ is the user-facing
`dcr_margin` parameter (default $1.0$). The interpretation is clean: $\mu = 1$
means *a generated row may not sit closer to a real row than real rows sit to
each other*. Larger $\mu$ is stricter. Both $\tilde{d}_{\text{real}}$ and the
effective $m$ are recorded in the run's diagnostics. An absolute mode is kept
behind `dcr_margin_relative=False`.

On Heart Disease, $\tilde{d}_{\text{real}} \approx 2.04$.

### 4.2 Soft one-hots are structurally far from hard one-hots

Even with the relative margin the penalty stayed at zero. The cause was the
representation of $\tilde{x}$: real rows contain exact one-hots such as
$(0, 0, 1, 0)$, while softmax outputs are soft, such as $(0.2, 0.1, 0.6, 0.1)$.
A generated row that would decode to the *identical* record is still
$\approx 0.5$ away from it on every discrete span. Heart Disease has eight
discrete columns plus five mode-indicator one-hots, so the minimum achievable
fake-to-real distance was around $5$ — well above any margin derived from the
real-to-real distances, which are measured between hard rows. The hinge could
never fire.

**Fix: straight-through Gumbel-softmax for the privacy term.** For each
one-hot span,

$$
y \;=\; \operatorname{one\_hot}\!\Big(\arg\max_k \big(\ell_k + g_k\big)\Big),
\qquad g_k \sim \text{Gumbel}(0, 1),
$$

is used in the forward pass, so $\tilde{x}^{\text{hard}}$ is a genuine one-hot
row comparable to $x$; in the backward pass the gradient of the soft
$\text{softmax}\big((\ell + g)/\tau\big)$ with $\tau = 1$ is used instead. The
hinge is now measuring the same kind of distance that the evaluation-time DCR
measures.

We keep soft activations for MMD and correlation: those terms compare
distributions rather than individual rows, soft outputs give lower-variance
gradients, and they were already effective before this change.

After both fixes, on Heart Disease at $\mu = 1$: $m \approx 2.04$, the mean
fake-to-real nearest-neighbour distance at the end of training is $\approx 2.7$,
and $\mathcal{L}_{\text{priv}} \approx 0.04$ — most generated rows sit outside
the margin and a few stragglers are being pushed out. That is the intended
regime.

### 4.3 Training budget masquerading as a result

The very first comparison (baseline vs. loss-aware, one seed, 100 epochs,
`batch_size=500` on a 237-row training set) showed synthetic F1 collapsing
from $0.75$ to $0.58$ under the loss-aware objective. This was not the loss.
With a batch size larger than the dataset, each epoch is a single gradient
step and 100 epochs is 100 steps; both models were badly undertrained and the
comparison was noise. With `batch_size=32` and 300 epochs, F1 discrepancy is
$\approx 0$ for every configuration.

The methodological point stands on its own: the first ablation on a new setup
tends to measure the training budget, not the idea.

---

## 5. Logging

Each batch appends a row to `loss_values` with the total loss and every
component separately: `Recon`, `KL`, `MMD`, `Corr`, `Priv`, plus `FakeNN`, the
mean fake-to-real nearest-neighbour distance used by the hinge. This is exposed
through `get_training_diagnostics()` together with all $\lambda$'s, $\mu$,
$\tilde{d}_{\text{real}}$ and the effective $m$, so every experiment report is
self-describing. Plotting the five loss components over epochs shows the terms
trading off against each other during training.

---

## 6. Experimental protocol

All experiments so far use Heart Disease (Cleveland), chosen because it is
small enough to iterate on (a 300-epoch run takes under a minute) and because
its size makes memorization — the failure mode the privacy term targets — most
likely.

Fixed settings: `batch_size=32`, `epochs=300`, `test_size=0.2`, 1000 synthetic
rows sampled per run. Every configuration is run with seeds 1, 2 and 3; the
seed controls the train/test split, the model initialization and the sampling.

**Ablation grid.**

| configuration | $\lambda_{\text{mmd}}$ | $\lambda_{\text{corr}}$ | $\lambda_{\text{priv}}$ |
|---|---|---|---|
| baseline | 0 | 0 | 0 |
| utility only | 1.0 | 0.5 | 0 |
| privacy only | 0 | 0 | 1.0 |
| both | 1.0 | 0.5 | 1.0 |

**Margin sweep.** The privacy-only and both configurations are additionally
run at $\mu \in \{0.5, 1.0, 1.5\}$ to trace the privacy–utility curve.

Results are evaluated with the full metric suite from `src/evaluation/`. Two
caveats apply when reading them:

- The evaluation DCR is computed in a *different* space from the training
  hinge: standardized label-encoded raw columns versus the transformer's
  one-hot space. The two are strongly correlated but not numerically
  comparable.
- With `sensitive_col = target_col`, the inference-risk metric and the
  synthetic-F1 utility metric are computed by the *same* procedure (train a
  classifier on synthetic, evaluate on the real test split, predict the
  target). They are numerically identical. On these datasets, downstream
  utility for the researcher and attribute-inference risk for the data subject
  are literally the same quantity; the metric only becomes an independent
  privacy signal if a different sensitive column is chosen.

---

## 7. Results of the first ablation (Heart Disease, 3 seeds, means)

These runs were made *before* the privacy-term fixes in §4, so the
privacy-only and both rows should be read as "hinge inactive".

| configuration | MMD | corr. distance | F1 discrepancy | DCR mean | DCR 5th pct | disclosure rate |
|---|---|---|---|---|---|---|
| baseline | 0.0193 | 2.66 | −0.03 | 1.72 | 0.74 | 1.0% |
| utility only | 0.0178 | 1.91 | −0.01 | 2.02 | 1.01 | 0.4% |
| privacy only | 0.0192 | 2.64 | −0.02 | 1.72 | 0.80 | 0.8% |
| both | 0.0179 | 1.84 | −0.02 | 2.02 | 0.99 | 0.2% |

Three observations.

**The optimized terms move.** MMD and correlation distance both improve under
the utility penalties; EMD, which is not in the loss, does not improve. F1
discrepancy is $\approx 0$ everywhere: on this dataset, synthetic data trained
with a sufficient budget is as useful as real data for the downstream task.

**The privacy term was inert**, for the reasons in §4; privacy-only is
indistinguishable from baseline.

**The utility terms improved privacy.** MMD + correlation alone raised mean
DCR from $1.72$ to $2.02$, raised the 5th-percentile DCR from $0.74$ to $1.01$,
and halved the disclosure rate. "Both" is essentially "utility only", so all
of the privacy gain came from the utility terms.

A plausible mechanism: on 237 training rows a plain VAE partially memorizes
individual points. Penalties that pull the decoder toward *population-level*
statistics spread samples across the distribution instead of clustering them
on training records; better generalization is better privacy. If this holds,
the privacy–utility relationship is not strictly zero-sum in the low-data
regime — a regularizer that improves distributional fidelity also reduces
memorization.

This is a hypothesis with supporting evidence from one small dataset, not a
finding. The natural test is the same grid on UCI Adult (≈37k training rows),
where memorization should matter much less: if the DCR gain from the utility
terms shrinks there, the memorization explanation is strengthened.

---

## 8. Ethical reading

The weights $\lambda_{\text{mmd}}, \lambda_{\text{corr}}, \lambda_{\text{priv}}$
and the margin $\mu$ are not tuning knobs in the usual sense. Setting
$\lambda_{\text{priv}}$ high relative to the utility terms is a statement
that the interests of the individuals in the dataset outrank the interests
of the researcher who will use the synthetic version. Setting it to zero is
the opposite statement, made implicitly by every generator that does not have
such a term.

The loss-aware formulation does not resolve that judgment; it makes it
explicit and forces it to be made. Where a stock generator embeds the
trade-off in architectural defaults that no stakeholder ever sees, this one
exposes it as four numbers that can be argued about, documented, and set by
whoever has the standing to set them. That is the principal ethical
contribution of the approach, independent of how the empirical results
resolve.

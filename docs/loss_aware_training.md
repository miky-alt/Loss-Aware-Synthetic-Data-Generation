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
\text{STOneHot}(\text{raw}_{\text{span}}) & \text{one-hot span, hard mode}
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

**Fix: straight-through one-hot for the privacy term.** For each one-hot span
with logits $\ell$ and soft probabilities $p = \text{softmax}(\ell)$,

$$
y \;=\; \operatorname{one\_hot}\!\big(\arg\max_k \ell_k\big) \;-\; \operatorname{sg}(p) \;+\; p,
$$

where $\operatorname{sg}$ is the stop-gradient operator. In the forward pass
$y$ is exactly a one-hot (the $-\operatorname{sg}(p) + p$ term is numerically
zero), so $\tilde{x}^{\text{hard}}$ is a genuine one-hot row comparable to
$x$. In the backward pass the gradient flows through $p$ only, i.e. the
softmax gradient. The hinge is now measuring the same kind of distance that
the evaluation-time DCR measures.

We first implemented this with PyTorch's `gumbel_softmax(hard=True)`, which
adds Gumbel noise before the argmax. That version crashed intermittently on
Apple's MPS backend (`scatter: index -1 is out of bounds`): the noise is
computed as $-\log(\text{Exponential}(1))$, which is $+\infty$ whenever the
exponential draw is exactly $0$; two infinities in one span produce a NaN
softmax, a garbage argmax, and a failed one-hot scatter. The noise is not
needed for our purpose — we want a hard sample comparable to real rows, not
an unbiased stochastic relaxation — so the deterministic estimator above
replaces it. A guard also skips any training step whose loss is non-finite
and records the count in the run diagnostics.

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
run at $\mu \in \{0.5, 1.0, 1.5\}$ with $\lambda_{\text{priv}} = 1$.

**Weight sweep.** The both configuration is run at
$\lambda_{\text{priv}} \in \{10, 50\}$, $\mu \in \{1.5, 2.0\}$ to find where
utility gives way, and then at $\lambda_{\text{priv}} \in \{2, 5, 10\}$,
$\mu = 1.5$ with five seeds to resolve the transition.

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
*Result (§7e): it did not shrink — the gain was +22% on Adult against +17% on
Heart. Memorization is not the mechanism, or not the only one; see §7e.*

---

## 7b. Margin sweep at $\lambda_{\text{priv}} = 1$ (Heart Disease, 3 seeds, means)

With the fixes of §4 in place, the hinge fires. Baseline and utility-only
rows repeated from §7 for reference.

| configuration | $\mu$ | corr. distance | F1 discrepancy | DCR mean | DCR 5th pct | disclosure rate |
|---|---|---|---|---|---|---|
| baseline | – | 2.66 | −0.03 | 1.72 | 0.74 | 1.0% |
| privacy only | 0.5 | 2.49 | 0.01 | 1.74 | 0.78 | 1.1% |
| privacy only | 1.0 | 2.35 | 0.00 | 1.77 | 0.80 | 1.0% |
| privacy only | 1.5 | 2.32 | −0.02 | 1.77 | 0.83 | 0.7% |
| utility only | – | 1.91 | −0.01 | 2.02 | 1.01 | 0.4% |
| both | 0.5 | 2.08 | 0.03 | 1.99 | 0.99 | 0.2% |
| both | 1.0 | 1.94 | −0.03 | 2.02 | 0.98 | 0.4% |
| both | 1.5 | 1.92 | −0.02 | 2.04 | 1.04 | 0.2% |

**The hinge behaves as a hinge.** Its effect is monotone in $\mu$ and
concentrated in the tail: 5th-percentile DCR rises $0.74 \to 0.78 \to 0.80
\to 0.83$ and the disclosure rate falls, while mean DCR barely moves. It
cleans up the closest samples without shifting the bulk — exactly what a
margin penalty should do.

**At $\lambda_{\text{priv}} = 1$ it is weak.** The reconstruction and KL
terms are of order $10$; the hinge is of order $0.1$. Its gradient is a
rounding error. The privacy-only effect is roughly a quarter of what the
utility terms achieve on their own.

**Utility does not give.** F1 discrepancy is $\approx 0$ in every row. This
is the free-lunch region: everything improves, nothing is paid for. The
trade-off has not yet been reached.

**An unexpected side effect.** Privacy-only also improved correlation
distance ($2.66 \to 2.32$). Pushing samples off individual training records
plausibly relieves the mild mode-collapse a small VAE exhibits, incidentally
restoring second-order structure. One sentence, not a claim.

---

## 7c. Weight sweep: finding the trade-off (Heart Disease, 3 seeds, means)

All rows are the "both" configuration
($\lambda_{\text{mmd}} = 1, \lambda_{\text{corr}} = 0.5$).

| $\lambda_{\text{priv}}$ | $\mu$ | MMD | mean EMD | corr. dist | F1 discrepancy | DCR mean | DCR 5th | NNDR mean | disclosure |
|---|---|---|---|---|---|---|---|---|---|
| 0 (baseline) | – | 0.019 | 1.3 | 2.66 | −0.03 | 1.72 | 0.74 | 0.83 | 1.0% |
| 1 | 1.5 | 0.018 | 1.2 | 1.92 | −0.02 | 2.04 | 1.04 | 0.85 | 0.2% |
| 10 | 1.5 | 0.018 | 1.9 | 2.34 | 0.17† | 2.67 | 1.37 | 0.88 | 0.1% |
| 10 | 2.0 | 0.025 | 23.6 | 3.91 | 0.22 | 6.23 | 3.23 | 0.93 | 0 |
| 50 | 1.5 | 0.021 | 30.5 | 3.23 | 0.08 | 6.27 | 3.64 | 0.90 | 0 |
| 50 | 2.0 | 0.032 | 24.6 | 3.48 | 0.21 | 6.36 | 5.12 | 0.95 | 0 |

The results fall into three regimes.

† The $\lambda_{\text{priv}} = 10, \mu = 1.5$ runs in this table used the
Gumbel-noise estimator of §4.2's first attempt; the five-seed replication in
§7d with the deterministic estimator gives $0.07 \pm 0.06$. The two are
different experiments and §7d's number supersedes this one.

**Regime I — free lunch** ($\lambda_{\text{priv}} \le 1$). Utility and
privacy improve together, as in §7 and §7b. The dominant effect is the
utility terms reducing memorization.

**Regime II — trade-off** ($\lambda_{\text{priv}} = 10, \mu = 1.5$). Mean
DCR rises by 30%, the disclosure rate is effectively zero, EMD and correlation
distance degrade only mildly — and F1 discrepancy jumps to $0.17$. Privacy
now has a utility price. This is the regime the method is *for*: the weights
are buying a specific amount of one thing with a specific amount of the other.
The per-seed spread ($0.22, 0.05, 0.22$) is large; three seeds do not
resolve the number, which motivates the five-seed follow-up in §7d.

**Regime III — collapse** ($\mu = 2$, or $\lambda_{\text{priv}} = 50$). Mean
DCR saturates near $6.3$ regardless of the exact setting, mean EMD grows
roughly twenty-fold, and in one run (seed 3, $\lambda = 50, \mu = 2$)
synthetic F1 equals the majority-class baseline exactly — the generated
target column has collapsed to a single value. The generator has left the
data manifold. Privacy is "perfect" because the samples no longer resemble
anyone. This is the *reductio* the ethical discussion needs (§8): total
privacy is trivially available by generating noise, so a privacy score has
no meaning without a utility floor attached to it.

Two methodological findings fall out of the collapse regime.

**Fixed-bandwidth MMD saturates.** MMD moved from $0.018$ to $0.032$ while
EMD moved from $1$ to $25$. With a fixed RBF bandwidth $\gamma$, once the two
distributions stop overlapping every cross term $k(\tilde{x}_i, x_j) \approx 0$
and

$$
\text{MMD}^2 \;\to\; \mathbb{E}\big[k(\tilde{x}, \tilde{x}')\big] + \mathbb{E}\big[k(x, x')\big] = \text{const},
$$

so the metric cannot distinguish "far" from "very far". This has two
consequences. As an evaluation metric, MMD under-reports gross distributional
failure; EMD, which is not in the loss, is what caught it. As a *loss term*,
$\mathcal{L}_{\text{mmd}}$'s gradient vanishes exactly when it is most needed,
which is why it could not hold the line against the hinge at
$\lambda_{\text{priv}} = 50$. A multi-scale kernel (a sum over several
$\gamma$) would mitigate both; we record it as a limitation rather than
change the objective at this stage.

**NNDR inverts off-manifold.** NNDR rose to $0.95$ in the collapse regime,
which [metrics/privacy.md](metrics/privacy.md) reads as "excellent
generalization". It is not. $\text{NNDR} \approx 1$ also occurs when a sample
is far from *all* real records, because then its first and second nearest
neighbours are equidistant trivially. NNDR is only interpretable jointly with
DCR: high NNDR at moderate DCR indicates generalization; high NNDR at very
high DCR indicates samples off the manifold.

---

## 7d. Resolving the transition ($\lambda_{\text{priv}} \in \{2, 5, 10\}$, $\mu = 1.5$, 5 seeds)

Both configuration, deterministic straight-through estimator, seeds 1–5.
Mean $\pm$ standard deviation across seeds where it matters.

| $\lambda_{\text{priv}}$ | F1 discrepancy | DCR mean | DCR 5th pct | mean EMD | corr. dist | MMD | disclosure |
|---|---|---|---|---|---|---|---|
| 0 (baseline, 3 seeds) | $-0.03$ | $1.72$ | $0.74$ | $1.3$ | $2.66$ | $0.019$ | 1.0% |
| 1 (3 seeds) | $-0.02$ | $2.04$ | $1.04$ | $1.2$ | $1.92$ | $0.018$ | 0.2% |
| 2 | $-0.01 \pm 0.07$ | $2.06 \pm 0.04$ | $0.98$ | $1.37$ | $2.00$ | $0.018$ | 0.3% |
| 5 | $-0.02 \pm 0.04$ | $2.13 \pm 0.05$ | $1.08$ | $1.47$ | $2.12$ | $0.018$ | 0.3% |
| 10 | $0.07 \pm 0.06$ | $2.73 \pm 0.19$ | $1.45$ | $2.46$ | $2.28$ | $0.019$ | 0.04% |

**Privacy is monotone and well resolved.** Mean DCR rises smoothly with
$\lambda_{\text{priv}}$ and the seed-to-seed spread is small. The 5th
percentile — the most exposed samples — rises faster than the mean
($0.74 \to 1.45$, roughly doubling), which is the tail behaviour a hinge is
built for. Disclosure rate reaches zero at $\lambda = 10$.

**The knee is between $\lambda = 5$ and $\lambda = 10$.** Up to $\lambda = 5$
nothing is paid: F1 discrepancy is indistinguishable from zero and the
distributional metrics move by a few percent. At $\lambda = 10$ the cost
appears on every axis at once: mean EMD $+80\%$, correlation distance
$+14\%$, F1 discrepancy $0.07$.

**F1 on a 60-row test set cannot resolve the trade-off.** The standard
deviation of F1 discrepancy is $\approx 0.06$ at *every* $\lambda$, including
$\lambda = 2$ where the mean is zero. That is the measurement floor of a
60-row test split, not a property of the generator. Any downstream-utility
effect smaller than $\pm 0.05$ is invisible on Heart Disease; the $\lambda = 10$
cost of $0.07$ is barely one standard deviation from zero. The distributional
metrics have no such floor and show the cost unambiguously. This is the
second time EMD has been the sensitive instrument (see the MMD saturation
note in §7c) and it should be stated as a methodological conclusion:
*on small datasets, distributional fidelity metrics resolve the
privacy–utility trade-off; downstream predictive metrics do not.* Adult, with
a test split of $\approx 9{,}000$ rows, is where F1 discrepancy becomes
meaningful.

**Operating point.** $\lambda_{\text{priv}} = 10$, $\mu = 1.5$ buys a 33%
increase in mean synthetic-to-real distance and a 47% increase in the
worst-case tail, with the disclosure rate at zero, for roughly $0.07$ of
downstream F1 and a visibly degraded marginal distribution. This is the
concrete offer the ethical discussion in §8 asks stakeholders to accept or
refuse.

**Figure.** The central figure of the report is F1 discrepancy (y) against
mean DCR (x), one point per $\lambda_{\text{priv}} \in \{0, 1, 2, 5, 10\}$,
error bars over seeds, with the collapse-regime points from §7c
($\lambda = 50$, DCR $\approx 6.3$) shown at the far right to mark where the
curve ends. A second panel plots the five loss components over epochs for one
$\lambda = 10$ run, showing $\mathcal{L}_{\text{priv}}$ decaying as the model
learns to keep its distance while $\mathcal{L}_{\text{recon}}$ rises slightly
to pay for it. The script `src/experiments/plot_tradeoff.py` produces both
panels from the run index.

---

## 7e. UCI Adult: does the trade-off depend on data density?

The same three configurations on Adult ($\approx 37{,}000$ training rows,
$\approx 9{,}600$ test rows), `batch_size=500`, `epochs=100`, seeds 1–3.

| configuration | F1 discrepancy | MMD | corr. dist | DCR mean | DCR 5th pct | disclosure |
|---|---|---|---|---|---|---|
| baseline | $0.042 \pm 0.012$ | $0.0035$ | $0.87$ | $1.11$ | $0.31$ | 13.4% |
| utility only | $0.048 \pm 0.001$ | $0.0021$ | $0.69$ | $1.36$ | $0.41$ | 8.5% |
| both, $\lambda_{\text{priv}} = 10$, $\mu = 1.5$ | $\mathbf{0.036 \pm 0.002}$ | $0.0022$ | $0.73$ | $\mathbf{2.37}$ | $\mathbf{0.78}$ | $\mathbf{1.8\%}$ |

**At $\lambda_{\text{priv}} = 10$ every metric improves.** Mean DCR more than
doubles, the 5th-percentile DCR grows $2.5\times$, the disclosure rate falls
from 13% to 2% — and F1 discrepancy *decreases*, with MMD and correlation
distance both better than baseline. On Heart Disease the same configuration
was the knee of the trade-off (§7d); on Adult it is still inside the
free-lunch region.

**Interpretation: the price of privacy is set by data density.** The hinge
pushes each generated row away from its nearest real row. On a dense manifold
the row lands next to *another* plausible but fictional record; the
distribution is preserved and nothing is paid. On a sparse one the nearest
empty space is off the manifold, and the row becomes an implausible record;
the distribution degrades. The same penalty, the same weight, opposite
outcomes — determined by the data, not the objective. Adult is dense because
it has many rows in few dimensions; Heart is sparse because it has few rows.
§7f shows that a dataset can be sparse for the opposite reason — many rows in
many dimensions — and that this is what actually matters.

**F1 discrepancy is now a precise instrument.** Seed-to-seed spread is
$\pm 0.002$ on the loss-aware runs against $\pm 0.06$ on Heart (§7d). The
60-row-test-set explanation for the Heart noise is confirmed by contrast.

**The memorization hypothesis of §7 is not supported.** The utility-only
DCR gain was predicted to shrink on a dataset too large to memorize. It grew:
$+22\%$ on Adult against $+17\%$ on Heart. MMD and correlation regularization
spread the generated distribution at both scales; whatever the mechanism, it
is not specific to small data.

**Distributional fidelity is not downstream utility.** Utility-only improved
MMD by 40% and correlation distance by 20% while F1 discrepancy moved
slightly the wrong way ($0.042 \to 0.048$; within the baseline's seed spread,
so not a firm result). Matching moments and correlations does not guarantee
that the decision boundary a downstream classifier needs is preserved. This
is the second time in the project that two "utility" metrics have pointed in
different directions (see the MMD/EMD divergence in §7c), and it argues
against reporting any single utility number.

**Training dynamics.** On Heart the privacy term decayed toward zero as the
model learned to keep its distance. On Adult it plateaus at a non-zero value
and the model simply carries it — a standing cost paid without visible
strain on reconstruction. The same free-lunch story, seen from inside the
training loop.

**Two evaluation caveats specific to this dataset.**

- *Mean EMD is in raw feature units.* On Adult it is dominated by `fnlwgt`
  (range to $1.5 \times 10^6$) and `capital-gain` (to $10^5$), giving values
  in the thousands that are not comparable to Heart's $\approx 1$. Within Adult
  the relative comparison across configurations is valid; the absolute
  number is not. A per-feature-standardized EMD would fix this and is left
  as future work.
- *The 0.5 disclosure threshold is absolute.* In a dense 37k-row dataset,
  synthetic rows naturally land within $0.5$ standardized units of *something*,
  which is why the baseline disclosure rate is 13% here against 1% on Heart.
  This is the same scale problem the training margin had before §4.1 made it
  relative; the disclosure threshold should eventually receive the same
  treatment. The relative comparison across configurations stands.

---

## 7f. Diabetes 130-US Hospitals: density is not row count

The same three configurations on Diabetes ($\approx 57{,}000$ training rows,
$\approx 14{,}000$ test rows, $\approx 40$ columns after cleaning),
`batch_size=500`, `epochs=100`, seeds 1–3.

| configuration | F1 discrepancy | F1 synthetic | MMD | mean EMD | corr. dist | DCR mean | DCR 5th pct |
|---|---|---|---|---|---|---|---|
| baseline | $0.004$ | $0.838$ | $0.0019$ | $3.85$ | $4.37$ | $2.48$ | $1.48$ |
| utility only | $0.004$ | $0.838$ | $0.0015$ | $2.23$ | $2.76$ | $3.13$ | $1.79$ |
| both, $\lambda_{\text{priv}} = 10$, $\mu = 1.5$ | $0.069 \pm 0.08$ | $0.773$ | $0.0016$ | $5.58$ | $3.05$ | $10.6 \pm 3.3$ | $9.13$ |

**F1 discrepancy is degenerate here, for a third distinct reason.** In the
baseline and utility-only rows, `f1_synthetic` equals the majority-class
baseline F1 ($0.8379$) in every seed: a classifier trained on synthetic data
predicts "not readmitted" for every patient, and the real-trained classifier
barely does better ($0.8425$). The target is $\approx 89/11$ imbalanced and
*weighted* F1 is dominated by the majority class, so the downstream task sits
on a ceiling where nothing can be distinguished. Heart's F1 was noise-limited
(§7d), Adult's was informative (§7e), Diabetes's is imbalance-limited.
Positive-class F1 or AUROC would have been the appropriate metric; this is a
limitation of the evaluation design. The utility axis for Diabetes is
effectively EMD and correlation distance.

**The utility terms improve privacy on all three datasets.** DCR $+26\%$
here, against $+17\%$ (Heart) and $+22\%$ (Adult), alongside a 37% reduction
in correlation distance and 42% in mean EMD. Three datasets spanning two
orders of magnitude in size, same direction. This is the project's most
robust empirical result: *distributional-fidelity regularization is itself a
privacy mechanism*, independent of dataset scale.

**At $\lambda_{\text{priv}} = 10$, Diabetes behaves like Heart, not Adult.**
Mean DCR quadruples, the 5th-percentile DCR ($9.1$) exceeds the baseline
*mean*, NNDR reaches $0.97$, mean EMD rises 45%, and in seed 2 synthetic F1
falls *below* the majority baseline — the synthetic target distribution
itself has been distorted. These are the Regime III signatures of §7c,
appearing at a weight that was free on Adult and only mildly costly on Heart.

**Density, not row count.** Diabetes has twice Adult's rows and pays for
privacy anyway. It also has three times Adult's columns, many of them
high-cardinality categoricals that mode-specific normalization expands into a
very wide transformed space. Volume grows exponentially with dimension;
$57{,}000$ rows in $\approx 40$ dimensions are far sparser than $37{,}000$ rows
in $14$. The three datasets therefore line up on a single axis:

| dataset | rows | columns | density | privacy at $\lambda = 10$ |
|---|---|---|---|---|
| Heart | 237 | 13 | sparse (few rows) | costs utility |
| Adult | 37,000 | 14 | dense | free |
| Diabetes | 57,000 | $\approx 40$ | sparse (many dims) | costs utility |

This refines the §7e claim: it is the density of the real data in the
generator's feature space — rows relative to dimensionality — that sets the
price of a privacy penalty, not the number of records.

**A confound that must be stated.** The hinge is
$\max(0, m - d_i)$. The margin $m$ is relative to the data's scale (§4.1),
but the *magnitude* of the penalty is not: in a wider transformed space both
$m$ and $d_i$ are larger, so a fixed $\lambda_{\text{priv}} = 10$ is
effectively a stronger penalty on Diabetes than on Adult. Part of the
Diabetes result may therefore be "$\lambda = 10$ means more here" rather than
"the manifold is sparser here"; the two are not separable with these runs.
The fix is to normalize the hinge by $m$,

$$
\mathcal{L}_{\text{priv}} \;=\; \frac{1}{B}\sum_i \max\!\Big(0,\; 1 - \frac{d_i}{m}\Big),
$$

which makes $\lambda_{\text{priv}}$ dimension-invariant. It is a one-line
change but would require re-running every experiment, so it is recorded as
future work.

**The Diabetes curve** ($\lambda_{\text{priv}} \in \{2, 5\}$, 3 seeds each,
added to the rows above):

| $\lambda_{\text{priv}}$ | F1 discrepancy | F1 synthetic | mean EMD | corr. dist | DCR mean | DCR 5th pct | NNDR |
|---|---|---|---|---|---|---|---|
| 0 (baseline) | $0.004$ | $0.838$ | $3.85$ | $4.37$ | $2.48$ | $1.48$ | $0.89$ |
| 0 (utility only) | $0.004$ | $0.838$ | $2.23$ | $2.76$ | $3.13$ | $1.79$ | $0.91$ |
| 2 | $0.004$ | $0.838$ | $\mathbf{1.66}$ | $\mathbf{2.65}$ | $3.35 \pm 0.04$ | $1.98$ | $0.91$ |
| 5 | $0.048 \pm 0.017$ | $0.794$ | $3.52$ | $3.29$ | $7.32 \pm 0.5$ | $3.86$ | $0.96$ |
| 10 | $0.069 \pm 0.08$ | $0.773$ | $5.58$ | $3.05$ | $10.6 \pm 3.3$ | $9.13$ | $0.97$ |

$\lambda = 2$ is the best configuration on the table: lowest EMD and
correlation distance of any row, DCR $+35\%$ over baseline, F1 untouched.
The transition to the collapse regime is *sharp* — DCR more than doubles
between $\lambda = 2$ and $\lambda = 5$, where the same interval on Heart
moved it from $2.06$ to $2.13$. The generator holds the manifold until it
cannot, then leaves it; NNDR jumping from $0.91$ to $0.96$ at the same point
is the off-manifold signature. At $\lambda \ge 5$ synthetic F1 falls *below*
the majority baseline ($0.79$ vs $0.84$): a classifier that does worse than
always guessing the majority has been trained on a table whose label marginal
is distorted. On Diabetes, F1 discrepancy measures corruption of the target
distribution, not loss of predictive signal.

**The knees order as Adult ($>10$) $>$ Heart ($5$–$10$) $>$ Diabetes
($2$–$5$).** The density account predicts this ordering because Diabetes is
the sparsest in its feature space. The hinge-scale confound predicts the same
ordering because $\lambda$ is effectively larger in a wider space. The
ordering is a robust result; its explanation is not yet unique.

**Absolute privacy numbers are not comparable across datasets.** Baseline
mean DCR is $1.1$ on Adult and $2.5$ on Diabetes, and baseline disclosure is
13% against 0%. Both differences are dimensionality, not privacy: distances
grow with the number of features and the $0.5$ threshold does not. Only
within-dataset relative comparisons carry meaning, which the §7e caveat
already anticipated and Diabetes confirms.

---

## 7g. Per-feature analysis: where the cost lands

`src/experiments/plot_feature_emd.py` averages `emd_per_feature` across
seeds for each $\lambda_{\text{priv}}$ and compares it with the plain
baseline. For a binary label-encoded column, EMD is $|p_{\text{real}} -
p_{\text{synth}}|$, so it reads directly as a shift in proportion. Figures:
`experiments/figures/feature_emd_*.png`; tables: the matching `.csv`.

### The Adult "free lunch" was not free

At $\lambda = 10$ on Adult every aggregate metric improved (§7e). Ranking
columns by standardized change ((EMD − baseline EMD) / real std), the top six
are **native-country, race, workclass, sex, income, relationship** — all
categorical, four protected or sensitive attributes, one the target. The
continuous columns sit at the bottom or improved.

| column | k | baseline EMD | $\lambda = 10$ EMD | reading (deviation from real; baseline in parentheses) |
|---|---|---|---|---|
| native-country | 42 | 1.4 | 9.1 | +1.0 sd, largest standardized shift |
| race | 5 | 0.16 | 0.91 | +0.9 sd |
| sex | 2 | 0.010 | 0.144 | sex ratio off by **14 pp** (1 pp) |
| income (target) | 2 | 0.013 | 0.111 | positive rate off by **11 pp** (1 pp) |
| fnlwgt, education, hours-per-week | — | — | — | *improved* (ratio 0.3–0.6) |

The hinge bought its privacy by distorting the marginals of exactly the
protected attributes and the target, while the high-cardinality continuous
columns got better. No aggregate metric registered it: weighted F1 is scored
on real test rows and a classifier trained on a table with 14 pp too many of
one sex still classifies real rows well; Pearson correlation is nearly
invariant to a marginal shift; fixed-bandwidth MMD in 14 dimensions barely
registers one binary coordinate moving. **The verdict "free" in §7e was
delivered by instruments structurally blind to the ethically salient
distortion.** §7e stands for the metrics it reports; its interpretation is
withdrawn.

### Heart and Diabetes

| dataset | config | column | baseline | after | reading |
|---|---|---|---|---|---|
| Heart | $\lambda = 10$ | num (target) | 0.013 | 0.229 | disease prevalence off by **23 pp** (1 pp) |
| Heart | $\lambda = 10$ | sex | 0.057 | 0.118 | sex ratio off by 12 pp (6 pp) |
| Heart | $\lambda = 50$ | chol | 6.3 | 280 | mean cholesterol off by 280 mg/dL |
| Heart | $\lambda = 50$ | trestbps / thalach / age | 5.3 / 3.7 / 2.0 | 59 / 62 / 20.6 | BP 59 mmHg, HR 62 bpm, age 20 yr |
| Diabetes | $\lambda = 2$ | diag_1 (k=656) | 46.5 | 8.5 | *improved* 5× |
| Diabetes | $\lambda = 2$ | gender | 0.125 | 0.012 | *improved* 10× |
| Diabetes | $\lambda = 10$ | readmitted (target) | 0.11 | 0.75 | readmission rate off by **75 pp** (11 pp) — majority class inverted |
| Diabetes | $\lambda = 10$ | diabetesMed | 0.016 | 0.39 | off by 39 pp (2 pp) |
| Diabetes | $\lambda = 10$ | acarbose (k=4) | 0.003 | 0.33 | rare drug (~0.3% real) in ~1/3 of rows |

The Regime II cost on Heart ("0.07 of F1", §7d) was a 23-point shift in
disease prevalence. The collapse regime on Heart is describable in physical
units: synthetic patients whose cholesterol is off by 280 mg/dL. On Diabetes
the 75-point target shift is why synthetic F1 fell below the majority
baseline (§7f): the synthetic table has the classes inverted.

### What breaks is the sparsest region of each column

The cardinality test was inconclusive ($r = +0.76$ Heart, $-0.56$ Adult,
undefined Diabetes) because cardinality is the wrong variable. What degrades
is the *minority value* of binaries (sex, the target), *rare categories*
(acarbose), and the *tails* of continuous columns (cholesterol). High-
cardinality columns sometimes improve. The hinge escapes real records by
moving mass into whatever region of each column is least populated. This is
the density account of §7f at the level of individual features, with a
mechanism attached — and it means the collapse regime does not merely produce
implausible records; it produces records that systematically over-represent
rare categories and minority classes.

### Low $\lambda$ is a genuine sweet spot

At $\lambda = 2$ on Diabetes nearly every column improves (diag_1 by 5×,
gender by 10×). At $\lambda \le 5$ on Heart most ratios are below one. The
free-lunch region is real; it is narrower than the aggregate metrics
suggested, and on Adult it does not extend to $\lambda = 10$.

### Consequence for the evaluation framework

Per-column marginal checks on protected attributes and on the target are not
optional extras to a utility evaluation. Without them, the framework in
`src/evaluation/` does not measure what it claims to. This should be treated
as a required addition, not future work.

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

The three regimes of §7c sharpen this. Regime I shows that the trade-off is
not always zero-sum — in the low-data setting a well-regularized generator is
both more useful and safer, so the common framing of "privacy *versus*
utility" is at best incomplete. Regime III shows that privacy metrics on
their own are trivially satisfiable and therefore meaningless as a target:
any claim that a synthetic dataset is "private" must be read alongside a
claim about what it is still good for. Regime II is where the actual decision
lives, and it is a decision — a 33% increase in the distance between
synthetic and real patients, purchased with roughly $0.07$ of downstream F1
and a measurably distorted marginal distribution —
that a data-protection officer, a clinical researcher and a patient
representative would each weigh differently. The contribution of the method
is that they now have a concrete number to disagree about.

Adult and Diabetes (§7e, §7f) add the sharpest point. The identical
configuration that costs utility on 237 patients costs nothing on 37,000
census respondents — and costs utility again on 57,000 hospital admissions
described by 40 attributes. Whether a privacy constraint has a price is not a
property of the constraint; it is a property of how densely the protected
population fills its own feature space. Small groups are sparse because they
are small; richly described groups are sparse because they are richly
described. Rare conditions, detailed clinical records, and under-represented
populations — exactly the settings in which privacy failures are most
consequential — are also the settings in which privacy protection is most
expensive in utility terms. Any policy that sets a single privacy–utility
weighting for all datasets will, as a consequence, either under-protect the
sparse population or over-pay on the dense one. The weighting has to be set
per dataset, and the data's density is one of the things whoever sets it
needs to see.

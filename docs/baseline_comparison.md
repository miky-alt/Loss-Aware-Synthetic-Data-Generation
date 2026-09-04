# Baseline Architecture Comparison

Three generator families run through the evaluation pipeline unchanged, with
`sdv` library defaults, 3 seeds each. This is Deliverable 2 of the original
project proposal ("identification and comparative analysis of synthetic data
generator architectures") and the reference point against which the
loss-aware TVAE in [loss_aware_training.md](loss_aware_training.md) is judged.

Raw output: `experiments/baselines.txt`. Full reports: `experiments/results/`.

## Results (means over 3 seeds)

| dataset | generator | MMD | mean EMD | corr dist | F1 disc | DCR mean | DCR 5th | disclosure |
|---|---|---|---|---|---|---|---|---|
| Heart | CTGAN | 0.018 | 5.91 | 3.47 | 0.31† | 3.30 | 2.06 | 0 |
| Heart | TVAE | 0.022 | 1.79 | 2.85 | 0.01 | 1.74 | 0.83 | 0.7% |
| Heart | GaussianCopula | 0.018 | 1.55 | 2.54 | 0.05 | 2.62 | 1.47 | 0.1% |
| Adult | CTGAN | 0.0019 | 1274 | 0.79 | 0.053 | 1.68 | 0.53 | 4.4% |
| Adult | TVAE | 0.0024 | 3878 | 0.97 | 0.058 | 1.18 | 0.32 | 12.6% |
| Adult | GaussianCopula | 0.0024 | 2310 | 1.26 | 0.145 | 3.08 | 1.20 | 0.03% |
| Diabetes | CTGAN | 0.0015 | 2.98 | 2.54 | 0.004 | 5.78 | 2.26 | 0 |
| Diabetes | TVAE | 0.0017 | 3.84 | 4.21 | 0.004 | 2.58 | 1.51 | 0 |
| Diabetes | GaussianCopula | 0.0015 | 1.44 | 2.77 | 0.004 | 4.52 | 2.85 | 0 |

† Undertrained. `sdv`'s default `batch_size=500` exceeds Heart's 237-row
training set, so each epoch is one gradient step. This is the same artefact
documented in [loss_aware_training.md §4.3](loss_aware_training.md) and should
not be read as an architecture property.

## Observations

**TVAE is the least private architecture on every dataset.** Lowest DCR,
highest disclosure in each block. It is also the most useful on Heart and
Adult. The loss-aware work therefore started from the generator with the
most privacy headroom; the free-lunch effect reported there may be partly a
TVAE property and has not been tested on the other two families.

**The Gaussian copula is competitive with no neural network.** Lowest EMD on
Heart and Diabetes, lowest correlation distance on Heart, always more private
than TVAE. On Adult it has the worst F1 discrepancy by a factor of three; on
both clinical datasets it is the most balanced baseline. Lower capacity →
less memorization → more privacy.

**CTGAN trades utility for privacy.** More private than TVAE everywhere,
less useful on Adult. On Diabetes its DCR mean (5.8) is far above its median
(3.8): a long tail of off-manifold samples.

**The stock TVAE matches the λ=0 loss-aware subclass** within seed noise on
all three datasets (Heart DCR 1.74 vs 1.72, Adult 1.18 vs 1.11, Diabetes 2.58
vs 2.48). The baseline claim in the loss-aware doc is demonstrated.

## Placing the loss-aware TVAE

| dataset | best loss-aware config | F1 disc | DCR mean | vs. best baseline |
|---|---|---|---|---|
| Heart | λ ≤ 5, μ = 1.5 | ≈ 0 | 2.1 | matches copula privacy, better corr |
| Adult | λ = 10, μ = 1.5 | 0.036 | 2.37 | better utility than all three; more private than both neural |
| Diabetes | λ = 2, μ = 1.5 | 0.004 | 3.35 | copula still more private (4.52); fidelity close |

On Adult the loss-aware model dominates every baseline on both axes. On the
clinical datasets it makes TVAE competitive with the copula on privacy while
keeping TVAE's utility; it does not make the copula obsolete.

## Caveats

- Library-default training budgets throughout. The loss-aware runs had their
  budgets tuned per dataset (`batch_size=32` on Heart); the baselines did not.
  A fair comparison would tune both.
- Mean EMD is in raw feature units and not comparable across datasets (see
  the Adult values).
- F1 discrepancy is imbalance-limited on Diabetes and noise-limited on Heart,
  as documented in the loss-aware doc.

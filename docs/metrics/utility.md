# Utility Metrics

Utility metrics answer the question: **how much statistical and predictive value is lost when replacing real data with synthetic data?**

All metrics in this module follow the same convention: **lower score = less utility loss = better synthetic data**. This makes them directly composable into a penalty term for the loss-aware training objective.

---

## 1. Maximum Mean Discrepancy (MMD)

### What it measures

MMD measures the **overall distributional distance** between real and synthetic data in a reproducing kernel Hilbert space (RKHS). Informally: if you project both datasets into a high-dimensional feature space defined by a kernel function, MMD is the distance between their mean embeddings.

A low MMD means the two datasets are statistically indistinguishable under the chosen kernel — their joint distributions are similar. A high MMD means a statistical test could easily tell them apart.

### Mathematical definition

Given real samples X = {x₁, ..., xₙ} and synthetic samples Y = {y₁, ..., yₘ}, the empirical MMD² with RBF kernel k(x, y) = exp(-γ ||x - y||²) is:

```
MMD²(X, Y) = E[k(x, x')] + E[k(y, y')] - 2·E[k(x, y)]
```

where expectations are over independent draws from each set. This is estimated as the average of the kernel matrix within each set, minus twice the average of the cross-kernel matrix.

### Implementation choices

- **RBF kernel with γ=1.0**: the RBF kernel is the standard choice for MMD because it is characteristic — meaning MMD=0 if and only if the two distributions are identical. Other kernels (e.g. polynomial) do not have this property. γ=1.0 is a reasonable default on normalized data; a more thorough study would use median heuristic bandwidth selection.
- **Normalization before kernel computation**: features are standardized (zero mean, unit variance) so that the kernel bandwidth γ=1.0 is meaningful across datasets with different feature scales.
- **Subsampling to 2000 records**: the kernel matrix computation is O(n²) in memory. Subsampling keeps it tractable while preserving statistical accuracy.
- **Clipping to 0**: MMD² is theoretically non-negative, but floating-point arithmetic can produce small negative values. We clip to 0 for numerical safety.

### Interpretation

| MMD value | Interpretation |
|-----------|---------------|
| ~0.0 | Distributions are nearly identical |
| 0.01–0.1 | Moderate distributional shift |
| >0.1 | Significant distributional mismatch |

These thresholds are approximate and depend on the dataset and kernel bandwidth.

**Saturation.** With a fixed bandwidth, MMD stops being informative once the two distributions no longer overlap: every cross-kernel term goes to zero and MMD² flattens to the constant E[k(x,x')] + E[k(y,y')]. It can therefore distinguish "close" from "far" but not "far" from "very far". We saw this in the loss-aware collapse regime, where MMD moved from 0.018 to 0.032 while mean EMD moved from 1 to 25 (see [loss_aware_training.md](../loss_aware_training.md#7c-weight-sweep-finding-the-trade-off-heart-disease-3-seeds-means)). MMD should be read together with EMD, which has no such ceiling; a multi-scale kernel (sum over several γ) would reduce the problem and is left as future work.

---

## 2. Earth Mover's Distance (EMD)

### What it measures

EMD, also known as the Wasserstein-1 distance, measures **how much work is required to transform one probability distribution into another**. Unlike MMD, which operates on the joint distribution, EMD is computed **per feature** and then averaged. This makes it more interpretable: you can see exactly which features are poorly preserved by the generator.

### Mathematical definition

For two 1D distributions P and Q with CDFs F_P and F_Q:

```
EMD(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
```

Intuitively, this is the area between the two cumulative distribution functions.

### Implementation choices

- **Per-feature computation**: we compute EMD separately for each numeric feature and return both the per-feature scores and the mean. This granularity is essential for diagnosing generator failures — a generator might faithfully reproduce most features while catastrophically failing on one.
- **`scipy.stats.wasserstein_distance`**: this is the standard, well-tested implementation of 1D Wasserstein distance. It operates on empirical samples directly (no density estimation required), which makes it robust on small datasets like Heart Disease.
- **No subsampling**: unlike kernel-based metrics, EMD on 1D empirical distributions does not require pairwise distance computation and scales linearly with sample size. Subsampling is therefore unnecessary.

### Interpretation

EMD is in the same units as the original feature values (before any normalization). This makes cross-feature comparison difficult for unnormalized data. For this reason, we use mean EMD as a relative indicator rather than an absolute threshold — the important comparison is between generators on the same dataset, or between datasets for the same generator.

---

## 3. Categorical Distribution Distance

### What it measures

Categorical Distribution Distance measures whether synthetic categorical and
boolean features preserve the frequency of each category. It uses the total
variation distance, which is half the L1 distance between the real and
synthetic category-frequency distributions.

```text
TVD(P, Q) = 1/2 · Σ_c |P(c) - Q(c)|
```

The score ranges from 0 (identical distributions) to 1 (disjoint
distributions). Categories present in only one dataset are included with zero
frequency in the other dataset. The report contains both the mean score and a
per-feature breakdown under `categorical_distance_per_feature`.

This metric complements EMD: EMD covers numeric columns, while this metric
ensures that columns such as `race`, `gender`, medication indicators, and
boolean targets are evaluated rather than silently omitted from distributional
utility.

## 4. F1 Discrepancy

### What it measures

F1 Discrepancy measures the **downstream predictive utility gap**: how much worse is a model trained on synthetic data compared to a model trained on real data, when both are evaluated on the same held-out real test set?

This is the most directly practically relevant metric. A synthetic dataset with low F1 discrepancy is a genuine substitute for real data in machine learning pipelines — the key use case for synthetic data in research and industry.

### Protocol

```
1. Split real data: 80% train, 20% test (stratified)
2. Train classifier A on real train → predict on real test → F1_real
3. Train classifier B on synthetic (full) → predict on real test → F1_synthetic
4. F1_discrepancy = F1_real - F1_synthetic
```

The test set is always drawn from real data. This is critical: evaluating on synthetic test data would not measure real-world utility — it would only measure whether the generator is internally consistent.

### Implementation choices

- **Random Forest with 100 estimators**: see [architecture.md](../architecture.md) for rationale. In brief: robust, consistent, and handles mixed feature types without preprocessing.
- **Weighted F1**: we use `average="weighted"` to account for class imbalance. All three datasets have imbalanced targets (especially Diabetes, where early readmission is rare), and weighted F1 correctly penalizes classifiers that simply predict the majority class.
- **Stratified split**: the train/test split is stratified by the target column to ensure both splits have representative class proportions, which is especially important on small datasets (Heart Disease).
- **Synthetic data used in full**: classifier B is trained on the entire synthetic dataset, not just 80% of it. This reflects the intended use case: a practitioner would use all available synthetic data for training.

### Interpretation

| F1 discrepancy | Interpretation |
|----------------|---------------|
| < 0.02 | Negligible utility loss — synthetic data is an excellent substitute |
| 0.02–0.05 | Acceptable utility loss for most applications |
| 0.05–0.10 | Moderate utility loss — may affect downstream model quality |
| > 0.10 | Significant utility loss — synthetic data is a poor substitute |

A negative discrepancy (synthetic F1 > real F1) is theoretically possible if the synthetic data acts as a regularizer or if the real training set is particularly noisy. This would be reported as-is.

---

## 5. Correlation Distance

### What it measures

Correlation Distance measures **how well feature relationships are preserved** by the generator. A synthetic dataset might match the marginal distribution of each feature perfectly (low EMD per feature) while completely destroying the correlations between features — which would make it useless for any task that depends on feature interactions.

### Mathematical definition

```
correlation_distance = ||Corr(X_real) - Corr(X_synthetic)||_F
```

where Corr is the Pearson correlation matrix and ||·||_F is the Frobenius norm (square root of the sum of squared element-wise differences).

### Implementation choices

- **Pearson correlation**: we use Pearson (linear) correlation because it is standard, interpretable, and sufficient for detecting whether generators preserve feature co-variation. Spearman or Kendall rank correlations would be more robust to non-linearity but are slower to compute and harder to embed in a loss function.
- **Frobenius norm**: this treats all pairwise correlations equally, summing the squared differences across the entire matrix. An alternative would be to weight correlations by their magnitude (giving more importance to strongly correlated feature pairs), but the uniform Frobenius norm is simpler and still sensitive to meaningful distortions.
- **`fillna(0)`**: if a feature has zero variance (constant column), its correlation with other features is undefined. We fill these with 0 rather than propagating NaN through the norm computation.

### Interpretation

The Frobenius norm of a p×p correlation matrix ranges from 0 (identical matrices) to approximately √(2p²) in the worst case (all correlations reversed). For our datasets with 13–50 features, a value below ~1.0 indicates good correlation preservation, and values above ~5.0 indicate substantial structural distortion.

---

## Unified Report

`compute_utility_report(real, synthetic, target_col)` runs all five metrics and returns a single dict:

```python
{
    "mmd": float,                          # overall distributional distance
    "mean_emd": float,                     # average per-feature Wasserstein distance
    "emd_per_feature": {col: float, ...},  # per-feature breakdown
    "mean_categorical_distance": float,    # average categorical TVD
    "categorical_distance_per_feature": {col: float, ...},
    "correlation_distance": float,         # Frobenius norm of correlation matrix diff
    "f1_real": float,                      # baseline F1 on real data
    "f1_synthetic": float,                 # F1 when trained on synthetic
    "f1_discrepancy": float,               # gap (higher = more utility loss)
}
```

The `emd_per_feature` dict is kept separate from the top-level keys to avoid cluttering the summary while still making it accessible for feature-level analysis.

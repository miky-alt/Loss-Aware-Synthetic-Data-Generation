# Privacy Metrics

Privacy metrics answer the question: **how much private information from the real dataset is leaked by the synthetic dataset?**

All metrics follow the convention: **higher score = more privacy risk = worse synthetic data**. This mirrors the utility metrics convention (higher = more utility loss), making both sets of metrics directly composable into a unified loss function.

An important caveat applies to all metrics in this module: they are **empirical approximations** of privacy risk, not formal mathematical guarantees. They simulate realistic attack scenarios and measure their success, but they do not provide the provable bounds that formal differential privacy (DP) would. See the [Relationship to Differential Privacy](#relationship-to-differential-privacy) section for discussion.

---

## 1. Distance to Closest Record (DCR)

### What it measures

DCR measures **re-identification risk** by asking: for each synthetic record, how close is the nearest real record? If synthetic records are very close to real ones, an attacker who obtains the synthetic dataset could use those near-matches to identify real individuals — particularly in combination with auxiliary information (a linkage attack).

A good synthetic dataset should have synthetic records distributed throughout the real data's feature space, not clustered around individual real records.

### Protocol

```
For each synthetic record s:
    find the nearest real record r = argmin_r dist(s, r)
    record distance d(s) = ||s - r||₂

Aggregate: mean, median, min, 5th percentile of {d(s)}
```

We report multiple statistics because the distribution of DCR scores matters, not just the average. A generator might achieve a high mean DCR while having a long tail of near-zero distances — a small number of near-copies that represent severe individual-level privacy violations. The minimum and 5th percentile capture this tail.

### Implementation choices

- **Euclidean distance on normalized features**: features are standardized before distance computation so that all features contribute equally to the distance. Without normalization, high-variance numerical features would dominate the distance and mask proximity in lower-variance dimensions.
- **NearestNeighbors with n_neighbors=1**: we use scikit-learn's efficient ball-tree nearest neighbor search rather than brute-force pairwise distance computation.
- **Subsampling to 2000 records**: nearest neighbor computation is O(n·m) where n and m are the sizes of the real and synthetic datasets. Subsampling keeps this tractable for the Diabetes dataset.

### Interpretation

DCR values are in standardized feature space (after StandardScaler), so they are dimensionless and comparable across datasets.

| DCR (mean) | Interpretation |
|------------|---------------|
| < 0.5 | High re-identification risk — synthetic records are close copies |
| 0.5–1.0 | Moderate risk |
| > 1.0 | Low risk — synthetic records are well separated from real ones |

These thresholds are heuristic. The more meaningful comparison is between generators on the same dataset: a generator with higher DCR is safer.

---

## 2. Nearest Neighbor Distance Ratio (NNDR)

### What it measures

NNDR detects **generator memorization**: cases where the generator has not learned the underlying data distribution but has instead memorized individual training examples and reproduces them with slight perturbation.

The insight is that a memorized synthetic record will be much closer to one specific real record than to any other — its nearest neighbor will be dramatically closer than its second nearest neighbor. A genuinely generalized synthetic record will sit in a region of the feature space where many real records exist, so the first and second nearest neighbors will be at similar distances.

### Mathematical definition

```
For each synthetic record s:
    d₁(s) = distance to nearest real record
    d₂(s) = distance to second nearest real record
    NNDR(s) = d₁(s) / d₂(s)
```

NNDR close to 1 → both nearest real records are equidistant → synthetic record is in a dense region → generalization (good).
NNDR close to 0 → first nearest neighbor is much closer than second → synthetic record is a near-copy of one real record → memorization (bad).

### Implementation choices

- **NearestNeighbors with n_neighbors=2**: we query the two nearest neighbors in a single pass for efficiency.
- **Epsilon for numerical stability**: we add a small epsilon (1e-10) to the denominator to prevent division by zero in the degenerate case where d₂=0 (which would imply two identical real records).
- **5th percentile as the primary risk indicator**: the mean NNDR summarizes the overall memorization level, but the 5th percentile identifies the worst-case tail — the fraction of synthetic records most likely to be near-copies.

### Interpretation

| NNDR (mean) | Interpretation |
|-------------|---------------|
| > 0.8 | Good generalization — synthetic records are not memorized |
| 0.5–0.8 | Some memorization present |
| < 0.5 | Significant memorization — generator is reproducing training examples |

On small datasets (Heart Disease, ~300 rows), NNDR is expected to be lower because the generator has fewer examples to generalize from. This is a known failure mode of generative models in low-data regimes and is one reason we include the Heart Disease dataset explicitly.

---

## 3. Inference Risk

### What it measures

Inference Risk simulates an **attribute inference attack**: an adversary who has access to the synthetic dataset attempts to infer a sensitive attribute of real individuals (e.g. income, diagnosis, disease status) using a classifier trained entirely on synthetic data.

If this classifier achieves significantly better-than-chance performance on real individuals, it means the synthetic data has encoded enough signal about the sensitive attribute to enable inference — a direct privacy violation.

### Protocol

```
1. Train classifier C on synthetic data to predict sensitive_col from other features
2. Evaluate C on real data → inference_f1
3. Compute majority-class baseline on real data → baseline_f1
4. inference_risk_delta = inference_f1 - baseline_f1
```

The baseline F1 represents what an attacker achieves by always guessing the most common class — no information required. Any lift above the baseline (`inference_risk_delta > 0`) represents genuine information leakage attributable to the synthetic data.

### Implementation choices

- **sensitive_col = target_col**: in our evaluation, we use the classification target (income, readmission, heart disease) as the sensitive attribute. This is a reasonable choice because it is the most practically sensitive attribute in each dataset and is the column most likely to be targeted by an adversary.
- **Weighted F1**: same rationale as in utility.py — class imbalance in the target variables requires weighted averaging to avoid misleadingly high accuracy from majority-class prediction.
- **No train/test split on synthetic data**: the attacker trains on all available synthetic data, representing the worst-case scenario where they have full access to the synthetic dataset.

### Interpretation

| inference_risk_delta | Interpretation |
|----------------------|---------------|
| ≤ 0.0 | No measurable inference leakage |
| 0.0–0.05 | Low leakage — marginal lift above baseline |
| 0.05–0.15 | Moderate leakage — attribute is partially recoverable |
| > 0.15 | High leakage — synthetic data substantially enables attribute inference |

---

## 4. Disclosure Protection

### What it measures

Disclosure Protection is the most direct privacy metric: it counts the **fraction of synthetic records that are near-copies of real records**, using a hard distance threshold. These records represent direct disclosure risks — an adversary could match them to real individuals with high confidence.

While DCR and NNDR characterize the distribution of distances, Disclosure Protection gives a binary answer per record (at-risk or not) and aggregates into a rate. This is the metric most directly interpretable to a non-technical audience: "X% of synthetic records are dangerously close to real ones."

### Protocol

```
For each synthetic record s:
    d(s) = distance to nearest real record
    at_risk(s) = 1 if d(s) < threshold else 0

disclosure_rate = mean(at_risk)
n_at_risk = sum(at_risk)
```

### Implementation choices

- **Default threshold of 0.5**: in standardized feature space (after StandardScaler), a distance of 0.5 corresponds to records that differ by less than half a standard deviation across features on average. This is a reasonably tight threshold — records at this distance would be near-indistinguishable to an attacker.
- **Threshold is configurable**: the threshold can be adjusted via the `disclosure_threshold` parameter. Stricter privacy requirements warrant a lower threshold (e.g. 0.3); more lenient requirements might use 1.0.
- **Reporting both rate and count**: the disclosure rate (fraction) is useful for comparing across datasets of different sizes; the absolute count (`n_at_risk`) is useful for understanding the scale of the problem.

### Interpretation

| disclosure_rate | Interpretation |
|-----------------|---------------|
| < 0.01 | Excellent — fewer than 1% of synthetic records are at risk |
| 0.01–0.05 | Acceptable for most applications |
| 0.05–0.20 | Elevated risk — review generator configuration |
| > 0.20 | High risk — generator is producing too many near-copies |

---

## Unified Report

`compute_privacy_report(real, synthetic, sensitive_col, disclosure_threshold)` runs all four metrics and returns a single dict:

```python
{
    # DCR
    "dcr_mean": float,
    "dcr_median": float,
    "dcr_min": float,
    "dcr_5th_percentile": float,

    # NNDR
    "nndr_mean": float,
    "nndr_median": float,
    "nndr_5th_percentile": float,

    # Inference Risk
    "inference_f1": float,
    "baseline_f1": float,
    "inference_risk_delta": float,

    # Disclosure Protection
    "disclosure_rate": float,
    "n_at_risk": int,
    "threshold_used": float,
}
```

---

## Relationship to Differential Privacy

Our privacy metrics are fundamentally different from formal **differential privacy (DP)**, and it is important to be explicit about this distinction.

Differential privacy provides a mathematical guarantee: a DP-trained generator ensures that the probability of any output changes by at most a factor of e^ε when any single training record is added or removed. This is a worst-case, adversary-agnostic guarantee that holds regardless of what auxiliary information an attacker has.

Our metrics, by contrast, simulate specific attack scenarios (nearest-neighbor linkage, attribute inference) and measure their empirical success. They are:

- **Attack-specific**: they only measure privacy against the attacks we implemented. A more sophisticated attacker might succeed where our simulated attacker fails.
- **Dataset-dependent**: our thresholds and interpretations depend on the specific datasets and their feature distributions.
- **Not composable**: DP has a composition theorem that lets you reason about privacy across multiple queries. Our empirical metrics do not compose in this way.

The advantage of our approach is interpretability and flexibility: DP requires modifying the training procedure (adding calibrated noise), while our metrics can be applied to any generator without modification. For the purposes of this project — comparing generators and designing a utility-privacy trade-off loss function — empirical metrics are sufficient and more directly actionable.

A complete production privacy framework would combine both: use our empirical metrics for iterative development and generator comparison, and use formal DP guarantees for deployment.

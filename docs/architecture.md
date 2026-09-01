# Architecture and Design Decisions

## Overview

The project is structured as a pipeline with three clearly separated concerns:

```
Data Loading → Synthetic Generation → Evaluation (Utility + Privacy)
```

Each stage is implemented as an independent module with a well-defined interface, so that either stage can be swapped or extended without affecting the others. This separation was intentional from the start and reflects a key design principle: **the evaluation framework should be generator-agnostic**.

---

## Module Map

```
src/
  data/
    loader.py           # Stage 1: data ingestion and preprocessing
  generators/           # Stage 2: synthetic data generation
    base.py             # SyntheticGenerator interface (fit/sample contract)
    baseline.py         # CTGANGenerator, TVAEGenerator (sdv-backed)
    registry.py         # GENERATORS dict + build_generator() — single extension point
  evaluation/
    utility.py          # Stage 3a: utility metrics
    privacy.py          # Stage 3b: privacy metrics
  experiments/          # Orchestration layer
    config.py           # TrainingConfig + ExperimentMode
    experiment.py       # run_experiment(): full pipeline entry point
    report.py           # build_report, save_report, append_to_index, summarize_report
    persistence.py      # save/load generator (.pkl), save_metadata, load_report
    query.py            # load_index, query_index — find runs by criteria
  main.py               # CLI: run / evaluate / show / find subcommands
```

See [generators.md](generators.md) for the rationale behind the generator interface and baseline choices, and [experiments.md](experiments.md) for the orchestration layer design.

---

## Data Flow

```
load_dataset(name)
    └─→ DatasetBundle(real, target_col, name, domain)
            │
            └─→ split_dataset(bundle, test_size, seed)
                    └─→ (train_bundle, test_bundle)   [stratified, deterministic]
                            │
                            ├─→ generator.fit(train_bundle.real)
                            │       └─→ synthetic: pd.DataFrame
                            │
                            ├─→ compute_utility_report(test_bundle.real, synthetic, target_col)
                            │       └─→ dict of utility scores
                            │
                            └─→ compute_privacy_report(test_bundle.real, synthetic, sensitive_col)
                                    └─→ dict of privacy scores
```

The generator only ever sees `train_bundle.real`; every metric is computed against `test_bundle.real`, which it never saw during training. See [experiments.md](experiments.md#why-the-traintest-split-matters) for why this split was introduced.

The only shared data structure between modules is `pd.DataFrame`. No module imports from another — the orchestration layer (`src/experiments/experiment.py`) is the only place that connects them. This makes each module independently testable with dummy data, which we verified during development.

---

## Key Design Decisions

### 1. DatasetBundle as the contract between stages

Rather than passing raw DataFrames and metadata separately throughout the codebase, we encapsulate everything a downstream module needs about a dataset into a single `DatasetBundle` dataclass. This has two benefits:

- **Clarity**: any function that receives a `DatasetBundle` has everything it needs. There is no implicit knowledge of column names, target variables, or domain.
- **Extensibility**: adding a new field (e.g. `sensitive_cols: list[str]` for multi-attribute privacy evaluation) requires changing only the dataclass definition and the loaders, not every function that uses dataset metadata.

### 2. All evaluation functions are pure and stateless

Every function in `utility.py` and `privacy.py` takes DataFrames as input and returns a dict or float. There is no global state, no file I/O, and no side effects. This makes the evaluation module easy to test, easy to parallelize, and easy to embed inside a training loop (which is exactly what the loss-aware generator will need to do — computing utility and privacy scores at each training step).

### 3. Subsampling at 2000 records for distance-based metrics

Several metrics (MMD, DCR, NNDR, Disclosure Protection) require computing pairwise distances between real and synthetic records. Naive computation is O(n²) in memory and time, which becomes prohibitive on the Diabetes dataset (~100k rows).

We subsample both real and synthetic data to a maximum of 2000 records before computing these metrics. 2000 was chosen as a balance point: large enough to give stable estimates of distributional statistics (as verified by running the metrics multiple times with different random seeds and observing low variance), and small enough to run in under a second on a standard laptop.

This subsampling is applied consistently across all distance-based metrics so that results are comparable across metrics within the same evaluation run.

### 4. StandardScaler normalization before distance computation

All distance-based metrics normalize features using `StandardScaler` (zero mean, unit variance) before computing distances. Without normalization, features with large absolute scales (e.g. `capital-gain` in the Adult dataset, which ranges 0–99999) would dominate Euclidean distances, making the metrics blind to variation in smaller-scale features.

The scaler is always fit on the real data and applied to both real and synthetic data. This is important: fitting the scaler on synthetic data would allow the synthetic data's own distribution to influence the normalization, which would bias the distance measurements.

### 5. Label encoding over one-hot encoding

See [datasets.md](datasets.md#design-decision-label-encoding-vs-one-hot-encoding) for full rationale. In short: label encoding keeps the feature space compact, which is important for the correctness of distance-based privacy metrics.

### 6. Random Forest as the canonical classifier

Both `compute_f1_discrepancy` (utility) and `compute_inference_risk` (privacy) use a Random Forest classifier with 100 estimators and `random_state=42`. The choice of Random Forest was motivated by:

- **Robustness**: it handles mixed feature types (numerical + label-encoded categorical) without preprocessing, is insensitive to feature scale, and rarely fails to converge.
- **Consistency**: using the same classifier architecture for both utility and privacy evaluation makes the two measurements directly comparable — any difference in F1 scores reflects the data, not architectural differences between classifiers.
- **Reproducibility**: fixing `random_state=42` ensures that repeated evaluations on the same data produce identical results.

A more thorough study might evaluate multiple classifier architectures, but for the scope of this project, consistency and reliability were prioritized over exhaustive comparison.

### 7. Separation of per-feature and aggregate metrics

`compute_emd` returns both per-feature EMD scores and a `mean_emd` aggregate. The per-feature scores are stored separately in the report under `emd_per_feature`. This was intentional: the aggregate is useful for high-level comparison across datasets and generators, but the per-feature breakdown is essential for diagnosing *where* utility loss occurs. A generator might achieve a low mean EMD overall but fail badly on a specific clinically important feature (e.g. `num_medications` in the Diabetes dataset). Hiding this in an aggregate would obscure the failure.

---

## What the Evaluation Framework Does Not Cover

For completeness, we document what is explicitly out of scope:

- **Formal differential privacy**: our privacy metrics are empirical approximations of privacy risk, not mathematical guarantees. A formally DP-trained generator would provide provable bounds; our framework measures the empirical manifestation of privacy risk without proving bounds. See [metrics/privacy.md](metrics/privacy.md) for discussion.
- **Fairness metrics**: the framework does not measure whether synthetic data preserves or distorts demographic parity, equalized odds, or other fairness criteria. This is an important limitation given that two of our datasets contain sensitive demographic attributes.
- **Temporal or sequential data**: all metrics assume i.i.d. tabular data. Time-series or sequential datasets would require different distributional metrics.
- **Continuous re-evaluation during training**: the current framework is designed for post-generation evaluation. The loss-aware generator (Stage 2) will need to call these functions — or differentiable approximations of them — inside the training loop.

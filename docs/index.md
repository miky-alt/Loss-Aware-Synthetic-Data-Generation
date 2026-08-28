# Loss-Aware Synthetic Data Generation

## Overview

This project investigates how to **measure and actively minimize utility loss** in synthetic data generation, while simultaneously enforcing strict privacy guarantees. It was developed as a course project for Ethics in AI.

The central argument is that the standard paradigm — generate synthetic data, then evaluate it — is ethically insufficient. Privacy-utility trade-offs should be **structurally enforced during training**, not left to post-hoc auditing. This project builds the evaluation infrastructure to make that argument empirically, and lays the groundwork for a loss-function-based prevention approach.

---

## Project Structure

```
src/
  data/
    loader.py         # Dataset loading and preprocessing (3 datasets)
  evaluation/
    utility.py        # Utility metrics: MMD, EMD, F1 discrepancy, correlation distance
    privacy.py        # Privacy metrics: DCR, NNDR, Inference Risk, Disclosure Protection
  generator/          # (colleague) Synthetic data generators: CTGAN, TVAE
  main.py             # Pipeline entrypoint — runs full evaluation across all datasets

docs/
  index.md            # This file
  datasets.md         # Dataset selection rationale and preprocessing decisions
  architecture.md     # Module design, data flow, and key design decisions
  metrics/
    utility.md        # Theory and implementation of utility metrics
    privacy.md        # Theory and implementation of privacy metrics
```

---

## How to Run

### Prerequisites

```bash
# Install dependencies
uv pip install -e .
```

### Run the full evaluation pipeline

```bash
uv run python src/main.py
```

### Run a quick smoke test on a single dataset

```bash
uv run python -c "
from data.loader import load_dataset
from evaluation.utility import compute_utility_report
from evaluation.privacy import compute_privacy_report
import numpy as np, pandas as pd

bundle = load_dataset('heart')
# Replace with real synthetic data once generator is available
synthetic = bundle.real.sample(frac=1.0, random_state=99).reset_index(drop=True)

u = compute_utility_report(bundle.real, synthetic, bundle.target_col)
p = compute_privacy_report(bundle.real, synthetic, bundle.target_col)
print(u)
print(p)
"
```

---

## Datasets

| Key | Name | Domain | Size |
|-----|------|--------|------|
| `adult` | UCI Adult (Census Income) | Socioeconomic | ~48k rows, 14 features |
| `diabetes` | Diabetes 130-US Hospitals | Medical | ~100k rows, 50 features |
| `heart` | Heart Disease (Cleveland) | Medical | ~300 rows, 14 features |

See [datasets.md](datasets.md) for full rationale.

---

## Metrics at a Glance

### Utility (lower = more utility loss)
| Metric | What it measures |
|--------|-----------------|
| MMD | Overall distributional divergence |
| EMD | Per-feature distributional distance |
| F1 Discrepancy | Downstream predictive performance gap |
| Correlation Distance | Feature relationship preservation |

### Privacy (higher = more privacy risk)
| Metric | What it measures |
|--------|-----------------|
| DCR | How close synthetic records are to real ones |
| NNDR | Generator memorization detection |
| Inference Risk | Attribute leakage via classifier attack |
| Disclosure Rate | Fraction of near-duplicate synthetic records |

See [metrics/utility.md](metrics/utility.md) and [metrics/privacy.md](metrics/privacy.md) for full details.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥2.5.2 | Numerical computation |
| `pandas` | ≥3.0.5 | Data manipulation |
| `scikit-learn` | ≥1.9.0 | Classifiers, nearest neighbors, preprocessing |
| `scipy` | ≥1.18.1 | Wasserstein distance (EMD) |
| `torch` | ≥2.13.0 | Generator training (colleague's module) |
| `ucimlrepo` | ≥0.0.7 | Automatic UCI dataset fetching |

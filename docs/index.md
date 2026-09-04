# Loss-Aware Synthetic Data Generation

## Overview

This project investigates how to **measure and actively minimize utility loss** in synthetic data generation, while simultaneously enforcing strict privacy guarantees. It was developed as a course project for Ethics in AI.

The central argument is that the standard paradigm — generate synthetic data, then evaluate it — is ethically insufficient. Privacy-utility trade-offs should be **structurally enforced during training**, not left to post-hoc auditing. This project builds the evaluation infrastructure to make that argument empirically, and lays the groundwork for a loss-function-based prevention approach.

---

## Project Structure

```
src/
  data/
    loader.py         # Dataset loading, preprocessing, and train/test splitting (3 datasets)
  evaluation/
    utility.py        # Utility metrics: MMD, EMD, F1 discrepancy, correlation distance
    privacy.py        # Privacy metrics: DCR, NNDR, Inference Risk, Disclosure Protection
  generators/         # Synthetic data generators: interface + CTGAN/TVAE/GaussianCopula baselines
    base.py           # SyntheticGenerator interface (fit/sample/get_training_diagnostics contract)
    baseline.py       # CTGANGenerator, TVAEGenerator, GaussianCopulaGenerator (sdv-backed)
    registry.py       # GENERATORS dict + build_generator() — single extension point
  experiments/        # Experiment orchestration layer
    config.py         # TrainingConfig + ExperimentMode
    experiment.py     # run_experiment(): full pipeline entry point
    report.py         # build_report, save_report, append_to_index, summarize_report
    persistence.py    # save/load generator (.pkl), save_metadata, load_report
    query.py          # load_index, query_index — find runs by criteria
  main.py             # CLI: run / evaluate / show / find subcommands

docs/
  index.md            # This file
  datasets.md         # Dataset selection rationale and preprocessing decisions
  architecture.md     # Module design, data flow, and key design decisions
  generators.md       # Generator interface and CTGAN/TVAE/GaussianCopula baseline design decisions
  experiments.md      # Experiment pipeline, config, persistence, run indexing, and CLI
  design-exploration.md  # sdv features investigated but not integrated, and why
  metrics/
    utility.md        # Theory and implementation of utility metrics
    privacy.md        # Theory and implementation of privacy metrics
```

---

## How to Run

### Prerequisites

```bash
uv sync
```

### Train a generator and evaluate it

```bash
uv run python -m src.main run --dataset adult --generator ctgan --kwarg epochs=300
uv run python -m src.main run --dataset heart --generator gaussian_copula
```

### Re-evaluate a pre-trained generator without retraining

```bash
uv run python -m src.main evaluate --pretrained-run adult_ctgan_seed42_<hash>_<timestamp> \
    --dataset adult --generator ctgan
```

### Print the summary of a saved report

```bash
uv run python -m src.main show adult_ctgan_seed42_<hash>_<timestamp>
```

### Find a run without knowing its exact name

```bash
uv run python -m src.main find --dataset adult --generator ctgan --kwarg epochs=300
```

See [experiments.md](experiments.md) for the full CLI reference and how `run_name`/`index.jsonl` work.

### Run configured experiments by dataset

The `scripts/` directory contains one editable runner per dataset. Each runner executes the configurations listed in its `EXPERIMENTS` list; see each script for its current set of generators and preprocessing options.

```powershell
uv run python scripts/run_adult.py
uv run python scripts/run_diabetes.py
uv run python scripts/run_heart.py
```

Edit `NUM_SAMPLES`, `SEED`, `TEST_SIZE`, `OUTPUT_DIR`, or the `EXPERIMENTS` list at the top of the relevant script to add generators or change their kwargs. Every configuration is trained, evaluated, and saved sequentially in `experiments/results/`. The Adult script uses `truncnorm` for `hours-per-week`; `gaussian_kde` is intentionally not the default there because its fit requires quadratic memory in the number of training rows.

For the complete Adult preprocessing comparison matrix, use:

```powershell
uv run python scripts/run_adult_preprocessing.py
```

This runs seven configurations: default and modified-distribution Gaussian Copula, default and `LogScaler(fnlwgt)` Gaussian Copula, and default plus `LogScaler(fnlwgt)` variants for CTGAN and TVAE. All runs use the same seed, train/test split, and sample count.

`scripts/run_diabetes.py` runs the equivalent seven-configuration matrix for Diabetes 130-US Hospitals, using `LogScaler(num_medications)` instead of `fnlwgt` as the preprocessing variant.

### Inspect a dataset and plot its distributions

```bash
uv run python -m src.data.analyze adult --head 5
uv run python -m src.data.analyze all --head 3 --plot-dir plots --analysis-dir analysis
```

The analysis command uses the same cleaning and target normalization as the experiment pipeline, while preserving categorical values and boolean targets. It detects metadata with `Metadata.detect_from_dataframe`, explicitly marks boolean columns with `metadata.update_columns_metadata`, and saves one metadata JSON per dataset. Text analyses are saved in `analysis/`, distribution plots in `plots/`, and SDV metadata in `metadata/`. Use `--metadata-dir` to change the metadata destination and `--no-plot` when only the textual analysis and metadata are needed.

In PowerShell, use backticks for multiline commands:

```powershell
uv run python -m src.data.analyze all `
  --head 3 `
  --plot-dir plots `
  --analysis-dir analysis
```

If `uv` is not available in `PATH`, use `& "$env:USERPROFILE\.local\bin\uv.exe" run` instead of `uv run`.

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
| `scikit-learn` | ≥1.9.0 | Classifiers, nearest neighbors, preprocessing, train/test splitting |
| `scipy` | ≥1.18.1 | Wasserstein distance (EMD) |
| `torch` | ≥2.13.0 | Neural-network generator training (used internally by `sdv`) |
| `sdv` | ≥1.17.0 | CTGAN/TVAE/GaussianCopula synthesizers, metadata detection |
| `ucimlrepo` | ≥0.0.7 | Automatic UCI dataset fetching |

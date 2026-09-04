# Loss-Aware Synthetic Data Generation
**When Does Privacy Cost Utility, and Who Decides?**

This repository contains the source code, experiment logs, documentation and final report for our Ethics in AI course project at the University of Bologna. The project investigates how to measure the utility lost when generating synthetic tabular data, how to write privacy and utility constraints directly into a generator's training objective, and what happens when you do.

## 👥 Authors
* **Michelangelo Urbano** - [michelangelo.urbano@studio.unibo.it](mailto:michelangelo.urbano@studio.unibo.it) (Student ID: 0001160217)
* **Gianlorenzo Urbano** - [gianlorenzo.urbano@studio.unibo.it](mailto:gianlorenzo.urbano@studio.unibo.it) (Student ID: 0001169323)

## 🎯 Project Overview
Synthetic data promises to share the statistical value of a sensitive dataset without exposing the people in it. The usual workflow trains a generator, samples from it, and only afterwards checks how much utility was lost and how much privacy risk remains. We argue this is ethically inadequate, because the privacy-utility trade-off ends up decided by whatever the architecture happens to do, and we built the alternative.

The project has three parts:

1. **Evaluation framework** (`src/evaluation/`): nine metrics covering distributional fidelity (MMD, EMD, and categorical total variation distance), structural fidelity (correlation distance), downstream utility (F1 discrepancy) and empirical privacy (DCR, NNDR, inference risk, disclosure rate).
2. **Baseline comparison** (`src/generators/baseline.py`): CTGAN, TVAE and GaussianCopula from `sdv`, run through the framework on three datasets.
3. **Loss-aware generators** (`src/generators/loss_aware.py`, `src/generators/loss_aware_ctgan.py`): TVAE and CTGAN with three differentiable penalty terms added to the training loss, an MMD term, a correlation term and a distance-to-closest-record hinge, with a self-calibrating privacy margin. With all weights at zero each is exactly the stock model.

### Datasets
| key | dataset | domain | train rows | columns |
|---|---|---|---|---|
| `heart` | Heart Disease (Cleveland) | clinical | 237 | 13 |
| `adult` | UCI Adult (Census Income) | socioeconomic | 37,000 | 14 |
| `diabetes` | Diabetes 130-US Hospitals | clinical | 57,000 | ~40 |

All three are fetched automatically from the UCI repository on first use.

## 🔍 Main Findings
Full details are in `docs/loss_aware_training.md` and in the report.

* **Fidelity regularization is a privacy mechanism, for generators that memorize.** On TVAE, the MMD and correlation penalties alone raised the distance between synthetic and real records by 17 to 26% on all three datasets while also improving utility. On a properly trained CTGAN, which does not memorize on Heart, the same terms gave no privacy gain.
* **Whether a privacy penalty costs utility depends on data density.** The same penalty (λ=10) was free by every aggregate metric on Adult, cost measurably on Heart, and collapsed the generator on Diabetes. Density means rows relative to dimensionality, not row count.
* **The cost lands on the sparsest part of each column.** Minority classes, rare categories and the target rate. On Adult, the six columns that moved most were native-country, race, workclass, sex, income and relationship.
* **Aggregate utility metrics cannot see this.** On Adult, MMD, correlation distance and F1 all improved while the sex ratio shifted 14 percentage points and the income rate 11. Per-column marginal checks on protected attributes are not optional.
* **Privacy scores are meaningless without a utility floor.** Every privacy indicator is trivially maximized by generating noise.

### Repeated-seed evaluation

The dataset runners repeat every Adult, Diabetes, and Heart configuration over
the same default seed set `(1, 2, 3)`, then report mean and 95% confidence intervals for all utility
and privacy metrics, including per-feature numeric EMD and categorical/boolean
TVD. Three seeds are suitable for an exploratory sweep; set `SEEDS` to five for
final results. This multiplies the number of trainings (roughly 54 runs
with three seeds or 90 with five for the current matrices), so CTGAN and TVAE
should be parallelized only when available memory allows it.

---

## ⚙️ Setup and Installation

The project is implemented in Python 3.12 using [uv](https://docs.astral.sh/uv/) for environment management.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/miky-alt/Loss-Aware-Synthetic-Data-Generation.git
   cd Loss-Aware-Synthetic-Data-Generation
   ```
2. Install uv following the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).
3. **Install dependencies:**
   ```bash
   uv sync
   ```
4. **Run the tests:**
   ```bash
   uv run pytest
   ```

## 🚀 Usage

### Run one experiment
```bash
# baseline TVAE (all penalty weights zero)
uv run python -m src.main run --dataset heart --generator tvae_loss_aware \
  --kwarg batch_size=32 --kwarg epochs=300 --seed 1

# loss-aware: utility terms plus privacy hinge
uv run python -m src.main run --dataset heart --generator tvae_loss_aware \
  --kwarg batch_size=32 --kwarg epochs=300 --seed 1 \
  --kwarg lambda_mmd=1 --kwarg lambda_corr=0.5 --kwarg lambda_priv=10 --kwarg dcr_margin=1.5

# a stock baseline
uv run python -m src.main run --dataset adult --generator ctgan --seed 1
```

Available generators: `ctgan`, `tvae`, `gaussian_copula`, `tvae_loss_aware`, `ctgan_loss_aware`.
Loss-aware kwargs: `lambda_mmd`, `lambda_corr`, `lambda_priv`, `dcr_margin` (relative to the median real-to-real nearest-neighbour distance; 1.0 means "no closer to a real row than real rows are to each other"), `mmd_gamma`. For `ctgan_loss_aware`, `batch_size` must be even and a multiple of 10 (use 50 on Heart), and it needs many more epochs than TVAE to converge (2000 on Heart).

Every run writes a JSON report to `experiments/results/<run_name>.json` (all metrics, per-batch loss components, effective margin, code version) and appends a line to `experiments/results/index.jsonl`.

### Inspect and query runs
```bash
uv run python -m src.main show <run_name>
uv run python -m src.main find --dataset heart --generator tvae_loss_aware
```

### Reproduce the figures and tables
```bash
# trade-off curve (F1 discrepancy vs DCR) and loss curves, plus aggregated CSV
uv run python -m src.experiments.plot_tradeoff --dataset heart --kwarg batch_size=32 --kwarg epochs=300 --exclude-collapse 4
uv run python -m src.experiments.plot_tradeoff --dataset adult --kwarg batch_size=500
uv run python -m src.experiments.plot_tradeoff --dataset diabetes --kwarg batch_size=500

# per-feature EMD heatmaps and standardized ranking
uv run python -m src.experiments.plot_feature_emd --dataset heart --kwarg batch_size=32 --kwarg epochs=300
uv run python -m src.experiments.plot_feature_emd --dataset adult --kwarg batch_size=500
uv run python -m src.experiments.plot_feature_emd --dataset diabetes --kwarg batch_size=500
```
Output goes to `experiments/figures/`. Both scripts keep the most recent run per (configuration, seed), so re-running a configuration after a code fix supersedes the old result automatically.

### Build the report
```bash
cd report && latexmk -pdf main.tex
```
See `report/README.md`.

## 📂 Repository Structure
```
src/
  data/loader.py              # dataset loading + preprocessing, DatasetBundle interface
  evaluation/
    utility.py                # MMD, EMD, correlation distance, F1 discrepancy
    privacy.py                # DCR, NNDR, inference risk, disclosure rate
  generators/
    base.py                   # SyntheticGenerator interface
    baseline.py               # CTGAN, TVAE, GaussianCopula (sdv-backed)
    loss_aware.py             # LossAwareTVAE: ctgan TVAE + MMD / correlation / DCR-hinge penalties; shared helpers
    loss_aware_ctgan.py       # LossAwareCTGAN: same penalties on CTGAN's generator objective
    registry.py               # name -> generator class
  experiments/
    config.py                 # TrainingConfig, deterministic run names
    experiment.py             # one run: load, train, evaluate, persist
    persistence.py            # save/load models and reports
    report.py                 # build report JSON, append to index
    query.py                  # filter index.jsonl
    plot_tradeoff.py          # aggregate across seeds, F1-vs-DCR figure, loss curves
    plot_feature_emd.py       # per-feature EMD heatmaps and ranking
  main.py                     # CLI: run / show / find

tests/                        # pytest suite for evaluation and generators
experiments/
  results/                    # one JSON per run + index.jsonl
  figures/                    # generated PNGs and aggregated CSVs
  *.txt                       # raw console logs of every sweep
docs/                         # technical documentation (see below)
.github/workflows/ci.yml      # runs pytest on every PR
```

## 📄 Documentation
| file | contents |
|---|---|
| `docs/index.md` | overview and entry point |
| `docs/datasets.md` | dataset choice, preprocessing decisions, label vs one-hot encoding |
| `docs/architecture.md` | module design, data flow, design decisions |
| `docs/metrics/utility.md` | utility metrics: theory, implementation, caveats (MMD saturation) |
| `docs/metrics/privacy.md` | privacy metrics: theory, implementation, caveats (NNDR inversion), relation to differential privacy |
| `docs/generators.md` | generator wrappers and registry |
| `docs/experiments.md` | experiment framework, persistence, querying |
| `docs/loss_aware_training.md` | the loss-aware TVAE: objective with derivations, the fixes that were needed, every sweep, per-feature analysis, ethical reading |
| `docs/baseline_comparison.md` | CTGAN vs TVAE vs GaussianCopula on all datasets |
| `docs/design-exploration.md` | sdv features investigated but not integrated |
| `docs/initial_report.tex` | the original project proposal |

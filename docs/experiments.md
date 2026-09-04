# Experiment Pipeline

This document describes the orchestration layer that ties dataset loading, train/test splitting, generator training, evaluation, and result persistence into a single reproducible experiment run. It lives in `src/experiments/` and is deliberately kept separate from both the generator implementations (`src/generators/`) and the metric functions (`src/evaluation/`).

---

## 1. Module map

```
src/experiments/
  config.py       — TrainingConfig (hyperparameters) + ExperimentMode (run mode)
  experiment.py   — run_experiment(): the single entry point for a full run
  report.py       — build_report(), save_report(), append_to_index(), summarize_report()
    multiseed.py    — repeated-seed execution and confidence-interval aggregation
  persistence.py  — save/load generator (.pkl), save_metadata(), load_report()
  query.py        — load_index(), query_index(): find runs by criteria
```

Each module has exactly one responsibility. `experiment.py` is the only file that imports from all the others — it is the composition root.

---

## 2. Configuration: `TrainingConfig` and `ExperimentMode`

### `TrainingConfig`

A `dataclass` holding every hyperparameter a training run needs:

```python
@dataclass
class TrainingConfig:
    dataset_name: str        # one of src.data.loader.LOADERS keys
    generator_name: str      # one of src.generators.registry.GENERATORS keys
    num_samples: int
    seed: int = 42
    test_size: float = 0.2   # fraction held out from generator.fit(), used for evaluation
    generator_kwargs: dict   # forwarded verbatim to the generator constructor
    created_at: str          # timestamp + random suffix, set automatically at creation
```

### Why a dataclass and not a YAML/JSON config file

A `dataclass` is type-checked, IDE-navigable, and importable. For the current scale of the project (a handful of experiments), it is simpler than adding a YAML parsing dependency. If the number of experiments grows, the `TrainingConfig` can be serialized to/from YAML without changing any calling code — the dataclass is the canonical representation either way.

### `ExperimentMode`

```python
class ExperimentMode(Enum):
    TRAIN_AND_EVALUATE  # fit a new generator, then evaluate (default)
    EVALUATE_ONLY       # load a pre-trained generator from disk, skip fit
```

`ExperimentMode` is a parameter of `run_experiment()`, not a field of `TrainingConfig`. The distinction is intentional: *how to run* the experiment is orthogonal to *which hyperparameters to use*. The same `TrainingConfig` can be passed to both modes.

### `run_name`: collision-proof by construction

Naively, `run_name = f"{dataset}_{generator}_seed{seed}"` looks reasonable — until you try two runs with different `generator_kwargs` (e.g. `epochs=50` vs `epochs=300`) on the same dataset/generator/seed: they'd silently overwrite each other's report and pickle. `run_name` is built from three parts to make this impossible:

```python
run_name = f"{dataset_name}_{generator_name}_seed{seed}_{_kwargs_hash}_{created_at}"
```

- **`_kwargs_hash`**: a 6-character SHA-256 hash of `(generator_kwargs, test_size)`, computed deterministically — different hyperparameters *always* produce a different hash, with no dependence on timing.
- **`created_at`**: `YYYYMMDD-HHMMSS` plus a 6-hex-character random suffix (`secrets.token_hex(3)`), generated once when the `TrainingConfig` is constructed. This guarantees uniqueness even for two configs with **identical** hyperparameters created in the same second (e.g. in a fast hyperparameter-sweep loop, or when rerunning the exact same experiment after a code change) — a timestamp with second-resolution alone would not be enough to prevent that collision.

The practical consequence: **`run_name` is never a good thing to guess by hand**. Use `find` (below) to look runs up instead of trying to reconstruct the filename.

---

## 3. `run_experiment()`

The single entry point for a full experiment:

```
load_dataset(config.dataset_name)
    └─→ DatasetBundle
            │
            └─→ split_dataset(bundle, test_size, seed)
                    └─→ (train_bundle, test_bundle)   [stratified on target_col]
                            │
                            ├─ [TRAIN_AND_EVALUATE] build_generator → fit(train_bundle.real)
                            │       └─→ save_generator (.pkl) + save_metadata (.metadata.json)
                            ├─ [EVALUATE_ONLY]      load_generator (.pkl)
                            │
                                └─→ generator.sample(config.num_samples)
                                    │
                                    └─→ build_report(train_bundle.real, test_bundle.real, synthetic, ...)
                                        ├─→ utility distribution metrics: test_bundle.real vs synthetic
                                        ├─→ F1: train_bundle.real / synthetic → test_bundle.real
                                        ├─→ privacy memorization metrics: train_bundle.real vs synthetic
                                        ├─→ save_report()       ← {run_name}.json
                                        └─→ append_to_index()   ← one line in index.jsonl
```

`save_generator()` / `save_metadata()` are called **before** `build_report()` in the training path, so the model and its schema are persisted even if the evaluation step raises an error.

### Why the train/test split matters

Earlier, `generator.fit()` saw the entire dataset, and `build_report()` evaluated the synthetic data against that same full dataset. This is methodologically weak in two ways:

- **Utility metrics** (MMD, EMD, categorical total variation distance, and correlation distance) measured how well the synthetic data resembled data the generator had already memorized, not how well it generalizes to unseen real data.
- **`EVALUATE_ONLY` had no well-defined held-out set** to compare against on a re-run.

`split_dataset()` (in `src/data/loader.py`) fixes this: `generator.fit()` only ever sees `train_bundle.real`. MMD, EMD, categorical total variation distance, and correlation distance compare synthetic data with `test_bundle.real`, which the generator never saw. F1 trains its real-data baseline on `train_bundle.real` and evaluates both that model and the synthetic-trained model on the same held-out `test_bundle.real`.

Privacy metrics answer a different question: DCR, NNDR, and disclosure protection compare synthetic records with `train_bundle.real`, because those are the records the generator could have memorized. Attribute inference trains on synthetic data and evaluates its attack on `test_bundle.real`. Because the split is deterministic — driven only by `(test_size, seed)`, both fixed in `TrainingConfig` — `EVALUATE_ONLY` reconstructs the **exact same** train/test split on every re-run, without ever retraining.

### Repeated runs and confidence intervals

A single run with `seed=42` is useful for debugging, but it is not enough for a
statistical comparison of generators or preprocessing choices. The configured
dataset runners (`scripts/run_adult.py`, `scripts/run_diabetes.py`, and
`scripts/run_heart.py`) should therefore execute every configuration with the
same set of independent seeds, for example `[1, 2, 3, 4, 5]`. The seed controls
the stratified split, generator randomness, and synthetic sampling. Using the
same seed list for every configuration makes comparisons paired: differences
between two configurations are measured under the same split and random-seed
conditions.

Each seed produces one complete report. The aggregation step must then compute
the mean and a 95% confidence interval for every scalar report metric:

- utility: MMD, mean EMD, mean categorical TVD, correlation distance, and F1
    discrepancy;
- privacy: DCR, NNDR, inference risk, and disclosure rate;
- per-feature diagnostics: EMD and categorical TVD for every numeric,
    categorical, and boolean column.

For a metric with values `x_1, ..., x_n` across seeds, the default interval is
the two-sided Student-t interval for the mean:

```text
mean(x) +/- t(0.975, n - 1) * sample_std(x) / sqrt(n)
```

The report should store the mean and CI half-width (or explicit lower and
upper bounds), together with `n_seeds`. With only three to five seeds this is
an empirical uncertainty interval, not a precise population-level guarantee;
the small sample size must be stated in tables and figures. A bootstrap
interval can be added later, but it does not remove the need for independent
repeated trainings.

The three dataset runners now execute every configuration for the configured
`SEEDS` tuple and write one aggregate JSON under
`experiments/results/aggregates/`. The individual seed reports remain
available in the normal results directory, so a failed seed can be inspected
or rerun independently. The existing loss-aware trade-off aggregation is a
separate post-processing tool and is not required by the Adult, Diabetes, and
Heart experiment matrices. At the end of each matrix, the runner automatically
creates utility and privacy plots with 95% CI error bars under
`experiments/results/figures/` using `src/experiments/plot_matrix.py`.

### Runtime trade-off

Repeated training increases cost linearly with the number of seeds. The current
matrices contain approximately 8 Adult, 8 Diabetes, and 2 Heart configurations.
With five seeds this becomes about 90 complete trainings instead of 18; with
three seeds it becomes about 54. Gaussian Copula runs are relatively cheap,
while CTGAN and TVAE at 500 epochs dominate the runtime, especially on
Diabetes. A practical protocol is therefore:

1. use three common seeds for a fast exploratory sweep;
2. use the default five seeds for the final comparison and narrower confidence intervals;
3. run configurations and seeds in parallel where memory allows;
4. keep the same `num_samples`, split fraction, and metric implementation for
     every seed.

This cost is justified for final claims about improvements: without repeated
seeds, an apparent preprocessing gain can be indistinguishable from training
or sampling noise.

---

## 4. Persistence: what gets saved per run

For each completed training run, files are written to `experiments/results/` (configurable via `--output-dir`):

| File | Contents |
|------|----------|
| `{run_name}.json` | Full evaluation report: config, `code_version`, utility metrics, privacy metrics, optional `artifacts` |
| `{run_name}.pkl` | Serialized generator object (pickle), ready for `EVALUATE_ONLY` re-use |
| `{run_name}.metadata.json` | The dataset's sdv `Metadata` (schema/sdtypes), detected from `train_bundle.real` |
| `index.jsonl` | One line appended per run — see [Section 6](#6-finding-runs-indexjsonl--the-find-command) |

All four are excluded from version control via `.gitignore` — they are reproducible artifacts, not source code.

### Why `save_metadata` takes the dataset, not the generator

Dataset schema (which columns are numerical/categorical/etc.) is a property of the **data**, not of whichever generator happens to be fit on it — two different generators trained on the same dataset would produce identical metadata. `save_metadata(real_data, run_name, output_dir)` therefore calls the shared `build_sdv_metadata(real_data)` helper from `src.data.loader`, independent of the generator instance. The helper detects the schema and explicitly preserves pandas boolean columns as SDV `boolean` columns. This costs one extra (cheap, schema-only) detection pass but keeps metadata construction consistent across dataset analysis, generator fitting, and persisted experiment results.

### `code_version`

`build_report()` captures `git rev-parse --short HEAD` (or `"unknown"` if not in a git repo / git unavailable) and stores it as `report["code_version"]`. This answers a specific reproducibility question: *if I rerun the exact same hyperparameters, how do I know whether a different result is due to randomness or because the code itself changed?* — `code_version` gives a direct answer without relying on `run_name` alone.

### `artifacts`

If the generator's `get_training_diagnostics()` (see [generators.md](generators.md)) returns a non-empty dict, `build_report()` includes it under the `"artifacts"` key. Baseline generators expose loss curves or learned distributions; the future loss-aware generator will expose its own per-epoch penalty terms through the same mechanism, with no changes needed here.

---

## 5. CLI

`src/main.py` exposes four subcommands:

```
python -m src.main run       --dataset DATASET --generator GEN [--kwarg KEY=VALUE ...] [options]
python -m src.main evaluate  --pretrained-run RUN_NAME --dataset DATASET --generator GEN [options]
python -m src.main show      RUN_NAME [--output-dir DIR]
python -m src.main find      [--dataset DATASET] [--generator GEN] [--kwarg KEY=VALUE ...] [options]
```

| Subcommand | What it does |
|------------|-------------|
| `run` | Trains a new generator (`TRAIN_AND_EVALUATE`), saves artifacts, prints summary |
| `evaluate` | Loads an existing `.pkl` (`EVALUATE_ONLY`), samples, evaluates, saves a new report |
| `show` | Loads and prints the summary of an existing `.json` report — no computation |
| `find` | Queries `index.jsonl` for runs matching given criteria — no report files opened |

`--kwarg KEY=VALUE` (repeatable) replaces a hardcoded `--epochs` flag: values are auto-cast to `int`/`float` where possible (`_parse_kwargs()` in `main.py`), falling back to string. This is what lets `--kwarg epochs=50 --kwarg batch_size=500` reach the underlying `sdv` synthesizer unchanged, and lets `gaussian_copula` runs simply omit `--kwarg` entirely (it takes no `epochs`).

### PowerShell usage

The examples above use Bash line continuation. In PowerShell, use a backtick at the end of each continued line. For structured JSON passed to `--kwargs-json`, escape the JSON double quotes with backslashes:

```powershell
uv run python -m src.main run `
    --dataset adult `
    --generator gaussian_copula `
    --num-samples 38096 `
    --kwargs-json '{\"numerical_distributions\":{\"age\":\"gamma\",\"fnlwgt\":\"gamma\",\"education-num\":\"truncnorm\",\"capital-gain\":\"gamma\",\"capital-loss\":\"gamma\",\"hours-per-week\":\"truncnorm\"}}'
```

If `uv` is not available in `PATH`, use the full executable path:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run python -m src.main find `
    --dataset adult `
    --generator gaussian_copula
```

The same quoting rule applies to structured JSON passed to `find`. Use `show RUN_NAME` after `find` to inspect a saved report.

`--kwarg KEY=VALUE` supports scalar values. For nested synthesizer parameters such as GaussianCopula's `numerical_distributions`, use `--kwargs-json` as shown above. The Python API described in [generators.md](generators.md#gaussiancopula-configuration) remains available when configuration needs to be assembled programmatically.

### Custom RDT transformers

Dataset runner scripts can configure RDT preprocessing per experiment with `transformer_specs`. The spec names a transformer from the installed RDT library and optionally supplies constructor kwargs:

```python
{
    "transformer_specs": {
        "fnlwgt": {"name": "LogScaler", "kwargs": {}},
    }
}
```

The runner resolves the class, calls `synthesizer.update_transformers(...)`, and then calls `fit()`. The selected transformer configuration is included in the report and in `index.jsonl`, so runs with different preprocessing remain distinguishable. The available transformer names are defined by the installed RDT version; examples include `LogScaler`, `GaussianNormalizer`, `ClusterBasedNormalizer`, `FloatFormatter`, `UniformEncoder`, `LabelEncoder`, and `BinaryEncoder`.

### Reproducing the default UCI Adult GaussianCopula baseline

The preprocessed Adult dataset contains 47,621 rows. With the default `test_size=0.2`, the generator trains on 38,096 real records and evaluates against 9,525 held-out records. The following command generates the same number of synthetic records as the training split, with every GaussianCopula synthesizer option at SDV's default:

```bash
uv run python -m src.main run --dataset adult --generator gaussian_copula --num-samples 38096
```

The generated artifacts are ignored by Git. Use `find` afterwards instead of trying to reconstruct the collision-proof run name manually:

```bash
uv run python -m src.main find --dataset adult --generator gaussian_copula
```

The `show` subcommand is backed by `summarize_report()`, which skips per-feature breakdowns (`emd_per_feature`) and the full config dump, printing only aggregate metrics plus a one-line summary of `artifacts` (e.g. `artifacts.loss_values: 300 entries`).

---

## 6. Finding runs: `index.jsonl` + the `find` command

With many hyperparameter combinations across 3 datasets and 3 generators, `run_name` alone (a hash + timestamp) is not something a human can usefully search by. `experiments/results/index.jsonl` solves this: `append_to_index()` writes one JSON line per completed run with only the queryable fields —

```json
{"run_name": "...", "created_at": "...", "code_version": "...", "dataset_name": "adult", "generator_name": "ctgan", "seed": 42, "test_size": 0.2, "num_samples": 1000, "generator_kwargs": {"epochs": 300}}
```

`src/experiments/query.py` reads and filters this file:

```python
query_index(output_dir=..., dataset_name="adult", generator_name="ctgan", kwargs={"epochs": 300})
```

`kwargs` uses a **superset match**: a query for `{"epochs": 100}` also matches a row with `{"epochs": 100, "batch_size": 500}` — you don't need to specify every hyperparameter the run had, only the ones you care about.

From the CLI:

```bash
python -m src.main find --dataset adult --generator ctgan --kwarg epochs=300
python -m src.main find --code-version abcdef1     # every run produced by a specific commit
```

This is deliberately a flat, append-only log rather than a database — reading it is just `pd.read_json("index.jsonl", lines=True)` if you want to do more advanced analysis (grouping, sorting, plotting) than the `find` command supports directly.

---

## 7. Generator registry

`src/generators/registry.py` is the single place where generator names are mapped to classes:

```python
GENERATORS = {
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
    "gaussian_copula": GaussianCopulaGenerator,
}
```

Adding the loss-aware generator requires adding one entry here. Nothing in `experiment.py`, `config.py`, or `main.py` needs to change.

---

## 8. Testing approach

- `tests/experiments/test_experiments.py`: `TrainingConfig.run_name` uniqueness (different kwargs, different test_size, identical config created twice), `save_report`/`load_report` roundtrip and enum serialization, `save_generator`/`load_generator` pickle roundtrip, `append_to_index()` (file creation, valid JSON per line, multiple runs appended), `summarize_report` metric selection and disk-loading path, `build_report` inclusion/omission of `artifacts`.
- `tests/experiments/test_query.py`: `load_index()` on a missing/populated index, and `query_index()` for every filter individually, combined filters, superset kwargs matching, and the no-match case.
- `tests/data/test_loader.py`: `split_dataset()` — correct sizes, no overlap between train/test, dataset metadata (`target_col`/`name`/`domain`) preserved on both splits, deterministic with the same seed, different with a different seed.

`run_experiment()` end-to-end is not tested in the fast suite — it requires real model training and dataset download, making it a `@pytest.mark.slow` integration test candidate.


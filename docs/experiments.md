# Experiment Pipeline

This document describes the orchestration layer that ties dataset loading, generator training, evaluation, and result persistence into a single reproducible experiment run. It lives in `src/experiments/` and is deliberately kept separate from both the generator implementations (`src/generators/`) and the metric functions (`src/evaluation/`).

---

## 1. Module map

```
src/experiments/
  config.py       — TrainingConfig (hyperparameters) + ExperimentMode (run mode)
  experiment.py   — run_experiment(): the single entry point for a full run
  report.py       — build_report(), save_report(), summarize_report()
  persistence.py  — save_generator(), load_generator(), load_report()
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
    generator_kwargs: dict   # forwarded verbatim to the generator constructor
```

The `run_name` property derives a unique identifier from these fields (`{dataset}_{generator}_seed{seed}`), which is used as the filename stem for all saved artifacts.

### Why a dataclass and not a YAML/JSON config file

A `dataclass` is type-checked, IDE-navigable, and importable. For the current scale of the project (a handful of experiments), it is simpler than adding a YAML parsing dependency. If the number of experiments grows, the `TrainingConfig` can be serialized to/from YAML without changing any calling code — the dataclass is the canonical representation either way.

### `ExperimentMode`

```python
class ExperimentMode(Enum):
    TRAIN_AND_EVALUATE  # fit a new generator, then evaluate (default)
    EVALUATE_ONLY       # load a pre-trained generator from disk, skip fit
```

`ExperimentMode` is a parameter of `run_experiment()`, not a field of `TrainingConfig`. The distinction is intentional: *how to run* the experiment is orthogonal to *which hyperparameters to use*. The same `TrainingConfig` can be passed to both modes.

---

## 3. `run_experiment()`

The single entry point for a full experiment:

```
load_dataset(config.dataset_name)
    └─→ DatasetBundle
            │
            ├─ [TRAIN_AND_EVALUATE] build_generator → fit → save_generator (.pkl)
            ├─ [EVALUATE_ONLY]      load_generator (.pkl)
            │
            └─→ generator.sample(config.num_samples)
                    │
                    ├─→ build_report()  ← utility + privacy metrics
                    └─→ save_report()   ← {run_name}.json
```

`save_generator()` is called **before** `build_report()` in the training path, so the model is persisted even if the evaluation step raises an error.

---

## 4. Persistence: what gets saved per run

For each completed run, two files are written to `experiments/results/` (configurable via `--output-dir`):

| File | Contents |
|------|----------|
| `{run_name}.json` | Full evaluation report: config, utility metrics, privacy metrics, optional `training_history` |
| `{run_name}.pkl` | Serialized generator object (pickle), ready for `EVALUATE_ONLY` re-use |

Both files are excluded from version control via `.gitignore` — they are reproducible artifacts, not source code.

### `training_history`

If the generator exposes a `training_history` attribute (a list of per-epoch metric dicts), `build_report()` includes it in the JSON under the key `training_history`. Baseline generators (CTGAN/TVAE) do not populate this field; the future loss-aware generator will, using this pattern:

```python
# inside the loss-aware generator's fit():
self.training_history.append({"epoch": e, "loss": total_loss, "mmd": mmd_val, ...})
```

No changes to `build_report()` or `run_experiment()` are needed to support this.

---

## 5. CLI

`src/main.py` exposes three subcommands:

```
python -m src.main run       --dataset DATASET --generator GEN [options]
python -m src.main evaluate  --pretrained-run RUN_NAME --dataset DATASET --generator GEN [options]
python -m src.main show      RUN_NAME [--output-dir DIR]
```

| Subcommand | What it does |
|------------|-------------|
| `run` | Trains a new generator (`TRAIN_AND_EVALUATE`), saves artifacts, prints summary |
| `evaluate` | Loads an existing `.pkl` (`EVALUATE_ONLY`), samples, evaluates, saves a new report |
| `show` | Loads and prints the summary of an existing `.json` report — no computation |

The `show` subcommand is backed by `summarize_report()`, which skips per-feature breakdowns (`emd_per_feature`) and the full config dump, printing only aggregate metrics.

---

## 6. Generator registry

`src/generators/registry.py` is the single place where generator names are mapped to classes:

```python
GENERATORS = {
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
}
```

Adding the loss-aware generator requires adding one entry here. Nothing in `experiment.py`, `config.py`, or `main.py` needs to change.

---

## 7. Testing approach

`tests/experiments/test_experiments.py` tests only the logic that lives in `src/experiments/` and is not covered elsewhere:

- `TrainingConfig.run_name` format and seed handling
- `save_report` / `load_report` roundtrip and JSON enum serialization
- `save_generator` / `load_generator` pickle roundtrip
- `summarize_report` metric selection, `emd_per_feature` exclusion, `training_history` count, and disk-loading path
- `build_report` inclusion/omission of `training_history`

`run_experiment()` end-to-end is not tested in the fast suite — it requires real model training and dataset download, making it a `@pytest.mark.slow` integration test candidate.

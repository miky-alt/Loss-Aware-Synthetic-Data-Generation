# Synthetic Data Generators

This module answers a different question than the evaluation framework: **given real data, how do we produce a synthetic replacement?** It is deliberately kept separate from `evaluation/` — the generator only needs to expose a `fit`/`sample` contract, and knows nothing about how its output will be scored.

---

## 1. Design goal: a generator-agnostic interface

### What it is

`SyntheticGenerator` (`src/generators/base.py`) is an abstract base class with two methods:

```python
class SyntheticGenerator(ABC):
    def fit(self, real_data: pd.DataFrame) -> "SyntheticGenerator": ...
    def sample(self, num_rows: int) -> pd.DataFrame: ...
```

### Why this shape

- **Symmetry with `DatasetBundle`**: the evaluation framework already treats datasets as an opaque contract (`DatasetBundle`). The generator side needed the same treatment — `main.py` (once wired up) should be able to swap generator implementations without changing the orchestration code.

- **`fit` returns `self`**: allows `generator = CTGANGenerator().fit(real_data)` in one line, matching the scikit-learn convention already used elsewhere in the evaluation code (`RandomForestClassifier`, `StandardScaler`).

- **This is the extension point for the loss-aware generator.** The core contribution of the project is a generator whose training loop is penalized by the utility/privacy metrics already implemented in `evaluation/`. That generator will implement the same `SyntheticGenerator` interface, so it can be evaluated with the exact same code path as the baselines below — the only thing that changes is what happens inside `fit`.

---

## 2. Baseline generators: CTGAN, TVAE, GaussianCopula

### What they are

`CTGANGenerator`, `TVAEGenerator`, and `GaussianCopulaGenerator` (`src/generators/baseline.py`) wrap the `sdv` library's `CTGANSynthesizer`, `TVAESynthesizer`, and `GaussianCopulaSynthesizer` behind the `SyntheticGenerator` interface. They exist to answer: *what does "off-the-shelf" synthetic data generation look like, before any loss-aware penalty is added?* Every utility/privacy number produced by the loss-aware generator will be compared against these baselines.

### Why `sdv` / CTGAN / TVAE / GaussianCopula specifically

- **Plug-and-play, well documented**: `sdv` handles metadata detection, mixed categorical/numeric columns, and sampling — the same preprocessing concerns already solved once in `data/loader.py` don't need to be re-solved for generation.
- **CTGAN and TVAE are the two standard deep generative baselines for tabular data**: CTGAN (conditional GAN) and TVAE (variational autoencoder) represent the two dominant generative paradigms (adversarial vs. likelihood-based). Comparing both against the loss-aware generator will show whether the penalty-term approach generalizes across generator architectures, or is specific to one.
- **GaussianCopulaSynthesizer adds a third, non-neural paradigm**: a classical statistical model (Gaussian copulas over per-column marginal distributions). It trains almost instantly compared to CTGAN/TVAE, giving a fast sanity-check baseline and a reference point for how much the neural approaches actually gain over simple statistical modeling.

### Implementation choices

- **Metadata auto-detection (`Metadata.detect_from_dataframe`)**: rather than requiring callers to hand-write column type metadata, the generator infers it directly from the DataFrame at `fit` time. This keeps the `fit(real_data)` signature identical to `SyntheticGenerator.fit`, with no extra required arguments.
- **Shared `_SDVSynthesizerGenerator` base class**: `CTGANGenerator`, `TVAEGenerator`, and `GaussianCopulaGenerator` differ only in which `sdv` synthesizer class they wrap (`_synthesizer_cls`) and what they expose via `get_training_diagnostics()`. All fit/sample/metadata logic is implemented once and inherited, so adding another `sdv` synthesizer later is a small, self-contained subclass.
- **`**synthesizer_kwargs` passthrough**: constructor arguments (e.g. `epochs`, `batch_size`) are forwarded directly to the underlying `sdv` synthesizer instead of being re-declared, so the wrapper never goes out of sync with `sdv`'s own parameters. From the CLI, this is exposed as repeatable `--kwarg KEY=VALUE` flags (see [experiments.md](experiments.md)).
- **`sample()` raises `RuntimeError` before `fit()`**: fails loudly and immediately rather than returning garbage or raising an unrelated `AttributeError` from inside `sdv`.
- **`src/generators/registry.py`** maps generator name strings (`"ctgan"`, `"tvae"`, `"gaussian_copula"`) to classes via `GENERATORS` + `build_generator(name, **kwargs)`. This is the single place to touch when registering a new generator — nothing in `experiments/` needs to change.

### `get_training_diagnostics()`: generator-specific post-fit output

`SyntheticGenerator.get_training_diagnostics()` (default: `{}`) lets each generator expose whatever diagnostic output is meaningful for *its own* training process, without forcing a one-size-fits-all schema:

| Generator | `get_training_diagnostics()` returns |
|---|---|
| `CTGANGenerator` | `{"loss_values": [...]}` — per-epoch generator/discriminator loss (`CTGANSynthesizer.get_loss_values()`) |
| `TVAEGenerator` | `{"loss_values": [...]}` — per-epoch/batch loss, no discriminator (`TVAESynthesizer.get_loss_values()`) |
| `GaussianCopulaGenerator` | `{"learned_distributions": {...}}` — per-column fitted distribution + parameters (`get_learned_distributions()`) |

This was named `get_training_diagnostics()` (not `get_run_artifacts()`, its original name) because "artifacts" is ambiguous with model/data artifacts elsewhere in the pipeline, while "training diagnostics" precisely describes *what kind* of information this method returns. `build_report()` includes the result under the `"artifacts"` key in the JSON report only if the dict is non-empty, so baseline reports without meaningful diagnostics stay clean.

The future loss-aware generator will override this method to expose its own per-epoch penalty terms (e.g. `{"epoch": e, "loss": total_loss, "mmd": mmd_val, "privacy_penalty": p}`), using the exact same integration point — no changes needed in `report.py` or `experiment.py`.

### Known limitations

- No control yet over training duration from the calling code beyond `**synthesizer_kwargs` — acceptable for baselines, but the loss-aware generator will need explicit hooks into the training loop (epoch-level callbacks) that `sdv`'s synthesizers do not expose by default.
- `get_training_diagnostics()` for CTGAN/TVAE only exposes the numeric loss curve, not the interactive plotly figure that `sdv` can also generate (`get_loss_values_plot()`) — that returns a `plotly.graph_objects.Figure`, which isn't JSON-serializable and doesn't fit the current text-based report format. If needed later, it would be saved as a separate `.html` file via `persistence.py`, not embedded in the JSON report.

---

## 3. Testing approach

`tests/generators/` mirrors the concerns above:

- `test_base.py` checks the *contract* in isolation, using a minimal dummy subclass — this does not require `sdv` to actually train anything, so it runs in milliseconds.
- `test_baseline.py` runs a real (but tiny, `epochs=1`) end-to-end fit/sample cycle for `CTGANGenerator` and `TVAEGenerator`, marked `@pytest.mark.slow` since they exercise real model training rather than pure logic. The same file also tests `get_training_diagnostics()` (empty before fit, populated with `loss_values` after) and `build_generator()`/`GENERATORS` coverage from the registry. Fast contract tests and slow integration tests are kept separate so CI can run the former on every push and the latter on demand.
- `test_artifacts.py` tests the `get_training_diagnostics()` default contract (`{}`) and `_parse_kwargs()` (the `--kwarg KEY=VALUE` CLI parser in `main.py`).

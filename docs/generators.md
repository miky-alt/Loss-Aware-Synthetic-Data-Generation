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

## 2. Baseline generators: CTGAN and TVAE

### What they are

`CTGANGenerator` and `TVAEGenerator` (`src/generators/baseline.py`) wrap the `sdv` library's `CTGANSynthesizer` and `TVAESynthesizer` behind the `SyntheticGenerator` interface. They exist to answer: *what does "off-the-shelf" synthetic data generation look like, before any loss-aware penalty is added?* Every utility/privacy number produced by the loss-aware generator will be compared against these baselines.

### Why `sdv` / CTGAN / TVAE specifically

- **Plug-and-play, well documented**: `sdv` handles metadata detection, mixed categorical/numeric columns, and sampling — the same preprocessing concerns already solved once in `data/loader.py` don't need to be re-solved for generation.
- **CTGAN and TVAE are the two standard deep generative baselines for tabular data**: CTGAN (conditional GAN) and TVAE (variational autoencoder) represent the two dominant generative paradigms (adversarial vs. likelihood-based). Comparing both against the loss-aware generator will show whether the penalty-term approach generalizes across generator architectures, or is specific to one.

### Implementation choices

- **Metadata auto-detection (`Metadata.detect_from_dataframe`)**: rather than requiring callers to hand-write column type metadata, the generator infers it directly from the DataFrame at `fit` time. This keeps the `fit(real_data)` signature identical to `SyntheticGenerator.fit`, with no extra required arguments.
- **Shared `_SDVSynthesizerGenerator` base class**: `CTGANGenerator` and `TVAEGenerator` differ only in which `sdv` synthesizer class they wrap (`_synthesizer_cls`). All fit/sample/metadata logic is implemented once and inherited, so adding another `sdv` synthesizer later (e.g. `GaussianCopulaSynthesizer`) is a two-line change.
- **`**synthesizer_kwargs` passthrough**: constructor arguments (e.g. `epochs`) are forwarded directly to the underlying `sdv` synthesizer instead of being re-declared, so the wrapper never goes out of sync with `sdv`'s own parameters.
- **`sample()` raises `RuntimeError` before `fit()`**: fails loudly and immediately rather than returning garbage or raising an unrelated `AttributeError` from inside `sdv`.

### Known limitations

- No control yet over training duration from the calling code beyond `**synthesizer_kwargs` — acceptable for baselines, but the loss-aware generator will need explicit hooks into the training loop (epoch-level callbacks) that `sdv`'s synthesizers do not expose by default.
- Baselines are evaluated only via `fit`/`sample` end-to-end; they do not expose intermediate training loss, so they cannot yet be used to sanity-check the evaluation metrics *during* training — only after generation completes.

---

## 3. Testing approach

`tests/generators/` mirrors the two concerns above:

- `test_base.py` checks the *contract* in isolation, using a minimal dummy subclass — this does not require `sdv` to actually train anything, so it runs in milliseconds.
- `test_baseline.py` runs a real (but tiny, `epochs=1`) end-to-end fit/sample cycle for both `CTGANGenerator` and `TVAEGenerator`, marked `@pytest.mark.slow` since they exercise real model training rather than pure logic. Fast contract tests and slow integration tests are kept separate so CI can run the former on every push and the latter on demand.

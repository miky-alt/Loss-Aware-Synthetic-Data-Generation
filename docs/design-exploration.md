# Design Exploration: SDV Features Considered but Not Implemented

This document records design decisions where an `sdv` feature was investigated, verified against the installed community-edition library, and deliberately **not** integrated into the current pipeline — together with the reasoning, so the decision isn't re-litigated from scratch later if a new dataset or requirement reopens the question.

---

## 1. Constraint-Augmented Generation (CAG)

### What it is

`sdv` synthesizers can enforce deterministic business rules ("checkout_date must be after checkin_date") on every row of generated data, via `synthesizer.add_constraints([...])`. Constraints can be auto-detected (`synthesizer.detect_constraints(data)`) or created manually (e.g. `sdv.cag.Inequality`, `Range`, `FixedCombinations`).

### What we verified

Auto-detection is gated behind a paid bundle in the installed community edition:

```python
>>> synth.detect_constraints(df)
AttributeError: 'GaussianCopulaSynthesizer' object has no attribute 'detect_constraints'
```

Manually-created constraints (`sdv.cag.Inequality` + `add_constraints()` + `fit()` + `sample()`) **do work** in the community edition — verified end-to-end with a toy `checkin < checkout` example.

### Why it isn't integrated

None of the three project datasets (`adult`, `diabetes`, `heart`) have a known, explicit business rule that must hold row-by-row — this was confirmed to be a genuine exploration, not a response to an observed data-quality issue. Adding a `constraints` parameter now would be unused code with no test coverage grounded in a real need.

### If this becomes relevant later

The integration point would be a `constraints: list | None` parameter on `_SDVSynthesizerGenerator.__init__`, applied via `add_constraints()` right after synthesizer construction and before `.fit()` — mirroring how `**synthesizer_kwargs` is already forwarded. Auto-detection should not be relied upon (it isn't available); constraints would need to be hand-specified per identified rule.

---

## 2. Custom Preprocessing / Transformer Overrides

### What it is

`sdv` auto-assigns a default RDT transformer per column based on detected `sdtype` (e.g. `FloatFormatter` for numerical, `LabelEncoder` for categorical). `synthesizer.update_transformers(column_name_to_transformer={...})` lets you override the transformer for specific columns after the defaults have been auto-assigned.

### Why a per-dataset "transformer script" was considered and rejected

The idea (one script per dataset defining "the transformers for that dataset") conflicts with how `sdv` is actually designed to be used: transformers are auto-assigned first, then **selectively patched** for the few columns that need something different — not defined wholesale. A separate script per dataset would either (a) have to re-implement the default assignment logic to avoid regressing every other column, or (b) leave most columns unconfigured and only be useful for the override case, making the "one script per dataset" framing misleading.

### Why it isn't integrated

No column in any of the three datasets has been observed to need a non-default transformer. Introducing an override mechanism without a concrete failing case would be speculative.

### If this becomes relevant later

The correct integration point (if a specific column's default transformer turns out to be wrong) is the same pattern as constraints: an optional `transformer_overrides: dict | None` parameter on `_SDVSynthesizerGenerator.__init__`, applied via `update_transformers()` after `Metadata.detect_from_dataframe()` and before `.fit()`.

---

## 3. Anonymization (`AnonymizedFaker` / `PseudoAnonymizedFaker`)

### What it is

`sdv` can replace values in columns detected as direct identifiers (email, address, SSN, etc.) with Faker-generated fake values **before** the generative model is trained, so the model never learns the real values. `AnonymizedFaker` is irreversible; `PseudoAnonymizedFaker` keeps an internal mapping back to the real value (useful for authorized tracing).

### Why it isn't relevant to this project

All three datasets are already de-identified, aggregated UCI datasets — none contain literal PII columns (names, emails, addresses, SSNs). `Metadata.detect_from_dataframe()` would detect their columns as plain `numerical`/`categorical` sdtypes, not PII sdtypes, so `AnonymizedFaker` would not be auto-assigned to any column even by default.

### Distinction from this project's actual privacy work

Anonymization is a **preprocessing-time** mitigation for literal, directly-identifying columns. This project's privacy metrics (DCR, NNDR, Inference Risk, Disclosure Protection — see [metrics/privacy.md](metrics/privacy.md)) instead measure **statistical memorization risk** on quasi-identifiers and distributional patterns, empirically, after generation. These are complementary techniques addressing different threat models, not alternatives to each other — but only the second is applicable given the datasets in scope.

---

## Summary

| Feature | Verified working? | Integrated? | Why not |
|---|---|---|---|
| `detect_constraints()` (auto-detect) | No (`AttributeError`, paid bundle) | No | Not available in community edition |
| `add_constraints()` (manual) | Yes | No | No known business rule in any of the 3 datasets |
| `update_transformers()` (custom preprocessing) | Not tested (no case needed it) | No | No column has an observed default-transformer problem |
| `AnonymizedFaker` / `PseudoAnonymizedFaker` | Not tested (not applicable) | No | Datasets have no literal PII columns |

All four remain documented extension points (see the generator-level `**synthesizer_kwargs` and the proposed `constraints`/`transformer_overrides` parameters above) should a future dataset or requirement need them.

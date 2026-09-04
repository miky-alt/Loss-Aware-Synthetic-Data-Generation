# Dataset Selection and Preprocessing

## Motivation for a Multi-Dataset Approach

A single dataset would limit the generalizability of our findings. An evaluation framework or loss function that only works well on one type of data is not a meaningful contribution. By testing across three datasets with different domains, sizes, and feature compositions, we can assess whether our metrics and generator architectures behave consistently — or whether they are sensitive to dataset characteristics.

The three datasets were chosen to cover two axes of variation:

- **Domain**: socioeconomic vs. medical. Medical data carries stronger re-identification risk and stricter ethical obligations (GDPR, HIPAA), making the privacy argument more concrete.
- **Size**: small (~300 rows), medium (~48k rows), large (~100k rows). Synthetic data generators are known to behave differently at different scales — they tend to memorize on small datasets and lose rare patterns on large ones. Testing all three exposes these failure modes.

---

## Datasets

### 1. UCI Adult (Census Income)
- **UCI Repository ID**: 2
- **Key**: `"adult"`
- **Size**: ~48,238 rows, 14 features
- **Domain**: Socioeconomic
- **Target**: Binary — whether an individual earns more than $50K/year (`income > 50K`)
- **Feature mix**: Categorical (workclass, education, marital status, occupation, relationship, race, sex, native country) and numerical (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week)

**Why this dataset:**
The Adult dataset is a standard benchmark in the fairness and privacy literature, which makes it easy to contextualize our results against prior work. More importantly, it contains demographically sensitive attributes — race, sex, and native country — that make it a meaningful testbed for privacy. A synthetic version that faithfully preserves these attributes risks encoding demographic biases; one that obscures them loses downstream fairness properties. This tension is ethically significant and maps directly onto our utility-privacy trade-off framework.

**Preprocessing decisions:**
- Target column (`income`) contains values like `">50K"` and `"<=50K"` with trailing periods in some versions of the dataset. We strip whitespace and periods before binarizing to `1` (`>50K`) and `0` (`<=50K`).
- Categorical values are preserved as strings/categories for SDV metadata detection; they are not treated as continuous measurements. This keeps the semantic category support available for the categorical TVD utility metric.
- Rows with missing values are dropped. The Adult dataset has a small fraction of missing values (marked as `"?"` in the original), and dropping them introduces minimal bias while keeping preprocessing simple.

---

### 2. Diabetes 130-US Hospitals
- **UCI Repository ID**: 296
- **Key**: `"diabetes"`
- **Size**: ~71k rows after cleaning (original ~100k), ~40–50 features after column pruning
- **Domain**: Medical
- **Target**: Binary — whether a patient was readmitted within 30 days (`readmitted == "<30"`)
- **Feature mix**: Demographics (age, gender, race), clinical features (time in hospital, number of procedures, diagnoses codes), and medication indicators

**Why this dataset:**
This dataset is derived from real hospital admissions records across 130 US hospitals between 1999 and 2008. It is the most ethically sensitive dataset in our evaluation: it contains patient demographics, diagnosis codes, and medication histories. The re-identification risk is substantial — a combination of age, gender, race, and rare diagnosis codes can uniquely identify individuals. This makes it the strongest testbed for our privacy metrics and the most compelling argument for why utility-privacy trade-offs in medical synthetic data must be handled carefully.

The readmission prediction task is also clinically meaningful: a synthetic dataset that preserves the statistical signal for early readmission prediction is genuinely useful for downstream ML research in healthcare.

**Preprocessing decisions:**
- Missing values in this dataset are encoded as `"?"` strings. We replace these with `pd.NA` before any processing.
- Columns with more than 40% missing values are dropped entirely. This threshold was chosen to balance data retention against imputation uncertainty — columns that are mostly missing carry little statistical signal and would distort the evaluation metrics. In practice this removes `weight`, `payer_code`, and `medical_specialty`.
- After column pruning, rows with any remaining missing values are dropped.
- The target column has three values: `"<30"` (readmitted within 30 days), `">30"` (readmitted after 30 days), and `"NO"` (not readmitted). We binarize to `1` for `"<30"` and `0` otherwise, focusing on the clinically critical early readmission case.
- String medication, diagnosis, demographic, and age-band columns remain categorical. Binary `gender`, `change`, and `diabetesMed` values are normalized to booleans, as is the binary `readmitted` target. Numeric admission IDs are converted to categorical codes because their values are labels, not measurements.

---

### 3. Heart Disease (Cleveland)
- **UCI Repository ID**: 45
- **Key**: `"heart"`
- **Size**: ~297 rows, 13 features
- **Domain**: Medical
- **Target**: Binary — presence of heart disease (`num > 0`)
- **Feature mix**: Numerical (age, resting blood pressure, cholesterol, max heart rate, ST depression) and categorical (sex, chest pain type, fasting blood sugar, resting ECG, exercise-induced angina, slope, number of vessels, thalassemia)

**Why this dataset:**
The Heart Disease dataset is intentionally small. At ~300 rows, it represents the most challenging scenario for synthetic data generation: generators trained on small datasets tend to memorize training examples rather than learning the underlying distribution. This manifests as very low DCR scores (synthetic records that are near-copies of real ones) and high disclosure rates — exactly the failure modes our privacy metrics are designed to detect.

Including this dataset allows us to demonstrate that our evaluation framework is sensitive to these failure modes, and motivates why the loss function must explicitly penalize memorization (via NNDR-based terms) in low-data regimes.

The loader also restores the semantic types that are hidden by the UCI numeric
codes: `sex`, `fbs`, and `exang` are normalized to booleans, while `cp`,
`restecg`, `slope`, `ca`, and `thal` are categorical. This prevents SDV from
modeling binary indicators and discrete category codes as continuous numeric
measurements. The Heart preprocessing matrix compares the default and
modified Gaussian Copula distributions, `LogScaler(chol)`, their combination,
and matching CTGAN/TVAE variants.

**Preprocessing decisions:**
- The original target column (`num`) has integer values 0–4 representing severity of heart disease. We binarize to `0` (no disease) and `1` (disease present, any severity) to keep the downstream classification task consistent with the other datasets.
- No columns are dropped — the dataset is small enough that all features are retained.
- Rows with missing values are dropped (very few in this dataset).
- Encoded binary columns (`sex`, `fbs`, and `exang`) are normalized to booleans. Encoded discrete columns (`cp`, `restecg`, `slope`, `ca`, and `thal`) are stored as categorical values so SDV and categorical TVD do not interpret their codes as continuous distances.

---

## Unified Interface: DatasetBundle

All loaders return a `DatasetBundle` dataclass:

```python
@dataclass
class DatasetBundle:
    real: pd.DataFrame      # preprocessed dataframe, all numeric
    target_col: str         # column name for downstream classification
    name: str               # human-readable name
    domain: str             # "medical" or "socioeconomic"
```

This interface ensures that all downstream code — evaluation metrics, generator wrappers, the main pipeline — is completely dataset-agnostic. Adding a fourth dataset in the future requires only implementing a new loader function and registering it in the `LOADERS` dict; no other code changes are needed.

---

## Design Decision: Label Encoding vs One-Hot Encoding

We consistently use label encoding (integer codes for categories) rather than one-hot encoding across all datasets. This was a deliberate choice with several motivations:

1. **Distance metrics**: DCR, NNDR, and MMD all operate on Euclidean distance in feature space. One-hot encoding inflates the dimensionality and distorts distances — a single categorical feature with 10 values becomes 10 binary dimensions, dominating the distance calculation. Label encoding keeps the feature space compact and distances meaningful.

2. **Classifier compatibility**: The Random Forest classifiers used in F1 discrepancy and Inference Risk evaluation handle integer-encoded categoricals natively and do not require one-hot encoding.

3. **Generator compatibility**: CTGAN and TVAE (the generators used in this project) have their own internal handling of categorical columns and do not require pre-encoded inputs.

The trade-off is that label encoding imposes an arbitrary ordinal relationship on nominal categories (e.g. `"Male"=0, "Female"=1` implies a numeric ordering that doesn't exist). For the purposes of this project — where we are evaluating distributional similarity, not making causal claims — this is acceptable. A production system would warrant more careful treatment.

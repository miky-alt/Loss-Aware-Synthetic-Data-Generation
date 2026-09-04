"""
Utility metrics for evaluating synthetic data quality.

Distribution metrics compare held-out real test data against synthetic data.
The F1 discrepancy additionally receives the real training split, so its real
baseline classifier and synthetic-data classifier are evaluated on the same
held-out test data.

And return a float (lower = more utility loss) or a dict of floats.
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric view of a frame with bool columns cast to int.

    The loaders keep binary columns (targets, sex, ...) as bool so that sdv's
    metadata detection treats them as categorical. pandas does not count bool
    as a number, so a plain select_dtypes("number") would silently drop them
    from every distance-based metric. Casting to int here keeps them in and
    reproduces exactly what the metrics computed when those columns were int.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(int)
    return out.select_dtypes(include="number")


def _shared_numeric(real: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r, s = _numeric(real), _numeric(synthetic)
    cols = r.columns.intersection(s.columns)
    return r[cols], s[cols]


# ---------------------------------------------------------------------------
# MMD — Maximum Mean Discrepancy
# ---------------------------------------------------------------------------

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Radial Basis Function kernel between rows of X and Y."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = XX + YY.T - 2 * X @ Y.T
    return np.exp(-gamma * dists)


def compute_mmd(real: pd.DataFrame, synthetic: pd.DataFrame, gamma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy (MMD) with RBF kernel.

    Measures the distance between the distributions of real and synthetic data
    in a reproducing kernel Hilbert space.

    Returns a float >= 0. Closer to 0 = distributions are more similar.
    """
    # Use only numeric columns present in both (bool cast to int)
    R, S = _shared_numeric(real, synthetic)
    X = R.values.astype(float)
    Y = S.values.astype(float)

    # Subsample for performance if datasets are large
    max_samples = 2000
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
    if len(Y) > max_samples:
        idx = np.random.choice(len(Y), max_samples, replace=False)
        Y = Y[idx]

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.transform(Y)

    K_XX = _rbf_kernel(X, X, gamma)
    K_YY = _rbf_kernel(Y, Y, gamma)
    K_XY = _rbf_kernel(X, Y, gamma)

    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return float(max(mmd, 0.0))  # numerical safety: MMD^2 >= 0


# ---------------------------------------------------------------------------
# EMD — Earth Mover's Distance (per feature, then averaged)
# ---------------------------------------------------------------------------

def compute_emd(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    """
    Earth Mover's Distance (Wasserstein-1) per numeric feature.

    Returns a dict mapping each feature name to its EMD score,
    plus an 'mean_emd' key with the average across all features.

    Lower = synthetic feature distribution is closer to real.
    """
    R, S = _shared_numeric(real, synthetic)

    scores: dict[str, float] = {}
    for col in R.columns:
        scores[col] = wasserstein_distance(
            R[col].dropna().values,
            S[col].dropna().values,
        )

    scores["mean_emd"] = float(np.mean(list(scores.values()))) if scores else 0.0
    return scores


# ---------------------------------------------------------------------------
# Categorical distribution distance
# ---------------------------------------------------------------------------

def compute_categorical_distance(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    """Compare categorical feature distributions with total variation distance.

    The score for each feature is half the L1 distance between the real and
    synthetic category-frequency distributions. It ranges from 0 (identical)
    to 1 (disjoint distributions), and includes categories present in only
    one of the two dataframes.
    """
    def categorical_columns(data: pd.DataFrame) -> list[str]:
        return [
            column
            for column in data.columns
            if (
                pd.api.types.is_object_dtype(data[column])
                or pd.api.types.is_string_dtype(data[column])
                or isinstance(data[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(data[column])
            )
        ]

    real_cols = categorical_columns(real)
    synthetic_cols = categorical_columns(synthetic)
    cols = [column for column in real_cols if column in synthetic_cols]

    scores: dict[str, float] = {}
    for col in cols:
        real_distribution = real[col].value_counts(normalize=True, dropna=False)
        synthetic_distribution = synthetic[col].value_counts(normalize=True, dropna=False)
        categories = real_distribution.index.union(synthetic_distribution.index)
        real_values = real_distribution.reindex(categories, fill_value=0.0)
        synthetic_values = synthetic_distribution.reindex(categories, fill_value=0.0)
        scores[col] = float(0.5 * (real_values - synthetic_values).abs().sum())

    scores["mean_categorical_distance"] = (
        float(np.mean(list(scores.values()))) if scores else 0.0
    )
    return scores


# ---------------------------------------------------------------------------
# F1 Discrepancy — downstream predictive performance gap
# ---------------------------------------------------------------------------

def compute_f1_discrepancy(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    random_state: int = 42,
) -> dict[str, float]:
    """
    F1 Discrepancy: measures how much predictive utility is lost when training
    on synthetic data vs real data, evaluated on a held-out real test set.

    Protocol:
    1. Train classifier A on the supplied real train split → evaluate on real test
    2. Train classifier B on synthetic data → evaluate on the same real test split
    3. Discrepancy = F1_real - F1_synthetic

    Returns:
    - f1_real: baseline F1 on real data
    - f1_synthetic: F1 when trained on synthetic
    - f1_discrepancy: the gap (higher = more utility loss)
    """
    features = [c for c in _numeric(train_real).columns if c != target_col]
    features = [c for c in features if c in synthetic.columns]

    X_train_real = _numeric(train_real)[features].values
    y_train_real = train_real[target_col].astype(int).values
    X_test = _numeric(test_real)[features].values
    y_test = test_real[target_col].astype(int).values

    X_synthetic = _numeric(synthetic)[features].values
    y_synthetic = synthetic[target_col].astype(int).values

    # Classifier A: trained on real
    clf_real = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf_real.fit(X_train_real, y_train_real)
    f1_real = f1_score(y_test, clf_real.predict(X_test), average="weighted")

    # Classifier B: trained on synthetic, tested on real test set
    clf_syn = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf_syn.fit(X_synthetic, y_synthetic)
    f1_syn = f1_score(y_test, clf_syn.predict(X_test), average="weighted")

    return {
        "f1_real": float(f1_real),
        "f1_synthetic": float(f1_syn),
        "f1_discrepancy": float(f1_real - f1_syn),
    }


# ---------------------------------------------------------------------------
# Correlation preservation
# ---------------------------------------------------------------------------

def compute_correlation_distance(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """
    Frobenius norm of the difference between real and synthetic correlation matrices.

    Measures how well feature relationships are preserved.
    Lower = better correlation preservation.
    """
    R, S = _shared_numeric(real, synthetic)

    corr_real = R.corr().fillna(0).values
    corr_syn = S.corr().fillna(0).values

    return float(np.linalg.norm(corr_real - corr_syn, ord="fro"))


# ---------------------------------------------------------------------------
# Unified summary
# ---------------------------------------------------------------------------

def compute_utility_report(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
) -> dict:
    """
    Run all utility metrics and return a single summary dict.

    MMD, EMD, and correlation distance compare `test_real` to `synthetic`.
    F1 discrepancy trains on `train_real` or synthetic data and evaluates on
    the same held-out `test_real` split.
    """
    emd = compute_emd(test_real, synthetic)
    categorical_distance = compute_categorical_distance(test_real, synthetic)

    return {
        "mmd": compute_mmd(test_real, synthetic),
        "mean_emd": emd["mean_emd"],
        "emd_per_feature": {k: v for k, v in emd.items() if k != "mean_emd"},
        "mean_categorical_distance": categorical_distance["mean_categorical_distance"],
        "categorical_distance_per_feature": {
            k: v for k, v in categorical_distance.items() if k != "mean_categorical_distance"
        },
        "correlation_distance": compute_correlation_distance(test_real, synthetic),
        **compute_f1_discrepancy(train_real, test_real, synthetic, target_col),
    }

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
    # Use only numeric columns present in both
    cols = real.select_dtypes(include="number").columns.intersection(
        synthetic.select_dtypes(include="number").columns
    )
    X = real[cols].values.astype(float)
    Y = synthetic[cols].values.astype(float)

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
    cols = real.select_dtypes(include="number").columns.intersection(
        synthetic.select_dtypes(include="number").columns
    )

    scores: dict[str, float] = {}
    for col in cols:
        scores[col] = wasserstein_distance(
            real[col].dropna().values,
            synthetic[col].dropna().values,
        )

    scores["mean_emd"] = float(np.mean(list(scores.values()))) if scores else 0.0
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
    cols = [c for c in train_real.columns if c != target_col]
    cols = train_real[cols].select_dtypes(include="number").columns.tolist()

    X_train_real = train_real[cols].values
    y_train_real = train_real[target_col].values
    X_test = test_real[cols].values
    y_test = test_real[target_col].values

    X_synthetic = synthetic[cols].values
    y_synthetic = synthetic[target_col].values

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
    cols = real.select_dtypes(include="number").columns.intersection(
        synthetic.select_dtypes(include="number").columns
    )

    corr_real = real[cols].corr().fillna(0).values
    corr_syn = synthetic[cols].corr().fillna(0).values

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

    return {
        "mmd": compute_mmd(test_real, synthetic),
        "mean_emd": emd["mean_emd"],
        "emd_per_feature": {k: v for k, v in emd.items() if k != "mean_emd"},
        "correlation_distance": compute_correlation_distance(test_real, synthetic),
        **compute_f1_discrepancy(train_real, test_real, synthetic, target_col),
    }

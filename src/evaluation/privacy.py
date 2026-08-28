"""
Privacy metrics for evaluating synthetic data safety.

All functions take:
- real: pd.DataFrame
- synthetic: pd.DataFrame

And return a float or a dict of floats.

Convention: higher privacy score = more privacy risk (worse).
This mirrors utility.py where higher = more utility loss.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _prepare_numeric(
    real: pd.DataFrame, synthetic: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and normalize shared numeric columns from real and synthetic.
    Returns (X_real, X_synthetic) as scaled numpy arrays.
    """
    cols = real.select_dtypes(include="number").columns.intersection(
        synthetic.select_dtypes(include="number").columns
    ).tolist()

    scaler = StandardScaler()
    X_real = scaler.fit_transform(real[cols].values.astype(float))
    X_syn = scaler.transform(synthetic[cols].values.astype(float))

    return X_real, X_syn


# ---------------------------------------------------------------------------
# DCR — Distance to Closest Record
# ---------------------------------------------------------------------------

def compute_dcr(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    """
    Distance to Closest Record (DCR).

    For each synthetic record, finds the nearest real record.
    A low DCR means synthetic records are very close to real ones —
    high re-identification risk.

    Returns:
    - dcr_mean: average distance (higher = safer)
    - dcr_median: median distance
    - dcr_min: minimum distance (worst case — the most exposed record)
    - dcr_5th_percentile: bottom 5% threshold
    """
    X_real, X_syn = _prepare_numeric(real, synthetic)

    # Subsample for performance
    max_samples = 2000
    if len(X_real) > max_samples:
        X_real = X_real[np.random.choice(len(X_real), max_samples, replace=False)]
    if len(X_syn) > max_samples:
        X_syn = X_syn[np.random.choice(len(X_syn), max_samples, replace=False)]

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(X_real)
    distances, _ = nn.kneighbors(X_syn)
    distances = distances.flatten()

    return {
        "dcr_mean": float(np.mean(distances)),
        "dcr_median": float(np.median(distances)),
        "dcr_min": float(np.min(distances)),
        "dcr_5th_percentile": float(np.percentile(distances, 5)),
    }


# ---------------------------------------------------------------------------
# NNDR — Nearest Neighbor Distance Ratio (Overfitting Protection)
# ---------------------------------------------------------------------------

def compute_nndr(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    """
    Nearest Neighbor Distance Ratio (NNDR).

    For each synthetic record, computes:
        NNDR = distance_to_1st_closest / distance_to_2nd_closest

    A ratio close to 1 means the two nearest real records are roughly
    equidistant — the synthetic record is well in the interior of the
    real distribution (good).

    A ratio close to 0 means the synthetic record is almost a duplicate
    of one real record — likely memorization (bad).

    Returns:
    - nndr_mean: average ratio (lower = more memorization risk)
    - nndr_median
    - nndr_5th_percentile: worst-case tail
    """
    X_real, X_syn = _prepare_numeric(real, synthetic)

    max_samples = 2000
    if len(X_real) > max_samples:
        X_real = X_real[np.random.choice(len(X_real), max_samples, replace=False)]
    if len(X_syn) > max_samples:
        X_syn = X_syn[np.random.choice(len(X_syn), max_samples, replace=False)]

    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(X_real)
    distances, _ = nn.kneighbors(X_syn)

    # Avoid division by zero
    eps = 1e-10
    ratios = distances[:, 0] / (distances[:, 1] + eps)

    return {
        "nndr_mean": float(np.mean(ratios)),
        "nndr_median": float(np.median(ratios)),
        "nndr_5th_percentile": float(np.percentile(ratios, 5)),
    }


# ---------------------------------------------------------------------------
# Inference Risk
# ---------------------------------------------------------------------------

def compute_inference_risk(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    sensitive_col: str,
) -> dict[str, float]:
    """
    Inference Risk.

    Simulates an attribute inference attack:
    1. Train a classifier on synthetic data to predict a sensitive column
    2. Evaluate it on real data

    If the attacker achieves high F1 on real data using only synthetic data
    for training, the synthetic data is leaking the sensitive attribute.

    Returns:
    - inference_f1: attacker's F1 on real data (higher = more risk)
    - baseline_f1: majority-class baseline F1 (for comparison)
    """
    if sensitive_col not in real.columns or sensitive_col not in synthetic.columns:
        raise ValueError(f"sensitive_col '{sensitive_col}' not found in both dataframes.")

    feature_cols = [
        c for c in real.select_dtypes(include="number").columns
        if c != sensitive_col
    ]
    feature_cols = [c for c in feature_cols if c in synthetic.columns]

    X_syn = synthetic[feature_cols].values
    y_syn = synthetic[sensitive_col].values

    X_real = real[feature_cols].values
    y_real = real[sensitive_col].values

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_syn, y_syn)

    y_pred = clf.predict(X_real)
    inference_f1 = f1_score(y_real, y_pred, average="weighted", zero_division=0)

    # Baseline: always predict majority class
    majority = np.bincount(y_real.astype(int)).argmax()
    baseline_f1 = f1_score(
        y_real,
        np.full_like(y_real, majority),
        average="weighted",
        zero_division=0,
    )

    return {
        "inference_f1": float(inference_f1),
        "baseline_f1": float(baseline_f1),
        "inference_risk_delta": float(inference_f1 - baseline_f1),
    }


# ---------------------------------------------------------------------------
# Disclosure Protection
# ---------------------------------------------------------------------------

def compute_disclosure_protection(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    distance_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Disclosure Protection.

    Fraction of synthetic records that fall within a tight distance
    threshold of any real record. These records are near-copies and
    could directly disclose real individuals.

    Returns:
    - disclosure_rate: fraction of synthetic records at risk (lower = safer)
    - n_at_risk: absolute count of at-risk synthetic records
    - threshold_used: the distance threshold applied
    """
    X_real, X_syn = _prepare_numeric(real, synthetic)

    max_samples = 2000
    if len(X_real) > max_samples:
        X_real = X_real[np.random.choice(len(X_real), max_samples, replace=False)]
    if len(X_syn) > max_samples:
        X_syn = X_syn[np.random.choice(len(X_syn), max_samples, replace=False)]

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(X_real)
    distances, _ = nn.kneighbors(X_syn)
    distances = distances.flatten()

    at_risk = distances < distance_threshold

    return {
        "disclosure_rate": float(at_risk.mean()),
        "n_at_risk": int(at_risk.sum()),
        "threshold_used": distance_threshold,
    }


# ---------------------------------------------------------------------------
# Unified summary
# ---------------------------------------------------------------------------

def compute_privacy_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    sensitive_col: str,
    disclosure_threshold: float = 0.5,
) -> dict:
    """
    Run all privacy metrics and return a single summary dict.
    """
    return {
        **compute_dcr(real, synthetic),
        **compute_nndr(real, synthetic),
        **compute_inference_risk(real, synthetic, sensitive_col),
        **compute_disclosure_protection(real, synthetic, disclosure_threshold),
    }

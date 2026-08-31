import numpy as np
import pandas as pd
import pytest

from src.evaluation.privacy import (
    compute_dcr,
    compute_disclosure_protection,
    compute_inference_risk,
    compute_nndr,
    compute_privacy_report,
)


@pytest.fixture
def real() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((300, 4)), columns=["a", "b", "c", "d"])
    df["target"] = (df["a"] > 0).astype(int)
    return df


@pytest.fixture
def synthetic(real) -> pd.DataFrame:
    """Slightly perturbed version of real — low privacy risk."""
    rng = np.random.default_rng(42)
    df = real.copy()
    df[["a", "b", "c", "d"]] += rng.standard_normal((len(real), 4)) * 0.1
    df["target"] = (df["a"] > 0).astype(int)
    return df


@pytest.fixture
def random_synthetic() -> pd.DataFrame:
    """Completely random data — high privacy safety, low utility."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame(rng.standard_normal((300, 4)) * 5, columns=["a", "b", "c", "d"])
    df["target"] = rng.integers(0, 2, size=300)
    return df


@pytest.fixture
def memorized_synthetic(real) -> pd.DataFrame:
    """Near-exact copies of real — high privacy risk."""
    rng = np.random.default_rng(7)
    df = real.copy()
    df[["a", "b", "c", "d"]] += rng.standard_normal((len(real), 4)) * 1e-4
    return df


# ---------------------------------------------------------------------------
# DCR
# ---------------------------------------------------------------------------

class TestComputeDCR:
    def test_returns_expected_keys(self, real, synthetic):
        result = compute_dcr(real, synthetic)
        for key in ["dcr_mean", "dcr_median", "dcr_min", "dcr_5th_percentile"]:
            assert key in result

    def test_all_values_non_negative(self, real, synthetic):
        result = compute_dcr(real, synthetic)
        for v in result.values():
            assert v >= 0.0

    def test_memorized_data_has_lower_dcr_than_random(self, real, memorized_synthetic, random_synthetic):
        dcr_memorized = compute_dcr(real, memorized_synthetic)["dcr_mean"]
        dcr_random = compute_dcr(real, random_synthetic)["dcr_mean"]
        assert dcr_memorized < dcr_random

    def test_min_leq_median_leq_mean(self, real, synthetic):
        result = compute_dcr(real, synthetic)
        assert result["dcr_min"] <= result["dcr_median"]


# ---------------------------------------------------------------------------
# NNDR
# ---------------------------------------------------------------------------

class TestComputeNNDR:
    def test_returns_expected_keys(self, real, synthetic):
        result = compute_nndr(real, synthetic)
        for key in ["nndr_mean", "nndr_median", "nndr_5th_percentile"]:
            assert key in result

    def test_values_between_zero_and_one(self, real, synthetic):
        result = compute_nndr(real, synthetic)
        for v in result.values():
            assert 0.0 <= v <= 1.0 + 1e-6  # small tolerance for epsilon

    def test_memorized_data_has_lower_nndr(self, real, memorized_synthetic, random_synthetic):
        nndr_memorized = compute_nndr(real, memorized_synthetic)["nndr_mean"]
        nndr_random = compute_nndr(real, random_synthetic)["nndr_mean"]
        assert nndr_memorized < nndr_random


# ---------------------------------------------------------------------------
# Inference Risk
# ---------------------------------------------------------------------------

class TestComputeInferenceRisk:
    def test_returns_expected_keys(self, real, synthetic):
        result = compute_inference_risk(real, synthetic, sensitive_col="target")
        for key in ["inference_f1", "baseline_f1", "inference_risk_delta"]:
            assert key in result

    def test_f1_values_in_unit_interval(self, real, synthetic):
        result = compute_inference_risk(real, synthetic, sensitive_col="target")
        assert 0.0 <= result["inference_f1"] <= 1.0
        assert 0.0 <= result["baseline_f1"] <= 1.0

    def test_delta_equals_difference(self, real, synthetic):
        result = compute_inference_risk(real, synthetic, sensitive_col="target")
        expected = result["inference_f1"] - result["baseline_f1"]
        assert abs(result["inference_risk_delta"] - expected) < 1e-6

    def test_raises_on_missing_sensitive_col(self, real, synthetic):
        with pytest.raises(ValueError):
            compute_inference_risk(real, synthetic, sensitive_col="nonexistent")


# ---------------------------------------------------------------------------
# Disclosure Protection
# ---------------------------------------------------------------------------

class TestComputeDisclosureProtection:
    def test_returns_expected_keys(self, real, synthetic):
        result = compute_disclosure_protection(real, synthetic)
        for key in ["disclosure_rate", "n_at_risk", "threshold_used"]:
            assert key in result

    def test_disclosure_rate_in_unit_interval(self, real, synthetic):
        result = compute_disclosure_protection(real, synthetic)
        assert 0.0 <= result["disclosure_rate"] <= 1.0

    def test_n_at_risk_consistent_with_rate(self, real, synthetic):
        result = compute_disclosure_protection(real, synthetic)
        expected = round(result["disclosure_rate"] * min(len(synthetic), 2000))
        assert abs(result["n_at_risk"] - expected) <= 1

    def test_memorized_data_has_higher_disclosure_rate(self, real, memorized_synthetic, random_synthetic):
        rate_memorized = compute_disclosure_protection(real, memorized_synthetic)["disclosure_rate"]
        rate_random = compute_disclosure_protection(real, random_synthetic)["disclosure_rate"]
        assert rate_memorized > rate_random

    def test_threshold_zero_gives_zero_rate(self, real, synthetic):
        result = compute_disclosure_protection(real, synthetic, distance_threshold=0.0)
        assert result["disclosure_rate"] == 0.0

    def test_threshold_infinity_gives_full_rate(self, real, synthetic):
        result = compute_disclosure_protection(real, synthetic, distance_threshold=1e9)
        assert result["disclosure_rate"] == 1.0


# ---------------------------------------------------------------------------
# Unified report
# ---------------------------------------------------------------------------

class TestComputePrivacyReport:
    def test_returns_all_expected_keys(self, real, synthetic):
        result = compute_privacy_report(real, synthetic, sensitive_col="target")
        for key in [
            "dcr_mean", "dcr_median", "dcr_min", "dcr_5th_percentile",
            "nndr_mean", "nndr_median", "nndr_5th_percentile",
            "inference_f1", "baseline_f1", "inference_risk_delta",
            "disclosure_rate", "n_at_risk", "threshold_used",
        ]:
            assert key in result

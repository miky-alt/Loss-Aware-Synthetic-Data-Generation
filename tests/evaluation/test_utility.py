import numpy as np
import pandas as pd
import pytest

from src.evaluation.utility import (
    compute_correlation_distance,
    compute_emd,
    compute_f1_discrepancy,
    compute_mmd,
    compute_utility_report,
)


@pytest.fixture
def real() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((300, 4)), columns=["a", "b", "c", "d"])
    df["target"] = (df["a"] > 0).astype(int)
    return df


@pytest.fixture
def synthetic(real) -> pd.DataFrame:
    """Slightly perturbed version of real — should have low but non-zero loss."""
    rng = np.random.default_rng(42)
    df = real.copy()
    df[["a", "b", "c", "d"]] += rng.standard_normal((len(real), 4)) * 0.1
    df["target"] = (df["a"] > 0).astype(int)
    return df


@pytest.fixture
def random_synthetic() -> pd.DataFrame:
    """Completely random data — should have high utility loss vs real."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame(rng.standard_normal((300, 4)) * 5, columns=["a", "b", "c", "d"])
    df["target"] = rng.integers(0, 2, size=300)
    return df


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

class TestComputeMMD:
    def test_returns_non_negative_float(self, real, synthetic):
        result = compute_mmd(real, synthetic)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_identical_data_gives_near_zero(self, real):
        result = compute_mmd(real, real.copy())
        assert result < 0.01

    def test_similar_data_lower_than_random(self, real, synthetic, random_synthetic):
        assert compute_mmd(real, synthetic) < compute_mmd(real, random_synthetic)

    def test_ignores_non_numeric_columns(self, real, synthetic):
        real_with_str = real.copy()
        real_with_str["label"] = "foo"
        synthetic_with_str = synthetic.copy()
        synthetic_with_str["label"] = "bar"
        result = compute_mmd(real_with_str, synthetic_with_str)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# EMD
# ---------------------------------------------------------------------------

class TestComputeEMD:
    def test_returns_dict_with_mean_and_per_feature(self, real, synthetic):
        result = compute_emd(real, synthetic)
        assert "mean_emd" in result
        for col in ["a", "b", "c", "d", "target"]:
            assert col in result

    def test_all_values_non_negative(self, real, synthetic):
        result = compute_emd(real, synthetic)
        for v in result.values():
            assert v >= 0.0

    def test_identical_data_gives_zero(self, real):
        result = compute_emd(real, real.copy())
        assert result["mean_emd"] < 1e-6

    def test_similar_data_lower_than_random(self, real, synthetic, random_synthetic):
        assert compute_emd(real, synthetic)["mean_emd"] < compute_emd(real, random_synthetic)["mean_emd"]


# ---------------------------------------------------------------------------
# F1 Discrepancy
# ---------------------------------------------------------------------------

class TestComputeF1Discrepancy:
    def test_returns_expected_keys(self, real, synthetic):
        result = compute_f1_discrepancy(real.iloc[:240], real.iloc[240:], synthetic, target_col="target")
        assert "f1_real" in result
        assert "f1_synthetic" in result
        assert "f1_discrepancy" in result

    def test_f1_values_in_unit_interval(self, real, synthetic):
        result = compute_f1_discrepancy(real.iloc[:240], real.iloc[240:], synthetic, target_col="target")
        assert 0.0 <= result["f1_real"] <= 1.0
        assert 0.0 <= result["f1_synthetic"] <= 1.0

    def test_discrepancy_equals_difference(self, real, synthetic):
        result = compute_f1_discrepancy(real.iloc[:240], real.iloc[240:], synthetic, target_col="target")
        assert abs(result["f1_discrepancy"] - (result["f1_real"] - result["f1_synthetic"])) < 1e-6

    def test_similar_data_lower_discrepancy_than_random(self, real, synthetic, random_synthetic):
        close = compute_f1_discrepancy(real.iloc[:240], real.iloc[240:], synthetic, target_col="target")["f1_discrepancy"]
        far = compute_f1_discrepancy(real.iloc[:240], real.iloc[240:], random_synthetic, target_col="target")["f1_discrepancy"]
        assert close < far

    def test_raises_on_missing_target_col(self, real, synthetic):
        with pytest.raises(Exception):
            compute_f1_discrepancy(real.iloc[:240], real.iloc[240:], synthetic, target_col="nonexistent")


# ---------------------------------------------------------------------------
# Correlation Distance
# ---------------------------------------------------------------------------

class TestComputeCorrelationDistance:
    def test_returns_non_negative_float(self, real, synthetic):
        result = compute_correlation_distance(real, synthetic)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_identical_data_gives_near_zero(self, real):
        result = compute_correlation_distance(real, real.copy())
        assert result < 1e-6

    def test_similar_data_lower_than_random(self, real, synthetic, random_synthetic):
        assert compute_correlation_distance(real, synthetic) < compute_correlation_distance(real, random_synthetic)


# ---------------------------------------------------------------------------
# Unified report
# ---------------------------------------------------------------------------

class TestComputeUtilityReport:
    def test_returns_all_expected_keys(self, real, synthetic):
        result = compute_utility_report(real.iloc[:240], real.iloc[240:], synthetic, target_col="target")
        for key in ["mmd", "mean_emd", "emd_per_feature", "correlation_distance",
                    "f1_real", "f1_synthetic", "f1_discrepancy"]:
            assert key in result

    def test_emd_per_feature_is_dict(self, real, synthetic):
        result = compute_utility_report(real.iloc[:240], real.iloc[240:], synthetic, target_col="target")
        assert isinstance(result["emd_per_feature"], dict)

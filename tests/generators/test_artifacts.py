"""Tests for _parse_kwargs and get_run_artifacts contract."""
import pytest

from src.generators.base import SyntheticGenerator
from src.generators.baseline import CTGANGenerator, GaussianCopulaGenerator
from src.main import _parse_kwargs


# --- _parse_kwargs ---

def test_parse_kwargs_empty():
    assert _parse_kwargs(None) == {}
    assert _parse_kwargs([]) == {}


def test_parse_kwargs_int_cast():
    assert _parse_kwargs(["epochs=50"]) == {"epochs": 50}


def test_parse_kwargs_float_cast():
    assert _parse_kwargs(["lr=0.001"]) == {"lr": 0.001}


def test_parse_kwargs_string_value():
    assert _parse_kwargs(["default_distribution=beta"]) == {"default_distribution": "beta"}


def test_parse_kwargs_multiple():
    result = _parse_kwargs(["epochs=100", "batch_size=500", "lr=0.002"])
    assert result == {"epochs": 100, "batch_size": 500, "lr": 0.002}


def test_parse_kwargs_missing_equals_raises():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _parse_kwargs(["epochs"])


# --- get_run_artifacts contract ---

def test_default_get_run_artifacts_returns_empty_dict():
    class _MinimalGenerator(SyntheticGenerator):
        def fit(self, real_data):
            return self
        def sample(self, num_rows):
            import pandas as pd
            return pd.DataFrame()

    assert _MinimalGenerator().get_run_artifacts() == {}


def test_gaussian_copula_artifacts_empty_before_fit():
    gen = GaussianCopulaGenerator()
    assert gen.get_run_artifacts() == {}


def test_ctgan_artifacts_empty_before_fit():
    gen = CTGANGenerator(epochs=1)
    # baseline SDV generators don't override get_run_artifacts — returns {}
    assert gen.get_run_artifacts() == {}

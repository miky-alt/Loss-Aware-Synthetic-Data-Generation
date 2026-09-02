"""Tests for _parse_kwargs and get_training_diagnostics contract."""
import pytest

from src.generators.base import SyntheticGenerator
from src.generators.baseline import CTGANGenerator, GaussianCopulaGenerator
from src.main import _merge_kwargs, _parse_kwargs, _parse_kwargs_json


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


# --- _parse_kwargs_json / _merge_kwargs ---

def test_parse_kwargs_json_empty():
    assert _parse_kwargs_json(None) == {}


def test_parse_kwargs_json_supports_nested_values():
    result = _parse_kwargs_json(
        '{"numerical_distributions": {"age": "norm"}, "enforce_rounding": false}'
    )
    assert result == {
        "numerical_distributions": {"age": "norm"},
        "enforce_rounding": False,
    }


def test_parse_kwargs_json_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        _parse_kwargs_json("{invalid")


def test_parse_kwargs_json_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        _parse_kwargs_json('["epochs", 50]')


def test_merge_kwargs_combines_distinct_keys():
    result = _merge_kwargs(
        {"epochs": 50},
        {"numerical_distributions": {"age": "norm"}},
    )
    assert result == {
        "epochs": 50,
        "numerical_distributions": {"age": "norm"},
    }


def test_merge_kwargs_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="Duplicate kwargs"):
        _merge_kwargs({"epochs": 50}, {"epochs": 300})


# --- get_training_diagnostics contract ---

def test_default_get_training_diagnostics_returns_empty_dict():
    class _MinimalGenerator(SyntheticGenerator):
        def fit(self, real_data):
            return self
        def sample(self, num_rows):
            import pandas as pd
            return pd.DataFrame()

    assert _MinimalGenerator().get_training_diagnostics() == {}


def test_gaussian_copula_diagnostics_empty_before_fit():
    gen = GaussianCopulaGenerator()
    assert gen.get_training_diagnostics() == {}


def test_ctgan_diagnostics_empty_before_fit():
    gen = CTGANGenerator(epochs=1)
    # baseline SDV generators don't override get_training_diagnostics — returns {}
    assert gen.get_training_diagnostics() == {}

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "dataset_name", "num_samples"),
    [
        ("scripts.run_adult", "adult", 38_096),
        ("scripts.run_diabetes", "diabetes", 47_303),
        ("scripts.run_heart", "heart", 238),
    ],
)
def test_dataset_runner_configuration(module_name, dataset_name, num_samples):
    module = importlib.import_module(module_name)

    assert module.NUM_SAMPLES == num_samples
    assert len(module.EXPERIMENTS) >= 2
    assert module.EXPERIMENTS[0]["generator_name"] == "gaussian_copula"
    assert module.EXPERIMENTS[1]["generator_name"] == "gaussian_copula"
    assert module.EXPERIMENTS[0]["generator_kwargs"] == {}
    assert module.EXPERIMENTS[1]["generator_kwargs"]["numerical_distributions"]
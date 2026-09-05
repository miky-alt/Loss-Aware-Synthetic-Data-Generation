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
    assert module.SEEDS == (1, 2, 3, 4, 5)
    assert len(module.EXPERIMENTS) == 3
    assert [item["generator_name"] for item in module.EXPERIMENTS] == [
        "gaussian_copula",
        "ctgan",
        "tvae",
    ]
    assert module.EXPERIMENTS[0]["generator_kwargs"] == {}
    assert all(item["transformer_specs"] == {} for item in module.EXPERIMENTS)
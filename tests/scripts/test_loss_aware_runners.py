import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "dataset_name", "num_samples"),
    [
        ("scripts.run_adult_loss_aware", "adult", 38_096),
        ("scripts.run_diabetes_loss_aware", "diabetes", 47_303),
        ("scripts.run_heart_loss_aware", "heart", 238),
    ],
)
def test_loss_aware_runner_matrix(module_name, dataset_name, num_samples):
    module = importlib.import_module(module_name)

    assert module.NUM_SAMPLES == num_samples
    assert module.SEEDS == (1, 2, 3, 4, 5)
    assert len(module.EXPERIMENTS) == 8
    assert {experiment["generator_name"] for experiment in module.EXPERIMENTS} == {
        "tvae_loss_aware",
        "ctgan_loss_aware",
    }
    assert {experiment["name"].split("_")[-1] for experiment in module.EXPERIMENTS} == {
        "baseline",
        "only",
        "full",
    }


def test_loss_aware_runner_regimes_have_expected_penalties():
    module = importlib.import_module("scripts.run_heart_loss_aware")

    regimes = {experiment["name"]: experiment["generator_kwargs"] for experiment in module.EXPERIMENTS}
    for generator_name in ("tvae_loss_aware", "ctgan_loss_aware"):
        assert regimes[f"{generator_name}_baseline"]["lambda_mmd"] == 0.0
        assert regimes[f"{generator_name}_baseline"]["lambda_corr"] == 0.0
        assert regimes[f"{generator_name}_baseline"]["lambda_priv"] == 0.0
        assert regimes[f"{generator_name}_utility_only"]["lambda_priv"] == 0.0
        assert regimes[f"{generator_name}_privacy_only"]["lambda_mmd"] == 0.0
        assert regimes[f"{generator_name}_full"]["lambda_mmd"] == 1.0
        assert regimes[f"{generator_name}_full"]["lambda_corr"] == 0.5
        assert regimes[f"{generator_name}_full"]["lambda_priv"] == 10.0

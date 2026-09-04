import importlib


def test_adult_preprocessing_matrix_has_eight_combinations():
    module = importlib.import_module("scripts.run_adult_preprocessing")

    assert len(module.EXPERIMENTS) == 8
    assert [item["generator_name"] for item in module.EXPERIMENTS] == [
        "gaussian_copula",
        "gaussian_copula",
        "gaussian_copula",
        "gaussian_copula",
        "ctgan",
        "ctgan",
        "tvae",
        "tvae",
    ]
    assert module.EXPERIMENTS[0]["transformer_specs"] == {}
    assert module.EXPERIMENTS[2]["transformer_specs"] == {
        "fnlwgt": {"name": "LogScaler", "kwargs": {}}
    }
    assert module.EXPERIMENTS[3]["transformer_specs"] == {
        "fnlwgt": {"name": "LogScaler", "kwargs": {}}
    }
    assert module.EXPERIMENTS[3]["generator_kwargs"] == module.EXPERIMENTS[1]["generator_kwargs"]
    assert module.EXPERIMENTS[4]["transformer_specs"] == module.EXPERIMENTS[6]["transformer_specs"]
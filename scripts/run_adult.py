"""Run the Adult preprocessing comparison matrix."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.config import TrainingConfig
from src.experiments.multiseed import run_experiment_matrix


NUM_SAMPLES = 38_096
SEEDS = (1, 2, 3)
SEED = SEEDS[0]
TEST_SIZE = 0.2
OUTPUT_DIR = "experiments/results"

EXPERIMENTS = [
    {
        "name": "gaussian_copula_default",
        "generator_name": "gaussian_copula",
        "generator_kwargs": {},
        "transformer_specs": {},
    },
    {
        "name": "gaussian_copula_modified_distributions",
        "generator_name": "gaussian_copula",
        "generator_kwargs": {
            "numerical_distributions": {
                "age": "gamma",
                "fnlwgt": "gamma",
                "education-num": "truncnorm",
                "capital-gain": "gamma",
                "capital-loss": "gamma",
                "hours-per-week": "truncnorm",
            }
        },
        "transformer_specs": {},
    },
    {
        "name": "gaussian_copula_log_fnlwgt",
        "generator_name": "gaussian_copula",
        "generator_kwargs": {},
        "transformer_specs": {
            "fnlwgt": {"name": "LogScaler", "kwargs": {}},
        },
    },
    {
        "name": "gaussian_copula_modified_log_fnlwgt",
        "generator_name": "gaussian_copula",
        "generator_kwargs": {
            "numerical_distributions": {
                "age": "gamma",
                "fnlwgt": "gamma",
                "education-num": "truncnorm",
                "capital-gain": "gamma",
                "capital-loss": "gamma",
                "hours-per-week": "truncnorm",
            }
        },
        "transformer_specs": {
            "fnlwgt": {"name": "LogScaler", "kwargs": {}},
        },
    },
    {
        "name": "ctgan_default",
        "generator_name": "ctgan",
        "generator_kwargs": {"epochs": 500, "verbose": True},
        "transformer_specs": {},
    },
    {
        "name": "ctgan_log_fnlwgt",
        "generator_name": "ctgan",
        "generator_kwargs": {"epochs": 500, "verbose": True},
        "transformer_specs": {
            "fnlwgt": {"name": "LogScaler", "kwargs": {}},
        },
    },
    {
        "name": "tvae_default",
        "generator_name": "tvae",
        "generator_kwargs": {"epochs": 500, "verbose": True},
        "transformer_specs": {},
    },
    {
        "name": "tvae_log_fnlwgt",
        "generator_name": "tvae",
        "generator_kwargs": {"epochs": 500, "verbose": True},
        "transformer_specs": {
            "fnlwgt": {"name": "LogScaler", "kwargs": {}},
        },
    },
]


def main() -> None:
    run_experiment_matrix(
        dataset_name="adult",
        experiments=EXPERIMENTS,
        num_samples=NUM_SAMPLES,
        seeds=SEEDS,
        test_size=TEST_SIZE,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
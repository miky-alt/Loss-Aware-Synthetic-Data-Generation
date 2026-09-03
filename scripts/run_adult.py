"""Run the configured Gaussian Copula experiments for UCI Adult."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.config import TrainingConfig
from src.experiments.experiment import run_experiment
from src.experiments.report import summarize_report


NUM_SAMPLES = 38_096
SEED = 42
TEST_SIZE = 0.2
OUTPUT_DIR = "experiments/results"

EXPERIMENTS = [
    {
        "name": "gaussian_copula_default",
        "generator_name": "gaussian_copula",
        "generator_kwargs": {},
    },
    {
        "name": "gaussian_copula_modified",
        "generator_name": "gaussian_copula",
        "generator_kwargs": {
            "numerical_distributions": {
                "age": "gamma",
                "fnlwgt": "gamma",
                "education-num": "truncnorm",
                "capital-gain": "gamma",
                "capital-loss": "gamma",
                # gaussian_kde is quadratic in the number of training rows.
                "hours-per-week": "truncnorm",
            }
        },
    },
]


def main() -> None:
    for experiment in EXPERIMENTS:
        config = TrainingConfig(
            dataset_name="adult",
            generator_name=experiment["generator_name"],
            num_samples=NUM_SAMPLES,
            seed=SEED,
            test_size=TEST_SIZE,
            generator_kwargs=experiment["generator_kwargs"],
        )
        report = run_experiment(config, output_dir=OUTPUT_DIR)
        print(f"\n=== {experiment['name']} ({config.run_name}) ===")
        print(summarize_report(report))


if __name__ == "__main__":
    main()
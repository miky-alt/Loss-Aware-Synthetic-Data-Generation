"""Run the configured Gaussian Copula experiments for Diabetes 130-US Hospitals."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.config import TrainingConfig
from src.experiments.experiment import run_experiment
from src.experiments.report import summarize_report


NUM_SAMPLES = 47_303
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
                "time_in_hospital": "gamma",
                "num_lab_procedures": "truncnorm",
                "num_medications": "gamma",
                "number_outpatient": "gamma",
                "number_emergency": "gamma",
                "number_inpatient": "gamma",
            }
        },
    },
]


def main() -> None:
    for experiment in EXPERIMENTS:
        config = TrainingConfig(
            dataset_name="diabetes",
            generator_name=experiment["generator_name"],
            num_samples=NUM_SAMPLES,
            seed=SEED,
            test_size=TEST_SIZE,
            generator_kwargs=experiment["generator_kwargs"],
            transformer_specs=experiment.get("transformer_specs", {}),
        )
        report = run_experiment(config, output_dir=OUTPUT_DIR)
        print(f"\n=== {experiment['name']} ({config.run_name}) ===")
        print(summarize_report(report))


if __name__ == "__main__":
    main()
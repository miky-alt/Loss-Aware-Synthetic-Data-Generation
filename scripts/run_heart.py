"""Run the configured Gaussian Copula experiments for Heart Disease."""

from src.experiments.config import TrainingConfig
from src.experiments.experiment import run_experiment
from src.experiments.report import summarize_report


NUM_SAMPLES = 238
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
                "trestbps": "truncnorm",
                "chol": "gamma",
                "thalach": "truncnorm",
                "oldpeak": "gamma",
            }
        },
    },
]


def main() -> None:
    for experiment in EXPERIMENTS:
        config = TrainingConfig(
            dataset_name="heart",
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
"""Run the configured Gaussian Copula experiments for Diabetes 130-US Hospitals."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.config import TrainingConfig
from src.experiments.multiseed import run_experiment_matrix


NUM_SAMPLES = 47_303
SEEDS = (1, 2, 3)
SEED = SEEDS[0]
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
            }
        },
        "transformer_specs": {
            "number_outpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_emergency": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_inpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
        },
    },
    {
        "name": "gaussian_copula_log_positive_counts",
        "generator_name": "gaussian_copula",
        "generator_kwargs": {},
        "transformer_specs": {
            "time_in_hospital": {"name": "LogScaler", "kwargs": {}},
            "num_medications": {"name": "LogScaler", "kwargs": {}},
            "number_outpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_emergency": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_inpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
        },
    },
    {
        "name": "gaussian_copula_modified_log_positive_counts",
        "generator_name": "gaussian_copula",
        "generator_kwargs": {
            "numerical_distributions": {
                "time_in_hospital": "gamma",
                "num_lab_procedures": "truncnorm",
                "num_medications": "gamma",
            }
        },
        "transformer_specs": {
            "time_in_hospital": {"name": "LogScaler", "kwargs": {}},
            "num_medications": {"name": "LogScaler", "kwargs": {}},
            "number_outpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_emergency": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_inpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
        },
    },
    {
        "name": "ctgan_default",
        "generator_name": "ctgan",
        "generator_kwargs": {"epochs": 500, "verbose": True},
        "transformer_specs": {},
    },
    {
        "name": "ctgan_log_positive_counts",
        "generator_name": "ctgan",
        "generator_kwargs": {"epochs": 500, "verbose": True},
        "transformer_specs": {
            "time_in_hospital": {"name": "LogScaler", "kwargs": {}},
            "num_medications": {"name": "LogScaler", "kwargs": {}},
            "number_outpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_emergency": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_inpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
        },
    },
    {
        "name": "tvae_default",
        "generator_name": "tvae",
        "generator_kwargs": {"epochs": 500, "verbose": True},
        "transformer_specs": {},
    },
    {
        "name": "tvae_log_positive_counts",
        "generator_name": "tvae",
        "generator_kwargs": {"epochs": 500, "verbose": True},
        "transformer_specs": {
            "time_in_hospital": {"name": "LogScaler", "kwargs": {}},
            "num_medications": {"name": "LogScaler", "kwargs": {}},
            "number_outpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_emergency": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
            "number_inpatient": {"name": "ClusterBasedNormalizer", "kwargs": {"max_clusters": 10}},
        },
    },
]


def main() -> None:
    run_experiment_matrix(
        dataset_name="diabetes",
        experiments=EXPERIMENTS,
        num_samples=NUM_SAMPLES,
        seeds=SEEDS,
        test_size=TEST_SIZE,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
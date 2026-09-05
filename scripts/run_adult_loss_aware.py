"""Run the Adult loss-aware generator ablation matrix."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.multiseed import run_experiment_matrix


NUM_SAMPLES = 38_096
SEEDS = (1, 2, 3, 4, 5)
TEST_SIZE = 0.2
OUTPUT_DIR = "experiments/results"
GENERATOR_KWARGS = {"batch_size": 500, "epochs": 100, "verbose": True}


def _regime(name: str, generator_name: str, **penalties: float) -> dict:
    return {
        "name": f"{generator_name}_{name}",
        "generator_name": generator_name,
        "generator_kwargs": {
            **GENERATOR_KWARGS,
            "lambda_mmd": 0.0,
            "lambda_corr": 0.0,
            "lambda_priv": 0.0,
            "dcr_margin": 1.5,
            **penalties,
        },
        "transformer_specs": {},
    }


EXPERIMENTS = [
    _regime("baseline", generator_name)
    for generator_name in ("tvae_loss_aware", "ctgan_loss_aware")
]
EXPERIMENTS += [
    _regime("utility_only", generator_name, lambda_mmd=1.0, lambda_corr=0.5)
    for generator_name in ("tvae_loss_aware", "ctgan_loss_aware")
]
EXPERIMENTS += [
    _regime("privacy_only", generator_name, lambda_priv=1.0, dcr_margin=1.5)
    for generator_name in ("tvae_loss_aware", "ctgan_loss_aware")
]
EXPERIMENTS += [
    _regime(
        "full",
        generator_name,
        lambda_mmd=1.0,
        lambda_corr=0.5,
        lambda_priv=10.0,
        dcr_margin=1.5,
    )
    for generator_name in ("tvae_loss_aware", "ctgan_loss_aware")
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

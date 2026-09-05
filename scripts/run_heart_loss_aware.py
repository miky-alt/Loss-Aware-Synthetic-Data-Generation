"""Run the Heart Disease loss-aware generator ablation matrix."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.multiseed import run_experiment_matrix


NUM_SAMPLES = 238
SEEDS = (1, 2, 3, 4, 5)
TEST_SIZE = 0.2
OUTPUT_DIR = "experiments/results"

# CTGAN's discriminator packs rows in groups of `pac` (default 10), so its
# batch_size must be divisible by 10; TVAE has no such constraint. 2000
# epochs for CTGAN because it needs far more steps than a VAE to converge on
# 237 rows (see docs/loss_aware_training.md §7h).
GENERATOR_KWARGS = {
    "tvae_loss_aware": {"batch_size": 32, "epochs": 300, "verbose": True},
    "ctgan_loss_aware": {"batch_size": 50, "epochs": 2000, "verbose": True},
}


def _regime(name: str, generator_name: str, **penalties: float) -> dict:
    return {
        "name": f"{generator_name}_{name}",
        "generator_name": generator_name,
        "generator_kwargs": {
            **GENERATOR_KWARGS[generator_name],
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
        dataset_name="heart",
        experiments=EXPERIMENTS,
        num_samples=NUM_SAMPLES,
        seeds=SEEDS,
        test_size=TEST_SIZE,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()

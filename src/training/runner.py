"""Ties dataset loading, generator fit/sample, and evaluation into one experiment run.

Kept separate from `src/evaluation/`: this module owns *orchestration*
(which generator, which dataset, where results land), while `evaluation/`
owns the metrics themselves. Swapping a generator or dataset here never
requires touching the metric code, and vice versa.
"""

import json
from pathlib import Path

from src.data.loader import load_dataset
from src.evaluation.privacy import compute_privacy_report
from src.evaluation.utility import compute_utility_report
from src.generators.baseline import CTGANGenerator, TVAEGenerator
from src.training.config import TrainingConfig

GENERATORS = {
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
}


def run_experiment(config: TrainingConfig, output_dir: str = "experiments/results") -> dict:
    """Fit `config.generator_name` on `config.dataset_name`, sample, evaluate, and persist the report."""
    bundle = load_dataset(config.dataset_name)

    if config.generator_name not in GENERATORS:
        raise ValueError(f"Unknown generator '{config.generator_name}'. Choose from: {list(GENERATORS.keys())}")
    generator = GENERATORS[config.generator_name](**config.generator_kwargs)
    generator.fit(bundle.real)
    synthetic = generator.sample(config.num_samples)

    report = {
        "config": vars(config),
        "utility": compute_utility_report(bundle.real, synthetic, bundle.target_col),
        # target_col doubles as the sensitive attribute: none of the current
        # datasets define a separate one.
        "privacy": compute_privacy_report(bundle.real, synthetic, bundle.target_col),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{config.run_name}.json", "w") as f:
        json.dump(report, f, indent=2)

    return report

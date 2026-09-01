"""Orchestrates one full experiment: load data, train, evaluate, persist.

Each step is delegated to a single-responsibility module:
- src/data/loader.py          -> dataset loading
- src/generators/registry.py  -> generator construction
- src/experiments/persistence.py -> generator serialization
- src/experiments/report.py   -> metrics + report persistence
"""

from src.data.loader import load_dataset, split_dataset
from src.experiments.config import ExperimentMode, TrainingConfig
from src.experiments.persistence import load_generator, save_generator, save_metadata
from src.experiments.report import append_to_index, build_report, save_report
from src.generators.registry import build_generator


def run_experiment(
    config: TrainingConfig,
    output_dir: str = "experiments/results",
    mode: ExperimentMode = ExperimentMode.TRAIN_AND_EVALUATE,
    pretrained_run_name: str | None = None,
) -> dict:
    bundle = load_dataset(config.dataset_name)
    # deterministic (test_size, seed) => identical split whether training now
    # or re-evaluating a previously trained generator later
    train_bundle, test_bundle = split_dataset(bundle, config.test_size, config.seed)

    if mode == ExperimentMode.EVALUATE_ONLY:
        if pretrained_run_name is None:
            raise ValueError("pretrained_run_name must be set when mode=EVALUATE_ONLY")
        generator = load_generator(pretrained_run_name, output_dir)
    else:
        generator = build_generator(config.generator_name, **config.generator_kwargs)
        generator.fit(train_bundle.real)
        save_generator(generator, config.run_name, output_dir)
        save_metadata(train_bundle.real, config.run_name, output_dir)

    synthetic = generator.sample(config.num_samples)
    report = build_report(config, test_bundle.real, synthetic, bundle.target_col, generator)
    save_report(report, config.run_name, output_dir)
    append_to_index(report, config.run_name, output_dir)

    return report

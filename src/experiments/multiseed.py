"""Repeated-seed experiment execution and confidence-interval aggregation."""

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy import stats

from src.experiments.config import TrainingConfig
from src.experiments.experiment import run_experiment
from src.experiments.report import _ReportEncoder, summarize_report


def _ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean, standard deviation, and Student-t CI half-width."""
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if len(values) < 2:
        return mean, standard_deviation, math.nan
    half_width = float(stats.t.ppf(0.975, len(values) - 1) * standard_deviation / math.sqrt(len(values)))
    return mean, standard_deviation, half_width


def _numeric_metrics(value: object, prefix: str = "") -> dict[str, float]:
    """Flatten numeric report values, including per-feature metric mappings."""
    if isinstance(value, dict):
        metrics = {}
        for key, nested_value in value.items():
            path = f"{prefix}.{key}" if prefix else key
            metrics.update(_numeric_metrics(nested_value, path))
        return metrics
    if isinstance(value, (bool, int, float, np.number)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def aggregate_reports(reports: Iterable[dict]) -> dict:
    """Aggregate all numeric utility and privacy report metrics across seeds."""
    reports = list(reports)
    if not reports:
        raise ValueError("at least one report is required")

    values_by_metric: dict[str, list[float]] = {}
    for report in reports:
        for section in ("utility", "privacy"):
            for metric, value in _numeric_metrics(report.get(section, {}), section).items():
                values_by_metric.setdefault(metric, []).append(value)

    metrics = {}
    for metric, values in sorted(values_by_metric.items()):
        mean, standard_deviation, half_width = _ci95(np.asarray(values, dtype=float))
        metrics[metric] = {
            "mean": mean,
            "std": standard_deviation,
            "ci95_half_width": half_width,
            "ci95_lower": mean - half_width if not math.isnan(half_width) else math.nan,
            "ci95_upper": mean + half_width if not math.isnan(half_width) else math.nan,
        }

    return {
        "n_seeds": len(reports),
        "seeds": [report["config"]["seed"] for report in reports],
        "metrics": metrics,
    }


def _aggregate_path(output_dir: str, dataset_name: str, experiment_name: str, seeds: tuple[int, ...]) -> Path:
    seed_label = "-".join(str(seed) for seed in seeds)
    safe_name = experiment_name.replace("/", "_").replace(" ", "_")
    return Path(output_dir) / "aggregates" / f"{dataset_name}_{safe_name}_seeds_{seed_label}.json"


def run_experiment_matrix(
    dataset_name: str,
    experiments: list[dict],
    num_samples: int,
    seeds: Iterable[int],
    test_size: float = 0.2,
    output_dir: str = "experiments/results",
) -> dict[str, dict]:
    """Run every experiment for every seed and persist aggregate CI reports."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")

    aggregates = {}
    for experiment in experiments:
        reports = []
        for seed in seeds:
            config = TrainingConfig(
                dataset_name=dataset_name,
                generator_name=experiment["generator_name"],
                num_samples=num_samples,
                seed=seed,
                test_size=test_size,
                generator_kwargs=experiment.get("generator_kwargs", {}),
                transformer_specs=experiment.get("transformer_specs", {}),
            )
            report = run_experiment(config, output_dir=output_dir)
            reports.append(report)
            print(f"\n=== {experiment['name']} seed={seed} ({config.run_name}) ===")
            print(summarize_report(report))

        aggregate = aggregate_reports(reports)
        aggregate["dataset_name"] = dataset_name
        aggregate["experiment_name"] = experiment["name"]
        aggregate["generator_name"] = experiment["generator_name"]
        aggregate_path = _aggregate_path(output_dir, dataset_name, experiment["name"], seeds)
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        with aggregate_path.open("w") as file:
            json.dump(aggregate, file, indent=2, cls=_ReportEncoder, allow_nan=False)
        aggregates[experiment["name"]] = aggregate
        print(f"\n=== aggregate {experiment['name']} ({aggregate_path}) ===")
        for metric, values in aggregate["metrics"].items():
            print(f"{metric}: {values['mean']} +/- {values['ci95_half_width']}")

    from src.experiments.plot_matrix import plot_matrix_aggregates

    plot_paths = plot_matrix_aggregates(dataset_name, output_dir, seeds)
    for path in plot_paths:
        print(f"wrote {path}")

    return aggregates
"""Plot confidence-interval aggregates for a dataset experiment matrix."""

import json
from pathlib import Path
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


UTILITY_METRICS = (
    "utility.mmd",
    "utility.mean_emd",
    "utility.mean_categorical_distance",
    "utility.correlation_distance",
    "utility.f1_discrepancy",
)
PRIVACY_METRICS = (
    "privacy.dcr_mean",
    "privacy.nndr_mean",
    "privacy.inference_f1",
    "privacy.disclosure_rate",
)


def _load_aggregates(dataset_name: str, output_dir: str, seeds: Iterable[int] | None = None) -> list[dict]:
    aggregate_dir = Path(output_dir) / "aggregates"
    seed_label = None if seeds is None else "-".join(str(seed) for seed in seeds)
    pattern = f"{dataset_name}_*_seeds_{seed_label}.json" if seed_label else f"{dataset_name}_*_seeds_*.json"
    reports = []
    for path in sorted(aggregate_dir.glob(pattern)):
        with path.open() as file:
            reports.append(json.load(file))
    return reports


def _plot_section(
    reports: list[dict],
    metrics: tuple[str, ...],
    section_name: str,
    dataset_name: str,
    output_dir: str,
) -> Path | None:
    available = [metric for metric in metrics if any(metric in report["metrics"] for report in reports)]
    if not available:
        return None

    labels = [report["experiment_name"] for report in reports]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(len(available), 1, figsize=(12, 3.2 * len(available)), squeeze=False)
    for axis, metric in zip(axes[:, 0], available):
        means = [report["metrics"].get(metric, {}).get("mean", np.nan) for report in reports]
        errors = [report["metrics"].get(metric, {}).get("ci95_half_width", 0.0) for report in reports]
        axis.errorbar(positions, means, yerr=errors, fmt="o", capsize=4)
        axis.set_title(metric.removeprefix(f"{section_name}."))
        axis.set_xticks(positions, labels=labels, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.3)

    figure.suptitle(f"{dataset_name} experiment matrix: {section_name} (95% CI)")
    figure.tight_layout()
    path = Path(output_dir) / f"{dataset_name}_{section_name}_matrix_ci95.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_matrix_aggregates(
    dataset_name: str,
    output_dir: str = "experiments/results",
    seeds: Iterable[int] | None = None,
) -> list[Path]:
    """Create utility and privacy CI plots from matrix aggregate JSON files."""
    reports = _load_aggregates(dataset_name, output_dir, seeds)
    if not reports:
        raise FileNotFoundError(f"no aggregate reports found for dataset '{dataset_name}'")

    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for metrics, section_name in ((UTILITY_METRICS, "utility"), (PRIVACY_METRICS, "privacy")):
        path = _plot_section(reports, metrics, section_name, dataset_name, str(figure_dir))
        if path is not None:
            paths.append(path)
    return paths

"""Inspect the preprocessed datasets used by the experiment pipeline."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sdv.metadata import Metadata

from src.data.loader import LOADERS, DatasetBundle, load_dataset


def describe_dataset(bundle: DatasetBundle, head_rows: int = 5) -> str:
    """Return a readable overview of a preprocessed DatasetBundle."""
    data = bundle.real
    lines = [
        f"Dataset: {bundle.name}",
        f"Domain: {bundle.domain}",
        f"Shape: {data.shape[0]} rows x {data.shape[1]} columns",
        f"Target: {bundle.target_col}",
        "",
        f"Head ({head_rows} rows):",
        data.head(head_rows).to_string(index=False),
        "",
        "Column dtypes:",
        data.dtypes.to_string(),
        "",
        "Missing values:",
        data.isna().sum().to_string(),
        "",
        "Target distribution:",
        data[bundle.target_col].value_counts(normalize=True).sort_index().to_string(),
        "",
        "Numeric summary:",
        data.describe().transpose().to_string(),
    ]
    return "\n".join(lines)


def _dataset_stem(bundle: DatasetBundle) -> str:
    return f"{bundle.name.lower().replace(' ', '_')}_distributions"


def save_analysis(
    bundle: DatasetBundle,
    description: str,
    output_dir: str = "analysis",
) -> Path:
    """Save the readable dataset analysis and return its path."""
    analysis_dir = Path(output_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_dir / f"{_dataset_stem(bundle)}.txt"
    analysis_path.write_text(description + "\n", encoding="utf-8")
    return analysis_path


def save_sdv_metadata(
    bundle: DatasetBundle,
    output_dir: str = "metadata",
) -> Path:
    """Detect SDV metadata from a preprocessed dataset and save it as JSON."""
    metadata = Metadata.detect_from_dataframe(bundle.real)
    metadata_dir = Path(output_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{bundle.name.lower().replace(' ', '_')}_metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
    metadata.save_to_json(str(metadata_path))
    return metadata_path


def plot_distributions(
    bundle: DatasetBundle,
    output_dir: str = "plots",
    bins: int = 30,
) -> Path:
    """Save one grid plot of every column distribution and return its path."""
    data = bundle.real
    columns = list(data.columns)
    n_columns = 3
    n_rows = (len(columns) + n_columns - 1) // n_columns
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, column in zip(axes, columns):
        values = data[column].dropna()
        if values.nunique() <= 20:
            values.value_counts().sort_index().plot.bar(ax=axis)
            axis.set_xlabel(column)
            axis.set_ylabel("Count")
        else:
            values.plot.hist(ax=axis, bins=bins)
            axis.set_xlabel(column)
            axis.set_ylabel("Count")
        axis.set_title(column)

    for axis in axes[len(columns):]:
        axis.remove()
    figure.suptitle(f"{bundle.name}: feature distributions", fontsize=16)
    figure.tight_layout()

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{_dataset_stem(bundle)}.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the preprocessed datasets used by the experiment pipeline."
    )
    parser.add_argument(
        "dataset",
        choices=[*LOADERS.keys(), "all"],
        help="dataset to inspect, or 'all' to inspect every dataset",
    )
    parser.add_argument("--head", type=int, default=5, help="number of initial rows to display")
    parser.add_argument("--bins", type=int, default=30, help="number of bins for continuous-column histograms")
    parser.add_argument("--plot-dir", default="plots", help="directory for distribution PNG files")
    parser.add_argument("--analysis-dir", default="analysis", help="directory for readable analysis text files")
    parser.add_argument("--metadata-dir", default="metadata", help="directory for SDV metadata JSON files")
    parser.add_argument("--no-plot", action="store_true", help="print the text analysis without saving plots")
    args = parser.parse_args()

    names = LOADERS.keys() if args.dataset == "all" else [args.dataset]
    for index, name in enumerate(names):
        if index:
            print("\n" + "=" * 80 + "\n")
        try:
            bundle = load_dataset(name)
        except ConnectionError as error:
            if args.dataset != "all":
                raise
            print(f"Could not load {name}: {error}")
            continue
        description = describe_dataset(bundle, head_rows=args.head)
        print(description)
        print(f"Analysis file: {save_analysis(bundle, description, args.analysis_dir)}")
        print(f"SDV metadata: {save_sdv_metadata(bundle, args.metadata_dir)}")
        if not args.no_plot:
            print(f"Distribution plot: {plot_distributions(bundle, args.plot_dir, args.bins)}")


if __name__ == "__main__":
    main()

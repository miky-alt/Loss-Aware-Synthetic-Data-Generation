"""Builds and persists the evaluation report for a completed training run.

Kept separate from `runner.py`: this module owns *what to measure and where
to store it*, while `runner.py` owns only fitting and sampling.
"""

import json
import subprocess
from enum import Enum
from pathlib import Path

import pandas as pd

from src.evaluation.privacy import compute_privacy_report
from src.evaluation.utility import compute_utility_report
from src.experiments.config import TrainingConfig
from src.experiments.persistence import load_report


def _get_code_version() -> str:
    """Short git commit hash of the code that produced this report, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


class _ReportEncoder(json.JSONEncoder):
    """Serializes Enum values as their names so reports are human-readable JSON."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return super().default(obj)


def build_report(
    config: TrainingConfig,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    generator=None,
) -> dict:
    report = {
        "config": vars(config),
        "code_version": _get_code_version(),
        "utility": compute_utility_report(real, synthetic, target_col),
        # target_col doubles as the sensitive attribute: none of the current
        # datasets define a separate one.
        "privacy": compute_privacy_report(real, synthetic, target_col),
    }
    if generator is not None:
        artifacts = generator.get_training_diagnostics()
        if artifacts:
            report["artifacts"] = artifacts
    return report


def save_report(report: dict, run_name: str, output_dir: str = "experiments/results") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, cls=_ReportEncoder)
    return out_path


def append_to_index(report: dict, run_name: str, output_dir: str = "experiments/results") -> Path:
    """Append a one-line summary of this run to index.jsonl for fast filtering
    across many runs, without opening every individual report file.
    """
    config = report["config"]
    row = {
        "run_name": run_name,
        "created_at": config.get("created_at"),
        "code_version": report.get("code_version"),
        "dataset_name": config.get("dataset_name"),
        "generator_name": config.get("generator_name"),
        "seed": config.get("seed"),
        "test_size": config.get("test_size"),
        "num_samples": config.get("num_samples"),
        "generator_kwargs": config.get("generator_kwargs"),
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    with open(index_path, "a") as f:
        f.write(json.dumps(row, cls=_ReportEncoder) + "\n")
    return index_path


def summarize_report(
    report: dict | str,
    output_dir: str = "experiments/results",
) -> str:
    """One line per aggregate metric; skips per-feature breakdowns and full config.

    `report` can be a run_name string (loaded from disk) or an already-loaded dict.
    """
    if isinstance(report, str):
        report = load_report(report, output_dir)
    lines = [f"config: {report['config'].get('dataset_name')} / {report['config'].get('generator_name')}"]
    for section in ("utility", "privacy"):
        for key, value in report.get(section, {}).items():
            if key == "emd_per_feature":
                continue
            lines.append(f"{section}.{key}: {value}")
    if "artifacts" in report:
        for key, value in report["artifacts"].items():
            if isinstance(value, list):
                lines.append(f"artifacts.{key}: {len(value)} entries")
            else:
                lines.append(f"artifacts.{key}: present")
    return "\n".join(lines)

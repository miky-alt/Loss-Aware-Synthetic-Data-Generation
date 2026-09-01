"""Serializes and deserializes trained generator artifacts."""

import json
import pickle
from pathlib import Path

import pandas as pd
from sdv.metadata import Metadata


def save_generator(generator, run_name: str, output_dir: str = "experiments/results") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(generator, f)
    return out_path


def load_generator(run_name: str, output_dir: str = "experiments/results"):
    out_path = Path(output_dir) / f"{run_name}.pkl"
    with open(out_path, "rb") as f:
        return pickle.load(f)


def save_metadata(real_data: pd.DataFrame, run_name: str, output_dir: str = "experiments/results") -> Path:
    """Detect and save the dataset's schema (sdtypes) — a property of the data, not the generator."""
    metadata = Metadata.detect_from_dataframe(real_data)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.metadata.json"
    metadata.save_to_json(str(out_path))
    return out_path


def load_report(run_name: str, output_dir: str = "experiments/results") -> dict:
    out_path = Path(output_dir) / f"{run_name}.json"
    with open(out_path) as f:
        return json.load(f)

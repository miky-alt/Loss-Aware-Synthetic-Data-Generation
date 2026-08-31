"""Serializes and deserializes trained generator artifacts."""

import json
import pickle
from pathlib import Path


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


def load_report(run_name: str, output_dir: str = "experiments/results") -> dict:
    out_path = Path(output_dir) / f"{run_name}.json"
    with open(out_path) as f:
        return json.load(f)

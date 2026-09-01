"""Query experiments/results/index.jsonl to find runs matching given criteria.

Kept separate from report.py: this module reads the run index for lookup,
while report.py owns building/writing individual run reports.
"""

import json
from pathlib import Path


def load_index(output_dir: str = "experiments/results") -> list[dict]:
    """Read all rows from index.jsonl, or [] if no runs have been recorded yet."""
    index_path = Path(output_dir) / "index.jsonl"
    if not index_path.exists():
        return []
    with open(index_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def query_index(
    output_dir: str = "experiments/results",
    dataset_name: str | None = None,
    generator_name: str | None = None,
    seed: int | None = None,
    test_size: float | None = None,
    code_version: str | None = None,
    kwargs: dict | None = None,
) -> list[dict]:
    """Return index rows matching every given (non-None) filter.

    `kwargs`, if given, matches rows whose generator_kwargs contains at least
    the provided key/value pairs (a superset match, not exact equality).
    """
    rows = load_index(output_dir)
    matches = []
    for row in rows:
        if dataset_name is not None and row.get("dataset_name") != dataset_name:
            continue
        if generator_name is not None and row.get("generator_name") != generator_name:
            continue
        if seed is not None and row.get("seed") != seed:
            continue
        if test_size is not None and row.get("test_size") != test_size:
            continue
        if code_version is not None and row.get("code_version") != code_version:
            continue
        if kwargs:
            row_kwargs = row.get("generator_kwargs") or {}
            if not all(row_kwargs.get(k) == v for k, v in kwargs.items()):
                continue
        matches.append(row)
    return matches

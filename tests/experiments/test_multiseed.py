import json

import pytest

from src.experiments import multiseed


def _report(seed: int, offset: float = 0.0) -> dict:
    return {
        "config": {"seed": seed},
        "utility": {
            "mmd": 0.1 + offset,
            "mean_emd": 0.2 + offset,
            "emd_per_feature": {"age": 0.3 + offset},
            "mean_categorical_distance": 0.4 + offset,
            "categorical_distance_per_feature": {"sex": 0.5 + offset},
            "correlation_distance": 0.6 + offset,
            "f1_discrepancy": 0.7 + offset,
        },
        "privacy": {
            "dcr_mean": 1.0 + offset,
            "nndr_mean": 0.8 + offset,
            "inference_f1": 0.2 + offset,
            "disclosure_rate": 0.1 + offset,
        },
    }


def test_aggregate_reports_includes_all_metric_levels():
    result = multiseed.aggregate_reports([_report(1), _report(2, 0.1), _report(3, 0.2)])

    assert result["n_seeds"] == 3
    assert result["seeds"] == [1, 2, 3]
    for metric in (
        "utility.mmd",
        "utility.emd_per_feature.age",
        "utility.mean_categorical_distance",
        "utility.categorical_distance_per_feature.sex",
        "privacy.dcr_mean",
    ):
        assert set(result["metrics"][metric]) == {
            "mean",
            "std",
            "ci95_half_width",
            "ci95_lower",
            "ci95_upper",
        }
        assert result["metrics"][metric]["ci95_half_width"] > 0


def test_aggregate_reports_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one report"):
        multiseed.aggregate_reports([])


def test_run_experiment_matrix_runs_each_configuration_for_each_seed(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(config, output_dir):
        calls.append((config.dataset_name, config.generator_name, config.seed, output_dir))
        return _report(config.seed)

    monkeypatch.setattr(multiseed, "run_experiment", fake_run_experiment)
    experiments = [{"name": "demo", "generator_name": "gaussian_copula", "generator_kwargs": {}}]

    result = multiseed.run_experiment_matrix(
        dataset_name="adult",
        experiments=experiments,
        num_samples=10,
        seeds=(1, 2, 3),
        output_dir=str(tmp_path),
    )

    assert [call[2] for call in calls] == [1, 2, 3]
    assert result["demo"]["n_seeds"] == 3
    aggregate_files = list((tmp_path / "aggregates").glob("*.json"))
    assert len(aggregate_files) == 1
    saved = json.loads(aggregate_files[0].read_text())
    assert saved["metrics"]["utility.mmd"]["ci95_lower"] < saved["metrics"]["utility.mmd"]["mean"]
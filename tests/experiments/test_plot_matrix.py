import json

from src.experiments.plot_matrix import plot_matrix_aggregates


def test_plot_matrix_aggregates_creates_utility_and_privacy_plots(tmp_path):
    aggregate_dir = tmp_path / "aggregates"
    aggregate_dir.mkdir()
    metrics = {
        "utility.mmd": {"mean": 0.1, "ci95_half_width": 0.01},
        "utility.mean_emd": {"mean": 0.2, "ci95_half_width": 0.02},
        "utility.mean_categorical_distance": {"mean": 0.3, "ci95_half_width": 0.03},
        "utility.correlation_distance": {"mean": 0.4, "ci95_half_width": 0.04},
        "utility.f1_discrepancy": {"mean": 0.5, "ci95_half_width": 0.05},
        "privacy.dcr_mean": {"mean": 1.0, "ci95_half_width": 0.1},
        "privacy.nndr_mean": {"mean": 0.8, "ci95_half_width": 0.08},
        "privacy.inference_f1": {"mean": 0.2, "ci95_half_width": 0.02},
        "privacy.disclosure_rate": {"mean": 0.1, "ci95_half_width": 0.01},
    }
    for name in ("default", "preprocessed"):
        path = aggregate_dir / f"adult_{name}_seeds_1-2-3.json"
        path.write_text(json.dumps({"experiment_name": name, "metrics": metrics}))

    paths = plot_matrix_aggregates("adult", str(tmp_path), seeds=(1, 2, 3))

    assert {path.name for path in paths} == {
        "adult_utility_matrix_ci95.png",
        "adult_privacy_matrix_ci95.png",
    }
    assert all(path.exists() for path in paths)

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import analyze
from src.data.analyze import describe_dataset, plot_distributions, save_analysis, save_sdv_metadata
from src.data.loader import DatasetBundle, load_uci_adult, split_dataset


@pytest.fixture
def bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"x": rng.random(n), "y": rng.integers(0, 2, n)})
    return DatasetBundle(real=df, target_col="y", name="fake", domain="test")


def test_split_dataset_sizes(bundle):
    train_bundle, test_bundle = split_dataset(bundle, test_size=0.2, seed=42)
    assert len(train_bundle.real) == 80
    assert len(test_bundle.real) == 20


def test_split_dataset_no_overlap(bundle):
    train_bundle, test_bundle = split_dataset(bundle, test_size=0.2, seed=42)
    # indices are reset in both splits, so compare on a value unique per row instead
    assert set(train_bundle.real["x"]).isdisjoint(set(test_bundle.real["x"]))


def test_split_dataset_preserves_metadata(bundle):
    train_bundle, test_bundle = split_dataset(bundle, test_size=0.2, seed=42)
    for b in (train_bundle, test_bundle):
        assert b.target_col == bundle.target_col
        assert b.name == bundle.name
        assert b.domain == bundle.domain


def test_split_dataset_deterministic_with_same_seed(bundle):
    train1, test1 = split_dataset(bundle, test_size=0.2, seed=7)
    train2, test2 = split_dataset(bundle, test_size=0.2, seed=7)
    pd.testing.assert_frame_equal(train1.real, train2.real)
    pd.testing.assert_frame_equal(test1.real, test2.real)


def test_split_dataset_different_seed_gives_different_split(bundle):
    train1, _ = split_dataset(bundle, test_size=0.2, seed=1)
    train2, _ = split_dataset(bundle, test_size=0.2, seed=2)
    assert not train1.real.equals(train2.real)


def test_adult_loader_preserves_categorical_values_for_sdv(monkeypatch):
    features = pd.DataFrame(
        {
            "workclass": ["Private", "State-gov"],
            "education": ["Bachelors", "HS-grad"],
            "occupation": ["Tech-support", "Exec-managerial"],
        }
    )
    targets = pd.DataFrame({"income": ["<=50K", ">50K"]})
    dataset = SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))
    monkeypatch.setattr("src.data.loader.fetch_ucirepo", lambda id: dataset)

    bundle = load_uci_adult()

    for column in ("workclass", "education", "occupation"):
        assert pd.api.types.is_string_dtype(bundle.real[column])
    assert bundle.real["education"].tolist() == ["Bachelors", "HS-grad"]
    assert bundle.real["income"].tolist() == [0, 1]


def test_describe_dataset_includes_head_and_summary(bundle):
    description = describe_dataset(bundle, head_rows=2)

    assert "Dataset: fake" in description
    assert "Head (2 rows):" in description
    assert "Target distribution:" in description
    assert "Numeric summary:" in description


def test_plot_distributions_creates_png(bundle, tmp_path):
    plot_path = plot_distributions(bundle, output_dir=str(tmp_path))

    assert plot_path.exists()
    assert plot_path.suffix == ".png"


def test_save_analysis_creates_readable_text_file(bundle, tmp_path):
    description = describe_dataset(bundle, head_rows=2)

    analysis_path = save_analysis(bundle, description, output_dir=str(tmp_path))

    assert analysis_path.exists()
    assert analysis_path.suffix == ".txt"
    assert analysis_path.read_text(encoding="utf-8").strip() == description


def test_save_sdv_metadata_creates_json(bundle, tmp_path):
    metadata_path = save_sdv_metadata(bundle, output_dir=str(tmp_path))

    assert metadata_path.exists()
    assert metadata_path.name == "fake_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "columns" in metadata["tables"]["table"]
    assert metadata["tables"]["table"]["columns"]["x"]["sdtype"] == "numerical"


def test_analyze_all_skips_dataset_connection_errors(monkeypatch, capsys):
    loaded = DatasetBundle(
        real=pd.DataFrame({"feature": [1, 2], "target": [0, 1]}),
        target_col="target",
        name="fake",
        domain="test",
    )

    def load_dataset(name):
        if name == "diabetes":
            raise ConnectionError("offline")
        return loaded

    monkeypatch.setattr(analyze, "LOADERS", {"adult": object(), "diabetes": object()})
    monkeypatch.setattr(analyze, "load_dataset", load_dataset)
    monkeypatch.setattr(analyze, "describe_dataset", lambda bundle, head_rows: bundle.name)
    monkeypatch.setattr(analyze, "plot_distributions", lambda *args: None)
    monkeypatch.setattr("sys.argv", ["analyze", "all", "--no-plot"])

    analyze.main()

    output = capsys.readouterr().out
    assert "fake" in output
    assert "Could not load diabetes: offline" in output

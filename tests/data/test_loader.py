import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import analyze
from src.data.analyze import describe_dataset, plot_distributions, save_analysis, save_sdv_metadata
from src.data.loader import (
    DatasetBundle,
    load_diabetes_130,
    load_heart_disease,
    load_uci_adult,
    split_dataset,
)


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
            "sex": ["Male", "Female"],
        }
    )
    targets = pd.DataFrame({"income": ["<=50K", ">50K"]})
    dataset = SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))
    monkeypatch.setattr("src.data.loader.fetch_ucirepo", lambda id: dataset)

    bundle = load_uci_adult()

    for column in ("workclass", "education", "occupation"):
        assert pd.api.types.is_string_dtype(bundle.real[column])
    assert bundle.real["education"].tolist() == ["Bachelors", "HS-grad"]
    assert pd.api.types.is_bool_dtype(bundle.real["sex"])
    assert bundle.real["sex"].tolist() == [True, False]
    assert pd.api.types.is_bool_dtype(bundle.real["income"])
    assert bundle.real["income"].tolist() == [False, True]


def test_diabetes_loader_normalizes_binary_columns_to_boolean(monkeypatch):
    features = pd.DataFrame(
        {
            "race": ["Caucasian", "AfricanAmerican"],
            "gender": ["Male", "Female"],
            "admission_type_id": [1, 2],
            "discharge_disposition_id": [22, 1],
            "admission_source_id": [7, 1],
            "change": ["Ch", "No"],
            "diabetesMed": ["Yes", "No"],
        }
    )
    targets = pd.DataFrame({"readmitted": ["<30", "NO"]})
    dataset = SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))
    monkeypatch.setattr("src.data.loader.fetch_ucirepo", lambda id: dataset)

    bundle = load_diabetes_130()

    assert pd.api.types.is_string_dtype(bundle.real["race"])
    for column in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        assert pd.api.types.is_string_dtype(bundle.real[column])
        assert all(isinstance(value, str) for value in bundle.real[column].dropna())
    for column in ("gender", "change", "diabetesMed", "readmitted"):
        assert pd.api.types.is_bool_dtype(bundle.real[column])
    assert bundle.real["gender"].tolist() == [True, False]
    assert bundle.real["change"].tolist() == [True, False]
    assert bundle.real["diabetesMed"].tolist() == [True, False]


def test_heart_loader_preserves_binary_and_categorical_semantics(monkeypatch):
    features = pd.DataFrame(
        {
            "age": [63, 67],
            "sex": [1, 0],
            "cp": [1, 4],
            "trestbps": [145, 160],
            "chol": [233, 286],
            "fbs": [1, 0],
            "restecg": [2, 0],
            "thalach": [150, 108],
            "exang": [0, 1],
            "oldpeak": [2.3, 1.5],
            "slope": [3, 2],
            "ca": [0.0, 3.0],
            "thal": [6.0, 3.0],
        }
    )
    targets = pd.DataFrame({"num": [0, 1]})
    dataset = SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))
    monkeypatch.setattr("src.data.loader.fetch_ucirepo", lambda id: dataset)

    bundle = load_heart_disease()

    for column in ("sex", "fbs", "exang", "num"):
        assert pd.api.types.is_bool_dtype(bundle.real[column])
    for column in ("cp", "restecg", "slope", "ca", "thal"):
        assert pd.api.types.is_string_dtype(bundle.real[column])
        assert all(isinstance(value, str) for value in bundle.real[column].dropna())


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


def test_plot_distributions_handles_high_cardinality_categoricals(tmp_path):
    data = pd.DataFrame(
        {
            "category": [f"value-{index}" for index in range(25)],
            "target": [False, True] * 12 + [False],
        }
    )
    bundle = DatasetBundle(real=data, target_col="target", name="categorical", domain="test")

    plot_path = plot_distributions(bundle, output_dir=str(tmp_path))

    assert plot_path.exists()


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


def test_save_sdv_metadata_marks_boolean_columns(bundle, tmp_path):
    boolean_bundle = DatasetBundle(
        real=bundle.real.assign(flag=bundle.real["y"].astype(bool)),
        target_col="y",
        name="boolean-data",
        domain="test",
    )

    metadata_path = save_sdv_metadata(boolean_bundle, output_dir=str(tmp_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["tables"]["table"]["columns"]["flag"]["sdtype"] == "boolean"


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

import numpy as np
import pandas as pd
import pytest

from src.data.analyze import describe_dataset, plot_distributions
from src.data.loader import DatasetBundle, split_dataset


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

import json
import pytest

from src.experiments.config import ExperimentMode, TrainingConfig
from src.experiments.persistence import load_generator, load_report, save_generator, save_metadata
from src.experiments.report import build_report, save_report, summarize_report


# --- TrainingConfig ---

def test_run_name_format():
    cfg = TrainingConfig(dataset_name="adult", generator_name="ctgan", num_samples=100)
    assert cfg.run_name == "adult_ctgan_seed42"


def test_run_name_uses_seed():
    cfg = TrainingConfig(dataset_name="heart", generator_name="tvae", num_samples=50, seed=7)
    assert cfg.run_name == "heart_tvae_seed7"


# --- save_report ---

def test_save_report_creates_file(tmp_path):
    report = {"config": {}, "utility": {}, "privacy": {}}
    out = save_report(report, run_name="test_run", output_dir=str(tmp_path))
    assert out == tmp_path / "test_run.json"
    assert out.exists()


def test_save_report_valid_json(tmp_path):
    report = {"config": {"dataset_name": "adult"}, "score": 0.5}
    out = save_report(report, run_name="run_json", output_dir=str(tmp_path))
    with open(out) as f:
        loaded = json.load(f)
    assert loaded == report


def test_save_report_serializes_enum(tmp_path):
    report = {"mode": ExperimentMode.EVALUATE_ONLY}
    out = save_report(report, run_name="run_enum", output_dir=str(tmp_path))
    with open(out) as f:
        loaded = json.load(f)
    assert loaded["mode"] == "EVALUATE_ONLY"


# --- summarize_report ---

def test_summarize_report_includes_aggregate_metrics():
    report = {
        "config": {"dataset_name": "adult", "generator_name": "ctgan"},
        "utility": {"mmd": 0.1, "mean_emd": 0.2, "emd_per_feature": {"age": 0.3, "income": 0.4}},
        "privacy": {"dcr_mean": 1.5},
    }
    summary = summarize_report(report)
    assert "utility.mmd: 0.1" in summary
    assert "utility.mean_emd: 0.2" in summary
    assert "privacy.dcr_mean: 1.5" in summary


def test_summarize_report_skips_per_feature_breakdown():
    report = {
        "config": {"dataset_name": "adult", "generator_name": "ctgan"},
        "utility": {"emd_per_feature": {"age": 0.3}},
        "privacy": {},
    }
    summary = summarize_report(report)
    assert "emd_per_feature" not in summary


def test_summarize_report_includes_training_history_count():
    report = {
        "config": {"dataset_name": "adult", "generator_name": "ctgan"},
        "utility": {},
        "privacy": {},
        "artifacts": {"training_history": [{"epoch": 0}, {"epoch": 1}]},
    }
    summary = summarize_report(report)
    assert "artifacts.training_history: 2 entries" in summary


def test_summarize_report_loads_from_disk(tmp_path):
    report = {
        "config": {"dataset_name": "heart", "generator_name": "tvae"},
        "utility": {"mmd": 0.05},
        "privacy": {},
    }
    save_report(report, run_name="disk_run", output_dir=str(tmp_path))
    summary = summarize_report("disk_run", output_dir=str(tmp_path))
    assert "heart / tvae" in summary
    assert "utility.mmd: 0.05" in summary


def test_save_report_creates_missing_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    save_report({}, run_name="r", output_dir=str(nested))
    assert (nested / "r.json").exists()


# --- build_report: artifacts ---

@pytest.fixture
def tiny_frames():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({"x": rng.random(n), "y": rng.integers(0, 2, n)})
    return df, df.copy()


def test_build_report_includes_artifacts(tiny_frames):
    real, synthetic = tiny_frames
    cfg = TrainingConfig(dataset_name="heart", generator_name="ctgan", num_samples=30)

    class _FakeGenerator:
        def get_training_diagnostics(self):
            return {"training_history": [{"epoch": 0, "loss": 1.2}]}

    report = build_report(cfg, real, synthetic, target_col="y", generator=_FakeGenerator())
    assert report["artifacts"] == {"training_history": [{"epoch": 0, "loss": 1.2}]}


def test_build_report_omits_artifacts_when_empty(tiny_frames):
    real, synthetic = tiny_frames
    cfg = TrainingConfig(dataset_name="heart", generator_name="ctgan", num_samples=30)
    report = build_report(cfg, real, synthetic, target_col="y", generator=None)
    assert "artifacts" not in report


# --- save_generator / load_generator ---

def test_save_generator_creates_pkl(tmp_path):
    obj = {"weights": [1, 2, 3]}
    out = save_generator(obj, run_name="run_gen", output_dir=str(tmp_path))
    assert out == tmp_path / "run_gen.pkl"
    assert out.exists()


def test_load_generator_roundtrip(tmp_path):
    obj = {"weights": [1, 2, 3]}
    save_generator(obj, run_name="run_rt", output_dir=str(tmp_path))
    loaded = load_generator(run_name="run_rt", output_dir=str(tmp_path))
    assert loaded == obj


def test_load_generator_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_generator(run_name="nonexistent", output_dir=str(tmp_path))


def test_load_report_roundtrip(tmp_path):
    report = {"config": {"dataset_name": "adult"}, "utility": {"mmd": 0.1}}
    save_report(report, run_name="run_load", output_dir=str(tmp_path))
    loaded = load_report("run_load", output_dir=str(tmp_path))
    assert loaded == report


# --- save_metadata ---

def test_save_metadata_writes_file(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"age": [23, 45, 31], "income": [2100.0, 5400.0, 3200.0]})
    out = save_metadata(df, run_name="run_meta", output_dir=str(tmp_path))
    assert out == tmp_path / "run_meta.metadata.json"
    assert out.exists()


def test_save_metadata_writes_valid_json(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"age": [23, 45, 31], "job": ["a", "b", "a"]})
    out = save_metadata(df, run_name="run_meta2", output_dir=str(tmp_path))
    with open(out) as f:
        loaded = json.load(f)
    assert "tables" in loaded or "columns" in loaded

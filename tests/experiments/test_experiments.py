import json
import pytest

from src.experiments.config import ExperimentMode, TrainingConfig
from src.experiments.persistence import load_generator, load_report, save_generator, save_metadata
from src.experiments.report import append_to_index, build_report, save_report, summarize_report


# --- TrainingConfig ---

def test_run_name_format():
    cfg = TrainingConfig(dataset_name="adult", generator_name="ctgan", num_samples=100)
    assert cfg.run_name.startswith("adult_ctgan_seed42_")


def test_run_name_uses_seed():
    cfg = TrainingConfig(dataset_name="heart", generator_name="tvae", num_samples=50, seed=7)
    assert cfg.run_name.startswith("heart_tvae_seed7_")


def test_run_name_differs_with_different_kwargs():
    # deterministic via _kwargs_hash: doesn't depend on timing, never flaky
    cfg1 = TrainingConfig(dataset_name="adult", generator_name="ctgan", num_samples=100,
                           generator_kwargs={"epochs": 50})
    cfg2 = TrainingConfig(dataset_name="adult", generator_name="ctgan", num_samples=100,
                           generator_kwargs={"epochs": 300})
    assert cfg1.run_name != cfg2.run_name


def test_run_name_differs_with_different_test_size():
    cfg1 = TrainingConfig(dataset_name="adult", generator_name="ctgan", num_samples=100, test_size=0.2)
    cfg2 = TrainingConfig(dataset_name="adult", generator_name="ctgan", num_samples=100, test_size=0.3)
    assert cfg1.run_name != cfg2.run_name


def test_run_name_unique_even_with_identical_config():
    # random suffix in created_at: two instances with the exact same
    # hyperparameters still never collide (e.g. rerunning after a code change)
    cfg1 = TrainingConfig(dataset_name="adult", generator_name="ctgan", num_samples=100)
    cfg2 = TrainingConfig(dataset_name="adult", generator_name="ctgan", num_samples=100)
    assert cfg1.run_name != cfg2.run_name


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


# --- append_to_index ---

def _fake_report(**config_overrides) -> dict:
    config = {
        "dataset_name": "adult",
        "generator_name": "ctgan",
        "seed": 42,
        "test_size": 0.2,
        "num_samples": 1000,
        "generator_kwargs": {"epochs": 50},
        "created_at": "20260901-120000-abc123",
    }
    config.update(config_overrides)
    return {"config": config, "code_version": "abcdef1"}


def test_append_to_index_creates_jsonl(tmp_path):
    index_path = append_to_index(_fake_report(), run_name="run1", output_dir=str(tmp_path))
    assert index_path == tmp_path / "index.jsonl"
    assert index_path.exists()


def test_append_to_index_row_is_valid_json(tmp_path):
    index_path = append_to_index(_fake_report(), run_name="run1", output_dir=str(tmp_path))
    with open(index_path) as f:
        row = json.loads(f.readline())
    assert row["run_name"] == "run1"
    assert row["dataset_name"] == "adult"
    assert row["generator_kwargs"] == {"epochs": 50}
    assert row["code_version"] == "abcdef1"


def test_append_to_index_appends_multiple_runs(tmp_path):
    append_to_index(_fake_report(), run_name="run1", output_dir=str(tmp_path))
    append_to_index(_fake_report(generator_kwargs={"epochs": 300}), run_name="run2", output_dir=str(tmp_path))
    index_path = tmp_path / "index.jsonl"
    with open(index_path) as f:
        lines = f.readlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["run_name"] == "run1"
    assert rows[1]["run_name"] == "run2"
    assert rows[1]["generator_kwargs"] == {"epochs": 300}


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

    report = build_report(cfg, real, synthetic, synthetic, target_col="y", generator=_FakeGenerator())
    assert report["artifacts"] == {"training_history": [{"epoch": 0, "loss": 1.2}]}


def test_build_report_omits_artifacts_when_empty(tiny_frames):
    real, synthetic = tiny_frames
    cfg = TrainingConfig(dataset_name="heart", generator_name="ctgan", num_samples=30)
    report = build_report(cfg, real, synthetic, synthetic, target_col="y", generator=None)
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


def test_save_metadata_marks_boolean_columns(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"age": [23, 45, 31], "income": [True, False, True]})
    out = save_metadata(df, run_name="run_bool_meta", output_dir=str(tmp_path))
    with open(out) as f:
        loaded = json.load(f)

    columns = loaded["tables"]["table"]["columns"]
    assert columns["income"]["sdtype"] == "boolean"

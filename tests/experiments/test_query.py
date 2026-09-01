import json

import pytest

from src.experiments.query import load_index, query_index


def _write_index(tmp_path, rows: list[dict]):
    index_path = tmp_path / "index.jsonl"
    with open(index_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return index_path


@pytest.fixture
def sample_rows():
    return [
        {
            "run_name": "adult_ctgan_seed42_a",
            "dataset_name": "adult",
            "generator_name": "ctgan",
            "seed": 42,
            "test_size": 0.2,
            "code_version": "abc1111",
            "generator_kwargs": {"epochs": 50},
        },
        {
            "run_name": "adult_ctgan_seed42_b",
            "dataset_name": "adult",
            "generator_name": "ctgan",
            "seed": 42,
            "test_size": 0.2,
            "code_version": "def2222",
            "generator_kwargs": {"epochs": 300},
        },
        {
            "run_name": "heart_tvae_seed7_c",
            "dataset_name": "heart",
            "generator_name": "tvae",
            "seed": 7,
            "test_size": 0.3,
            "code_version": "abc1111",
            "generator_kwargs": {"epochs": 100, "batch_size": 500},
        },
    ]


# --- load_index ---

def test_load_index_empty_when_missing(tmp_path):
    assert load_index(str(tmp_path)) == []


def test_load_index_reads_all_rows(tmp_path, sample_rows):
    _write_index(tmp_path, sample_rows)
    assert load_index(str(tmp_path)) == sample_rows


# --- query_index ---

def test_query_index_no_filters_returns_all(tmp_path, sample_rows):
    _write_index(tmp_path, sample_rows)
    assert len(query_index(output_dir=str(tmp_path))) == 3


def test_query_index_by_dataset(tmp_path, sample_rows):
    _write_index(tmp_path, sample_rows)
    matches = query_index(output_dir=str(tmp_path), dataset_name="heart")
    assert len(matches) == 1
    assert matches[0]["run_name"] == "heart_tvae_seed7_c"


def test_query_index_by_dataset_and_generator(tmp_path, sample_rows):
    _write_index(tmp_path, sample_rows)
    matches = query_index(output_dir=str(tmp_path), dataset_name="adult", generator_name="ctgan")
    assert len(matches) == 2


def test_query_index_by_kwargs(tmp_path, sample_rows):
    _write_index(tmp_path, sample_rows)
    matches = query_index(output_dir=str(tmp_path), kwargs={"epochs": 300})
    assert len(matches) == 1
    assert matches[0]["run_name"] == "adult_ctgan_seed42_b"


def test_query_index_by_kwargs_partial_match(tmp_path, sample_rows):
    _write_index(tmp_path, sample_rows)
    # epochs=100 alone should match the row that also has batch_size=500
    matches = query_index(output_dir=str(tmp_path), kwargs={"epochs": 100})
    assert len(matches) == 1
    assert matches[0]["run_name"] == "heart_tvae_seed7_c"


def test_query_index_by_code_version(tmp_path, sample_rows):
    _write_index(tmp_path, sample_rows)
    matches = query_index(output_dir=str(tmp_path), code_version="abc1111")
    assert len(matches) == 2


def test_query_index_no_matches(tmp_path, sample_rows):
    _write_index(tmp_path, sample_rows)
    assert query_index(output_dir=str(tmp_path), dataset_name="diabetes") == []


def test_query_index_combined_filters(tmp_path, sample_rows):
    _write_index(tmp_path, sample_rows)
    matches = query_index(
        output_dir=str(tmp_path),
        dataset_name="adult",
        generator_name="ctgan",
        kwargs={"epochs": 50},
    )
    assert len(matches) == 1
    assert matches[0]["run_name"] == "adult_ctgan_seed42_a"

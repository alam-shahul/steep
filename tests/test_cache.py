from pathlib import Path

from omegaconf import OmegaConf

from steep.cache import CACHE_SCHEMA_VERSION, TRAINING_CACHE_KEYS, cache_hash_vars, dataset_fingerprint
from steep.utils._general import get_fully_qualified_cache_paths


def test_dataset_fingerprint_uses_sorted_h5ad_names_and_sizes(tmp_path: Path):
    (tmp_path / "b.h5ad").write_bytes(b"12")
    (tmp_path / "a.h5ad").write_bytes(b"123")
    first = dataset_fingerprint(tmp_path)

    (tmp_path / "ignored.txt").write_bytes(b"different")
    assert dataset_fingerprint(tmp_path) == first

    (tmp_path / "a.h5ad").write_bytes(b"1234")
    assert dataset_fingerprint(tmp_path) != first


def test_cache_hash_changes_with_schema_or_data_fingerprint(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "slide.h5ad").write_bytes(b"first")
    cfg = OmegaConf.create({"seed": 3})

    first = get_fully_qualified_cache_paths(
        cfg,
        tmp_path / "cache",
        keys=("seed",),
        hash_vars=cache_hash_vars(data_dir),
        mkdir=False,
    )
    second = get_fully_qualified_cache_paths(
        cfg,
        tmp_path / "cache",
        keys=("seed",),
        hash_vars={
            **cache_hash_vars(data_dir),
            "schema_version": CACHE_SCHEMA_VERSION + 1,
        },
        mkdir=False,
    )

    assert first != second


def test_training_cache_ignores_run_name(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "seed": 3,
            "run_name": "first",
            "dataset": {"args": {"data_directory": str(data_dir)}},
        },
    )
    first = get_fully_qualified_cache_paths(
        cfg,
        tmp_path / "cache",
        keys=TRAINING_CACHE_KEYS,
        hash_vars=cache_hash_vars(data_dir),
        mkdir=False,
    )
    cfg.run_name = "second"
    second = get_fully_qualified_cache_paths(
        cfg,
        tmp_path / "cache",
        keys=TRAINING_CACHE_KEYS,
        hash_vars=cache_hash_vars(data_dir),
        mkdir=False,
    )

    assert first == second

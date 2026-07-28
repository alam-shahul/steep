from pathlib import Path

import anndata as ad
import numpy as np

from steep.sketcher import HopperSketcher, RandomSubsampleSketcher


def _make_adata(num_cells: int = 10, num_genes: int = 4) -> ad.AnnData:
    adata = ad.AnnData(X=np.arange(num_cells * num_genes, dtype=np.float32).reshape(num_cells, num_genes))
    adata.obsm["spatial"] = np.stack(
        [np.linspace(0.0, 1.0, num_cells), np.linspace(1.0, 2.0, num_cells)],
        axis=1,
    )
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)
    for idx in range(num_cells - 1):
        adjacency[idx, idx + 1] = 1.0
        adjacency[idx + 1, idx] = 1.0
    adata.obsp["adjacency_matrix"] = adjacency
    return adata


def test_random_subsample_sketcher_fit_transform_reduces_cells():
    adata = _make_adata(num_cells=10)
    sketcher = RandomSubsampleSketcher(retention_ratio=0.5, random_seed=7)

    sketched = sketcher.fit_transform(adata)

    assert sketched.n_obs == 5
    assert sketched.n_vars == adata.n_vars


def test_random_subsample_sketcher_fit_transform_to_disk_returns_metadata(tmp_path: Path):
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.h5ad"
    _make_adata(num_cells=10).write_h5ad(input_path)

    sketcher = RandomSubsampleSketcher(retention_ratio=0.4, random_seed=3)
    metadata = sketcher.fit_transform_to_disk(input_path, output_path)

    written = ad.read_h5ad(output_path)
    assert written.n_obs == 4
    assert metadata.original_num_cells == 10
    assert metadata.sketched_num_cells == 4
    assert metadata.compression_ratio == 0.4
    assert metadata.original_disk_bytes is not None
    assert metadata.sketched_disk_bytes is not None
    assert metadata.sketch_time_seconds is not None


def test_hopper_sketcher_fit_transform_reduces_cells_deterministically():
    adata = _make_adata(num_cells=10)
    sketcher = HopperSketcher(retention_ratio=0.5, random_seed=7)

    first = sketcher.fit_transform(adata)
    second = sketcher.fit_transform(adata)

    assert first.n_obs == 5
    assert second.n_obs == 5
    assert first.obs_names.tolist() == second.obs_names.tolist()


def test_hopper_sketcher_fit_transform_to_disk_returns_metadata(tmp_path: Path):
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.h5ad"
    _make_adata(num_cells=10).write_h5ad(input_path)

    sketcher = HopperSketcher(retention_ratio=0.4, random_seed=3)
    metadata = sketcher.fit_transform_to_disk(input_path, output_path)

    written = ad.read_h5ad(output_path)
    assert written.n_obs == 4
    assert metadata.original_num_cells == 10
    assert metadata.sketched_num_cells == 4
    assert metadata.compression_ratio == 0.4

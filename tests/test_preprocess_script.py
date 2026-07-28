import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.sparse import csr_array


def _load_preprocess_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "preprocess.py"
    spec = importlib.util.spec_from_file_location("scripts.preprocess", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_main_processes_all_h5ad_files_in_directory(tmp_path, monkeypatch):
    module = _load_preprocess_module()

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    for filename in ("a.h5ad", "b.h5ad"):
        adata = ad.AnnData(X=csr_array(np.array([[1.0, 2.0]], dtype=np.float32)))
        adata.obsm["spatial"] = np.array([[0.0, 0.0]], dtype=np.float32)
        adata.write_h5ad(input_dir / filename)

    processed_files = []

    def fake_preprocess(adata):
        processed_files.append(adata.shape)
        return adata

    monkeypatch.setattr(module, "preprocess", fake_preprocess)
    monkeypatch.setattr(
        "sys.argv",
        ["preprocess.py", str(input_dir), str(output_dir), "--num-workers", "1"],
    )

    module.main()

    assert processed_files == [(1, 2), (1, 2)]
    assert sorted(path.name for path in output_dir.glob("*.h5ad")) == ["a.h5ad", "b.h5ad"]


def test_main_can_randomly_sample_subset_of_h5ad_files(tmp_path, monkeypatch):
    module = _load_preprocess_module()

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    for filename in ("a.h5ad", "b.h5ad", "c.h5ad"):
        adata = ad.AnnData(X=csr_array(np.array([[1.0, 2.0]], dtype=np.float32)))
        adata.obsm["spatial"] = np.array([[0.0, 0.0]], dtype=np.float32)
        adata.write_h5ad(input_dir / filename)

    def fake_preprocess(adata):
        return adata

    monkeypatch.setattr(module, "preprocess", fake_preprocess)

    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocess.py",
            str(input_dir),
            str(output_dir),
            "--num-slides",
            "2",
            "--seed",
            "7",
            "--num-workers",
            "1",
        ],
    )

    module.main()

    assert sorted(path.name for path in output_dir.glob("*.h5ad")) == ["a.h5ad", "b.h5ad"]

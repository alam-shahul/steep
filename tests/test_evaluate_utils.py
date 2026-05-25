import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anndata as ad
import numpy as np
import pytest
import torch

from steep.utils._cluster import (
    LabelMetricStore,
    accumulate_label_results,
    cluster_embeddings,
    summarize_cluster_scores,
)
from steep.utils._embedding import extract_slide_embedding_records


def test_cluster_embeddings_scanpy_backend_returns_one_label_per_embedding():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )

    labels = cluster_embeddings(
        embeddings,
        random_seed=0,
        n_neighbors=2,
        leiden_resolution=1.0,
        clustering_backend="scanpy",
    )

    assert labels.shape == (embeddings.shape[0],)


def test_cluster_embeddings_rapids_backend_requires_rapids_singlecell():
    embeddings = np.ones((5, 2), dtype=np.float32)

    with patch.dict(sys.modules, {"rapids_singlecell": None}):
        with pytest.raises(ImportError, match="uv sync --group rapids-cu12"):
            cluster_embeddings(
                embeddings,
                random_seed=0,
                n_neighbors=2,
                leiden_resolution=1.0,
                clustering_backend="rapids",
            )


def test_cluster_embeddings_rapids_backend_uses_rapids_singlecell_api():
    embeddings = np.ones((5, 2), dtype=np.float32)
    calls = []

    def anndata_to_gpu(adata, convert_all):
        calls.append(("to_gpu", convert_all, "X_stagate" in adata.obsm))

    def neighbors(adata, use_rep, n_neighbors, random_state):
        calls.append(("neighbors", use_rep, n_neighbors, random_state))

    def leiden(adata, key_added, resolution, random_state):
        calls.append(("leiden", key_added, resolution, random_state))
        adata.obs[key_added] = ["0", "0", "1", "1", "1"]

    fake_rapids = SimpleNamespace(
        get=SimpleNamespace(anndata_to_GPU=anndata_to_gpu),
        pp=SimpleNamespace(neighbors=neighbors),
        tl=SimpleNamespace(leiden=leiden),
    )

    with patch.dict(sys.modules, {"rapids_singlecell": fake_rapids}):
        labels = cluster_embeddings(
            embeddings,
            random_seed=3,
            n_neighbors=2,
            leiden_resolution=0.5,
            clustering_backend="rapids",
        )

    assert labels.tolist() == ["0", "0", "1", "1", "1"]
    assert calls == [
        ("to_gpu", True, True),
        ("neighbors", "X_stagate", 2, 3),
        ("leiden", "predicted_leiden", 0.5, 3),
    ]


def test_cluster_metric_store_accumulates_and_summarizes():
    metric_store = {"cell_type": LabelMetricStore()}
    accumulate_label_results(
        metric_store,
        {"cell_type": {"ari": 0.5, "nmi": 0.6, "count": 10}},
    )
    accumulate_label_results(
        metric_store,
        {"cell_type": {"ari": 0.7, "nmi": 0.8, "count": 20}},
    )

    summary = summarize_cluster_scores(["cell_type"], metric_store)
    metrics = summary["metrics_by_label_key"]["cell_type"]

    assert metrics["ari_mean"] == 0.6
    assert metrics["nmi_mean"] == 0.7
    assert metrics["ari_weighted_mean"] == (0.5 * 10 + 0.7 * 20) / 30
    assert metrics["nmi_weighted_mean"] == (0.6 * 10 + 0.8 * 20) / 30
    assert metrics["evaluated_slides"] == 2


def test_extract_slide_embedding_records_returns_expected_fields(tmp_path: Path):
    adata = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
    adata.obs["cell_type"] = ["a", "b", "a"]
    adata.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32)
    data_path = tmp_path / "slide.h5ad"
    adata.write_h5ad(data_path)

    class DummyGraph:
        def to(self, device):
            del device
            return self

    class DummyDataset:
        def __init__(self, path):
            self.data_paths = [path]

        def __getitem__(self, idx):
            assert idx == 0
            return DummyGraph()

    class DummyModel:
        def to(self, device):
            del device
            return self

        def eval(self):
            return self

        def __call__(self, graph):
            del graph
            return {"embedding": torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)}

    trainer = SimpleNamespace(
        data=DummyDataset(data_path),
        model=DummyModel(),
        lightning_module=None,
        device="cpu",
    )

    records = extract_slide_embedding_records(trainer, label_keys=["cell_type"])

    assert len(records) == 1
    record = records[0]
    assert record.slide_name == "slide.h5ad"
    assert record.embeddings.shape == (3, 1)
    assert record.labels_by_key["cell_type"].tolist() == ["a", "b", "a"]

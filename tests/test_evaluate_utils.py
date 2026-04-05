from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import torch

from steep.utils._cluster import LabelMetricStore, accumulate_label_results, summarize_cluster_scores
from steep.utils._embedding import extract_slide_embedding_records


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

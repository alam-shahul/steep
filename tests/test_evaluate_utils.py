import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anndata as ad
import numpy as np
import pytest
import torch

import steep.utils._classification as classification_utils
from steep.utils._classification import evaluate_split_classification, summarize_split_classification_scores
from steep.utils._cluster import (
    LabelMetricStore,
    accumulate_label_results,
    cluster_embeddings,
    summarize_cluster_scores,
)
from steep.utils._embedding import extract_slide_embedding_records
from steep.utils._plot import save_slide_clustering_plot


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


def test_save_slide_clustering_plot_sets_squidpy_library_id(tmp_path: Path):
    captured = {}

    def spatial_scatter(adata, library_id, **kwargs):
        captured["library_id"] = library_id
        captured["uns_spatial"] = adata.uns["spatial"]
        captured["kwargs"] = kwargs

    with patch("steep.utils._plot.sq.pl.spatial_scatter", side_effect=spatial_scatter):
        with patch("steep.utils._plot.plt.savefig"):
            save_slide_clustering_plot(
                slide_name="batch_001.h5ad",
                obs_names=np.array(["a", "b"]),
                spatial=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                predicted_labels=np.array(["0", "1"]),
                output_path=tmp_path / "plot.png",
            )

    assert captured["library_id"] == "batch_001"
    assert captured["uns_spatial"] == {"batch_001": {}}
    assert captured["kwargs"]["color"] == "predicted_leiden"


def test_extract_slide_embedding_records_returns_expected_fields(tmp_path: Path):
    adata = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
    adata.obs["cell_type"] = ["a", "b", "a"]
    adata.obs["subclass"] = ["L2/3 IT Glut", "Lamp5 Gaba", "vascular endothelial"]
    adata.obs["class"] = ["excitatory", "inhibitory", "endothelial"]
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

    records = extract_slide_embedding_records(trainer, label_keys=["cell_type", "cell_type_coarse"])

    assert len(records) == 1
    record = records[0]
    assert record.slide_name == "slide.h5ad"
    assert record.embeddings.shape == (3, 1)
    assert record.labels_by_key["cell_type"].tolist() == ["a", "b", "a"]
    assert record.labels_by_key["cell_type_coarse"].tolist() == [
        "Excitatory neuron",
        "Inhibitory neuron",
        "Endothelial",
    ]


def test_evaluate_split_classification_scores_test_slides_and_saves_confusion_matrix(tmp_path: Path):
    train_records = [
        SimpleNamespace(
            embeddings=np.array([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32),
            labels_by_key={"cell_type": np.array(["a", "a", "b", "b"], dtype=object)},
        ),
    ]
    test_records = [
        SimpleNamespace(
            embeddings=np.array([[0.2], [10.2], [20.0]], dtype=np.float32),
            labels_by_key={"cell_type": np.array(["a", "b", "c"], dtype=object)},
        ),
    ]

    results = evaluate_split_classification(
        train_records=train_records,
        test_records=test_records,
        label_keys=["cell_type"],
        n_neighbors=1,
        confusion_matrix_dir=tmp_path,
        stage="baseline",
    )

    result = results["cell_type"]
    assert result["accuracy"] == 1.0
    assert result["count"] == 2
    assert result["classifier"] == "KNeighborsClassifier"
    assert result["classification_backend"] == "sklearn"
    assert result["n_test_dropped_unknown_class"] == 1
    assert result["split"] == "train_slide_test_slide"
    assert Path(result["confusion_matrix_path"]).exists()


def test_evaluate_split_classification_rapids_backend_encodes_labels(tmp_path: Path):
    captured = {}

    class FakeCuMLKNeighborsClassifier:
        def __init__(self, n_neighbors, output_type):
            captured["init"] = (n_neighbors, output_type)

        def fit(self, X, y):
            captured["fit_X_dtype"] = X.dtype
            captured["fit_y"] = y.copy()
            return self

        def predict(self, X):
            captured["predict_X_dtype"] = X.dtype
            return np.array([0, 1], dtype=np.int32)

    train_records = [
        SimpleNamespace(
            embeddings=np.array([[0.0], [0.1], [10.0], [10.1]], dtype=np.float64),
            labels_by_key={"cell_type": np.array(["a", "a", "b", "b"], dtype=object)},
        ),
    ]
    test_records = [
        SimpleNamespace(
            embeddings=np.array([[0.2], [10.2]], dtype=np.float64),
            labels_by_key={"cell_type": np.array(["a", "b"], dtype=object)},
        ),
    ]
    fake_cuml = SimpleNamespace(neighbors=SimpleNamespace(KNeighborsClassifier=FakeCuMLKNeighborsClassifier))

    with patch.dict(sys.modules, {"cuml": fake_cuml, "cuml.neighbors": fake_cuml.neighbors}):
        results = evaluate_split_classification(
            train_records=train_records,
            test_records=test_records,
            label_keys=["cell_type"],
            n_neighbors=1,
            confusion_matrix_dir=tmp_path,
            stage="baseline",
            backend="rapids",
        )

    result = results["cell_type"]
    assert captured["init"] == (1, "numpy")
    assert captured["fit_X_dtype"] == np.float32
    assert captured["predict_X_dtype"] == np.float32
    assert np.issubdtype(captured["fit_y"].dtype, np.integer)
    assert result["accuracy"] == 1.0
    assert result["classifier"] == "CuMLKNeighborsClassifier"
    assert result["classification_backend"] == "rapids"
    assert Path(result["confusion_matrix_path"]).exists()


def test_evaluate_split_classification_skips_large_confusion_matrix(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(classification_utils, "MAX_CONFUSION_MATRIX_CLASSES", 2)
    train_records = [
        SimpleNamespace(
            embeddings=np.array([[float(idx)] for idx in range(6)], dtype=np.float32),
            labels_by_key={"cell_type": np.array(["a", "a", "b", "b", "c", "c"], dtype=object)},
        ),
    ]
    test_records = [
        SimpleNamespace(
            embeddings=np.array([[0.1], [2.1], [4.1]], dtype=np.float32),
            labels_by_key={"cell_type": np.array(["a", "b", "c"], dtype=object)},
        ),
    ]

    results = evaluate_split_classification(
        train_records=train_records,
        test_records=test_records,
        label_keys=["cell_type"],
        n_neighbors=1,
        confusion_matrix_dir=tmp_path,
        stage="baseline",
    )

    result = results["cell_type"]
    assert result["confusion_matrix_path"] is None
    assert not list(tmp_path.glob("*.png"))


def test_summarize_split_classification_scores_preserves_metadata():
    summary = summarize_split_classification_scores(
        ["cell_type"],
        {
            "cell_type": {
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "weighted_f1": 0.45,
                "count": 8,
                "n_test_slides": 2,
                "split": "train_slide_test_slide",
            },
        },
    )

    metrics = summary["classification_metrics_by_label_key"]["cell_type"]
    assert metrics["accuracy_mean"] == 0.5
    assert metrics["weighted_f1_weighted_mean"] == 0.45
    assert metrics["evaluated_slides"] == 2
    assert metrics["split"] == "train_slide_test_slide"

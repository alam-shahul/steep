import warnings
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

CLUSTER_METRIC_NAMES = ("ari", "nmi")


@dataclass
class MetricAccumulator:
    scores: list[float] = field(default_factory=list)
    weighted_scores: list[tuple[float, int]] = field(default_factory=list)

    def add(self, score: float, count: int) -> None:
        self.scores.append(float(score))
        self.weighted_scores.append((float(score), int(count)))

    def mean(self) -> float | None:
        if not self.scores:
            return None
        return float(np.mean(self.scores))

    def weighted_mean(self) -> float | None:
        if not self.weighted_scores:
            return None
        total_weight = sum(weight for _, weight in self.weighted_scores)
        return sum(score * weight for score, weight in self.weighted_scores) / total_weight


@dataclass
class LabelMetricStore:
    ari: MetricAccumulator = field(default_factory=MetricAccumulator)
    nmi: MetricAccumulator = field(default_factory=MetricAccumulator)

    def add_result(self, result: dict[str, float | int]) -> None:
        count = int(result["count"])
        self.ari.add(float(result["ari"]), count)
        self.nmi.add(float(result["nmi"]), count)

    def to_summary(self) -> dict[str, float | int | None]:
        if not self.ari.scores:
            return {
                "ari_mean": None,
                "ari_weighted_mean": None,
                "nmi_mean": None,
                "nmi_weighted_mean": None,
                "evaluated_slides": 0,
            }

        return {
            "ari_mean": self.ari.mean(),
            "ari_weighted_mean": self.ari.weighted_mean(),
            "nmi_mean": self.nmi.mean(),
            "nmi_weighted_mean": self.nmi.weighted_mean(),
            "evaluated_slides": len(self.ari.scores),
        }


def cluster_embeddings(
    embeddings: np.ndarray,
    random_seed: int = 0,
    n_neighbors: int = 15,
    leiden_resolution: float = 1.0,
    clustering_backend: str = "scanpy",
) -> np.ndarray:

    clustering_adata = ad.AnnData(X=np.asarray(embeddings, dtype=np.float32))
    clustering_adata.obsm["X_stagate"] = np.asarray(embeddings, dtype=np.float32)

    num_cells = clustering_adata.n_obs
    effective_neighbors = max(2, min(int(n_neighbors), num_cells - 1))
    if clustering_backend == "scanpy":
        sc.pp.neighbors(clustering_adata, use_rep="X_stagate", n_neighbors=effective_neighbors)
        sc.tl.leiden(
            clustering_adata,
            key_added="predicted_leiden",
            resolution=float(leiden_resolution),
            random_state=int(random_seed),
        )
    elif clustering_backend == "rapids":
        try:
            import rapids_singlecell as rsc
        except ImportError as exc:
            raise ImportError(
                "RAPIDS clustering backend requires rapids-singlecell. "
                "Install it with `uv sync --group rapids-cu12`.",
            ) from exc

        rsc.get.anndata_to_GPU(clustering_adata, convert_all=True)
        rsc.pp.neighbors(
            clustering_adata,
            use_rep="X_stagate",
            n_neighbors=effective_neighbors,
            random_state=int(random_seed),
        )
        rsc.tl.leiden(
            clustering_adata,
            key_added="predicted_leiden",
            resolution=float(leiden_resolution),
            random_state=int(random_seed),
        )
    else:
        raise ValueError(f"Unsupported clustering_backend: {clustering_backend!r}.")

    return clustering_adata.obs["predicted_leiden"].to_numpy()


def evaluate_slide_embeddings(
    labels_by_key: dict[str, np.ndarray],
    predicted: np.ndarray,
) -> dict[str, dict[str, float | int] | None]:
    warnings.simplefilter("ignore")

    results = {}
    for label_key, labels in labels_by_key.items():
        valid_mask = ~pd.isna(labels)
        if valid_mask.sum() < 2:
            results[label_key] = None
            continue

        filtered_labels = labels[valid_mask]
        filtered_predicted = predicted[valid_mask]
        if len(np.unique(filtered_labels)) < 2 or filtered_predicted.shape[0] < 3:
            results[label_key] = None
            continue

        results[label_key] = {
            "ari": float(adjusted_rand_score(filtered_labels, filtered_predicted)),
            "nmi": float(normalized_mutual_info_score(filtered_labels, filtered_predicted)),
            "count": int(len(filtered_labels)),
        }

    return results


def evaluate_slide_cluster_agreement(
    reference_labels: np.ndarray,
    target_embeddings: np.ndarray,
    random_seed: int,
    n_neighbors: int,
    leiden_resolution: float,
    clustering_backend: str = "scanpy",
) -> dict[str, float | int] | None:
    warnings.simplefilter("ignore")

    if len(reference_labels) < 3 or target_embeddings.shape[0] < 3:
        return None
    if len(np.unique(reference_labels)) < 2:
        return None

    predicted = cluster_embeddings(
        target_embeddings,
        random_seed=random_seed,
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        clustering_backend=clustering_backend,
    )
    return {
        "ari": float(adjusted_rand_score(reference_labels, predicted)),
        "nmi": float(normalized_mutual_info_score(reference_labels, predicted)),
        "count": int(len(reference_labels)),
    }


def accumulate_label_results(
    metric_store: dict[str, LabelMetricStore],
    slide_results: dict[str, dict[str, float | int] | None],
) -> None:
    for label_key, result in slide_results.items():
        if result is None:
            continue

        metric_store[label_key].add_result(result)


def summarize_cluster_scores(
    label_keys: list[str] | tuple[str, ...],
    metric_store: dict[str, LabelMetricStore],
) -> dict[str, object]:
    metrics_by_label_key = {}
    for label_key in label_keys:
        metrics_by_label_key[label_key] = metric_store[label_key].to_summary()

    return {
        "label_keys": list(label_keys),
        "metrics_by_label_key": metrics_by_label_key,
    }

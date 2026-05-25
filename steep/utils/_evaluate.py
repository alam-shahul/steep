from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import pandas as pd
import torch
from tqdm import tqdm

from steep.utils._cluster import (
    LabelMetricStore,
    accumulate_label_results,
    cluster_embeddings,
    evaluate_slide_cluster_agreement,
    evaluate_slide_embeddings,
    summarize_cluster_scores,
)
from steep.utils._embedding import extract_slide_embedding_records
from steep.utils._general import to_builtin
from steep.utils._plot import save_slide_clustering_plot


@dataclass
class DatasetSummary:
    """Basic size summary for a benchmark dataset directory."""

    name: str
    data_directory: str
    num_slides: int = 0
    num_cells: int = 0
    disk_bytes: int = 0

    def __post_init__(self) -> None:
        """Populate slide, cell, and disk totals from the data directory."""
        data_directory = Path(self.data_directory)
        input_paths = sorted(data_directory.glob("*.h5ad"))

        total_cells = 0
        total_disk_bytes = 0
        for input_path in input_paths:
            adata = ad.read_h5ad(input_path, backed="r")
            total_cells += int(adata.n_obs)
            total_disk_bytes += input_path.stat().st_size
            adata.file.close()

        self.data_directory = str(data_directory)
        self.num_slides = len(input_paths)
        self.num_cells = total_cells
        self.disk_bytes = total_disk_bytes

    def to_dict(self) -> dict[str, int | str]:
        """Convert the dataset summary to a plain dictionary."""
        return asdict(self)


def compute_peak_gpu_memory(device: str) -> int | None:
    """Return peak allocated GPU memory for the active device."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def extract_loss_metrics(trainer) -> dict[str, float | None]:
    """Collect final train, validation, and test losses from Lightning."""
    callback_metrics = trainer.lightning_trainer.callback_metrics if trainer.lightning_trainer is not None else {}
    metric_names = {
        "train_loss": ("train_loss_epoch", "train_loss"),
        "val_loss": ("val_loss",),
        "test_loss": ("test_loss",),
    }

    extracted = {}
    for output_name, candidates in metric_names.items():
        value = None
        for candidate in candidates:
            if candidate in callback_metrics:
                value = to_builtin(callback_metrics[candidate])
                break
        extracted[output_name] = value

    return extracted


def evaluate_and_plot_slide_embeddings(
    labels_by_key,
    embeddings,
    random_seed,
    n_neighbors,
    leiden_resolution,
    clustering_backend="scanpy",
    slide_name=None,
    obs_names=None,
    spatial=None,
    plot_path=None,
):
    """Cluster one slide, score label agreement, and optionally save a plot."""
    predicted = cluster_embeddings(
        embeddings,
        random_seed=random_seed,
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        clustering_backend=clustering_backend,
    )
    results = evaluate_slide_embeddings(
        labels_by_key=labels_by_key,
        predicted=predicted,
    )

    if plot_path is not None and slide_name is not None and obs_names is not None:
        save_slide_clustering_plot(
            slide_name=slide_name,
            obs_names=obs_names,
            spatial=spatial,
            predicted_labels=predicted,
            output_path=plot_path,
        )

    return results


def iter_evaluation_results(
    job_callable,
    job_kwargs_list: list[dict[str, object]],
    max_workers: int,
    progress_desc: str,
):
    """Yield completed evaluation results, optionally in parallel."""
    if max_workers == 1:
        iterator = (job_callable(**job_kwargs) for job_kwargs in job_kwargs_list)
        yield from tqdm(iterator, total=len(job_kwargs_list), desc=progress_desc, unit="slide")
        return

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=torch.multiprocessing.get_context("spawn"),
    ) as executor:
        pending = {executor.submit(job_callable, **job_kwargs) for job_kwargs in job_kwargs_list}
        with tqdm(total=len(pending), desc=progress_desc, unit="slide") as pbar:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                pbar.update(len(done))
                for future in done:
                    yield future.result()


def evaluate_sketch_cluster_agreement(
    reference_trainer,
    reference_data,
    target_trainer,
    target_data,
    embedding_eval_num_workers: int,
    random_seed: int,
    n_neighbors: int,
    leiden_resolution: float,
    clustering_backend: str = "scanpy",
    label_key: str = "original_leiden",
) -> dict[str, object]:
    """Compare sketch-model clusters against clusters from the full-data
    model."""
    reference_records = extract_slide_embedding_records(
        trainer=reference_trainer,
        evaluation_data=reference_data,
        progress_desc="Extracting reference embeddings",
    )
    target_records = extract_slide_embedding_records(
        trainer=target_trainer,
        evaluation_data=target_data,
        progress_desc="Extracting sketch embeddings",
    )
    if not reference_records or not target_records:
        return {"label_keys": [label_key], "metrics_by_label_key": {}}

    reference_clusterings = {}
    for record in tqdm(reference_records, desc="Clustering reference embeddings", unit="slide"):
        predicted = cluster_embeddings(
            record.embeddings,
            random_seed=random_seed,
            n_neighbors=n_neighbors,
            leiden_resolution=leiden_resolution,
            clustering_backend=clustering_backend,
        )
        reference_clusterings[record.slide_name] = pd.Series(predicted, index=record.obs_names)

    slide_jobs = []
    for record in target_records:
        reference_labels = reference_clusterings.get(record.slide_name)
        if reference_labels is None:
            continue

        matched_reference_labels = reference_labels.reindex(record.obs_names)
        valid_mask = ~matched_reference_labels.isna().to_numpy()
        if valid_mask.sum() < 3:
            continue

        slide_jobs.append(
            {
                "reference_labels": matched_reference_labels.to_numpy()[valid_mask],
                "target_embeddings": record.embeddings[valid_mask],
                "random_seed": random_seed,
                "n_neighbors": n_neighbors,
                "leiden_resolution": leiden_resolution,
                "clustering_backend": clustering_backend,
            },
        )

    metric_store = {label_key: LabelMetricStore()}
    for result in iter_evaluation_results(
        evaluate_slide_cluster_agreement,
        slide_jobs,
        max_workers=max(1, embedding_eval_num_workers),
        progress_desc="Scoring Sketch Agreement",
    ):
        accumulate_label_results(metric_store, {label_key: result})

    return summarize_cluster_scores([label_key], metric_store)

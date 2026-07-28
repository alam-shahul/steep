from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


@dataclass
class SlideEmbeddingRecord:
    slide_name: str
    obs_names: np.ndarray
    spatial: np.ndarray | None
    embeddings: np.ndarray
    labels_by_key: dict[str, np.ndarray]


COARSE_CELL_TYPE_LABEL_KEY = "cell_type_coarse"
COARSE_CELL_TYPE_SOURCE_COLUMN = "class"
COARSE_CELL_TYPE_PATTERNS = {
    "Excitatory neuron": ("glut", "excit"),
    "Inhibitory neuron": ("gaba", "inhib"),
    "Astrocyte": ("astro",),
    "OPC": ("opc",),
    "Oligodendrocyte": ("oligo",),
    "Microglia": ("micro",),
    "Endothelial": ("endo",),
    "Pericyte": ("peri",),
    "Ependymal": ("ependy",),
    "Immune": ("b cell", "t cell", "immune", "macrophage"),
}


def _derive_coarse_cell_type_labels(adata: ad.AnnData) -> np.ndarray | None:
    if COARSE_CELL_TYPE_SOURCE_COLUMN not in adata.obs:
        return None

    labels = []
    for value in adata.obs[COARSE_CELL_TYPE_SOURCE_COLUMN]:
        values = [] if pd.isna(value) else [str(value)]
        matched_label = "Other"

        for coarse_label, patterns in COARSE_CELL_TYPE_PATTERNS.items():
            if any(pattern in value.lower() for value in values for pattern in patterns):
                matched_label = coarse_label
                break

        labels.append(matched_label)

    return np.asarray(labels, dtype=object)


def _extract_labels_by_key(adata: ad.AnnData, label_keys: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
    labels_by_key = {}
    for label_key in label_keys:
        if label_key in adata.obs:
            labels_by_key[label_key] = adata.obs[label_key].to_numpy()
        elif label_key == COARSE_CELL_TYPE_LABEL_KEY:
            derived_labels = _derive_coarse_cell_type_labels(adata)
            if derived_labels is not None:
                labels_by_key[label_key] = derived_labels
    return labels_by_key


def extract_slide_embedding_records(
    trainer,
    evaluation_data=None,
    label_keys: list[str] | tuple[str, ...] = (),
    progress_desc: str = "Extracting Embeddings",
) -> list[SlideEmbeddingRecord]:
    """Extract per-slide embedding records from a trained model.

    Args:
        trainer: Trainer object that owns the trained model and target device.
        evaluation_data: Dataset to iterate over for graph construction and inference.
            Defaults to ``trainer.data`` when omitted.
        label_keys: Observation-column names to copy from each slide's ``adata.obs``
            into the returned record when present.
        progress_desc: Description shown in the tqdm progress bar during embedding
            extraction.

    Returns:
        A list of :class:`SlideEmbeddingRecord` objects, one per successfully
        processed slide.

    """
    if evaluation_data is None:
        evaluation_data = trainer.data

    if not hasattr(evaluation_data, "data_paths"):
        return []

    model = trainer.lightning_module.model if trainer.lightning_module is not None else trainer.model
    model = model.to(trainer.device)
    slide_records = []

    model.eval()
    with torch.no_grad():
        iterator = enumerate(evaluation_data.data_paths)
        total = len(evaluation_data.data_paths)
        for idx, data_path in tqdm(iterator, total=total, desc=progress_desc, unit="slide"):
            adata = ad.read_h5ad(data_path)
            graph = evaluation_data[idx].to(trainer.device)
            outputs = model(graph)
            if "embedding" not in outputs:
                continue

            slide_records.append(
                SlideEmbeddingRecord(
                    slide_name=Path(data_path).name,
                    obs_names=adata.obs_names.to_numpy(copy=True),
                    spatial=adata.obsm["spatial"].copy() if "spatial" in adata.obsm else None,
                    embeddings=outputs["embedding"].detach().cpu().numpy(),
                    labels_by_key=_extract_labels_by_key(adata, label_keys),
                ),
            )

    return slide_records

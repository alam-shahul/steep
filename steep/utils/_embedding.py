from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class SlideEmbeddingRecord:
    slide_name: str
    obs_names: np.ndarray
    spatial: np.ndarray | None
    embeddings: np.ndarray
    labels_by_key: dict[str, np.ndarray]


def extract_slide_embedding_records(
    model,
    evaluation_data,
    device: str | torch.device,
    label_keys: list[str] | tuple[str, ...] = (),
    progress_desc: str = "Extracting Embeddings",
) -> list[SlideEmbeddingRecord]:
    """Extract per-slide embedding records from a trained model.

    Args:
        model: Trained model used to produce embeddings.
        evaluation_data: Dataset to iterate over for graph construction and inference.
        device: Device on which inference is performed.
        label_keys: Observation-column names to copy from each slide's ``adata.obs``
            into the returned record when present.
        progress_desc: Description shown in the tqdm progress bar during embedding
            extraction.

    Returns:
        A list of :class:`SlideEmbeddingRecord` objects, one per successfully
        processed slide.

    """
    if not hasattr(evaluation_data, "data_paths"):
        return []

    model = model.to(device)
    slide_records = []

    model.eval()
    with torch.no_grad():
        iterator = enumerate(evaluation_data.data_paths)
        total = len(evaluation_data.data_paths)
        for idx, data_path in tqdm(iterator, total=total, desc=progress_desc, unit="slide"):
            adata = ad.read_h5ad(data_path)
            graph = evaluation_data[idx].to(device)
            outputs = model(graph)
            if "embedding" not in outputs:
                continue

            available_label_keys = [label_key for label_key in label_keys if label_key in adata.obs]
            slide_records.append(
                SlideEmbeddingRecord(
                    slide_name=Path(data_path).name,
                    obs_names=adata.obs_names.to_numpy(copy=True),
                    spatial=adata.obsm["spatial"].copy() if "spatial" in adata.obsm else None,
                    embeddings=outputs["embedding"].detach().cpu().numpy(),
                    labels_by_key={label_key: adata.obs[label_key].to_numpy() for label_key in available_label_keys},
                ),
            )

    return slide_records

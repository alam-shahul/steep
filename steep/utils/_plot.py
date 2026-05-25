import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squidpy as sq


def save_slide_clustering_plot(
    slide_name: str,
    obs_names: np.ndarray,
    spatial: np.ndarray | None,
    predicted_labels: np.ndarray,
    output_path: str | Path,
) -> None:
    if spatial is None:
        return

    plot_adata = ad.AnnData(obs=pd.DataFrame(index=obs_names))
    plot_adata.obsm["spatial"] = np.asarray(spatial)
    plot_adata.obs["predicted_leiden"] = pd.Categorical(np.asarray(predicted_labels).astype(str))
    library_id = Path(slide_name).stem
    plot_adata.uns["spatial"] = {library_id: {}}

    squidpy_logger = logging.getLogger("squidpy")
    previous_level = squidpy_logger.level
    squidpy_logger.setLevel(logging.ERROR)
    try:
        sq.pl.spatial_scatter(
            plot_adata,
            library_id=library_id,
            color="predicted_leiden",
            title=slide_name,
            shape=None,
        )
    finally:
        squidpy_logger.setLevel(previous_level)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close("all")

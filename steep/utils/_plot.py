import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squidpy as sq

# def save_slide_clustering_plot(
#     slide_name: str,
#     obs_names: np.ndarray,
#     spatial: np.ndarray | None,
#     predicted_labels: np.ndarray,
#     output_path: str | Path,
# ) -> None:
#     if spatial is None:
#         return

#     plot_adata = ad.AnnData(obs=pd.DataFrame(index=obs_names))
#     plot_adata.obsm["spatial"] = np.asarray(spatial)
#     plot_adata.obs["predicted_leiden"] = pd.Categorical(np.asarray(predicted_labels).astype(str))

#     squidpy_logger = logging.getLogger("squidpy")
#     previous_level = squidpy_logger.level
#     squidpy_logger.setLevel(logging.ERROR)
#     try:
#         sq.pl.spatial_scatter(
#             plot_adata,
#             color="predicted_leiden",
#             title=slide_name,
#             shape=None,
#         )
#     finally:
#         squidpy_logger.setLevel(previous_level)
#     plt.savefig(output_path, bbox_inches="tight", dpi=200)
#     plt.close("all")


def save_slide_clustering_plot(
    slide_name: str,
    obs_names: np.ndarray,
    spatial: np.ndarray | None,
    predicted_labels: np.ndarray,
    output_path: str | Path,
) -> None:
    del obs_names

    if spatial is None:
        return

    spatial = np.asarray(spatial)
    predicted_labels = np.asarray(predicted_labels).astype(str)

    x = spatial[:, 0]
    y = spatial[:, 1]

    categories = pd.Categorical(predicted_labels)
    codes = categories.codes

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=codes, s=4, linewidths=0)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.title(slide_name)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

import anndata as ad
import awkward as ak
import numpy as np
import squidpy as sq


def construct_spatial_graph(
    adata: ad.AnnData,
    k: int | None = None,
    radius: float | None = None,
    adjacency_matrix_key: str = "adjacency_matrix",
    adjacency_list_key: str = "adjacency_list",
):
    """Construct cell-cell graph from spatial coordinates.

    Can use spatial coordinates.

    Args:
        adata: the spatial dataset
        k: number of nearest-neighbors to
        cutoff:

    Returns:
        The input dataset,

    """
    if radius is None and k is None:
        raise ValueError("At least one of `k` or `radius` must be specified.")

    if radius is not None:
        radius = [0, radius]

    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        radius=radius,
        n_neighs=k,
    )

    adata.obsp[adjacency_matrix_key] = adata.obsp["spatial_connectivities"]
    num_cells, _ = adata.obsp[adjacency_matrix_key].shape

    adjacency_list = [[] for _ in range(num_cells)]
    for x, y in zip(*adata.obsp[adjacency_matrix_key].nonzero()):
        adjacency_list[x].append(y)

    adata.obsm[adjacency_list_key] = ak.Array(adjacency_list)

    return adata


def compute_spatial_neighbors(
    adata: ad.AnnData,
    threshold: float = 94.5,
    adjacency_matrix_key: str = "adjacency_matrix",
    adjacency_list_key: str = "adjacency_list",
):
    r"""Compute neighbor graph based on spatial coordinates.

    Stores resulting graph in ``adata.obs[adjacency_matrix_key]``.

    """

    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    distance_matrix = adata.obsp["spatial_distances"]
    distances = distance_matrix.data
    cutoff = np.percentile(distances, threshold)

    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        delaunay=True,
        radius=[0, cutoff],
    )
    adata.obsp[adjacency_matrix_key] = adata.obsp["spatial_connectivities"]

    num_cells, _ = adata.obsp[adjacency_matrix_key].shape

    adjacency_list = [[] for _ in range(num_cells)]
    for x, y in zip(*adata.obsp[adjacency_matrix_key].nonzero()):
        adjacency_list[x].append(y)

    adata.obsm[adjacency_list_key] = ak.Array(adjacency_list)

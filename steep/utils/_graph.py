import os

import anndata as ad
import awkward as ak
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import squidpy as sq
import torch


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


def visualize_bundle(pt_path, out_path="test_outputs", undirected=False, show=True):
    """Visualize original vs sparsified graph and save figure.

    Parameters
    ----------
    pt_path : str
        Path to saved bundle (.pt)
    out_path : str
        Directory to save figure
    undirected : bool
        Whether to treat graph as undirected
    show : bool
        Whether to display figure

    """

    def edge_list(edge_index):
        return list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    os.makedirs(out_path, exist_ok=True)

    b = torch.load(pt_path, map_location="cpu")

    orig = b["orig"]
    sparse = b["sparse"]

    x = orig["x"]
    edge_index_orig = orig["edge_index"]
    edge_index_sparse = sparse["edge_index"]

    num_nodes = x.size(0)

    graph = nx.graphraph() if undirected else nx.Digraphraph()
    graph.add_nodes_from(range(num_nodes))

    orig_edges = edge_list(edge_index_orig)
    sparse_edges = set(edge_list(edge_index_sparse))

    if orig.get("pos") is not None:
        pos = {i: orig["pos"][i].tolist() for i in range(num_nodes)}
    else:
        pos = nx.spring_layout(graph, seed=0)

    plt.figure(figsize=(6, 6))

    nx.draw_networkx_nodes(graph, pos, node_size=600)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=orig_edges,
        arrows=not undirected,
        width=1.0,
        alpha=0.25,
        edge_color="gray",
    )

    kept_edges = [e for e in orig_edges if e in sparse_edges]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=kept_edges,
        arrows=not undirected,
        width=2.5,
        alpha=0.9,
        edge_color="red",
    )

    plt.axis("off")

    base_name = os.path.splitext(os.path.basename(pt_path))[0]
    save_file = os.path.join(out_path, f"{base_name}_graph.pdf")

    plt.savefig(save_file, dpi=500, bbox_inches="tight")
    print(f"Saved figure to: {save_file}")

    if show:
        plt.show()
    else:
        plt.close()

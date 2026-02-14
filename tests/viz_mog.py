import os
import torch
import networkx as nx
import matplotlib.pyplot as plt


def edge_list(edge_index):
    return list(zip(edge_index[0].tolist(), edge_index[1].tolist()))


def visualize_bundle(pt_path, out_path="test_outputs", undirected=False, show=True):
    """
    Visualize original vs sparsified graph and save figure.

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

    os.makedirs(out_path, exist_ok=True)

    b = torch.load(pt_path, map_location="cpu")

    orig = b["orig"]
    sparse = b["sparse"]

    x = orig["x"]
    edge_index_orig = orig["edge_index"]
    edge_index_sparse = sparse["edge_index"]

    num_nodes = x.size(0)

    G = nx.Graph() if undirected else nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    orig_edges = edge_list(edge_index_orig)
    sparse_edges = set(edge_list(edge_index_sparse))

    if orig.get("pos") is not None:
        pos = {i: orig["pos"][i].tolist() for i in range(num_nodes)}
    else:
        pos = nx.spring_layout(G, seed=0)

    plt.figure(figsize=(6, 6))

    nx.draw_networkx_nodes(G, pos, node_size=600)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=orig_edges,
        arrows=not undirected,
        width=1.0,
        alpha=0.25,
        edge_color="gray",
    )

    kept_edges = [e for e in orig_edges if e in sparse_edges]
    nx.draw_networkx_edges(
        G,
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


if __name__ == "__main__":
    out_path = "test_outputs"
    visualize_bundle(
        f"{out_path}/final_sparsity_check.pt",
        out_path=out_path,
        undirected=False,
        show=True,
    )
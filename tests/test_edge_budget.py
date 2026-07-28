import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from steep.sketcher import (
    MoGSketcher,
    RandomEdgeSketcher,
    canonical_edge_pairs,
    mean_pair_scores,
    num_undirected_edge_pairs,
    target_budget_count,
    topk_pair_mask,
)


def test_canonical_edge_pairs_map_asymmetric_edges_and_ignore_self_loops():
    edge_index = torch.tensor(
        [
            [2, 0, 1, 1, 3],
            [1, 1, 2, 1, 2],
        ],
    )

    pairs, edge_to_pair = canonical_edge_pairs(edge_index, num_nodes=4)

    assert pairs.tolist() == [[0, 1, 2], [1, 2, 3]]
    assert edge_to_pair.tolist() == [1, 0, 1, -1, 2]


def test_mean_pair_scores_average_available_directions():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    scores = torch.tensor([0.2, 0.8, 0.4, 0.6])
    pairs, edge_to_pair = canonical_edge_pairs(edge_index, num_nodes=3)

    pair_scores = mean_pair_scores(scores, edge_to_pair, pairs.size(1))

    assert torch.allclose(pair_scores, torch.tensor([0.5, 0.5]))


def test_topk_pair_mask_uses_exact_budget_and_stable_ties():
    scores = torch.tensor([0.5, 0.5, 0.2, 0.1])

    mask = topk_pair_mask(scores, retention_ratio=0.25)

    assert target_budget_count(scores.numel(), 0.25) == 1
    assert mask.tolist() == [True, False, False, False]


def test_mog_edge_budget_operates_on_undirected_pairs():
    sketcher = MoGSketcher(
        mog_args={"retention_ratio": 0.5},
        device="cpu",
        use_topo=False,
        use_expr_prior=False,
    )
    edge_index = torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 3]])
    edge_scores = torch.tensor([0.1, 0.9, 0.4, 0.6, 0.7])

    pairs, mask = sketcher._pairs_from_score(edge_scores, edge_index, num_nodes=4)

    assert pairs.tolist() == [[0, 1, 2], [1, 2, 3]]
    assert mask.sum().item() == 2
    assert pairs[:, mask].tolist() == [[0, 2], [1, 3]]


def test_edge_baseline_retains_exact_symmetric_pair_budget(tmp_path):
    adata = ad.AnnData(X=np.ones((5, 2), dtype=np.float32))
    adjacency = sp.csr_matrix(
        (
            np.ones(9),
            (
                [0, 1, 1, 2, 2, 3, 3, 4, 4],
                [1, 0, 2, 1, 3, 2, 4, 3, 4],
            ),
        ),
        shape=(5, 5),
    )
    adata.obsp["adjacency_matrix"] = adjacency
    sketcher = RandomEdgeSketcher(
        retention_ratio=0.5,
        random_seed=3,
        drop_isolated_cells=False,
    )

    result = sketcher.transform(adata)
    result_adjacency = result.obsp["adjacency_matrix"]

    assert num_undirected_edge_pairs(result_adjacency) == 2
    assert (result_adjacency != result_adjacency.T).nnz == 0

    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.h5ad"
    adata.write_h5ad(input_path)
    metadata = sketcher.fit_transform_to_disk(input_path, output_path)

    assert metadata.original_num_edge_pairs == 4
    assert metadata.sketched_num_edge_pairs == 2

import torch

from steep.models._sparsify import MoG


def test_mog_learner_scores_edges_and_returns_loss_terms():
    model = MoG(
        num_features=3,
        device=torch.device("cpu"),
        k_list=[0.1, 0.3, 0.5],
        hidden_spl=8,
        num_layers_spl=2,
        expert_select=2,
    )
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1))

    model.learner.get_topo_val(edge_index)
    model.learner.expr_prior = torch.linspace(0.0, 1.0, edge_index.size(1))
    output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    assert output["edge_score"].shape == (edge_index.size(1),)
    assert output["loss"].requires_grad
    assert set(output) == {"edge_score", "loss", "loss_balance", "loss_topo", "loss_expr"}

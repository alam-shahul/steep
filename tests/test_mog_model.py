import pytest
import torch

from steep.models._sparsify import MoG, _edge_scores_in_input_order


def test_topology_scores_follow_original_edge_order():
    class FakeGraph:
        edge_ids = {(2, 0): 1, (0, 1): 2, (1, 2): 0}

        def edgeId(self, src, dst):  # noqa: N802
            return self.edge_ids[(src, dst)]

    scores = (
        [10, 20, 30],
        [100, 200, 300],
    )

    ordered = _edge_scores_in_input_order(
        FakeGraph(),
        [(2, 0), (0, 1), (1, 2)],
        scores,
    )

    assert ordered == [[20, 200], [30, 300], [10, 100]]


def test_networkit_topology_scores_match_edge_count():
    pytest.importorskip("networkit")
    model = MoG(
        num_features=2,
        device=torch.device("cpu"),
        k_list=[0.1, 0.3, 0.5],
        hidden_spl=4,
        num_layers_spl=2,
        expert_select=2,
    )
    edge_index = torch.tensor(
        [[2, 0, 1, 0, 1, 2], [0, 1, 2, 2, 0, 1]],
        dtype=torch.long,
    )

    model.learner.get_topo_val(edge_index, random_seed=11)

    assert model.learner.topo_val.shape == (edge_index.size(1), 4)
    assert torch.isfinite(model.learner.topo_val).all()


def test_mog_learner_scores_edges_and_returns_loss_terms():
    model = MoG(
        num_features=3,
        device=torch.device("cpu"),
        k_list=[0.1, 0.3, 0.5],
        hidden_spl=8,
        num_layers_spl=2,
        expert_select=2,
        expr_loss_coef=0.5,
    )
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1))

    model.learner.expr_prior = torch.linspace(0.0, 1.0, edge_index.size(1))
    output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    assert output["edge_score"].shape == (edge_index.size(1),)
    assert output["edge_mask"].shape == (edge_index.size(1),)
    assert output["loss"].requires_grad
    assert output["loss_expr"].item() > 0
    assert set(output) == {"edge_score", "edge_mask", "loss", "loss_balance", "loss_topo", "loss_expr"}


def test_mog_expression_prior_can_augment_edge_scores():
    model = MoG(
        num_features=3,
        device=torch.device("cpu"),
        k_list=[0.1, 0.3, 0.5],
        hidden_spl=8,
        num_layers_spl=2,
        expert_select=2,
        expr_aug_coef=2.0,
    )
    model.eval()
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1))
    expr_prior = torch.linspace(0.0, 1.0, edge_index.size(1))

    model.learner.expr_prior = None
    base_output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    model.learner.expr_prior = expr_prior
    expr_output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    assert torch.allclose(expr_output["edge_score"], base_output["edge_score"] + 2.0 * expr_prior)


def test_mog_expression_prior_shapes_topology_loss():
    model = MoG(
        num_features=3,
        device=torch.device("cpu"),
        k_list=[0.1, 0.3, 0.5],
        hidden_spl=8,
        num_layers_spl=2,
        expert_select=2,
        topo_loss_coef=1.0,
        expr_loss_coef=0.0,
        expr_topo_mode="both",
    )
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1))

    model.learner.topo_val = torch.ones(edge_index.size(1), 4)
    model.learner.expr_prior = torch.linspace(0.0, 1.0, edge_index.size(1))

    output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    assert output["loss_topo"].item() > 0


def test_mog_learner_handles_nodes_without_outgoing_edges():
    model = MoG(
        num_features=3,
        device=torch.device("cpu"),
        k_list=[0.1, 0.3, 0.5],
        hidden_spl=8,
        num_layers_spl=2,
        expert_select=2,
    )
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1))

    output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    assert output["edge_score"].shape == (edge_index.size(1),)
    assert output["edge_mask"].shape == (edge_index.size(1),)

import os
import statistics
import time

import pytest
import torch

from steep.models._sparsify import MoG


def _make_mog(num_features: int) -> MoG:
    return _make_mog_with_hidden(num_features=num_features, hidden_spl=8)


def _make_mog_with_hidden(num_features: int, hidden_spl: int) -> MoG:
    return MoG(
        num_features=num_features,
        device=torch.device("cpu"),
        k_list=[0.1, 0.3, 0.5],
        hidden_spl=hidden_spl,
        num_layers_spl=2,
        expert_select=2,
    )


def test_mog_learner_scores_edges_and_returns_loss_terms():
    model = _make_mog(num_features=3)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1))

    model.learner.expr_prior = torch.linspace(0.0, 1.0, edge_index.size(1))
    output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    assert output["edge_score"].shape == (edge_index.size(1),)
    assert output["edge_mask"].shape == (edge_index.size(1),)
    assert torch.equal(output["edge_mask"], output["mask"])
    assert output["loss"].requires_grad
    assert {"edge_score", "edge_mask", "loss", "loss_balance", "loss_topo", "loss_expr"}.issubset(output)


def test_mog_learner_handles_nodes_without_outgoing_edges():
    model = _make_mog(num_features=3)
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1))

    output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    assert output["edge_score"].shape == (edge_index.size(1),)
    assert output["edge_mask"].shape == (edge_index.size(1),)


def test_mog_edge_mlp_input_width_matches_full_feature_graph():
    model = _make_mog(num_features=1122)
    x = torch.randn(4, 1122)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1))

    output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    assert model.learner.experts[0].layers[0].in_features == 2245
    assert output["edge_score"].shape == (edge_index.size(1),)


def test_mog_edge_mlp_input_width_matches_pca_feature_graph():
    model = _make_mog(num_features=50)
    x = torch.randn(4, 50)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.size(1))

    output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)

    assert model.learner.experts[0].layers[0].in_features == 101
    assert output["edge_score"].shape == (edge_index.size(1),)


@pytest.mark.skipif(
    os.environ.get("RUN_MOG_BENCHMARK") != "1",
    reason="Manual MoG microbenchmark; set RUN_MOG_BENCHMARK=1 to run.",
)
def test_mog_forward_runtime_full_features_vs_pca_features():
    torch.manual_seed(0)
    num_nodes = 26_395
    num_edges = 149_570
    hidden_spl = 64
    repeats = 1
    balanced_edge_ids = torch.arange(num_edges, dtype=torch.long)
    balanced_src = balanced_edge_ids // 10
    balanced_dst = (balanced_src * 17 + balanced_edge_ids % 10 + 1) % num_nodes
    balanced_edge_index = torch.stack([balanced_src, balanced_dst], dim=0)

    high_degree_sources = 100
    high_degree_span = (num_edges + high_degree_sources - 1) // high_degree_sources
    high_degree_src = torch.div(balanced_edge_ids, high_degree_span, rounding_mode="floor")
    high_degree_dst = (high_degree_src * 997 + balanced_edge_ids % high_degree_span + 1) % num_nodes
    high_degree_edge_index = torch.stack([high_degree_src, high_degree_dst], dim=0)

    def time_forward(edge_index: torch.Tensor, num_features: int) -> float:
        model = _make_mog_with_hidden(num_features=num_features, hidden_spl=hidden_spl)
        x = torch.randn(num_nodes, num_features)
        edge_attr = torch.ones(edge_index.size(1))

        start = time.perf_counter()
        output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr, training=False)
        elapsed = time.perf_counter() - start

        assert output["edge_score"].shape == (num_edges,)
        return elapsed

    def time_train_step(
        edge_index: torch.Tensor,
        num_features: int,
        topo_val: torch.Tensor | None,
        expr_prior: torch.Tensor | None,
    ) -> float:
        model = _make_mog_with_hidden(num_features=num_features, hidden_spl=hidden_spl)
        optimizer = torch.optim.Adam(model.learner.parameters(), lr=1e-3)
        model.learner.topo_val = None if topo_val is None else topo_val.clone()
        model.learner.expr_prior = None if expr_prior is None else expr_prior.clone()
        x = torch.randn(num_nodes, num_features)
        edge_attr = torch.ones(edge_index.size(1))

        model.train()
        start = time.perf_counter()
        optimizer.zero_grad()
        output = model.learner(x=x, edge_index=edge_index, temp=1.0, edge_attr=edge_attr)
        output["loss"].backward()
        optimizer.step()
        elapsed = time.perf_counter() - start

        assert output["edge_score"].shape == (num_edges,)
        return elapsed

    def repeat_timing(fn) -> tuple[float, float]:
        elapsed = [fn() for _ in range(repeats)]
        return statistics.mean(elapsed), 0.0 if len(elapsed) == 1 else statistics.stdev(elapsed)

    def benchmark_graph(graph_name: str, edge_index: torch.Tensor) -> None:
        topo_model = _make_mog_with_hidden(num_features=50, hidden_spl=hidden_spl)

        def time_topo() -> float:
            topo_model.learner.topo_val = None
            start = time.perf_counter()
            topo_model.learner.get_topo_val(edge_index)
            return time.perf_counter() - start

        topo_mean, topo_std = repeat_timing(time_topo)
        topo_val = topo_model.learner.topo_val.detach().clone()
        expr_prior = torch.rand(edge_index.size(1))
        full_mean, full_std = repeat_timing(lambda: time_forward(edge_index=edge_index, num_features=1122))
        pca_mean, pca_std = repeat_timing(lambda: time_forward(edge_index=edge_index, num_features=50))
        full_train_mean, full_train_std = repeat_timing(
            lambda: time_train_step(
                edge_index=edge_index,
                num_features=1122,
                topo_val=topo_val,
                expr_prior=expr_prior,
            ),
        )
        pca_train_mean, pca_train_std = repeat_timing(
            lambda: time_train_step(
                edge_index=edge_index,
                num_features=50,
                topo_val=topo_val,
                expr_prior=expr_prior,
            ),
        )

        print(f"MoG benchmark graph: {graph_name}")
        print(f"MoG get_topo_val: {topo_mean:.3f}s +/- {topo_std:.3f}s")
        print(f"MoG forward full features: {full_mean:.3f}s +/- {full_std:.3f}s")
        print(f"MoG forward PCA features: {pca_mean:.3f}s +/- {pca_std:.3f}s")
        print(
            "MoG train step full features with topo/expression priors: "
            f"{full_train_mean:.3f}s +/- {full_train_std:.3f}s",
        )
        print(
            "MoG train step PCA features with topo/expression priors: "
            f"{pca_train_mean:.3f}s +/- {pca_train_std:.3f}s",
        )

    print(f"MoG benchmark repeats: {repeats}")
    print(f"MoG benchmark scale: nodes={num_nodes}, edges={num_edges}, hidden_spl={hidden_spl}")
    benchmark_graph("balanced_10_edges_per_source", balanced_edge_index)
    benchmark_graph("high_degree_100_sources", high_degree_edge_index)

import torch

from steep.sketcher import MoGSketcher


def test_mog_rng_scope_is_reproducible_and_restores_global_state(monkeypatch):
    sketcher = MoGSketcher(
        mog_args={"retention_ratio": 0.5},
        device="cpu",
        use_topo=False,
        use_expr_prior=False,
    )

    def fake_train_impl(**kwargs):
        del kwargs
        return torch.rand(4)

    monkeypatch.setattr(sketcher, "_train_mog_impl", fake_train_impl)
    features = torch.zeros(2, 1)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.ones(2)
    state_before = torch.random.get_rng_state().clone()

    first = sketcher._train_mog(features, edge_index, edge_attr, random_seed=17)
    state_after = torch.random.get_rng_state()
    second = sketcher._train_mog(features, edge_index, edge_attr, random_seed=17)

    assert torch.equal(state_before, state_after)
    assert torch.equal(first, second)

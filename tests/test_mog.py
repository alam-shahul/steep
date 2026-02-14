import os
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from steep.models._sparsify import MoG


SAVE_DIR = "test_outputs"


def save_orig_and_sparsified(data, mask, tag):
    os.makedirs(SAVE_DIR, exist_ok=True)

    keep = mask.detach().bool()

    orig_edge_index = data.edge_index.detach().cpu()
    sparse_edge_index = data.edge_index[:, keep].detach().cpu()

    bundle = {
        "orig": {
            "x": data.x.detach().cpu(),
            "edge_index": orig_edge_index,
            "pos": data.pos.detach().cpu() if hasattr(data, "pos") and data.pos is not None else None,
            "edge_attr": data.edge_attr.detach().cpu()
            if hasattr(data, "edge_attr") and data.edge_attr is not None
            else None,
            "batch": data.batch.detach().cpu() if hasattr(data, "batch") and data.batch is not None else None,
            "y": data.y.detach().cpu() if hasattr(data, "y") and data.y is not None else None,
        },
        "sparse": {
            "x": data.x.detach().cpu(),
            "edge_index": sparse_edge_index,
            "pos": data.pos.detach().cpu() if hasattr(data, "pos") and data.pos is not None else None,
            "edge_attr": data.edge_attr[keep].detach().cpu()
            if hasattr(data, "edge_attr") and data.edge_attr is not None
            else None,
            "batch": data.batch.detach().cpu() if hasattr(data, "batch") and data.batch is not None else None,
            "y": data.y.detach().cpu() if hasattr(data, "y") and data.y is not None else None,
        },
        "mask": mask.detach().cpu(),
        "kept_edge_idx": keep.nonzero(as_tuple=False).view(-1).detach().cpu(),
    }

    torch.save(bundle, os.path.join(SAVE_DIR, f"{tag}.pt"))


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda:0")


@pytest.fixture
def mock_args():
    return {
        "k_list": [0.25, 0.5, 0.75],
        "hidden_spl": 16,
        "num_layers_spl": 3,
        "expert_select": 2,
        "lam": 0.1,
        "use_topo": False,
    }


@pytest.fixture
def tiny_graph(device):
    torch.manual_seed(0)

    num_nodes = 6
    in_dim = 8

    x = torch.randn(num_nodes, in_dim, device=device)

    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 5],
            [1, 2, 0, 3, 3, 4, 4, 5, 5, 4],
        ],
        dtype=torch.long,
        device=device,
    )

    pos = torch.randn(num_nodes, 2, device=device)
    edge_attr = torch.randn(edge_index.size(1), 2, device=device)

    y = torch.tensor([1], dtype=torch.long, device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=y,
        batch=batch,
    )


@pytest.fixture
def mog_model(mock_args, tiny_graph, device):
    in_dim = tiny_graph.x.size(1)
    out_classes = 3

    model = MoG(
        in_dim=in_dim,
        emb_dim=mock_args["hidden_spl"],
        out_channels=out_classes,
        edge_dim=0,
        args=mock_args,
        device=device,
    ).to(device)

    model.eval()
    return model


def test_mog_forward_shapes(mog_model, tiny_graph, mock_args, device):
    data = tiny_graph.to(device)

    x, edge_index = data.x, data.edge_index
    if mock_args.get("use_topo", False):
        mog_model.learner.get_topo_val(edge_index)

    mask, add_loss = mog_model.learner(
        x=x,
        edge_index=edge_index,
        temp=1e-3,
        edge_attr=None,
        training=False,
    )

    assert mask.dim() == 1
    assert mask.numel() == edge_index.size(1)

    assert torch.is_tensor(add_loss)
    assert add_loss.numel() == 1

    unique_vals = set(mask.detach().cpu().tolist())
    assert unique_vals.issubset({0.0, 1.0})

    out = mog_model.gnn(data, mask)

    assert out.dim() == 2
    assert out.size(0) == 1

    prob_sum = out.exp().sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5)


def test_mog_one_train_step_decreases_backward_ok(mog_model, tiny_graph, mock_args, device):
    model = mog_model
    model.train()
    data = tiny_graph.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x, edge_index = data.x, data.edge_index
    if mock_args.get("use_topo", False):
        model.learner.get_topo_val(edge_index)

    mask, add_loss = model.learner(
        x=x,
        edge_index=edge_index,
        temp=1e-3,
        edge_attr=None,
        training=True,
    )
    out = model.gnn(data, mask)

    loss = F.nll_loss(out, data.y) + add_loss * 1e-1

    opt.zero_grad()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)

    opt.step()


def test_mask_sparsity_reasonable(mog_model, tiny_graph, device):
    model = mog_model
    model.eval()
    data = tiny_graph.to(device)

    mask, _ = model.learner(
        x=data.x,
        edge_index=data.edge_index,
        temp=1e-3,
        edge_attr=None,
        training=False,
    )

    kept = int(mask.sum().item())
    total = mask.numel()
    assert 0 < kept < total

    save_orig_and_sparsified(data, mask, "final_sparsity_check")
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

from steep.training import SteepDataModule, SteepLightningModule


class _DummyGraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        logits = data.x * self.scale
        return {"logits": logits, "embedding": logits}


def _graphs(count: int):
    return [
        Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            pos=torch.randn(4, 2),
        )
        for _ in range(count)
    ]


def test_data_module_allows_test_only_random_split():
    datamodule = SteepDataModule(
        data=_graphs(3),
        batch_size=1,
        train_ratio=0.0,
        val_ratio=0.0,
    )

    assert len(datamodule.datasets["train"]) == 0
    assert len(datamodule.datasets["val"]) == 0
    assert len(datamodule.datasets["test"]) == 3
    assert len(datamodule.train_dataloader()) == 0
    assert len(datamodule.val_dataloader()) == 0
    assert len(datamodule.test_dataloader()) == 3


def test_scheduler_uses_lightning_estimated_optimizer_steps(monkeypatch):
    cfg = OmegaConf.create(
        {
            "optimizer": {"type": "torch.optim.AdamW", "args": {"lr": 1e-3}},
            "scheduler": {"name": "constant", "args": {"warmup_ratio": 0.25}},
        },
    )
    module = SteepLightningModule(cfg, _DummyGraphModel(), Mock())
    module._trainer = Mock(estimated_stepping_batches=8)
    scheduler = Mock()
    get_scheduler = Mock(return_value=scheduler)
    monkeypatch.setattr("steep.training.get_scheduler", get_scheduler)

    configured = module.configure_optimizers()

    assert configured["lr_scheduler"]["scheduler"] is scheduler
    get_scheduler.assert_called_once_with(
        name="constant",
        optimizer=configured["optimizer"],
        num_warmup_steps=2,
        num_training_steps=8,
    )


def test_spatial_block_split_creates_non_overlapping_subgraphs():
    width = 4
    positions = torch.tensor([[float(x), float(y)] for y in range(width) for x in range(width)])
    features = torch.arange(width * width, dtype=torch.float32).unsqueeze(1)
    edges = []
    for node in range(width * width):
        x, y = node % width, node // width
        if x + 1 < width:
            edges.extend(((node, node + 1), (node + 1, node)))
        if y + 1 < width:
            edges.extend(((node, node + width), (node + width, node)))
    graph = Data(x=features, pos=positions, edge_index=torch.tensor(edges).T.contiguous())
    datamodule = SteepDataModule(
        data=[graph],
        batch_size=1,
        train_ratio=0.5,
        val_ratio=0.25,
        split_type="spatial_block",
        spatial_block_grid_size=2,
    )

    node_ids = [set(datamodule.datasets[name][0].x[:, 0].tolist()) for name in ("train", "val", "test")]
    assert all(node_ids)
    assert node_ids[0].isdisjoint(node_ids[1])
    assert node_ids[0].isdisjoint(node_ids[2])
    assert node_ids[1].isdisjoint(node_ids[2])
    assert set.union(*node_ids) == set(range(width * width))

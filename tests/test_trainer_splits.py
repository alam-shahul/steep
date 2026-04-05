import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

from steep.trainer import PyGTrainer


class _DummyGraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        logits = data.x * self.scale
        return {"logits": logits, "embedding": logits}


def _cfg():
    return OmegaConf.create(
        {
            "optimizer": {
                "type": "torch.optim.AdamW",
                "args": {"lr": 1e-3, "weight_decay": 0.0},
            },
            "scheduler": {
                "name": "constant",
                "args": {"warmup_ratio": 0.0},
            },
            "loss": {
                "type": "steep.loss.NodeMSELoss",
            },
        },
    )


def test_pyg_trainer_allows_test_only_random_split():
    dataset = [
        Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            pos=torch.randn(4, 2),
        )
        for _ in range(3)
    ]

    trainer = PyGTrainer(
        cfg=_cfg(),
        model=_DummyGraphModel(),
        data=dataset,
        batchsize=1,
        epochs=1,
        device="cpu",
        shuffle=True,
        train_ratio=0.0,
        val_ratio=0.0,
        split_type="random",
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    assert len(trainer.datasets["train"]) == 0
    assert len(trainer.datasets["val"]) == 0
    assert len(trainer.datasets["test"]) == len(dataset)
    assert len(trainer.dataloaders["train"]) == 0
    assert len(trainer.dataloaders["val"]) == 0
    assert len(trainer.dataloaders["test"]) == len(dataset)

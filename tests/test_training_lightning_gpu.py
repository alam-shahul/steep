from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

from steep.trainer import PyGTrainer
from steep.utils import instantiate_from_config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU integration tests."),
]


def _distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class ToyAutoencoder(nn.Module):
    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(self, data: Data):
        return {
            "logits": self.proj(data.x),
        }


class _IdentityLatentModel:
    def __call__(self, data: Data):
        return {
            "embedding": data.x.float(),
        }

    def decode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        return x

    def eval(self):
        return self

    def train(self):
        return self


def _make_toy_graph(offset: float, in_dim: int = 4) -> Data:
    pos = torch.tensor(
        [
            [0.0 + offset, 0.0],
            [1.0 + offset, 0.0],
            [0.5 + offset, 0.75],
        ],
        dtype=torch.float32,
    )
    x = torch.stack(
        [
            torch.linspace(0.1, 0.4, steps=in_dim),
            torch.linspace(0.2, 0.5, steps=in_dim),
            torch.linspace(0.3, 0.6, steps=in_dim),
        ],
        dim=0,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2],
        ],
        dtype=torch.long,
    )
    return Data(x=x, edge_index=edge_index, pos=pos)


def _make_toy_dataset(num_graphs: int = 5, in_dim: int = 4) -> list[Data]:
    return [_make_toy_graph(float(idx), in_dim=in_dim) for idx in range(num_graphs)]


def _make_trainer_cfg(tmp_path: Path, pretrained_ckpt_path: str | None = None):
    cfg = OmegaConf.create(
        {
            "cache_dir": str(tmp_path / "cache"),
            "project": "tests",
            "run_name": "lightning-gpu-smoke",
            "entity": "tests",
            "model": {"type": "tests.test_training_lightning_gpu.ToyAutoencoder"},
            "dataset": {"name": "toy", "args": {"data_directory": "toy-data"}},
            "optimizer": {"type": "torch.optim.AdamW", "args": {"lr": 1e-3}},
            "scheduler": {"name": "linear", "args": {"warmup_ratio": 0.0}},
            "loss": {"type": "steep.loss.NodeMSELoss"},
        },
    )
    if pretrained_ckpt_path is not None:
        cfg.pretrained_ckpt_path = pretrained_ckpt_path
    return cfg


def test_pyg_trainer_gpu_fit(tmp_path):
    cfg = _make_trainer_cfg(tmp_path)
    trainer = PyGTrainer(
        cfg=cfg,
        model=ToyAutoencoder().cuda(),
        data=_make_toy_dataset(),
        batchsize=2,
        epochs=1,
        device="cuda",
        train_ratio=0.6,
        val_ratio=0.2,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        num_sanity_val_steps=0,
        run_wandb=False,
    )

    trainer.fit(resume_from_checkpoint=False)

    assert trainer.device.startswith("cuda")
    assert trainer.step > 0
    assert (trainer.results_folder / "last.ckpt").exists()
    assert not _distributed_is_initialized()


def test_pyg_trainer_deepspeed_single_gpu_fit(tmp_path):
    cfg = _make_trainer_cfg(tmp_path)
    trainer = PyGTrainer(
        cfg=cfg,
        model=ToyAutoencoder().cuda(),
        data=_make_toy_dataset(),
        batchsize=2,
        epochs=1,
        device="cuda",
        train_ratio=0.6,
        val_ratio=0.2,
        accelerator="gpu",
        devices=1,
        strategy="deepspeed_stage_2",
        precision="32-true",
        num_sanity_val_steps=0,
        run_wandb=False,
    )

    trainer.fit(resume_from_checkpoint=False)

    assert trainer.device.startswith("cuda")
    assert trainer.step > 0
    assert (trainer.results_folder / "last.ckpt").exists()
    assert not _distributed_is_initialized()

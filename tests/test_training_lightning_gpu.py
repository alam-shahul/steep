from pathlib import Path

import pytest
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

from steep.lightning import build_lightning_callbacks, cleanup_distributed
from steep.loss import NodeMSELoss
from steep.training import SteepDataModule, SteepLightningModule

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU integration tests."),
]


class ToyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, data: Data):
        return {"logits": self.proj(data.x)}


def _graphs():
    return [
        Data(
            x=torch.randn(3, 4),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            pos=torch.randn(3, 2),
        )
        for _ in range(5)
    ]


def _run_gpu_fit(tmp_path: Path, strategy: str):
    cfg = OmegaConf.create(
        {
            "optimizer": {"type": "torch.optim.AdamW", "args": {"lr": 1e-3}},
            "scheduler": {"name": "linear", "args": {"warmup_ratio": 0.0}},
            "trainer": {"args": {"accelerator": "gpu", "strategy": strategy}},
        },
    )
    module = SteepLightningModule(cfg, ToyAutoencoder(), NodeMSELoss())
    datamodule = SteepDataModule(data=_graphs(), batch_size=2, train_ratio=0.6, val_ratio=0.2)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        precision="32-true",
        logger=False,
        callbacks=build_lightning_callbacks(tmp_path, enable_lr_monitor=False, enable_progress_bar=False),
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    try:
        trainer.fit(module, datamodule=datamodule)
    finally:
        cleanup_distributed()
    assert trainer.global_step > 0
    assert (tmp_path / "last.ckpt").exists()


def test_lightning_gpu_fit(tmp_path):
    _run_gpu_fit(tmp_path, "auto")


def test_lightning_deepspeed_single_gpu_fit(tmp_path):
    _run_gpu_fit(tmp_path, "deepspeed_stage_2")

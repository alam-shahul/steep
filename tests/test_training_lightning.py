from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

from steep.lightning import build_lightning_callbacks, cleanup_distributed
from steep.training import SteepDataModule, SteepLightningModule, load_pretrained_weights, prepare_resume_checkpoint
from steep.utils import instantiate_from_config


class ToyAutoencoder(nn.Module):
    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(self, data: Data):
        return {"logits": self.proj(data.x)}


def _make_toy_graph(offset: float, in_dim: int = 4) -> Data:
    return Data(
        x=torch.stack(
            [
                torch.linspace(0.1, 0.4, steps=in_dim),
                torch.linspace(0.2, 0.5, steps=in_dim),
                torch.linspace(0.3, 0.6, steps=in_dim),
            ],
        ),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]),
        pos=torch.tensor([[offset, 0.0], [offset + 1.0, 0.0], [offset + 0.5, 0.75]]),
    )


def _make_toy_dataset(num_graphs: int = 5) -> list[Data]:
    return [_make_toy_graph(float(idx)) for idx in range(num_graphs)]


def _cfg():
    return OmegaConf.create(
        {
            "optimizer": {"type": "torch.optim.AdamW", "args": {"lr": 1e-3}},
            "scheduler": {"name": "linear", "args": {"warmup_ratio": 0.0}},
            "datamodule": {
                "type": "steep.training.SteepDataModule",
                "args": {"batch_size": 2, "train_ratio": 0.6, "val_ratio": 0.2, "num_workers": 0},
            },
            "lightning_module": {"type": "steep.training.SteepLightningModule", "args": {}},
            "trainer": {
                "type": "lightning.pytorch.Trainer",
                "args": {
                    "max_epochs": 1,
                    "accelerator": "cpu",
                    "devices": 1,
                    "logger": False,
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "num_sanity_val_steps": 0,
                },
            },
        },
    )


def _instantiate_stack(tmp_path: Path):
    cfg = _cfg()
    datamodule = instantiate_from_config(cfg.datamodule, data=_make_toy_dataset())
    model = ToyAutoencoder()
    module = instantiate_from_config(
        cfg.lightning_module,
        cfg=cfg,
        model=model,
        loss_function=instantiate_from_config({"type": "steep.loss.NodeMSELoss"}),
    )
    trainer = instantiate_from_config(
        cfg.trainer,
        default_root_dir=str(tmp_path),
        callbacks=build_lightning_callbacks(tmp_path, enable_lr_monitor=False, enable_progress_bar=False),
    )
    return model, module, datamodule, trainer


def test_configured_components_fit_validate_and_test(tmp_path):
    _, module, datamodule, trainer = _instantiate_stack(tmp_path)

    trainer.fit(module, datamodule=datamodule)
    fit_metrics = dict(trainer.callback_metrics)
    test_results = trainer.test(module, datamodule=datamodule)

    assert isinstance(trainer, Trainer)
    assert isinstance(module, SteepLightningModule)
    assert isinstance(datamodule, SteepDataModule)
    assert "val_loss" in fit_metrics
    assert "test_loss" in test_results[0]
    assert (tmp_path / "last.ckpt").exists()


def test_epoch_timer_synchronizes_cuda_at_boundaries(tmp_path, monkeypatch):
    import steep.lightning as lightning_utils

    timer = next(
        callback
        for callback in build_lightning_callbacks(
            tmp_path,
            enable_lr_monitor=False,
            enable_progress_bar=False,
        )
        if hasattr(callback, "epoch_times_seconds")
    )
    synchronize_calls = []
    timestamps = iter((10.0, 13.5))
    module = SimpleNamespace(device=torch.device("cuda:0"))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronize_calls.append(device))
    monkeypatch.setattr(lightning_utils, "perf_counter", lambda: next(timestamps))

    timer.on_fit_start(None, module)
    timer.on_train_epoch_start(None, module)
    timer.on_train_epoch_end(None, module)

    assert synchronize_calls == [torch.device("cuda:0"), torch.device("cuda:0")]
    assert timer.epoch_times_seconds == [3.5]


def test_scheduler_counts_partial_batches_and_accumulated_optimizer_steps(tmp_path, monkeypatch):
    import steep.training as training

    cfg = _cfg()
    captured = {}
    original_get_scheduler = training.get_scheduler

    def capture_scheduler(**kwargs):
        captured.update(kwargs)
        return original_get_scheduler(**kwargs)

    monkeypatch.setattr(training, "get_scheduler", capture_scheduler)
    module = SteepLightningModule(cfg, ToyAutoencoder(), instantiate_from_config({"type": "steep.loss.NodeMSELoss"}))
    datamodule = SteepDataModule(
        data=_make_toy_dataset(7),
        batch_size=2,
        train_ratio=1.0,
        val_ratio=0.0,
        num_workers=0,
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(module, datamodule=datamodule)

    assert len(datamodule.train_dataloader()) == 4
    assert trainer.estimated_stepping_batches == 6
    assert captured["num_training_steps"] == 6


def test_load_pretrained_weights_from_lightning_checkpoint(tmp_path):
    model, module, datamodule, trainer = _instantiate_stack(tmp_path)
    trainer.fit(module, datamodule=datamodule)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    reloaded = ToyAutoencoder()
    for parameter in reloaded.parameters():
        nn.init.zeros_(parameter)

    assert load_pretrained_weights(reloaded, tmp_path)
    for key, value in reloaded.state_dict().items():
        assert torch.allclose(value, expected[key])


def test_legacy_resume_loads_weights_and_resets_loop_state(tmp_path):
    expected_model = ToyAutoencoder()
    legacy_path = tmp_path / "model-2.pt"
    torch.save(
        {
            "model": expected_model.state_dict(),
            "optimizer": {"legacy": True},
            "epoch": 2,
            "step": 9,
        },
        legacy_path,
    )
    (tmp_path / "milestone.txt").write_text("2")
    actual_model = ToyAutoencoder()
    for parameter in actual_model.parameters():
        nn.init.zeros_(parameter)

    with pytest.warns(UserWarning, match="optimizer, scheduler, epoch, and loop state were reset"):
        resume_path, reused = prepare_resume_checkpoint(actual_model, tmp_path, enabled=True)

    assert resume_path is None
    assert reused
    for key, value in actual_model.state_dict().items():
        assert torch.allclose(value, expected_model.state_dict()[key])


def test_lightning_resume_path_is_returned_for_exact_resume(tmp_path):
    checkpoint_path = tmp_path / "last.ckpt"
    checkpoint_path.touch()

    resume_path, reused = prepare_resume_checkpoint(ToyAutoencoder(), tmp_path, enabled=True)

    assert resume_path == str(checkpoint_path)
    assert reused


def test_data_module_small_dataset_keeps_nonzero_splits():
    datamodule = SteepDataModule(data=_make_toy_dataset(8), batch_size=1, train_ratio=0.8, val_ratio=0.1)
    sizes = [len(datamodule.datasets[name]) for name in ("train", "val", "test")]
    assert all(size > 0 for size in sizes)
    assert sum(sizes) == 8


def test_cleanup_distributed_resets_deepspeed_global_state(monkeypatch):
    deepspeed_comm = pytest.importorskip("deepspeed.comm")
    deepspeed_groups = pytest.importorskip("deepspeed.utils.groups")
    sentinel = object()
    monkeypatch.setattr(deepspeed_comm, "cdb", object(), raising=False)
    for name in (
        "_WORLD_GROUP",
        "_DATA_PARALLEL_GROUP",
        "_MODEL_PARALLEL_GROUP",
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_ZERO_PARAM_INTRA_PARALLEL_GROUP",
    ):
        monkeypatch.setattr(deepspeed_groups, name, sentinel, raising=False)
    for name in (
        "_EXPERT_PARALLEL_GROUP",
        "_EXPERT_PARALLEL_GROUP_RANKS",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP_RANKS",
        "_ALL_TO_ALL_GROUP",
    ):
        monkeypatch.setattr(deepspeed_groups, name, {"sentinel": sentinel}, raising=False)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    cleanup_distributed()

    assert deepspeed_comm.cdb is None
    assert deepspeed_groups._WORLD_GROUP is None
    assert deepspeed_groups._DATA_PARALLEL_GROUP is None
    assert deepspeed_groups._MODEL_PARALLEL_GROUP is None
    assert deepspeed_groups._TENSOR_MODEL_PARALLEL_GROUP is None
    assert deepspeed_groups._ZERO_PARAM_INTRA_PARALLEL_GROUP is None
    assert deepspeed_groups._EXPERT_PARALLEL_GROUP == {}
    assert deepspeed_groups._EXPERT_PARALLEL_GROUP_RANKS == {}
    assert deepspeed_groups._EXPERT_DATA_PARALLEL_GROUP == {}
    assert deepspeed_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS == {}
    assert deepspeed_groups._ALL_TO_ALL_GROUP == {}

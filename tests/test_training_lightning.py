from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

from steep.lightning import cleanup_distributed
from steep.trainer import PyGTrainer
from steep.utils import instantiate_from_config


class ToyAutoencoder(nn.Module):
    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(self, data: Data):
        return {
            "logits": self.proj(data.x),
        }


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


def _make_block_graph(width: int = 4, height: int = 4, in_dim: int = 4) -> Data:
    pos = []
    features = []
    edge_src = []
    edge_dst = []
    node_index = 0
    for y_idx in range(height):
        for x_idx in range(width):
            pos.append([float(x_idx), float(y_idx)])
            feature = torch.full((in_dim,), float(node_index), dtype=torch.float32)
            features.append(feature)

            if x_idx + 1 < width:
                right = node_index + 1
                edge_src.extend([node_index, right])
                edge_dst.extend([right, node_index])
            if y_idx + 1 < height:
                down = node_index + width
                edge_src.extend([node_index, down])
                edge_dst.extend([down, node_index])
            node_index += 1

    return Data(
        x=torch.stack(features, dim=0),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float32),
    )


def _make_trainer_cfg(tmp_path, pretrained_ckpt_path: str | None = None):
    cfg = OmegaConf.create(
        {
            "cache_dir": str(tmp_path / "cache"),
            "project": "tests",
            "run_name": "lightning-smoke",
            "entity": "tests",
            "model": {"type": "tests.test_training_lightning.ToyAutoencoder"},
            "dataset": {"name": "toy", "args": {"data_directory": "toy-data"}},
            "optimizer": {"type": "torch.optim.AdamW", "args": {"lr": 1e-3}},
            "scheduler": {"name": "linear", "args": {"warmup_ratio": 0.0}},
            "loss": {"type": "steep.loss.NodeMSELoss"},
        },
    )
    if pretrained_ckpt_path is not None:
        cfg.pretrained_ckpt_path = pretrained_ckpt_path
    return cfg


def test_pyg_trainer_lightning_fit_saves_last_checkpoint(tmp_path):
    cfg = _make_trainer_cfg(tmp_path)
    trainer = PyGTrainer(
        cfg=cfg,
        model=ToyAutoencoder(),
        data=_make_toy_dataset(),
        batchsize=2,
        epochs=1,
        device="cpu",
        train_ratio=0.6,
        val_ratio=0.2,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        num_sanity_val_steps=0,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        run_wandb=False,
    )

    trainer.fit(resume_from_checkpoint=False)

    assert trainer.lightning_trainer is not None
    assert trainer.lightning_module is not None
    assert trainer.step > 0
    assert (trainer.results_folder / "last.ckpt").exists()


def test_pyg_trainer_fresh_fit_removes_existing_checkpoint_directory(tmp_path):
    cfg = _make_trainer_cfg(tmp_path)
    trainer = PyGTrainer(
        cfg=cfg,
        model=ToyAutoencoder(),
        data=_make_toy_dataset(),
        batchsize=2,
        epochs=1,
        device="cpu",
        train_ratio=0.6,
        val_ratio=0.2,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        num_sanity_val_steps=0,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        run_wandb=False,
    )
    stale_file = trainer.get_checkpoint_directory() / "stale.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old checkpoint state")

    trainer.fit(resume_from_checkpoint=False)

    assert not stale_file.exists()
    assert (trainer.results_folder / "last.ckpt").exists()


def test_pyg_trainer_load_pretrained_from_lightning_checkpoint(tmp_path):
    cfg = _make_trainer_cfg(tmp_path)
    trainer = PyGTrainer(
        cfg=cfg,
        model=ToyAutoencoder(),
        data=_make_toy_dataset(),
        batchsize=2,
        epochs=1,
        device="cpu",
        train_ratio=0.6,
        val_ratio=0.2,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        num_sanity_val_steps=0,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        run_wandb=False,
    )
    trainer.fit(resume_from_checkpoint=False)

    pretrained_path = trainer.results_folder / "last.ckpt"
    reloaded_cfg = _make_trainer_cfg(tmp_path, pretrained_ckpt_path=str(pretrained_path))
    reloaded = PyGTrainer(
        cfg=reloaded_cfg,
        model=ToyAutoencoder(),
        data=_make_toy_dataset(),
        batchsize=2,
        epochs=1,
        device="cpu",
        train_ratio=0.6,
        val_ratio=0.2,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        num_sanity_val_steps=0,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        run_wandb=False,
    )

    for parameter in reloaded.model.parameters():
        nn.init.zeros_(parameter)

    reloaded.load_pretrained()

    expected_state = trainer.model.state_dict()
    actual_state = reloaded.model.state_dict()
    assert expected_state.keys() == actual_state.keys()
    for key in expected_state:
        assert torch.allclose(actual_state[key], expected_state[key])


def test_pyg_trainer_resolves_checkpoint_directory_to_best_ckpt(tmp_path):
    cfg = _make_trainer_cfg(tmp_path)
    trainer = PyGTrainer(
        cfg=cfg,
        model=ToyAutoencoder(),
        data=_make_toy_dataset(),
        batchsize=2,
        epochs=1,
        device="cpu",
        train_ratio=0.6,
        val_ratio=0.2,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        num_sanity_val_steps=0,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        run_wandb=False,
    )
    trainer.fit(resume_from_checkpoint=False)

    reloaded_cfg = _make_trainer_cfg(tmp_path, pretrained_ckpt_path=str(trainer.results_folder))
    reloaded = PyGTrainer(
        cfg=reloaded_cfg,
        model=ToyAutoencoder(),
        data=_make_toy_dataset(),
        batchsize=2,
        epochs=0,
        device="cpu",
        train_ratio=0.6,
        val_ratio=0.2,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        num_sanity_val_steps=0,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        run_wandb=False,
    )

    assert reloaded.get_pretrained_load_path() == trainer.results_folder / "best.ckpt"


def test_pyg_trainer_rejects_deepspeed_on_cpu(tmp_path):
    cfg = _make_trainer_cfg(tmp_path)
    trainer = PyGTrainer(
        cfg=cfg,
        model=ToyAutoencoder(),
        data=_make_toy_dataset(),
        batchsize=2,
        epochs=1,
        device="cpu",
        train_ratio=0.6,
        val_ratio=0.2,
        accelerator="cpu",
        devices=1,
        strategy="deepspeed_stage_2",
        precision="32-true",
        num_sanity_val_steps=0,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        run_wandb=False,
    )
    trainer.initialize_checkpointing()

    with pytest.raises(ValueError, match="DeepSpeed strategy requires `accelerator=gpu`"):
        trainer._build_lightning_trainer()


def test_pyg_trainer_small_dataset_keeps_nonzero_val_and_test(tmp_path):
    cfg = _make_trainer_cfg(tmp_path)
    trainer = PyGTrainer(
        cfg=cfg,
        model=ToyAutoencoder(),
        data=_make_toy_dataset(num_graphs=8),
        batchsize=1,
        epochs=0,
        device="cpu",
        train_ratio=0.8,
        val_ratio=0.1,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        num_sanity_val_steps=0,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        run_wandb=False,
    )

    assert len(trainer.datasets["train"]) > 0
    assert len(trainer.datasets["val"]) > 0
    assert len(trainer.datasets["test"]) > 0
    assert len(trainer.datasets["train"]) + len(trainer.datasets["val"]) + len(trainer.datasets["test"]) == 8


def test_pyg_trainer_spatial_block_split_creates_non_overlapping_subgraphs(tmp_path):
    cfg = _make_trainer_cfg(tmp_path)
    trainer = PyGTrainer(
        cfg=cfg,
        model=ToyAutoencoder(),
        data=[_make_block_graph()],
        batchsize=1,
        epochs=0,
        device="cpu",
        train_ratio=0.5,
        val_ratio=0.25,
        split_type="spatial_block",
        spatial_block_grid_size=2,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        num_sanity_val_steps=0,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        run_wandb=False,
    )

    train_graph = trainer.datasets["train"][0]
    val_graph = trainer.datasets["val"][0]
    test_graph = trainer.datasets["test"][0]

    train_ids = set(train_graph.x[:, 0].tolist())
    val_ids = set(val_graph.x[:, 0].tolist())
    test_ids = set(test_graph.x[:, 0].tolist())
    all_ids = train_ids | val_ids | test_ids

    assert train_ids
    assert val_ids
    assert test_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert all_ids == set(range(16))


def test_cleanup_distributed_resets_deepspeed_global_state(monkeypatch):
    deepspeed_comm = pytest.importorskip("deepspeed.comm")
    deepspeed_groups = pytest.importorskip("deepspeed.utils.groups")

    sentinel_group = object()
    sentinel_dict = {"sentinel": object()}

    monkeypatch.setattr(deepspeed_comm, "cdb", object(), raising=False)
    monkeypatch.setattr(deepspeed_groups, "_WORLD_GROUP", sentinel_group, raising=False)
    monkeypatch.setattr(deepspeed_groups, "_DATA_PARALLEL_GROUP", sentinel_group, raising=False)
    monkeypatch.setattr(deepspeed_groups, "_MODEL_PARALLEL_GROUP", sentinel_group, raising=False)
    monkeypatch.setattr(deepspeed_groups, "_TENSOR_MODEL_PARALLEL_GROUP", sentinel_group, raising=False)
    monkeypatch.setattr(deepspeed_groups, "_ZERO_PARAM_INTRA_PARALLEL_GROUP", sentinel_group, raising=False)
    monkeypatch.setattr(deepspeed_groups, "_EXPERT_PARALLEL_GROUP", sentinel_dict.copy(), raising=False)
    monkeypatch.setattr(deepspeed_groups, "_EXPERT_PARALLEL_GROUP_RANKS", sentinel_dict.copy(), raising=False)
    monkeypatch.setattr(deepspeed_groups, "_EXPERT_DATA_PARALLEL_GROUP", sentinel_dict.copy(), raising=False)
    monkeypatch.setattr(deepspeed_groups, "_EXPERT_DATA_PARALLEL_GROUP_RANKS", sentinel_dict.copy(), raising=False)
    monkeypatch.setattr(deepspeed_groups, "_ALL_TO_ALL_GROUP", sentinel_dict.copy(), raising=False)
    monkeypatch.setattr(deepspeed_groups, "expert_tensor_parallel_world_size", 4, raising=False)
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
    assert deepspeed_groups.expert_tensor_parallel_world_size == 1

"""Lightning-native training components and checkpoint utilities."""

import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import lightning as L  # noqa: N812
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from steep.dataset import SpatialBlockSubsetDataset, build_spatial_block_node_subsets
from steep.lightning import (
    capture_rng_state,
    default_dataloader_num_workers,
    extract_submodule_state_dict,
    get_resume_checkpoint_path,
    restore_rng_state,
    write_config_snapshot,
)
from steep.utils import get_fully_qualified_cache_paths, get_scheduler, instantiate_from_config

CHECKPOINT_KEYS = (
    "run_name",
    "model",
    "dataset.args.data_directory",
    "datamodule.args.train_ratio",
    "datamodule.args.val_ratio",
    "datamodule.args.split_type",
    "datamodule.args.spatial_block_grid_size",
    "sketcher.type",
    "sketcher.args.retention_ratio",
    "sketcher.args.random_seed",
)
LEGACY_CHECKPOINT_KEYS = (
    "run_name",
    "model",
    "dataset.args.data_directory",
    "trainer.args.train_ratio",
    "trainer.args.val_ratio",
    "trainer.args.split_type",
    "trainer.args.spatial_block_grid_size",
    "sketcher.type",
    "sketcher.args.retention_ratio",
    "sketcher.args.random_seed",
)


class SteepDataModule(L.LightningDataModule):
    """Create deterministic PyG dataset splits and dataloaders."""

    def __init__(
        self,
        data: Dataset,
        batch_size: int,
        shuffle: bool = True,
        random_seed: int = 0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        split_type: str = "random",
        spatial_block_grid_size: int = 4,
        num_workers: int | None = None,
        persistent_workers: bool | None = None,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        self.split_type = split_type
        self.spatial_block_grid_size = spatial_block_grid_size
        self.num_workers = default_dataloader_num_workers() if num_workers is None else num_workers
        self.persistent_workers = self.num_workers > 0 if persistent_workers is None else persistent_workers
        self.pin_memory = pin_memory
        self.datasets: dict[str, Dataset] = {}
        self.setup()

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.datasets:
            return
        if self.split_type == "random":
            sizes = self._resolve_split_sizes(len(self.data))
            train, val, test = random_split(
                self.data,
                sizes,
                generator=torch.Generator().manual_seed(self.random_seed),
            )
        elif self.split_type == "spatial_block":
            train, val, test = self._build_spatial_block_datasets()
        else:
            raise ValueError(f"Unsupported split type: {self.split_type}")
        self.datasets = {"train": train, "val": val, "test": test, "full": self.data}

    def _resolve_split_sizes(self, total_size: int) -> tuple[int, int, int]:
        if total_size <= 0:
            raise ValueError("Dataset must contain at least one sample.")
        requested = {
            "train": max(float(self.train_ratio), 0.0),
            "val": max(float(self.val_ratio), 0.0),
            "test": max(float(self.test_ratio), 0.0),
        }
        active = [name for name, ratio in requested.items() if ratio > 0]
        if total_size < len(active):
            raise ValueError(f"Dataset size {total_size} is too small for non-empty active splits {active}.")

        sizes = dict.fromkeys(requested, 0)
        for name in active:
            sizes[name] = 1
        remaining = total_size - len(active)
        if remaining > 0 and active:
            ratio_sum = sum(requested[name] for name in active)
            raw = {name: remaining * requested[name] / ratio_sum for name in active}
            fractions = []
            for name in active:
                extra = int(raw[name])
                sizes[name] += extra
                fractions.append((raw[name] - extra, name))
            leftover = total_size - sum(sizes.values())
            for _, name in sorted(fractions, reverse=True):
                if leftover <= 0:
                    break
                sizes[name] += 1
                leftover -= 1
        return sizes["train"], sizes["val"], sizes["test"]

    def _build_spatial_block_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        subsets: dict[str, list[tuple[int, torch.Tensor]]] = {"train": [], "val": [], "test": []}
        for base_idx in range(len(self.data)):
            split_ids = build_spatial_block_node_subsets(
                self.data[base_idx],
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                grid_size=self.spatial_block_grid_size,
            )
            for split, node_ids in split_ids.items():
                if node_ids.numel() > 0:
                    subsets[split].append((base_idx, node_ids.cpu()))
        return tuple(SpatialBlockSubsetDataset(self.data, subsets[name]) for name in ("train", "val", "test"))

    def _dataloader(self, split: str, *, shuffle: bool = False) -> DataLoader:
        dataset = self.datasets[split]
        kwargs: dict[str, Any] = {}
        if self.num_workers > 0:
            kwargs["multiprocessing_context"] = "spawn"
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and len(dataset) > 0,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            pin_memory=self.pin_memory,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train", shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader("test")

    def full_dataloader(self) -> DataLoader:
        return self._dataloader("full")


class SteepLightningModule(L.LightningModule):
    """Train a Steep graph model with Lightning."""

    def __init__(self, cfg: DictConfig, model: nn.Module, loss_function: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_function = loss_function

    def forward(self, inputs: Data):
        return self.model(inputs)

    def _shared_step(self, batch: Data):
        outputs = self.model(batch)
        return outputs, self.loss_function(batch, outputs)

    def training_step(self, batch: Data, batch_idx: int):
        del batch_idx
        _, loss = self._shared_step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=getattr(batch, "num_graphs", 1),
            sync_dist=self.trainer.world_size > 1,
        )
        return loss

    def validation_step(self, batch: Data, batch_idx: int):
        del batch_idx
        _, loss = self._shared_step(batch)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=getattr(batch, "num_graphs", 1),
            sync_dist=self.trainer.world_size > 1,
        )
        return loss

    def test_step(self, batch: Data, batch_idx: int):
        del batch_idx
        _, loss = self._shared_step(batch)
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            batch_size=getattr(batch, "num_graphs", 1),
            sync_dist=self.trainer.world_size > 1,
        )
        return loss

    def configure_optimizers(self):
        optimizer = instantiate_from_config(self.cfg.optimizer, self.model.parameters())
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(float(self.cfg.scheduler.args.warmup_ratio) * total_steps)
        scheduler = get_scheduler(
            name=self.cfg.scheduler.name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        checkpoint.update(capture_rng_state())

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        restore_rng_state(checkpoint)


def infer_num_genes(cfg: DictConfig) -> int:
    """Read the input feature count used to instantiate configured models."""
    data_directory = Path(cfg.dataset.args.data_directory)
    first_path = next(data_directory.glob("*.h5ad"))
    adata = ad.read_h5ad(first_path, backed="r")
    try:
        return int(adata.n_vars)
    finally:
        adata.file.close()


def get_checkpoint_directory(cfg: DictConfig) -> Path:
    """Return the deterministic checkpoint directory for a training config."""
    return get_fully_qualified_cache_paths(
        cfg,
        Path(cfg.cache_dir) / "checkpoints",
        keys=CHECKPOINT_KEYS,
        mkdir=False,
    )


def _get_legacy_checkpoint_directory(cfg: DictConfig) -> Path:
    legacy_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for name in ("train_ratio", "val_ratio", "split_type", "spatial_block_grid_size"):
        OmegaConf.update(
            legacy_cfg,
            f"trainer.args.{name}",
            OmegaConf.select(cfg, f"datamodule.args.{name}"),
            force_add=True,
        )
    return get_fully_qualified_cache_paths(
        legacy_cfg,
        Path(cfg.cache_dir) / "checkpoints",
        keys=LEGACY_CHECKPOINT_KEYS,
        mkdir=False,
    )


def initialize_checkpointing(cfg: DictConfig) -> Path:
    """Create the checkpoint directory and write its resolved configuration."""
    results_folder = get_checkpoint_directory(cfg)
    legacy_results_folder = _get_legacy_checkpoint_directory(cfg)
    if not results_folder.exists() and legacy_results_folder.exists():
        results_folder = legacy_results_folder
    results_folder.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(results_folder, cfg)
    return results_folder


def load_pretrained_weights(model: nn.Module, checkpoint_value: str | Path | None, device: str = "cpu") -> bool:
    """Load model weights from a legacy or Lightning checkpoint."""
    if checkpoint_value is None:
        return False
    path = Path(checkpoint_value)
    if path.is_dir():
        path = next(
            (path / name for name in ("best.ckpt", "last.ckpt", "checkpoint.pt") if (path / name).exists()),
            path,
        )
    if not path.is_file():
        warnings.warn(
            f"Checkpoint {checkpoint_value} does not exist; pretrained weights were not loaded.",
            stacklevel=2,
        )
        return False
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = extract_submodule_state_dict(checkpoint, legacy_key="model", lightning_prefix="model.")
    model.load_state_dict(state, strict=False)
    return True


def prepare_resume_checkpoint(
    model: nn.Module,
    results_folder: Path,
    *,
    enabled: bool,
    device: str = "cpu",
) -> tuple[str | None, bool]:
    """Resolve an exact Lightning resume or migrate legacy model weights."""
    if not enabled:
        return None, False
    resume_path = get_resume_checkpoint_path(results_folder)
    if resume_path is None:
        return None, False
    if resume_path.endswith(".ckpt"):
        return resume_path, True
    load_pretrained_weights(model, resume_path, device=device)
    warnings.warn(
        "Loaded model weights from a legacy checkpoint; optimizer, scheduler, epoch, and loop state were reset.",
        stacklevel=2,
    )
    return None, True

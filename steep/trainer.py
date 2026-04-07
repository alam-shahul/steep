import random
from collections import defaultdict
from pathlib import Path
from pprint import pformat

import anndata as ad
import lightning as L  # noqa: N812
import numpy as np
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from steep.dataset import SpatialBlockSubsetDataset, build_spatial_block_node_subsets
from steep.lightning import (
    build_lightning_callbacks,
    build_wandb_logger,
    capture_rng_state,
    cleanup_distributed,
    default_dataloader_num_workers,
    extract_epoch_and_step,
    extract_submodule_state_dict,
    finalize_lightning_logger,
    get_resume_checkpoint_path,
    infer_accelerator,
    restore_rng_state,
    suppress_deepspeed_probe_logging,
    write_config_snapshot,
)
from steep.utils import get_fully_qualified_cache_paths, get_scheduler, instantiate_from_config


class _PyGLightningModule(L.LightningModule):
    def __init__(self, owner: "PyGTrainer"):
        super().__init__()
        self.owner = owner
        self.model = owner.model
        self.loss_function = owner.loss_function

    def forward(self, inputs: Data):
        return self.model(inputs)

    def _shared_step(self, batch: Data):
        outputs = self.model(batch)
        loss = self.loss_function(batch, outputs)
        return outputs, loss

    def _should_sync_dist(self) -> bool:
        trainer = self.trainer
        if trainer is None:
            return False
        accelerator_connector = getattr(trainer, "_accelerator_connector", None)
        if accelerator_connector is not None:
            return bool(getattr(accelerator_connector, "is_distributed", False))
        strategy = getattr(trainer, "strategy", None)
        return bool(getattr(strategy, "is_distributed", False))

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
            sync_dist=self._should_sync_dist(),
        )
        return loss

    def validation_step(self, batch: Data, batch_idx: int, dataloader_idx: int = 0):
        del batch_idx
        _, loss = self._shared_step(batch)
        metric_name = "val_loss" if dataloader_idx == 0 else "test_loss"
        self.log(
            metric_name,
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=metric_name == "val_loss",
            add_dataloader_idx=False,
            batch_size=getattr(batch, "num_graphs", 1),
            sync_dist=self._should_sync_dist(),
        )
        return loss

    def configure_optimizers(self):
        scheduler_config = {
            "scheduler": self.owner.lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [self.owner.optimizer], [scheduler_config]

    def on_fit_start(self):
        self.owner.device = str(self.device)
        self.owner.model = self.model

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        del outputs, batch, batch_idx
        self.owner.step = int(self.global_step)

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        checkpoint["step"] = int(self.global_step)
        checkpoint.update(capture_rng_state())

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        restore_rng_state(checkpoint, rng=self.owner.rng)
        _, step = extract_epoch_and_step(checkpoint)
        self.owner.step = step


def _resolve_checkpoint_candidate(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_dir():
        for candidate in ("best.ckpt", "last.ckpt", "checkpoint.pt"):
            candidate_path = path / candidate
            if candidate_path.exists():
                return candidate_path
        return None

    if path.exists():
        return path
    return None


class PyGTrainer:
    """PyG trainer."""

    CHECKPOINT_KEYS = (
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

    def __init__(
        self,
        cfg: OmegaConf,
        model: nn.Module,
        data: Dataset,
        batchsize: int,
        epochs: int,
        device: str,
        shuffle: bool = True,
        random_seed: int = 0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        split_type: str = "random",
        spatial_block_grid_size: int = 4,
        grad_norm_clip: float = 1.0,
        accelerator: str = "auto",
        devices: int | str | list[int] = 1,
        strategy: str = "auto",
        precision: str = "32-true",
        accumulate_grad_batches: int = 1,
        log_every_n_steps: int = 1,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = False,
        num_sanity_val_steps: int = 0,
        num_workers: int | None = None,
        persistent_workers: bool | None = None,
        pin_memory: bool | None = None,
        run_wandb: bool = False,
    ):
        self.cfg = cfg
        self.model = model

        # Training loop hyperparameters
        self.batchsize = batchsize
        self.epochs = epochs
        self.device = device
        self.model.to(self.device)
        self._instantiate_loss()

        # Dataset hyperparameters
        self.shuffle = shuffle
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        self.split_type = split_type
        self.spatial_block_grid_size = spatial_block_grid_size
        self.num_workers = default_dataloader_num_workers() if num_workers is None else num_workers
        self.persistent_workers = self.num_workers > 0 if persistent_workers is None else persistent_workers
        self.pin_memory = device.startswith("cuda") if pin_memory is None else pin_memory

        # Random seeds
        self.random_seed = random_seed
        self.rng = torch.Generator().manual_seed(random_seed)

        # Dataloader setup
        self.data = data
        self.step = 0

        # Optimizer setup
        self.grad_norm_clip = grad_norm_clip
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.precision = precision
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_every_n_steps = log_every_n_steps
        self.enable_progress_bar = enable_progress_bar
        self.enable_model_summary = enable_model_summary
        self.num_sanity_val_steps = num_sanity_val_steps
        self.lightning_module = None
        self.lightning_trainer = None
        self._legacy_resume_epoch = 0
        self._initialize_optimizer()
        self._initialize_lr_scheduler()

        # WandB setup
        self.run_wandb = run_wandb

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._setup_dataloaders()

    def _setup_dataloaders(self):
        """Set up dataloaders for all splits given an SRTDataset.

        Automatically called when setting the `data` property.

        """
        if self.split_type == "random":
            total_size = len(self._data)
            train_size, val_size, test_size = self._resolve_split_sizes(total_size)
            train_dataset, val_dataset, test_dataset = random_split(
                self._data,
                [train_size, val_size, test_size],
                generator=self.rng,
            )
        elif self.split_type == "spatial_block":
            train_dataset, val_dataset, test_dataset = self._build_spatial_block_datasets()
        else:
            raise ValueError(f"Unsupported split type: {self.split_type}")

        self.datasets = {
            "train": train_dataset,
            "test": test_dataset,
            "val": val_dataset,
            "full": self.data,
        }
        self.dataloaders = {}
        for split, split_dataset in self.datasets.items():
            dataloader_kwargs = {}
            if self.num_workers > 0:
                dataloader_kwargs["multiprocessing_context"] = "spawn"
            split_shuffle = self.shuffle if split in {"train", "full"} and len(split_dataset) > 0 else False
            dataloader = DataLoader(
                split_dataset,
                batch_size=self.batchsize,
                shuffle=split_shuffle,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                pin_memory=self.pin_memory,
                **dataloader_kwargs,
            )
            self.dataloaders[split] = dataloader
            print(f"{split} DataLoader has size {len(dataloader)}")

    def _build_spatial_block_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        node_subsets = {
            "train": [],
            "val": [],
            "test": [],
        }
        for base_idx in range(len(self._data)):
            data = self._data[base_idx]
            split_node_ids = build_spatial_block_node_subsets(
                data,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                grid_size=self.spatial_block_grid_size,
            )
            for split, node_ids in split_node_ids.items():
                if node_ids.numel() == 0:
                    continue
                node_subsets[split].append((base_idx, node_ids.cpu()))

        return (
            SpatialBlockSubsetDataset(self._data, node_subsets["train"]),
            SpatialBlockSubsetDataset(self._data, node_subsets["val"]),
            SpatialBlockSubsetDataset(self._data, node_subsets["test"]),
        )

    def _resolve_split_sizes(self, total_size: int) -> tuple[int, int, int]:
        if total_size <= 0:
            raise ValueError("Dataset must contain at least one sample.")

        requested = {
            "train": max(float(self.train_ratio), 0.0),
            "val": max(float(self.val_ratio), 0.0),
            "test": max(float(self.test_ratio), 0.0),
        }
        active_splits = [name for name, ratio in requested.items() if ratio > 0]

        if total_size < len(active_splits):
            raise ValueError(
                f"Dataset size {total_size} is too small for non-empty active splits {active_splits}.",
            )

        sizes = dict.fromkeys(requested, 0)
        for name in active_splits:
            sizes[name] = 1

        remaining = total_size - len(active_splits)
        if remaining > 0 and active_splits:
            total_ratio = sum(requested[name] for name in active_splits)
            raw_allocations = {name: remaining * requested[name] / total_ratio for name in active_splits}
            fractional_parts = []
            for name in active_splits:
                extra = int(raw_allocations[name])
                sizes[name] += extra
                fractional_parts.append((raw_allocations[name] - extra, name))

            leftover = total_size - sum(sizes.values())
            for _, name in sorted(fractional_parts, reverse=True):
                if leftover <= 0:
                    break
                sizes[name] += 1
                leftover -= 1

        return sizes["train"], sizes["val"], sizes["test"]

    def _initialize_optimizer(self):
        self.optimizer = instantiate_from_config(
            self.cfg.optimizer,
            self.model.parameters(),
        )

    def _initialize_wandb(self, **wandb_kwargs):
        print("==> Starting a new WANDB run")
        new_tags = (self.cfg.dataset.name,)
        wandb_kwargs = {
            "tags": new_tags,
            "name": self.cfg.run_name,
            "entity": self.cfg.entity,
            **wandb_kwargs,
        }

        wandb.init(
            project=self.cfg.project,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            **wandb_kwargs,
        )
        print("==> Initialized Run")

    def _initialize_lr_scheduler(self):
        global_batch_size = self.batchsize
        total_steps = len(self.dataloaders["train"].dataset) // global_batch_size * self.epochs
        warmup_ratio = self.cfg.scheduler.args.warmup_ratio
        warmup_step = int(warmup_ratio * total_steps)

        self.lr_scheduler = get_scheduler(
            name=self.cfg.scheduler.name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=total_steps,
        )

    def _instantiate_loss(self):
        loss_kwargs = {}  # Placeholder in case we need special loss function parameters/shapes
        self.loss_function = instantiate_from_config(self.cfg.loss, **loss_kwargs)

    def _resolve_lightning_devices(self, accelerator: str):
        if accelerator == "cpu" and self.devices == "auto":
            return 1
        return self.devices

    def _validate_lightning_strategy(self, accelerator: str) -> None:
        if not isinstance(self.strategy, str) or not self.strategy.startswith("deepspeed"):
            return
        suppress_deepspeed_probe_logging()
        if accelerator != "gpu":
            raise ValueError("DeepSpeed strategy requires `accelerator=gpu`.")
        try:
            import deepspeed  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "DeepSpeed strategy was requested but the `deepspeed` package is not installed. "
                "Install it with `pip install -e .[deepspeed]`.",
            ) from exc

    def _build_lightning_trainer(self) -> Trainer:
        accelerator = infer_accelerator(self.device, self.accelerator)
        self._validate_lightning_strategy(accelerator)
        logger = build_wandb_logger(self.cfg, self.run_wandb) if self.run_wandb else False
        callbacks = build_lightning_callbacks(self.results_folder, enable_lr_monitor=bool(logger))

        return Trainer(
            default_root_dir=str(self.results_folder),
            max_epochs=self.epochs,
            accelerator=accelerator,
            devices=self._resolve_lightning_devices(accelerator),
            strategy=self.strategy,
            precision=self.precision,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=self.grad_norm_clip,
            accumulate_grad_batches=self.accumulate_grad_batches,
            log_every_n_steps=self.log_every_n_steps,
            enable_progress_bar=self.enable_progress_bar,
            enable_model_summary=self.enable_model_summary,
            num_sanity_val_steps=self.num_sanity_val_steps,
        )

    def _prepare_resume_checkpoint(self, resume_from_checkpoint: bool) -> str | None:
        self._legacy_resume_epoch = 0
        if not resume_from_checkpoint:
            return None

        resume_path = get_resume_checkpoint_path(self.results_folder)
        if resume_path is None:
            return None

        if resume_path.endswith(".ckpt"):
            print(f"> Resuming Lightning checkpoint from {resume_path}")
            return resume_path

        print(f"> Loading legacy checkpoint state from {resume_path}")
        self._legacy_resume_epoch = self.load_checkpoint()
        print("> Loaded legacy trainer state; continuing under Lightning without loop-state resume.")
        return None

    def save_checkpoint(self, epoch):
        """Save model checkpoint at the given epoch."""
        self.initialize_checkpointing()

        data = {
            "epoch": epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # Save checkpoint
        checkpoint_path = self.results_folder / f"model-{epoch}.pt"
        torch.save(data, str(checkpoint_path))
        print(f"> Saved checkpoint to {checkpoint_path}")

        # Overwrite 'milestone.txt' with the new milestone
        milestone_file = self.results_folder / "milestone.txt"
        with open(milestone_file, "w") as f:
            f.write(str(epoch))
        print(f"> Updated milestone.txt to milestone {epoch}")

        config_path = self.results_folder / "config.txt"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

    def load_trainer_state(self, data):
        self.optimizer.load_state_dict(data["optimizer"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])
        restore_rng_state(data, rng=self.rng)

        epoch = data["epoch"]
        self.step = data["step"]

        return epoch

    def get_pretrained_load_path(self):
        config = self.cfg
        load_path = None
        if "pretrained_ckpt_path" in config:
            load_path = _resolve_checkpoint_candidate(config.pretrained_ckpt_path)
            if load_path is None:
                print(
                    f"> Checkpoint file {config.pretrained_ckpt_path} does not exist. Check the value of "
                    f"`{config.pretrained_ckpt_path=}` for correctness.",
                )
                return None

        return load_path

    def load_pretrained(self):
        load_path = self.get_pretrained_load_path()
        if load_path is None:
            return

        print(f"> Loading pretrained model state from {load_path}")

        data = torch.load(str(load_path), map_location=self.device, weights_only=False)
        model_state = extract_submodule_state_dict(data, legacy_key="model", lightning_prefix="model.")
        self.model.load_state_dict(model_state, strict=False)

        trainer_state_keys = {
            "optimizer",
            "lr_scheduler",
            "python_rng_state",
            "numpy_rng_state",
            "torch_rng_state",
            "cuda_rng_state_all",
            "epoch",
            "step",
        }
        if isinstance(data, dict) and trainer_state_keys.issubset(data):
            self.load_trainer_state(data)
        elif isinstance(data, dict):
            _, self.step = extract_epoch_and_step(data)

        print(f">Finished loading pretrained params loaded from {load_path}")

    def load_checkpoint(self):
        """Load the most recent checkpoint."""

        # Ensure results folder is initialized
        self.initialize_checkpointing()

        milestone_file = self.results_folder / "milestone.txt"
        if not milestone_file.exists():
            print("> No milestone.txt found. Starting from scratch.")
            return 0

        # Read the milestone number
        with open(milestone_file) as f:
            milestone_str = f.read().strip()
            if not milestone_str.isdigit():
                print("milestone.txt is invalid. Starting from scratch.")
                return 0
            milestone = int(milestone_str)

        # Load the checkpoint
        load_path = self.results_folder / f"model-{milestone}.pt"
        if not load_path.exists():
            print(f"> Checkpoint file {load_path} does not exist. Starting from scratch.")
            return 0

        print(f"> Loading checkpoint from {load_path}")

        data = torch.load(str(load_path), map_location=self.device, weights_only=False)

        self.model.load_state_dict(data["model"])
        epoch = self.load_trainer_state(data)

        print(f"> Resumed from epoch {epoch + 1}, step {self.step}")

        return epoch + 1

    def get_checkpoint_directory(self, additional_keys: tuple = (), hash_vars: tuple = ()):
        cache_dir = Path(self.cfg.cache_dir)
        keys = self.CHECKPOINT_KEYS + additional_keys
        checkpoint_directory = get_fully_qualified_cache_paths(
            self.cfg,
            cache_dir / "checkpoints",
            keys=keys,
            hash_vars=hash_vars,
            mkdir=False,
        )

        return checkpoint_directory

    def initialize_checkpointing(self, additional_keys: tuple = (), hash_vars: tuple = ()):
        """Initialize checkpoint directory."""
        self.results_folder = self.get_checkpoint_directory(additional_keys=additional_keys, hash_vars=hash_vars)

        try:
            self.results_folder.mkdir(parents=True, exist_ok=False)
            print(f"> Checkpoint directory initialized at {self.results_folder}")
        except FileExistsError:
            print(f"> Checkpoint directory already exists at {self.results_folder}")

    def get_outputs_and_loss(self, inputs: Data):
        """Run batch through model.

        Args:
            inputs: a PyTorch Geometric dataset.

        """
        outputs = self.model(inputs)
        loss = self.loss_function(inputs, outputs)

        return outputs, loss

    def iterate_dataloader(
        self,
        dataloader,
        loss=None,
        epoch=None,
        keys_to_keep: list[str] | None = None,
        log_every: int = 1,
    ):
        """Iterate through `DataLoader` (fo training or validation)."""
        training = epoch is not None

        if training:
            step = len(dataloader) * epoch
        else:
            step = 0

        if keys_to_keep is None:
            keys_to_keep = []

        outputs = defaultdict(list)
        with tqdm(dataloader) as pbar:
            for batch in pbar:
                batch = batch.to(self.device)
                step += 1

                lr = self.lr_scheduler.get_last_lr()[0]

                batch_outputs, loss = self.get_outputs_and_loss(batch)
                for key in keys_to_keep:
                    outputs[key].extend(batch_outputs[key].detach().cpu().numpy())
                del batch_outputs

                # batch = batch.to("cpu")
                del batch

                if training:
                    loss.backward()

                    loss = loss.detach().cpu().item()

                    grad_norm = clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_norm_clip,
                    )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1

                    log = {
                        "train_loss": loss,
                        "global_step": self.step,
                        "learning_rate": lr,
                        "epoch": epoch,
                        "grad_norm": grad_norm.item(),
                    }
                    if self.run_wandb:
                        wandb.log(log)

                    pbar.set_description(
                        f"Epoch: {epoch} "
                        f"Step {self.step} "
                        f"Loss: {loss:.4f} "
                        f"LR: {lr:.1e} "
                        f"grad_norm: {grad_norm:.4f} ",
                    )

                    loss = None
                else:
                    loss = loss.detach().cpu().item()
                    pbar.set_description(
                        f"Batch loss: {loss:.4f} ",
                    )

        return outputs, loss

    def validation_epoch(self, dataloader, dataloader_type):
        """Validate model (without updating model weights)."""
        self.model.eval()

        if len(dataloader) == 0:
            raise ValueError("`DataLoader` length cannot be zero. Check custom sampler implementation.")

        with torch.no_grad():
            _, loss = self.iterate_dataloader(dataloader)

        torch.cuda.empty_cache()

        log = {
            f"{dataloader_type}_loss": loss,
        }
        if self.run_wandb:
            wandb.log(log)
        else:
            print(f"log = {pformat(log)}")

        return log

    def train_epoch(self, epoch):
        """Run one epoch of training."""
        self.model.train()

        self.iterate_dataloader(
            self.dataloaders["train"],
            epoch=epoch,
        )

    def warmup_dataloaders(self) -> None:
        """Warm one batch from each dataloader without updating training
        state."""
        if len(self.dataloaders["train"]) == 0:
            return

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for split, dataloader in self.dataloaders.items():
            if len(dataloader) == 0:
                continue

            batch = next(iter(dataloader))
            batch = batch.to(self.device)
            batch_outputs, loss = self.get_outputs_and_loss(batch)
            if split == "train":
                loss.backward()

            del batch_outputs
            del loss
            del batch

        self.optimizer.zero_grad(set_to_none=True)

        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def fit(
        self,
        resume_from_checkpoint=True,
        start_epoch=0,
    ):
        """Fit the model to the train DataLoader."""
        del start_epoch

        self.last_fit_reused_checkpoint = False
        self.last_fit_skipped_training = False

        self.initialize_checkpointing()
        write_config_snapshot(self.results_folder, self.cfg)

        if self.epochs <= 0:
            self.last_fit_skipped_training = True
            return

        ckpt_path = self._prepare_resume_checkpoint(resume_from_checkpoint)
        self.last_fit_reused_checkpoint = bool(ckpt_path) or self._legacy_resume_epoch > 0
        if ckpt_path is None and self._legacy_resume_epoch >= self.epochs:
            self.last_fit_skipped_training = True
            return
        self.lightning_module = _PyGLightningModule(self)
        self.lightning_trainer = self._build_lightning_trainer()

        try:
            self.lightning_trainer.fit(
                self.lightning_module,
                train_dataloaders=self.dataloaders["train"],
                val_dataloaders=[self.dataloaders["val"], self.dataloaders["test"]],
                ckpt_path=ckpt_path,
            )
        finally:
            finalize_lightning_logger(self.lightning_trainer.logger if self.lightning_trainer is not None else None)
            cleanup_distributed()


def setup_trainer(config):
    # Inferring num_genes
    data_directory = Path(config.dataset.args.data_directory)
    first_filepath = next(data_directory.glob("*.h5ad"))
    first_sample = ad.read_h5ad(first_filepath)
    _, num_genes = first_sample.shape

    data = instantiate_from_config(config.dataset)
    model = instantiate_from_config(config.model, in_dim=num_genes)
    trainer = instantiate_from_config(
        config.trainer,
        cfg=config,
        model=model,
        data=data,
    )
    trainer.load_pretrained()

    return trainer

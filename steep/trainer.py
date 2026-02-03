import random
from pathlib import Path
from pprint import pformat

import anndata as ad
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from steep.utils import get_fully_qualified_cache_paths, instantiate_from_config


class PyGTrainer:
    """PyG trainer."""

    CHECKPOINT_KEYS = ("model", "dataset.args.data_directory")

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
        grad_norm_clip: float = 1.0,
        run_wandb: bool = False,
    ):
        self.cfg = cfg
        self.model = model

        self.batchsize = batchsize
        self.epochs = epochs
        self.device = device
        self.model.to(self.device)

        self.shuffle = shuffle
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio

        self.random_seed = random_seed
        self.rng = torch.Generator().manual_seed(random_seed)

        self.data = data
        self.step = 0

        self.grad_norm_clip = grad_norm_clip
        self._initialize_optimizer()
        self._initialize_lr_scheduler()

        self.run_wandb = run_wandb
        if run_wandb:
            self._initialize_wandb()

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
        total_size = len(self._data)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            self._data,
            [train_size, val_size, test_size],
            generator=self.rng,
        )
        self.datasets = {
            "train": train_dataset,
            "test": test_dataset,
            "val": val_dataset,
        }
        self.dataloaders = {}
        for split, split_dataset in self.datasets.items():
            dataloader = DataLoader(
                split_dataset,
                batch_size=self.batchsize,
                shuffle=self.shuffle,
            )
            self.dataloaders[split] = dataloader
            print(f"{split} DataLoader has size {len(dataloader)}")

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

    def fit(
        self,
        resume_from_checkpoint=True,
        start_epoch=0,
    ):
        """Train the model with automatic checkpointing and resumption."""
        # Try to resume from checkpoint if requested
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint()

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
        # Restore optimizer and scheduler states

        # clean_opt_sd = clean_optimizer_state_for_current_model(data["optimizer"], self.optimizer, verbose=True)
        # self.optimizer.load_state_dict(clean_opt_sd)
        self.optimizer.load_state_dict(data["optimizer"])

        self.lr_scheduler.load_state_dict(data["lr_scheduler"])

        # Restore random states
        random.setstate(data["python_rng_state"])
        np.random.set_state(data["numpy_rng_state"])

        # Handle torch RNG state
        self.rng.set_state(data["torch_rng_state"].cpu())
        # if isinstance(torch_rng_state, torch.Tensor) and torch_rng_state.device.type != "cpu":
        #     torch_rng_state = torch_rng_state.cpu()
        # torch.random.set_rng_state(torch_rng_state)

        # Handle CUDA RNG states
        if data["cuda_rng_state_all"] is not None and torch.cuda.is_available():
            num_visible_devices = torch.cuda.device_count()
            if len(data["cuda_rng_state_all"]) != num_visible_devices:
                print(
                    "Warning: Number of visible CUDA devices does not match the number of saved CUDA RNG states. "
                    "Skipping CUDA RNG state restoration.",
                )
            else:
                new_cuda_states = []
                for state in data["cuda_rng_state_all"]:
                    if isinstance(state, torch.Tensor) and state.device.type != "cpu":
                        state = state.cpu()
                    new_cuda_states.append(state)
                torch.cuda.set_rng_state_all(new_cuda_states)

        epoch = data["epoch"]
        self.step = data["step"]

        return epoch

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

    def get_loss(self, batch):
        """Run batch through model."""
        _, output = self.model(batch.x, batch.edge_index)
        loss = F.mse_loss(batch.x, output)
        return loss

    def iterate_dataloader(
        self,
        dataloader,
        loss=None,
        epoch=None,
        log_every: int = 1,
    ):
        """Iterate through `DataLoader` (fo training or validation)."""
        training = epoch is not None

        if training:
            step = len(dataloader) * epoch
        else:
            step = 0

        with tqdm(dataloader) as pbar:
            for batch in pbar:
                batch = batch.to(self.device)
                step += 1

                is_logging = step % log_every == 0
                lr = self.lr_scheduler.get_last_lr()[0]

                loss = self.get_loss(batch)
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

        return loss

    def validation_epoch(self, dataloader, dataloader_type):
        """Validate model (without updating model weights)."""
        self.model.eval()

        if len(dataloader) == 0:
            raise ValueError("`DataLoader` length cannot be zero. Check custom sampler implementation.")

        with torch.no_grad():
            loss = self.iterate_dataloader(dataloader)

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

    def fit(
        self,
        resume_from_checkpoint=True,
        start_epoch=0,
    ):
        """Fit the model to the train DataLoader."""
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint()
        if start_epoch >= self.epochs:
            return

        for epoch in range(start_epoch, self.epochs):
            # Validation and test evaluation
            valid_log = self.validation_epoch(self.dataloaders["val"], "val")
            test_log = self.validation_epoch(self.dataloaders["test"], "test")

            # self.save_checkpoint(epoch)

            # Train for one epoch
            self.train_epoch(epoch)

            self.save_checkpoint(epoch)
            print(f"> Saved regular checkpoint at epoch {epoch}")

        if self.run_wandb:
            wandb.finish()


def setup_trainer(config):
    # Inferring num_genes
    data_directory = Path(config.dataset.args.data_directory)
    first_filepath = next(data_directory.iterdir())
    first_sample = ad.read_h5ad(first_filepath)
    _, num_genes = first_sample.shape

    data = instantiate_from_config(config.dataset)
    model = instantiate_from_config(config.model, in_dim=num_genes, out_dim=num_genes)
    trainer = instantiate_from_config(
        config.trainer,
        cfg=config,
        model=model,
        data=data,
    )

    return trainer

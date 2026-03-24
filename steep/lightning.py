import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf


def infer_accelerator(device: str | None = None, accelerator: str | None = None) -> str:
    if accelerator not in (None, "auto"):
        return accelerator
    if device in {"cuda", "gpu"}:
        return "gpu"
    if device in {"cpu", "mps", "tpu"}:
        return device
    return "auto"


def default_dataloader_num_workers(max_workers: int = 8) -> int:
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    available_cpus = None
    if slurm_cpus_per_task is not None:
        try:
            available_cpus = int(slurm_cpus_per_task)
        except ValueError:
            available_cpus = None

    if available_cpus is None:
        available_cpus = os.cpu_count() or 1

    if available_cpus <= 1:
        return 0

    return min(max_workers, max(1, available_cpus - 1))


def build_wandb_logger(cfg: OmegaConf, enabled: bool):
    if not enabled:
        return None

    extra_tags = OmegaConf.select(cfg, "wandb_tags", default=[]) or []
    if isinstance(extra_tags, str):
        extra_tags = [extra_tags]
    tags = [cfg.dataset.name, *extra_tags]

    return WandbLogger(
        project=cfg.project,
        name=cfg.run_name,
        entity=cfg.entity,
        group=OmegaConf.select(cfg, "wandb_group", default=None),
        job_type=OmegaConf.select(cfg, "wandb_job_type", default=None),
        tags=tuple(dict.fromkeys(tags)),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def finalize_lightning_logger(logger, status: str = "success") -> None:
    if logger is None:
        return

    finalize = getattr(logger, "finalize", None)
    if callable(finalize):
        finalize(status)


def cleanup_distributed() -> None:
    deepspeed_comm = None
    deepspeed_groups = None
    try:
        import deepspeed.comm as deepspeed_comm
        import deepspeed.utils.groups as deepspeed_groups
    except ImportError:
        pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if deepspeed_comm is not None:
            try:
                deepspeed_comm.destroy_process_group()
            except Exception:
                torch.distributed.destroy_process_group()
        else:
            torch.distributed.destroy_process_group()

    if deepspeed_comm is not None and hasattr(deepspeed_comm, "cdb"):
        deepspeed_comm.cdb = None

    if deepspeed_groups is None:
        return

    for attr in (
        "_WORLD_GROUP",
        "_ZERO_PARAM_INTRA_PARALLEL_GROUP",
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_MODEL_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP",
        "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE",
        "_MPU_TENSOR_MODEL_PARALLEL_RANK",
        "mesh_device",
        "mpu",
    ):
        if hasattr(deepspeed_groups, attr):
            setattr(deepspeed_groups, attr, None)

    for attr in (
        "_EXPERT_PARALLEL_GROUP",
        "_EXPERT_PARALLEL_GROUP_RANKS",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP_RANKS",
        "_ALL_TO_ALL_GROUP",
    ):
        if hasattr(deepspeed_groups, attr):
            setattr(deepspeed_groups, attr, {})

    if hasattr(deepspeed_groups, "expert_tensor_parallel_world_size"):
        deepspeed_groups.expert_tensor_parallel_world_size = 1


def build_lightning_callbacks(results_folder: Path, enable_lr_monitor: bool = True) -> list:
    callbacks = [
        ModelCheckpoint(
            dirpath=str(results_folder),
            filename="epoch{epoch:04d}",
            auto_insert_metric_name=False,
            save_last=True,
            save_top_k=-1,
            every_n_epochs=1,
        ),
        ModelCheckpoint(
            dirpath=str(results_folder),
            filename="best",
            auto_insert_metric_name=False,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]
    if enable_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


def write_config_snapshot(results_folder: Path, cfg: OmegaConf) -> None:
    config_path = results_folder / "config.txt"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def capture_rng_state() -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(checkpoint: dict[str, Any], rng: torch.Generator | None = None) -> None:
    python_rng_state = checkpoint.get("python_rng_state")
    if python_rng_state is not None:
        random.setstate(python_rng_state)

    numpy_rng_state = checkpoint.get("numpy_rng_state")
    if numpy_rng_state is not None:
        np.random.set_state(numpy_rng_state)

    torch_rng_state = checkpoint.get("torch_rng_state")
    if torch_rng_state is not None:
        torch_rng_state = torch_rng_state.cpu()
        if rng is not None:
            rng.set_state(torch_rng_state)
        torch.random.set_rng_state(torch_rng_state)

    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if cuda_rng_state_all is None or not torch.cuda.is_available():
        return

    num_visible_devices = torch.cuda.device_count()
    if len(cuda_rng_state_all) != num_visible_devices:
        print(
            "Warning: Number of visible CUDA devices does not match the number of saved CUDA RNG states. "
            "Skipping CUDA RNG state restoration.",
        )
        return

    new_cuda_states = []
    for state in cuda_rng_state_all:
        if isinstance(state, torch.Tensor) and state.device.type != "cpu":
            state = state.cpu()
        new_cuda_states.append(state)
    torch.cuda.set_rng_state_all(new_cuda_states)


def get_resume_checkpoint_path(results_folder: Path) -> str | None:
    last_checkpoint = results_folder / "last.ckpt"
    if last_checkpoint.exists():
        return str(last_checkpoint)

    milestone_file = results_folder / "milestone.txt"
    if not milestone_file.exists():
        return None

    with open(milestone_file) as f:
        milestone_str = f.read().strip()
        if not milestone_str.isdigit():
            return None

    legacy_checkpoint = results_folder / f"model-{int(milestone_str)}.pt"
    if legacy_checkpoint.exists():
        return str(legacy_checkpoint)

    return None


def extract_submodule_state_dict(
    checkpoint: Any,
    legacy_key: str,
    lightning_prefix: str,
):
    if not isinstance(checkpoint, dict):
        return checkpoint

    if legacy_key in checkpoint:
        return checkpoint[legacy_key]

    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, dict):
        extracted_state = {
            key[len(lightning_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(lightning_prefix)
        }
        if extracted_state:
            return extracted_state

    return checkpoint


def extract_epoch_and_step(checkpoint: dict[str, Any]) -> tuple[int, int]:
    epoch = int(checkpoint.get("epoch", 0))
    step = int(checkpoint.get("step", checkpoint.get("global_step", 0)))
    return epoch, step

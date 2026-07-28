import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _constant_schedule_with_warmup_lambda(current_step: int, *, num_warmup_steps: int) -> float:
    if num_warmup_steps > 0 and current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return 1.0


def _linear_schedule_with_warmup_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
) -> float:
    if num_warmup_steps > 0 and current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    remaining_steps = num_training_steps - current_step
    decay_steps = max(1, num_training_steps - num_warmup_steps)
    return max(0.0, float(remaining_steps) / float(decay_steps))


def _cosine_schedule_with_warmup_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> float:
    if num_warmup_steps > 0 and current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int = 0,
    num_training_steps: int | None = None,
):
    normalized_name = name.lower()

    if normalized_name == "constant":
        lr_lambda = lambda current_step: _constant_schedule_with_warmup_lambda(  # noqa: E731
            current_step,
            num_warmup_steps=num_warmup_steps,
        )
        return LambdaLR(optimizer, lr_lambda)

    if num_training_steps is None:
        raise ValueError(f"`num_training_steps` is required for scheduler '{name}'.")

    if normalized_name == "linear":
        lr_lambda = lambda current_step: _linear_schedule_with_warmup_lambda(  # noqa: E731
            current_step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return LambdaLR(optimizer, lr_lambda)

    if normalized_name == "cosine":
        lr_lambda = lambda current_step: _cosine_schedule_with_warmup_lambda(  # noqa: E731
            current_step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unsupported scheduler name: {name}")

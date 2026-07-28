import torch

from steep.utils import get_scheduler


def _optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.AdamW([parameter], lr=1.0)


def test_constant_scheduler_with_zero_warmup_stays_constant():
    optimizer = _optimizer()
    scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=10)

    values = [scheduler.lr_lambdas[0](step) for step in range(5)]

    assert values == [1.0] * 5


def test_linear_scheduler_decays_to_zero():
    optimizer = _optimizer()
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=4)

    values = [scheduler.lr_lambdas[0](step) for step in range(5)]

    assert values == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_cosine_scheduler_warmup_then_decay():
    optimizer = _optimizer()
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=2, num_training_steps=6)

    values = [scheduler.lr_lambdas[0](step) for step in range(6)]

    assert values[0] == 0.0
    assert values[1] == 0.5
    assert values[2] == 1.0
    assert 0.0 <= values[-1] <= 1.0

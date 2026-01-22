import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture
def mock_dim():
    return 16


@pytest.fixture
def torch_rng():
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)

    return rng


@pytest.fixture
def mock_data(mock_dim, torch_rng):
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1]],
        dtype=torch.long,
    )

    x = torch.rand((3, mock_dim), device="cpu", generator=torch_rng)

    return Data(x=x, edge_index=edge_index.t().contiguous())

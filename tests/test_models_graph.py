import pytest

from steep.models import STAGATE, STAGATEVAE


@pytest.fixture
def stagate(mock_dim):
    in_dim = mock_dim
    num_hidden = mock_dim
    d_model = 2

    model = STAGATE(
        in_dim=in_dim,
        num_hidden=num_hidden,
        d_model=d_model,
    )

    return model


@pytest.fixture
def stagate_vae(mock_dim):
    in_dim = mock_dim
    num_hidden = mock_dim
    d_model = 2

    model = STAGATEVAE(
        in_dim=in_dim,
        num_hidden=num_hidden,
        d_model=d_model,
    )

    return model


def test_stagate(stagate, mock_dim, mock_data):
    num_nodes, in_dim = mock_data.x.size()
    outputs = stagate(mock_data)
    assert outputs["embedding"].size() == (num_nodes, 2)
    assert outputs["logits"].size() == (num_nodes, mock_dim)


def test_stagate_vae(stagate_vae, mock_dim, mock_data):
    num_nodes, in_dim = mock_data.x.size()
    outputs = stagate_vae(mock_data)
    assert outputs["embedding"].size() == (num_nodes, 2)
    assert outputs["mean"].size() == (num_nodes, 2)
    assert outputs["logvar"].size() == (num_nodes, 2)
    assert outputs["logits"].size() == (num_nodes, mock_dim)

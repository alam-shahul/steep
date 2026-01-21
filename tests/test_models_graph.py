import pytest

from steep.models import STAGATE


@pytest.fixture
def stagate(mock_dim):
    in_dim = mock_dim
    num_hidden = mock_dim
    out_dim = 2

    model = STAGATE(
        in_dim=in_dim,
        num_hidden=num_hidden,
        out_dim=out_dim,
    )

    return model


def test_stagate(stagate, mock_dim, mock_data):
    num_nodes, in_dim = mock_data.x.size()
    node_embeddings, reconstruction = stagate(mock_data.x, mock_data.edge_index)
    assert node_embeddings.size() == (num_nodes, 2)
    assert reconstruction.size() == (num_nodes, mock_dim)

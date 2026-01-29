import os

import anndata as ad
import numpy as np
import pytest
import torch

# from Heimdall.utils import convert_to_ensembl_ids # Exclude Heimdall dependency for now
from pytest import fixture
from scipy.sparse import csr_array
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


@fixture(scope="module")
def gene_names():
    return ["A1BG", "A1CF", "fake_gene", "A2M"]


@fixture
def mock_anndata(gene_names):

    mock_expression = csr_array(
        np.array(
            [
                [1, 4, 3, 2],
                [2, 1, 4, 3],
                [3, 2, 1, 4],
                [4, 3, 2, 1],
            ],
        ),
    )
    num_cells, num_genes = mock_expression.shape

    rng = np.random.default_rng(0)
    mock_dataset = ad.AnnData(X=mock_expression)
    mock_dataset.var_names = gene_names
    mock_dataset.obsm["spatial"] = rng.random((num_cells, 2))
    # convert_to_ensembl_ids(mock_dataset, data_dir=os.environ["DATA_PATH"])

    return mock_dataset


@fixture(scope="session")
def session_cache_dir(tmp_path_factory):
    # Create the directory using tmp_path_factory
    cache_dir = tmp_path_factory.mktemp("cache")
    yield cache_dir

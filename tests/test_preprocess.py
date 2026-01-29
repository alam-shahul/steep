import pytest

from steep.utils import anndata_to_pyg, compute_spatial_neighbors, construct_spatial_graph


def test_construct_spatial_graph(mock_anndata):
    assert "adjacency_matrix" not in mock_anndata.obsp
    construct_spatial_graph(mock_anndata, k=2)
    assert "adjacency_matrix" in mock_anndata.obsp


def test_compute_spatial_neighbors(mock_anndata):
    assert "adjacency_matrix" not in mock_anndata.obsp
    compute_spatial_neighbors(mock_anndata)
    assert "adjacency_matrix" in mock_anndata.obsp


def test_anndata_to_pyg(mock_anndata):
    compute_spatial_neighbors(mock_anndata)
    data = anndata_to_pyg(mock_anndata)
    print(data.x.size)
    print(data.edge_index)

import anndata as ad

from steep.utils._graph import compute_spatial_neighbors


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    compute_spatial_neighbors(adata)
    return adata

from pathlib import Path

import anndata as ad
import scanpy as sc
from torch_geometric.data import Dataset

from steep.utils import anndata_to_pyg


class SRTDataset(Dataset):
    def __init__(self, data_directory: str, transform=None):
        super().__init__(None, transform=transform)
        self.data_directory = Path(data_directory)
        self.data_paths = list(self.data_directory.iterdir())

    def len(self):  # noqa: A003
        return len(self.data_paths)

    def get(self, idx):
        adata = ad.read_h5ad(self.data_paths[idx])
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        data = anndata_to_pyg(adata)
        # print(f'{data.x.size()=}')
        # print(f'{data.edge_index.size()=}')
        return data

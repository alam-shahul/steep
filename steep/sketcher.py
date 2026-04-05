import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
from torch_geometric.data import Data

from steep.utils import hopper_sketch_indices, num_edges_from_adata

try:
    from steep.trainer import PyGTrainer
except ModuleNotFoundError as exc:
    _MOG_IMPORT_ERROR = exc

    class PyGTrainer:  # type: ignore[no-redef]
        pass

else:
    _MOG_IMPORT_ERROR = None


@dataclass
class SketchMetadata:
    original_num_cells: int
    sketched_num_cells: int
    compression_ratio: float
    original_num_edges: int | None = None
    sketched_num_edges: int | None = None
    original_disk_bytes: int | None = None
    sketched_disk_bytes: int | None = None
    sketch_time_seconds: float | None = None


class AnnDataSketcher(ABC):
    """Base API for sketchers that shrink processed AnnData objects."""

    def fit(self, adata: ad.AnnData) -> "AnnDataSketcher":
        del adata
        return self

    @abstractmethod
    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        """Return a smaller AnnData object."""

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        self.fit(adata)
        return self.transform(adata)

    def fit_transform_to_disk(self, input_path: str | Path, output_path: str | Path) -> SketchMetadata:
        input_path = Path(input_path)
        output_path = Path(output_path)

        start_time = time.perf_counter()
        backed_adata = ad.read_h5ad(input_path, backed="r")
        adata = backed_adata.to_memory()
        backed_adata.file.close()
        sketched_adata = self.fit_transform(adata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sketched_adata.write_h5ad(output_path)
        sketch_time_seconds = time.perf_counter() - start_time

        original_num_cells = int(adata.n_obs)
        sketched_num_cells = int(sketched_adata.n_obs)
        compression_ratio = sketched_num_cells / original_num_cells if original_num_cells > 0 else 0.0

        return SketchMetadata(
            original_num_cells=original_num_cells,
            sketched_num_cells=sketched_num_cells,
            compression_ratio=compression_ratio,
            original_num_edges=num_edges_from_adata(adata),
            sketched_num_edges=num_edges_from_adata(sketched_adata),
            original_disk_bytes=input_path.stat().st_size,
            sketched_disk_bytes=output_path.stat().st_size,
            sketch_time_seconds=sketch_time_seconds,
        )


class RandomSubsampleSketcher(AnnDataSketcher):
    """Uniformly sample a fraction of cells/spots from a processed AnnData."""

    def __init__(self, retention_ratio: float, random_seed: int = 0):
        if not 0 < retention_ratio <= 1:
            raise ValueError(f"`retention_ratio` must be in (0, 1], got {retention_ratio}.")
        self.retention_ratio = float(retention_ratio)
        self.random_seed = int(random_seed)

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        num_cells = int(adata.n_obs)
        if num_cells == 0:
            return adata.copy()

        sample_size = max(1, int(np.ceil(num_cells * self.retention_ratio)))
        if sample_size >= num_cells:
            return adata.copy()

        rng = np.random.default_rng(self.random_seed)
        selected_indices = np.sort(rng.choice(num_cells, size=sample_size, replace=False))
        return adata[selected_indices, :].copy()


class HopperSketcher(AnnDataSketcher):
    """Farthest-first sketching baseline over the expression matrix only."""

    def __init__(self, retention_ratio: float, random_seed: int = 0):
        if not 0 < retention_ratio <= 1:
            raise ValueError(f"`retention_ratio` must be in (0, 1], got {retention_ratio}.")
        self.retention_ratio = float(retention_ratio)
        self.random_seed = int(random_seed)

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        num_cells = int(adata.n_obs)
        if num_cells == 0:
            return adata.copy()

        sample_size = max(1, int(np.ceil(num_cells * self.retention_ratio)))
        if sample_size >= num_cells:
            return adata.copy()

        selected_indices = np.sort(
            hopper_sketch_indices(
                adata.X,
                num_points=sample_size,
                random_seed=self.random_seed,
            ),
        )
        return adata[selected_indices, :].copy()


class MoGSketcher(PyGTrainer):
    """Placeholder for training-time sketching algorithms such as MoG."""

    def __init__(self, *args, **kwargs):
        if _MOG_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "MoGSketcher requires the training stack dependencies to be installed.",
            ) from _MOG_IMPORT_ERROR
        super().__init__(*args, **kwargs)

    def get_outputs_and_loss(self, inputs: Data, training: bool):
        del inputs, training
        raise NotImplementedError("MoGSketcher is not implemented yet.")

    def _instantiate_loss(self):
        raise NotImplementedError("MoGSketcher is not implemented yet.")

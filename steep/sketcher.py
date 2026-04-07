import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
from geosketch import gs
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


class SpatialHopperSketcher(AnnDataSketcher):
    """Farthest-point sampling based on spatial coordinates (x, y).

    Ensures a spatially uniform coverage of the tissue.

    """

    def __init__(self, retention_ratio: float, random_seed: int = 0, spatial_key: str = "spatial"):
        super().__init__()
        self.retention_ratio = retention_ratio
        self.random_seed = random_seed
        self.spatial_key = spatial_key

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        num_cells = int(adata.n_obs)
        sample_size = max(1, int(np.ceil(num_cells * self.retention_ratio)))

        if sample_size >= num_cells:
            return adata.copy()

        if self.spatial_key not in adata.obsm:
            raise KeyError(f"Spatial coordinates not found in adata.obsm['{self.spatial_key}']")

        coords = adata.obsm[self.spatial_key]
        if sp.issparse(coords):
            coords = coords.toarray()

        selected_indices = np.sort(
            hopper_sketch_indices(
                coords,
                num_points=sample_size,
                random_seed=self.random_seed,
            ),
        )
        return adata[selected_indices, :].copy()


class GeoSketcher(AnnDataSketcher):
    """Geometric sketching to sample uniformly from high-dimensional space."""

    def __init__(self, retention_ratio: float, random_seed: int = 0):
        self.retention_ratio = retention_ratio
        self.random_seed = random_seed

    def transform(self, adata: ad.AnnData) -> ad.AnnData:

        X = adata.obsm["X_pca"] if "X_pca" in adata.obsm else adata.X
        num_cells = adata.n_obs
        sample_size = max(1, int(np.ceil(num_cells * self.retention_ratio)))

        selected_indices = gs(X, sample_size, seed=self.random_seed)
        return adata[np.sort(selected_indices), :].copy()


class LeverageScoreSketcher(AnnDataSketcher):
    """Samples cells based on the statistical leverage scores in PCA space."""

    def __init__(self, retention_ratio: float, random_seed: int = 0, n_components: int = 50):
        self.retention_ratio = retention_ratio
        self.random_seed = random_seed
        self.n_components = n_components

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        if "X_pca" not in adata.obsm:
            raise KeyError("Please run PCA (sc.tl.pca) before using LeverageScoreSketcher.")

        U = adata.obsm["X_pca"][:, : self.n_components]
        leverage_scores = np.sum(U**2, axis=1)
        probs = leverage_scores / np.sum(leverage_scores)

        num_cells = adata.n_obs
        sample_size = max(1, int(np.ceil(num_cells * self.retention_ratio)))

        rng = np.random.default_rng(self.random_seed)
        selected_indices = np.sort(
            rng.choice(num_cells, size=sample_size, replace=False, p=probs),
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

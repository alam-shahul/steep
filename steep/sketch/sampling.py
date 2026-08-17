"""Offline node-sampling sketchers."""

import anndata as ad
import numpy as np
import scanpy as sc

from steep.sketch.base import AnnDataSketcher
from steep.utils import hopper_sketch_indices


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
        # if sp.issparse(coords):
        # coords = coords.toarray()

        selected_indices = np.sort(
            hopper_sketch_indices(
                coords,
                num_points=sample_size,
                random_seed=self.random_seed,
            ),
        )
        return adata[selected_indices, :].copy()


class JointHopperSketcher(AnnDataSketcher):
    """Farthest-first sketching considering both spatial and expression
    similarity."""

    def __init__(self, retention_ratio: float, alpha: float = 0.5, random_seed: int = 0):
        self.retention_ratio = retention_ratio
        self.alpha = alpha
        self.random_seed = random_seed

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        from sklearn.preprocessing import StandardScaler

        spatial = StandardScaler().fit_transform(adata.obsm["spatial"])
        sc.tl.pca(adata)
        pca = StandardScaler().fit_transform(adata.obsm["X_pca"])

        joint_features = np.hstack(
            [
                self.alpha * spatial,
                (1 - self.alpha) * pca,
            ],
        )

        num_cells = adata.n_obs
        sample_size = max(1, int(np.ceil(num_cells * self.retention_ratio)))

        indices = hopper_sketch_indices(
            joint_features,
            num_points=sample_size,
            random_seed=self.random_seed,
        )
        return adata[np.sort(indices), :].copy()


class GeoSketcher(AnnDataSketcher):
    """Geometric sketching to sample uniformly from high-dimensional space."""

    def __init__(self, retention_ratio: float, random_seed: int = 0):
        self.retention_ratio = retention_ratio
        self.random_seed = random_seed

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        from geosketch import gs

        sc.tl.pca(adata)
        features = adata.obsm["X_pca"]
        num_cells = adata.n_obs
        sample_size = max(1, int(np.ceil(num_cells * self.retention_ratio)))

        selected_indices = gs(features, sample_size, seed=self.random_seed)
        return adata[np.sort(selected_indices), :].copy()


class LeverageScoreSketcher(AnnDataSketcher):
    """Samples cells based on the statistical leverage scores in PCA space."""

    def __init__(self, retention_ratio: float, random_seed: int = 0, n_components: int = 50):
        self.retention_ratio = retention_ratio
        self.random_seed = random_seed
        self.n_components = n_components

    def transform(self, adata: ad.AnnData) -> ad.AnnData:

        sc.tl.pca(adata)
        components = adata.obsm["X_pca"][:, : self.n_components]
        leverage_scores = np.sum(components**2, axis=1)
        probs = leverage_scores / np.sum(leverage_scores)

        num_cells = adata.n_obs
        sample_size = max(1, int(np.ceil(num_cells * self.retention_ratio)))

        rng = np.random.default_rng(self.random_seed)
        selected_indices = np.sort(
            rng.choice(num_cells, size=sample_size, replace=False, p=probs),
        )
        return adata[selected_indices, :].copy()

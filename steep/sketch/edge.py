"""Edge-scoring sketchers."""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import issparse

from steep.sketch.base import AnnDataSketcher, SketchMetadata


class EdgeScoreSketcherBase(AnnDataSketcher, ABC):
    """Base class for edge-only sketchers.

    Workflow:
    1. Keep all cells initially
    2. Score each undirected edge
    3. Keep top-k edges according to retention_ratio
    4. Rebuild adjacency matrix
    5. Optionally drop isolated cells

    Supports score caching:
    - compute edge scores once per slide / seed / config
    - reuse cached scores for multiple retention ratios

    """

    def __init__(
        self,
        retention_ratio: float,
        random_seed: int = 0,
        adjacency_matrix_key: str = "adjacency_matrix",
        spatial_key: str = "spatial",
        symmetrize: bool = True,
        drop_isolated_cells: bool = True,
        cache_scores: bool = True,
        cache_directory: str | None = None,
    ):
        if not 0 < retention_ratio <= 1:
            raise ValueError(f"`retention_ratio` must be in (0, 1], got {retention_ratio}.")
        self.retention_ratio = float(retention_ratio)
        self.random_seed = int(random_seed)
        self.adjacency_matrix_key = adjacency_matrix_key
        self.spatial_key = spatial_key
        self.symmetrize = bool(symmetrize)
        self.drop_isolated_cells = bool(drop_isolated_cells)
        self.cache_scores = bool(cache_scores)
        self.cache_directory = cache_directory

    def _to_numpy_dense(self, x):
        if issparse(x):
            x = x.toarray()
        return np.asarray(x)

    def _get_undirected_edges(self, adata: ad.AnnData):
        """Return unique undirected edges as (src, dst, values) with src <
        dst."""
        if self.adjacency_matrix_key not in adata.obsp:
            raise KeyError(f"Adjacency matrix not found in adata.obsp['{self.adjacency_matrix_key}'].")

        adj = adata.obsp[self.adjacency_matrix_key].tocsr()
        adj = adj.maximum(adj.T).tocsr()

        upper = sp.triu(adj, k=1).tocoo()
        src = upper.row.astype(np.int64)
        dst = upper.col.astype(np.int64)
        values = upper.data.astype(np.float32)

        return src, dst, values, adj.shape[0]

    def _normalize_01(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        return (x - x_min) / (x_max - x_min + 1e-12)

    def _topk_mask(self, scores: np.ndarray) -> np.ndarray:
        """Keep top retention_ratio fraction of edges."""
        num_edges = len(scores)
        if num_edges == 0:
            return np.zeros(0, dtype=bool)

        k = max(1, int(np.ceil(num_edges * self.retention_ratio)))
        k = min(k, num_edges)

        order = np.argsort(-scores, kind="mergesort")
        keep_idx = order[:k]

        mask = np.zeros(num_edges, dtype=bool)
        mask[keep_idx] = True
        return mask

    def _build_adjacency(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        num_nodes: int,
        edge_values: np.ndarray | None = None,
    ) -> sp.csr_matrix:
        """Build adjacency from undirected edge list."""
        if edge_values is None:
            edge_values = np.ones(len(src), dtype=np.float32)
        else:
            edge_values = np.asarray(edge_values, dtype=np.float32)

        adj = sp.csr_matrix(
            (edge_values, (src, dst)),
            shape=(num_nodes, num_nodes),
        )

        if self.symmetrize:
            adj = adj.maximum(adj.T).tocsr()

        return adj

    def _drop_isolated_cells(self, adata: ad.AnnData, adj: sp.csr_matrix) -> ad.AnnData:
        if adj.nnz == 0:
            raise RuntimeError("All edges were removed; no connected cells remain.")

        degree = np.asarray(adj.getnnz(axis=1)).ravel()
        keep_nodes = degree > 0

        if keep_nodes.all():
            out = adata.copy()
            out.obsp[self.adjacency_matrix_key] = adj
            return out

        out = adata[keep_nodes].copy()
        out.obsp[self.adjacency_matrix_key] = adj[keep_nodes][:, keep_nodes].tocsr()
        return out

    def _score_cache_key(self, adata: ad.AnnData, input_path: str | Path | None = None) -> str:
        """Cache key should depend on everything that affects edge scores, but
        NOT on retention_ratio."""
        signature = {
            "input_path": None if input_path is None else str(Path(input_path).resolve()),
            "random_seed": self.random_seed,
            "adjacency_matrix_key": self.adjacency_matrix_key,
            "spatial_key": self.spatial_key,
            "symmetrize": self.symmetrize,
            "drop_isolated_cells": self.drop_isolated_cells,
            "class_name": self.__class__.__name__,
            "score_params": self._score_signature(),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
        }
        raw = json.dumps(signature, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def _score_signature(self) -> dict:
        """Override if subclass has additional score-defining params."""
        return {}

    def _get_score_cache_path(self, adata: ad.AnnData, input_path: str | Path | None = None) -> Path | None:
        if not self.cache_scores or self.cache_directory is None:
            return None

        cache_dir = Path(self.cache_directory)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._score_cache_key(adata, input_path=input_path)
        return cache_dir / f"{cache_key}.npz"

    def _save_score_cache(
        self,
        cache_path: Path,
        src: np.ndarray,
        dst: np.ndarray,
        scores: np.ndarray,
        num_nodes: int,
    ) -> None:
        np.savez_compressed(
            cache_path,
            src=src.astype(np.int64),
            dst=dst.astype(np.int64),
            scores=scores.astype(np.float32),
            num_nodes=np.asarray([num_nodes], dtype=np.int64),
        )

    def _load_score_cache(self, cache_path: Path):
        data = np.load(cache_path)
        src = data["src"].astype(np.int64)
        dst = data["dst"].astype(np.int64)
        scores = data["scores"].astype(np.float32)
        num_nodes = int(data["num_nodes"][0])
        return src, dst, scores, num_nodes

    @abstractmethod
    def _score_edges(
        self,
        adata: ad.AnnData,
        src: np.ndarray,
        dst: np.ndarray,
        edge_values: np.ndarray,
    ) -> np.ndarray:
        """Return one score per undirected edge.

        Larger score = more likely to keep.

        """

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        return self._transform_impl(adata=adata, input_path=None)

    def _transform_impl(
        self,
        adata: ad.AnnData,
        input_path: str | Path | None = None,
    ) -> ad.AnnData:
        adata = adata.copy()

        src, dst, edge_values, num_nodes = self._get_undirected_edges(adata)

        cache_path = self._get_score_cache_path(adata, input_path=input_path)

        if cache_path is not None and cache_path.exists():
            cached_src, cached_dst, scores, cached_num_nodes = self._load_score_cache(cache_path)

            if cached_num_nodes != num_nodes:
                raise RuntimeError("Cached num_nodes does not match current graph.")
            if not (np.array_equal(cached_src, src) and np.array_equal(cached_dst, dst)):
                raise RuntimeError("Cached edge list does not match current graph.")
        else:
            scores = self._score_edges(adata, src, dst, edge_values)
            scores = np.asarray(scores, dtype=np.float32)

            if scores.shape[0] != src.shape[0]:
                raise ValueError(
                    f"Score length mismatch: got {scores.shape[0]} scores for {src.shape[0]} edges.",
                )

            if cache_path is not None:
                self._save_score_cache(
                    cache_path=cache_path,
                    src=src,
                    dst=dst,
                    scores=scores,
                    num_nodes=num_nodes,
                )

        keep = self._topk_mask(scores)

        new_src = src[keep]
        new_dst = dst[keep]

        new_values = np.ones(len(new_src), dtype=np.float32)

        new_adj = self._build_adjacency(
            src=new_src,
            dst=new_dst,
            num_nodes=num_nodes,
            edge_values=new_values,
        )

        if self.drop_isolated_cells:
            new_adata = self._drop_isolated_cells(adata, new_adj)
        else:
            new_adata = adata.copy()
            new_adata.obsp[self.adjacency_matrix_key] = new_adj

        orig_edges = int(adata.obsp[self.adjacency_matrix_key].nnz)
        final_edges = int(new_adata.obsp[self.adjacency_matrix_key].nnz)

        new_adata.uns["edge_sketcher_type"] = self.__class__.__name__
        new_adata.uns["edge_retention_ratio_target"] = float(self.retention_ratio)
        new_adata.uns["original_num_edges"] = orig_edges
        new_adata.uns["sketched_num_edges"] = final_edges
        new_adata.uns["original_num_cells"] = int(adata.n_obs)
        new_adata.uns["sketched_num_cells"] = int(new_adata.n_obs)

        return new_adata

    def fit_transform_to_disk(self, input_path: str | Path, output_path: str | Path) -> SketchMetadata:
        input_path = Path(input_path)
        output_path = Path(output_path)

        start_time = time.perf_counter()

        backed_adata = ad.read_h5ad(input_path, backed="r")
        adata = backed_adata.to_memory()
        backed_adata.file.close()

        sketched_adata = self._transform_impl(adata=adata, input_path=input_path)

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
            original_num_edges=int(adata.obsp[self.adjacency_matrix_key].nnz),
            sketched_num_edges=int(sketched_adata.obsp[self.adjacency_matrix_key].nnz),
            original_disk_bytes=input_path.stat().st_size,
            sketched_disk_bytes=output_path.stat().st_size,
            sketch_time_seconds=sketch_time_seconds,
        )


class RandomEdgeSketcher(EdgeScoreSketcherBase):
    """Randomly keep a fraction of edges."""

    def _score_edges(
        self,
        adata: ad.AnnData,
        src: np.ndarray,
        dst: np.ndarray,
        edge_values: np.ndarray,
    ) -> np.ndarray:
        num_edges = len(src)
        del adata, src, dst, edge_values
        rng = np.random.default_rng(self.random_seed)
        return rng.random(num_edges, dtype=np.float32)

    def _score_signature(self) -> dict:
        return {}


class SpatialShortEdgeSketcher(EdgeScoreSketcherBase):
    """Keep the shortest spatial edges."""

    def _score_edges(
        self,
        adata: ad.AnnData,
        src: np.ndarray,
        dst: np.ndarray,
        edge_values: np.ndarray,
    ) -> np.ndarray:
        del edge_values
        if self.spatial_key not in adata.obsm:
            raise KeyError(f"Spatial coordinates not found in adata.obsm['{self.spatial_key}'].")

        coords = np.asarray(adata.obsm[self.spatial_key], dtype=np.float32)
        dist = np.linalg.norm(coords[src] - coords[dst], axis=1)

        dist01 = self._normalize_01(dist)
        return 1.0 - dist01

    def _score_signature(self) -> dict:
        return {}


class SpatialLongEdgeSketcher(EdgeScoreSketcherBase):
    """Keep the longest spatial edges."""

    def _score_edges(
        self,
        adata: ad.AnnData,
        src: np.ndarray,
        dst: np.ndarray,
        edge_values: np.ndarray,
    ) -> np.ndarray:
        del edge_values
        if self.spatial_key not in adata.obsm:
            raise KeyError(f"Spatial coordinates not found in adata.obsm['{self.spatial_key}'].")

        coords = np.asarray(adata.obsm[self.spatial_key], dtype=np.float32)
        dist = np.linalg.norm(coords[src] - coords[dst], axis=1)

        # longer edge = larger score
        return self._normalize_01(dist)

    def _score_signature(self) -> dict:
        return {}


class ExpressionSimilarityEdgeSketcher(EdgeScoreSketcherBase):
    """Keep edges connecting expression-similar cells.

    By default uses PCA coordinates if available/needed.

    """

    def __init__(
        self,
        retention_ratio: float,
        random_seed: int = 0,
        adjacency_matrix_key: str = "adjacency_matrix",
        spatial_key: str = "spatial",
        symmetrize: bool = True,
        drop_isolated_cells: bool = True,
        expr_key: str | None = "X_pca",
        expr_dim: int = 16,
        cache_scores: bool = True,
        cache_directory: str | None = None,
    ):
        super().__init__(
            retention_ratio=retention_ratio,
            random_seed=random_seed,
            adjacency_matrix_key=adjacency_matrix_key,
            spatial_key=spatial_key,
            symmetrize=symmetrize,
            drop_isolated_cells=drop_isolated_cells,
            cache_scores=cache_scores,
            cache_directory=cache_directory,
        )
        self.expr_key = expr_key
        self.expr_dim = int(expr_dim)

    def _get_expr_features(self, adata: ad.AnnData) -> np.ndarray:
        if self.expr_key is not None:
            if self.expr_key == "X_pca":
                if "X_pca" not in adata.obsm:
                    sc.tl.pca(adata)
                z = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
            else:
                if self.expr_key not in adata.obsm:
                    raise KeyError(f"Expression features not found in adata.obsm['{self.expr_key}'].")
                z = np.asarray(adata.obsm[self.expr_key], dtype=np.float32)
        else:
            z = self._to_numpy_dense(adata.X).astype(np.float32)

        if z.shape[1] > self.expr_dim:
            z = z[:, : self.expr_dim]
        return z

    def _score_edges(
        self,
        adata: ad.AnnData,
        src: np.ndarray,
        dst: np.ndarray,
        edge_values: np.ndarray,
    ) -> np.ndarray:
        del edge_values
        z = self._get_expr_features(adata)

        z_src = z[src]
        z_dst = z[dst]

        num = np.sum(z_src * z_dst, axis=1)
        den = np.linalg.norm(z_src, axis=1) * np.linalg.norm(z_dst, axis=1) + 1e-12
        cos = num / den

        return ((cos + 1.0) / 2.0).astype(np.float32)

    def _score_signature(self) -> dict:
        return {
            "expr_key": self.expr_key,
            "expr_dim": self.expr_dim,
        }


class HybridSpatialExpressionEdgeSketcher(EdgeScoreSketcherBase):
    """
    Keep edges favored by a weighted combination of:
    - expression similarity
    - short spatial distance
    """

    def __init__(
        self,
        retention_ratio: float,
        alpha: float = 0.5,
        random_seed: int = 0,
        adjacency_matrix_key: str = "adjacency_matrix",
        spatial_key: str = "spatial",
        symmetrize: bool = True,
        drop_isolated_cells: bool = True,
        expr_key: str | None = "X_pca",
        expr_dim: int = 16,
        cache_scores: bool = True,
        cache_directory: str | None = None,
    ):
        super().__init__(
            retention_ratio=retention_ratio,
            random_seed=random_seed,
            adjacency_matrix_key=adjacency_matrix_key,
            spatial_key=spatial_key,
            symmetrize=symmetrize,
            drop_isolated_cells=drop_isolated_cells,
            cache_scores=cache_scores,
            cache_directory=cache_directory,
        )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"`alpha` must be in [0, 1], got {alpha}.")
        self.alpha = float(alpha)
        self.expr_key = expr_key
        self.expr_dim = int(expr_dim)

    def _get_expr_features(self, adata: ad.AnnData) -> np.ndarray:
        if self.expr_key is not None:
            if self.expr_key == "X_pca":
                if "X_pca" not in adata.obsm:
                    sc.tl.pca(adata)
                z = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
            else:
                if self.expr_key not in adata.obsm:
                    raise KeyError(f"Expression features not found in adata.obsm['{self.expr_key}'].")
                z = np.asarray(adata.obsm[self.expr_key], dtype=np.float32)
        else:
            z = self._to_numpy_dense(adata.X).astype(np.float32)

        if z.shape[1] > self.expr_dim:
            z = z[:, : self.expr_dim]
        return z

    def _score_edges(
        self,
        adata: ad.AnnData,
        src: np.ndarray,
        dst: np.ndarray,
        edge_values: np.ndarray,
    ) -> np.ndarray:
        del edge_values

        if self.spatial_key not in adata.obsm:
            raise KeyError(f"Spatial coordinates not found in adata.obsm['{self.spatial_key}'].")

        # expression similarity
        z = self._get_expr_features(adata)
        z_src = z[src]
        z_dst = z[dst]
        num = np.sum(z_src * z_dst, axis=1)
        den = np.linalg.norm(z_src, axis=1) * np.linalg.norm(z_dst, axis=1) + 1e-12
        expr_sim = (num / den + 1.0) / 2.0
        expr_sim = expr_sim.astype(np.float32)

        # short spatial distance score
        coords = np.asarray(adata.obsm[self.spatial_key], dtype=np.float32)
        dist = np.linalg.norm(coords[src] - coords[dst], axis=1)
        short_spatial_score = 1.0 - self._normalize_01(dist)

        return self.alpha * expr_sim + (1.0 - self.alpha) * short_spatial_score

    def _score_signature(self) -> dict:
        return {
            "alpha": self.alpha,
            "expr_key": self.expr_key,
            "expr_dim": self.expr_dim,
        }

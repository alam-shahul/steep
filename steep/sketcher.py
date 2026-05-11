import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from geosketch import gs

from steep.utils import hopper_sketch_indices, num_edges_from_adata

try:
    from steep.trainer import PyGTrainer
except ModuleNotFoundError as exc:
    _MOG_IMPORT_ERROR = exc

    class PyGTrainer:  # type: ignore[no-redef]
        pass

else:
    _MOG_IMPORT_ERROR = None

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse import issparse

from steep.models._sparsify import MoG


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


def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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

        sc.tl.pca(adata)
        X = adata.obsm["X_pca"]
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

        sc.tl.pca(adata)
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


class MoGSketcher(AnnDataSketcher):
    """MoG-based sketcher.

    Supports two modes:
    - edge: sparsify edges by learned edge score
    - node: aggregate learned edge scores into node scores, then subset cells

    Supports score caching:
    - train MoG once per slide / seed / config
    - reuse cached edge scores for multiple retention ratios

    """

    def __init__(
        self,
        mog_args: dict,
        random_seed: int = 0,
        device: str = "cuda",
        spatial_key: str = "spatial",
        adjacency_matrix_key: str = "adjacency_matrix",
        feature_key: str | None = None,
        expr_prior_key: str | None = "X_pca",
        expr_prior_dim: int = 16,
        use_topo: bool = True,
        use_expr_prior: bool = True,
        edge_attr_mode: str = "adjacency",
        symmetrize: bool = True,
        drop_isolated_cells: bool = True,
        epochs: int = 100,
        lr: float = 1e-3,
        temp_r: float = 1e-3,
        temp_N: int = 1,
        cache_scores: bool = True,
        cache_directory: str | None = None,
        sketch_mode: str = "edge",  # "edge" / "node"
        node_score_mode: str = "source_mean",  # "source_mean" / "incident_mean"
    ):
        self.mog_args = dict(mog_args)
        self.random_seed = int(random_seed)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.spatial_key = spatial_key
        self.adjacency_matrix_key = adjacency_matrix_key
        self.feature_key = feature_key
        self.expr_prior_key = expr_prior_key
        self.expr_prior_dim = int(expr_prior_dim)
        self.use_topo = bool(use_topo)
        self.use_expr_prior = bool(use_expr_prior)
        self.edge_attr_mode = edge_attr_mode
        self.symmetrize = bool(symmetrize)
        self.drop_isolated_cells = bool(drop_isolated_cells)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.temp_r = float(temp_r)
        self.temp_N = int(temp_N)
        self.cache_scores = bool(cache_scores)
        self.cache_directory = cache_directory

        self.sketch_mode = sketch_mode
        self.node_score_mode = node_score_mode

        if self.sketch_mode not in {"edge", "node"}:
            raise ValueError(f"Unsupported sketch_mode: {self.sketch_mode}")

        if self.node_score_mode not in {"source_mean", "incident_mean"}:
            raise ValueError(f"Unsupported node_score_mode: {self.node_score_mode}")

    def _to_numpy_dense(self, x):
        if issparse(x):
            x = x.toarray()
        return np.asarray(x)

    def _get_feature_matrix(self, adata: ad.AnnData) -> torch.Tensor:
        if self.feature_key is None:
            feat = adata.X
        else:
            feat = adata.obsm[self.feature_key]
        feat = self._to_numpy_dense(feat)
        return torch.tensor(feat, dtype=torch.float32, device=self.device)

    def _get_edge_index_and_adj_values(self, adata: ad.AnnData):
        adj = adata.obsp[self.adjacency_matrix_key].tocsr()
        coo = adj.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long, device=self.device)
        col = torch.tensor(coo.col, dtype=torch.long, device=self.device)
        edge_index = torch.stack([row, col], dim=0)
        adj_values = torch.tensor(coo.data, dtype=torch.float32, device=self.device)
        return edge_index, adj_values

    def _normalize_01(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + 1e-12)

    def _build_edge_attr(
        self,
        adata: ad.AnnData,
        edge_index: torch.Tensor,
        adj_values: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.edge_attr_mode

        if mode == "adjacency":
            edge_attr = adj_values.float()

        elif mode == "ones":
            edge_attr = torch.ones(edge_index.size(1), dtype=torch.float32, device=self.device)

        elif mode in {"spatial_distance", "inv_spatial_distance"}:
            coords = adata.obsm[self.spatial_key]
            coords = np.asarray(coords)
            coords = torch.tensor(coords, dtype=torch.float32, device=self.device)

            src = edge_index[0]
            dst = edge_index[1]
            dist = torch.norm(coords[src] - coords[dst], dim=1)

            if mode == "spatial_distance":
                edge_attr = self._normalize_01(dist)
            else:
                edge_attr = 1.0 / (dist + 1e-8)
                edge_attr = self._normalize_01(edge_attr)

        else:
            raise ValueError(f"Unsupported edge_attr_mode: {mode}")

        return edge_attr

    def _get_expr_prior_features(
        self,
        adata: ad.AnnData,
        fallback_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.expr_prior_key is not None and self.expr_prior_key in adata.obsm:
            z = adata.obsm[self.expr_prior_key]
            z = np.asarray(z)
            z = z[:, : self.expr_prior_dim]
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
            return z

        z = fallback_features
        if z.size(1) > self.expr_prior_dim:
            z = z[:, : self.expr_prior_dim]
        return z

    def _compute_expr_prior(
        self,
        adata: ad.AnnData,
        edge_index: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        z = self._get_expr_prior_features(adata, features)
        src = edge_index[0]
        dst = edge_index[1]
        expr_prior = F.cosine_similarity(z[src], z[dst], dim=1)
        expr_prior = (expr_prior + 1.0) / 2.0
        return expr_prior

    def _train_mog(self, features, edge_index, edge_attr):
        num_features = features.size(1)

        model = MoG(
            num_features=num_features,
            device=self.device,
            k_list=self.mog_args["k_list"],
            hidden_spl=self.mog_args["hidden_spl"],
            num_layers_spl=self.mog_args["num_layers_spl"],
            expert_select=self.mog_args["expert_select"],
            lam=self.mog_args.get("lam", 1.0),
            topo_loss_coef=self.mog_args.get("topo_loss_coef", 1.0),
            expr_loss_coef=self.mog_args.get("expr_loss_coef", 0.1),
            retention_ratio=self.mog_args.get("retention_ratio", None),
            expr_aug_coef=self.mog_args.get("expr_aug_coef", 0.0),
            expr_topo_mode=self.mog_args.get("expr_topo_mode", "both"),
        ).to(self.device)

        optimizer = torch.optim.Adam(model.learner.parameters(), lr=self.lr)

        if self.use_topo:
            model.learner.get_topo_val(edge_index)
        else:
            model.learner.topo_val = None

        if self.use_expr_prior:
            model.learner.expr_prior = self._cached_expr_prior
        else:
            model.learner.expr_prior = None

        best_loss = float("inf")
        best_score = None
        best_aux = None

        for epoch in range(1, self.epochs + 1):
            if (epoch - 1) % self.temp_N == 0:
                decay_temp = np.exp(-1.0 * self.temp_r * epoch)
                temp = max(0.05, decay_temp)

            model.train()
            optimizer.zero_grad()

            out = model.learner(
                x=features,
                edge_index=edge_index,
                temp=temp,
                edge_attr=edge_attr,
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                eval_out = model.learner(
                    x=features,
                    edge_index=edge_index,
                    temp=temp,
                    edge_attr=edge_attr,
                )

            eval_loss = float(eval_out["loss"].item())
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_score = eval_out["edge_score"].detach().clone()
                best_aux = {
                    "loss_balance": float(eval_out["loss_balance"].item()),
                    "loss_topo": float(eval_out["loss_topo"].item()),
                    "loss_expr": float(eval_out["loss_expr"].item()),
                }

        if best_score is None:
            raise RuntimeError("MoG did not produce a valid score.")

        return best_score, best_loss, best_aux

    def _edge_index_to_adj(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_values: torch.Tensor | None = None,
    ):
        edge_index_np = edge_index.detach().cpu().numpy()
        if edge_values is None:
            values_np = np.ones(edge_index_np.shape[1], dtype=np.float32)
        else:
            values_np = edge_values.detach().cpu().numpy().astype(np.float32)

        adj = sp.csr_matrix(
            (values_np, (edge_index_np[0], edge_index_np[1])),
            shape=(num_nodes, num_nodes),
        )
        return adj

    def _drop_isolated_cells(self, adata: ad.AnnData, adj: sp.csr_matrix) -> ad.AnnData:
        if adj.nnz == 0:
            raise RuntimeError("All edges were removed; no connected cells remain.")

        if self.symmetrize:
            degree = np.asarray(adj.getnnz(axis=1)).ravel()
            keep_nodes = degree > 0
        else:
            out_degree = np.asarray(adj.getnnz(axis=1)).ravel()
            in_degree = np.asarray(adj.getnnz(axis=0)).ravel()
            keep_nodes = (out_degree + in_degree) > 0

        if keep_nodes.all():
            out = adata.copy()
            out.obsp[self.adjacency_matrix_key] = adj
            return out

        out = adata[keep_nodes].copy()
        adj_sub = adj[keep_nodes][:, keep_nodes].tocsr()
        out.obsp[self.adjacency_matrix_key] = adj_sub
        return out

    def _mask_from_score(self, edge_score: torch.Tensor) -> torch.Tensor:
        ratio = self.mog_args.get("retention_ratio", None)
        num_edges = edge_score.numel()

        if num_edges == 0:
            return torch.zeros_like(edge_score, dtype=torch.bool)

        if ratio is None:
            return edge_score > 0

        ratio = float(ratio)
        if not (0 < ratio <= 1):
            raise ValueError(f"retention_ratio must be in (0, 1], got {ratio}")

        k = max(1, int(np.ceil(num_edges * ratio)))
        k = min(k, num_edges)

        top_idx = torch.topk(edge_score, k=k, largest=True).indices
        mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_score.device)
        mask[top_idx] = True
        return mask

    def _node_mask_from_edge_score(
        self,
        edge_score: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        ratio = self.mog_args.get("retention_ratio", None)
        if ratio is None:
            raise ValueError("Node mode requires mog_args.retention_ratio.")

        ratio = float(ratio)
        if not (0 < ratio <= 1):
            raise ValueError(f"retention_ratio must be in (0, 1], got {ratio}")

        src = edge_index[0]
        dst = edge_index[1]

        node_score_sum = torch.zeros(num_nodes, device=edge_score.device, dtype=edge_score.dtype)
        node_score_count = torch.zeros(num_nodes, device=edge_score.device, dtype=edge_score.dtype)

        if self.node_score_mode == "source_mean":
            node_score_sum.scatter_add_(0, src, edge_score)
            node_score_count.scatter_add_(0, src, torch.ones_like(edge_score))

        elif self.node_score_mode == "incident_mean":
            one = torch.ones_like(edge_score)
            node_score_sum.scatter_add_(0, src, edge_score)
            node_score_sum.scatter_add_(0, dst, edge_score)
            node_score_count.scatter_add_(0, src, one)
            node_score_count.scatter_add_(0, dst, one)

        node_score = node_score_sum / torch.clamp(node_score_count, min=1.0)
        node_score = torch.where(
            node_score_count > 0,
            node_score,
            torch.full_like(node_score, -float("inf")),
        )

        target_keep = max(1, int(np.ceil(num_nodes * ratio)))
        target_keep = min(target_keep, num_nodes)

        top_idx = torch.topk(node_score, k=target_keep, largest=True).indices
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_score.device)
        node_mask[top_idx] = True
        return node_mask

    def _score_cache_key(self, adata: ad.AnnData, input_path: str | Path | None = None) -> str:
        train_signature = {
            "input_path": None if input_path is None else str(Path(input_path).resolve()),
            "random_seed": self.random_seed,
            "spatial_key": self.spatial_key,
            "adjacency_matrix_key": self.adjacency_matrix_key,
            "feature_key": self.feature_key,
            "expr_prior_key": self.expr_prior_key,
            "expr_prior_dim": self.expr_prior_dim,
            "use_topo": self.use_topo,
            "use_expr_prior": self.use_expr_prior,
            "edge_attr_mode": self.edge_attr_mode,
            "epochs": self.epochs,
            "lr": self.lr,
            "temp_r": self.temp_r,
            "temp_N": self.temp_N,
            "mog_args": {k: v for k, v in self.mog_args.items() if k != "retention_ratio"},
            "expr_topo_mode": self.mog_args.get("expr_topo_mode", "both"),
            "loss_version": "expr_topo_v1",
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
        }
        raw = json.dumps(train_signature, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_score_cache_path(
        self,
        adata: ad.AnnData,
        input_path: str | Path | None = None,
    ) -> Path | None:
        if not self.cache_scores:
            return None
        if self.cache_directory is None:
            return None

        cache_dir = Path(self.cache_directory)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._score_cache_key(adata, input_path=input_path)
        return cache_dir / f"{cache_key}.pt"

    def _save_score_cache(
        self,
        cache_path: Path,
        edge_index: torch.Tensor,
        best_score: torch.Tensor,
        best_loss: float,
        best_aux: dict,
    ) -> None:
        payload = {
            "edge_index": edge_index.detach().cpu(),
            "best_score": best_score.detach().cpu(),
            "best_loss": float(best_loss),
            "best_aux": best_aux,
        }
        torch.save(payload, cache_path)

    def _load_score_cache(self, cache_path: Path):
        payload = torch.load(cache_path, map_location="cpu")
        return (
            payload["edge_index"],
            payload["best_score"],
            payload["best_loss"],
            payload["best_aux"],
        )

    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        return self._transform_impl(adata=adata, input_path=None)

    def _transform_impl(
        self,
        adata: ad.AnnData,
        input_path: str | Path | None = None,
    ) -> ad.AnnData:
        fix_seed(self.random_seed)

        adata = adata.copy()

        need_pca = (self.feature_key == "X_pca") or (self.use_expr_prior and self.expr_prior_key == "X_pca")
        if need_pca and "X_pca" not in adata.obsm:
            sc.tl.pca(adata)

        features = self._get_feature_matrix(adata)
        edge_index, adj_values = self._get_edge_index_and_adj_values(adata)
        edge_attr = self._build_edge_attr(adata, edge_index, adj_values)

        if self.use_expr_prior:
            self._cached_expr_prior = self._compute_expr_prior(adata, edge_index, features)
        else:
            self._cached_expr_prior = None

        cache_path = self._get_score_cache_path(adata, input_path=input_path)

        if cache_path is not None and cache_path.exists():
            cached_edge_index, best_score, best_loss, best_aux = self._load_score_cache(cache_path)
            cached_edge_index = cached_edge_index.to(edge_index.device)
            best_score = best_score.to(edge_index.device)

            if cached_edge_index.shape != edge_index.shape or not torch.equal(cached_edge_index, edge_index):
                raise RuntimeError("Cached edge_index does not match current edge_index.")
        else:
            best_score, best_loss, best_aux = self._train_mog(
                features=features,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            if cache_path is not None:
                self._save_score_cache(
                    cache_path=cache_path,
                    edge_index=edge_index,
                    best_score=best_score,
                    best_loss=best_loss,
                    best_aux=best_aux,
                )

        orig_edges = int(adata.obsp[self.adjacency_matrix_key].nnz)

        if self.sketch_mode == "edge":
            edge_mask = self._mask_from_score(best_score)
            new_edge_index = edge_index[:, edge_mask]

            new_edge_values = torch.ones(
                int(edge_mask.sum().item()),
                dtype=torch.float32,
                device=edge_index.device,
            )

            new_adj = self._edge_index_to_adj(
                edge_index=new_edge_index,
                num_nodes=adata.n_obs,
                edge_values=new_edge_values,
            )

            if self.symmetrize:
                new_adj = new_adj.maximum(new_adj.T).tocsr()

            if self.drop_isolated_cells:
                new_adata = self._drop_isolated_cells(adata, new_adj)
            else:
                new_adata = adata.copy()
                new_adata.obsp[self.adjacency_matrix_key] = new_adj

            selected_node_ratio = new_adata.n_obs / adata.n_obs if adata.n_obs > 0 else 0.0

        elif self.sketch_mode == "node":
            node_mask = self._node_mask_from_edge_score(
                edge_score=best_score,
                edge_index=edge_index,
                num_nodes=adata.n_obs,
            )
            node_mask_np = node_mask.detach().cpu().numpy()

            old_adj = adata.obsp[self.adjacency_matrix_key].tocsr()
            new_adata = adata[node_mask_np].copy()
            new_adata.obsp[self.adjacency_matrix_key] = old_adj[node_mask_np][:, node_mask_np].tocsr()

            selected_node_ratio = float(node_mask.float().mean().item())

        else:
            raise ValueError(f"Unsupported sketch_mode: {self.sketch_mode}")

        final_edges = int(new_adata.obsp[self.adjacency_matrix_key].nnz)

        new_adata.uns["mog_best_loss"] = float(best_loss)
        new_adata.uns["mog_edge_retention_ratio"] = final_edges / orig_edges if orig_edges > 0 else 0.0
        new_adata.uns["mog_node_retention_ratio"] = float(selected_node_ratio)
        new_adata.uns["mog_original_num_edges"] = orig_edges
        new_adata.uns["mog_sketched_num_edges"] = final_edges
        new_adata.uns["mog_original_num_cells"] = int(adata.n_obs)
        new_adata.uns["mog_sketched_num_cells"] = int(new_adata.n_obs)
        new_adata.uns["mog_loss_balance"] = best_aux["loss_balance"]
        new_adata.uns["mog_loss_topo"] = best_aux["loss_topo"]
        new_adata.uns["mog_loss_expr"] = best_aux["loss_expr"]
        new_adata.uns["mog_edge_attr_mode"] = self.edge_attr_mode
        new_adata.uns["mog_sketch_mode"] = self.sketch_mode
        new_adata.uns["mog_node_score_mode"] = self.node_score_mode

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
        l = len(src)
        del adata, src, dst, edge_values
        rng = np.random.default_rng(self.random_seed)
        return rng.random(l, dtype=np.float32)

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

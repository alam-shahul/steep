"""Mixture-of-graphs sketcher."""

import hashlib
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.sparse import issparse

from steep.sketch.base import AnnDataSketcher, SketchMetadata


def _fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
        temp_N: int = 1,  # noqa: N803
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
        from steep.models._sparsify import MoG

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
        _fix_seed(self.random_seed)

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

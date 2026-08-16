"""Train the MoG pruner with a signal that actually reaches its experts.

The sequential pipeline trains MoG on a surrogate objective and only then trains
STAGATE on the resulting sketch. That surrogate is where the method loses: the
topology target it regresses onto separates same-cell-type edges at AUROC ~0.50
on a Delaunay spatial graph, so the pruner is fitting noise.

Both trainers here replace the surrogate with a signal tied to the downstream
model:

`JointSparsifier`
    One optimizer over the pruner and STAGATE together, as in the upstream MoG.
    The soft edge mask multiplies the attention weights, so STAGATE's
    reconstruction loss backpropagates into the pruner and the experts learn
    which edges the downstream model needs.

`DistillSparsifier`
    Keeps the two-stage structure, but trains the pruner to make the sparsified
    graph reproduce the embeddings of a STAGATE already trained on the full
    graph. The teacher is fitted once and reused, and the target is exactly what
    the benchmark scores, so no labels are involved.

Both expose `edge_scores()`, which `MoGSketcher` can consume in place of its own
`_train_mog`.

"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from steep.models._graph import STAGATE
from steep.models._sparsify import MoG


def _temperature(epoch: int, temp_r: float, temp_n: int) -> float:
    """Same annealing schedule the sequential trainer uses."""
    step = epoch - (epoch % max(temp_n, 1))
    return max(0.05, float(np.exp(-1.0 * temp_r * step)))


class _SparsifierBase(nn.Module):
    def __init__(
        self,
        num_features: int,
        mog_args: dict,
        device: torch.device,
        num_hidden: int = 128,
        d_model: int = 30,
        lr: float = 1e-3,
        epochs: int = 100,
        temp_r: float = 1e-3,
        temp_n: int = 1,
        use_topo: bool = False,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.device = device
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.temp_r = float(temp_r)
        self.temp_n = int(temp_n)
        self.use_topo = bool(use_topo)
        self.grad_clip = float(grad_clip)

        self.mog = MoG(
            num_features=num_features,
            device=device,
            k_list=mog_args["k_list"],
            hidden_spl=mog_args["hidden_spl"],
            num_layers_spl=mog_args["num_layers_spl"],
            expert_select=mog_args["expert_select"],
            lam=mog_args.get("lam", 1.0),
            topo_loss_coef=mog_args.get("topo_loss_coef", 1.0),
            expr_loss_coef=mog_args.get("expr_loss_coef", 0.1),
            retention_ratio=mog_args.get("retention_ratio"),
            expr_aug_coef=mog_args.get("expr_aug_coef", 0.0),
            expr_topo_mode=mog_args.get("expr_topo_mode", "both"),
        ).to(device)

        self.stagate = STAGATE(in_dim=num_features, num_hidden=num_hidden, d_model=d_model).to(device)

    def _prepare_priors(self, edge_index, expr_prior):
        if self.use_topo:
            self.mog.learner.get_topo_val(edge_index)
        else:
            self.mog.learner.topo_val = None
        self.mog.learner.expr_prior = expr_prior

    def _pruner_step(self, features, edge_index, edge_attr, temp):
        """One MoE pass; returns the soft mask and the auxiliary loss."""
        out = self.mog.learner(x=features, edge_index=edge_index, temp=temp, edge_attr=edge_attr)
        # `mask` is the hard selection, `gated_prob` the differentiable score.
        # Multiplying them keeps the forward sparse while letting gradients flow.
        soft_mask = out["mask"] * out["gated_prob"]
        return soft_mask, out

    @torch.no_grad()
    def edge_scores(self, features, edge_index, edge_attr):
        """Final per-edge score and the per-node mask, for the sketcher."""
        self.eval()
        out = self.mog.learner(
            x=features,
            edge_index=edge_index,
            temp=0.05,
            edge_attr=edge_attr,
        )
        return out["edge_score"].detach(), out["local_mask"].detach().bool()


class JointSparsifier(_SparsifierBase):
    """Pruner and STAGATE under one optimizer, as upstream MoG does it."""

    def fit(self, features, edge_index, edge_attr, expr_prior=None):
        self._prepare_priors(edge_index, expr_prior)
        optimizer = torch.optim.Adam(
            list(self.mog.learner.parameters()) + list(self.stagate.parameters()),
            lr=self.lr,
        )

        history = []
        for epoch in range(1, self.epochs + 1):
            temp = _temperature(epoch, self.temp_r, self.temp_n)
            self.train()
            optimizer.zero_grad()

            soft_mask, out = self._pruner_step(features, edge_index, edge_attr, temp)
            reconstruction = self.stagate.forward(
                _Graph(features, edge_index),
                edge_weight=soft_mask,
            )["logits"]
            task_loss = F.mse_loss(reconstruction, features)

            loss = task_loss + out["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            optimizer.step()

            history.append({"epoch": epoch, "task": float(task_loss), "aux": float(out["loss"])})

        return history


class DistillSparsifier(_SparsifierBase):
    """Two-stage, but the pruner is taught by a full-graph STAGATE.

    Stage 1 fits the teacher on the dense graph. Stage 2 freezes it and trains
    the pruner so the sparsified graph reproduces the teacher's embeddings.

    """

    def fit_teacher(self, features, edge_index, teacher_epochs: int | None = None):
        epochs = self.epochs if teacher_epochs is None else int(teacher_epochs)
        optimizer = torch.optim.Adam(self.stagate.parameters(), lr=self.lr)
        graph = _Graph(features, edge_index)

        self.stagate.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = F.mse_loss(self.stagate(graph)["logits"], features)
            loss.backward()
            nn.utils.clip_grad_norm_(self.stagate.parameters(), self.grad_clip)
            optimizer.step()

        self.stagate.eval()
        with torch.no_grad():
            return self.stagate(graph)["embedding"].detach()

    def fit(self, features, edge_index, edge_attr, expr_prior=None, teacher_embedding=None):
        if teacher_embedding is None:
            teacher_embedding = self.fit_teacher(features, edge_index)

        self._prepare_priors(edge_index, expr_prior)
        for parameter in self.stagate.parameters():
            parameter.requires_grad_(False)
        self.stagate.eval()

        optimizer = torch.optim.Adam(self.mog.learner.parameters(), lr=self.lr)
        graph = _Graph(features, edge_index)

        history = []
        for epoch in range(1, self.epochs + 1):
            temp = _temperature(epoch, self.temp_r, self.temp_n)
            self.mog.learner.train()
            optimizer.zero_grad()

            soft_mask, out = self._pruner_step(features, edge_index, edge_attr, temp)
            student = self.stagate.forward(graph, edge_weight=soft_mask)["embedding"]
            distill_loss = F.mse_loss(student, teacher_embedding)

            loss = distill_loss + out["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(self.mog.learner.parameters(), self.grad_clip)
            optimizer.step()

            history.append({"epoch": epoch, "distill": float(distill_loss), "aux": float(out["loss"])})

        return history


class _Graph:
    """Minimal stand-in for a PyG `Data` -- STAGATE only reads
    `x`/`edge_index`."""

    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index
        self.edge_weight = None

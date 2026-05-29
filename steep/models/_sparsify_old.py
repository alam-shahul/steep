import networkit as nk
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch_geometric.utils import softmax

eps = 1e-8


# ====================
# MoG
# ====================
class MoG(nn.Module):
    def __init__(
        self,
        num_features: int,
        device: torch.device,
        # ---- MoE core ----
        k_list,
        hidden_spl: int,
        num_layers_spl: int,
        expert_select: int,
        # ---- losses / behavior ----
        lam: float = 1.0,
        topo_loss_coef: float = 1.0,
        expr_loss_coef: float = 0.1,
        retention_ratio: float | None = None,
        expr_aug_coef: float = 0.0,
        expr_topo_mode: str = "both",
    ):
        super().__init__()

        self.device = device
        self.k_list = torch.tensor(k_list, device=device, dtype=torch.float32)
        self.learner = MoE(
            input_size=num_features,
            hidden_size=hidden_spl,
            num_experts=self.k_list.size(0),
            nlayers=num_layers_spl,
            activation=nn.ReLU(),
            k_list=self.k_list,
            expert_select=expert_select,
            lam=lam,
            topo_loss_coef=topo_loss_coef,
            expr_loss_coef=expr_loss_coef,
            retention_ratio=retention_ratio,
            expr_aug_coef=expr_aug_coef,
            expr_topo_mode=expr_topo_mode,
        )


# ====================
# MoE
# ====================


class MoE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        nlayers,
        activation,
        k_list,
        expert_select,
        noisy_gating=True,
        coef=1e-2,
        lam=1.0,
        # newly added input
        topo_loss_coef=1.0,
        expr_loss_coef=0.1,
        retention_ratio=None,
        expr_aug_coef=0.0,
        expr_topo_mode: str = "both",  # "none" / "intra" / "inter" / "both"
    ):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = expert_select
        self.loss_coef = coef
        self.k_list = k_list

        k_list_python = k_list.detach().cpu().tolist()
        self.experts = nn.ModuleList(
            [
                SpLearner(
                    nlayers=nlayers,
                    in_dim=input_size * 2 + 1,
                    hidden=hidden_size,
                    activation=activation,
                    k=float(k),
                )
                for k in k_list_python
            ],
        )

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.lam = lam
        # newly added initialization
        self.topo_loss_coef = float(topo_loss_coef)
        self.expr_loss_coef = float(expr_loss_coef)
        self.retention_ratio = retention_ratio
        self.expr_aug_coef = float(expr_aug_coef)

        self.topo_val = None
        # newly added feature
        self.expr_prior = None
        self.expr_topo_mode = expr_topo_mode

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

        if retention_ratio is not None:
            if not (0 < float(retention_ratio) <= 1):
                raise ValueError(f"retention_ratio must be in (0, 1], got {retention_ratio}")

    # newly added helper
    def _group_topk_mask(self, scores, src_index, keep_counts):
        """
        scores: [num_edges]
        src_index: [num_edges]
        keep_counts: [num_unique_src], aligned with unique_consecutive(src_sorted)
        """
        num_edges = scores.numel()
        if num_edges == 0:
            return torch.zeros_like(scores)

        perm = torch.argsort(src_index, stable=True)
        src_sorted = src_index[perm]
        scores_sorted = scores[perm]

        unique_src, counts = torch.unique_consecutive(src_sorted, return_counts=True)
        assert keep_counts.numel() == counts.numel()

        mask_sorted = torch.zeros_like(scores_sorted)
        start = 0
        for i, cnt in enumerate(counts.tolist()):
            end = start + cnt
            seg = scores_sorted[start:end]
            k_keep = int(keep_counts[i].item())
            k_keep = max(1, min(k_keep, cnt))
            top_idx = torch.topk(seg, k=k_keep, largest=True).indices
            mask_sorted[start + top_idx] = 1.0
            start = end

        mask = torch.zeros_like(mask_sorted)
        mask[perm] = mask_sorted
        return mask

    # newly added helper
    def _enforce_retention_ratio(self, scores, mask):
        """Hard enforce a global edge retention ratio.

        If current kept edges > target: trim. If current kept edges < target:
        supplement from highest-scoring unkept edges.

        """
        if self.retention_ratio is None:
            return mask

        num_edges = scores.numel()
        if num_edges == 0:
            return mask

        target_keep = max(1, int(np.ceil(num_edges * float(self.retention_ratio))))
        target_keep = min(target_keep, num_edges)

        current_keep = int(mask.sum().item())
        if current_keep == target_keep:
            return mask

        final_mask = torch.zeros_like(mask)

        kept_idx = torch.nonzero(mask > 0, as_tuple=False).view(-1)
        not_kept_idx = torch.nonzero(mask <= 0, as_tuple=False).view(-1)

        if current_keep > target_keep:
            keep_scores = scores[kept_idx]
            chosen_local = torch.topk(keep_scores, k=target_keep, largest=True).indices
            chosen_idx = kept_idx[chosen_local]
            final_mask[chosen_idx] = 1.0
            return final_mask

        final_mask[kept_idx] = 1.0
        need = target_keep - current_keep
        if need > 0 and not_kept_idx.numel() > 0:
            need = min(need, not_kept_idx.numel())
            extra_scores = scores[not_kept_idx]
            chosen_local = torch.topk(extra_scores, k=need, largest=True).indices
            chosen_idx = not_kept_idx[chosen_local]
            final_mask[chosen_idx] = 1.0

        return final_mask

    # newly added helper
    @staticmethod
    def _minmax_by_col(tensor):
        min_val = tensor.min(dim=0, keepdim=True).values
        max_val = tensor.max(dim=0, keepdim=True).values
        return (tensor - min_val) / (max_val - min_val + eps)

    # add topo feature
    def get_topo_val(self, edge_index):
        G = nx.DiGraph()
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
        G = nk.nxadapter.nx2nk(G)
        G.indexEdges()

        lds = nk.sparsification.LocalDegreeScore(G).run().scores()
        ffs = nk.sparsification.ForestFireScore(G, 0.6, 5.0).run().scores()
        triangles = nk.sparsification.TriangleEdgeScore(G).run().scores()
        lss = nk.sparsification.LocalSimilarityScore(G, triangles).run().scores()
        scan = nk.sparsification.SCANStructuralSimilarityScore(G, triangles).run().scores()

        topo_val = torch.tensor([lds, ffs, lss, scan], device=edge_index.device).t().float()
        topo_val = self._minmax_by_col(topo_val)
        self.topo_val = topo_val

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.

        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.

        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.

        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]

        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.

        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].

        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, edge_index, temp, edge_attr=None):

        num_edges = edge_index.size(1)

        if self.topo_val is not None:
            assert (
                self.topo_val.size(0) == num_edges
            ), f"topo_val has {self.topo_val.size(0)} edges, expected {num_edges}"

        if self.expr_prior is not None:
            assert (
                self.expr_prior.size(0) == num_edges
            ), f"expr_prior has {self.expr_prior.size(0)} edges, expected {num_edges}"

        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        else:
            edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)

        # --------------------------------------------------
        # 1) node-level expert routing
        # --------------------------------------------------
        node_gates, load = self.noisy_top_k_gating(x, self.training)

        importance = node_gates.sum(0)
        loss_balance = self.cv_squared(importance) + self.cv_squared(load)
        loss_balance = loss_balance * self.loss_coef

        src = edge_index[0]
        edge_gates = torch.index_select(node_gates, dim=0, index=src)

        # --------------------------------------------------
        # 2) expert edge scoring
        # --------------------------------------------------
        expert_probs = []
        expert_masked_scores = []

        for i in range(self.num_experts):
            raw_scores_i, edge_prob_i, expert_mask_i, masked_scores_i = self.experts[i](
                features=x,
                indices=edge_index,
                values=edge_attr,
                temperature=temp,
            )
            expert_probs.append(edge_prob_i)
            expert_masked_scores.append(masked_scores_i)

        expert_probs = torch.stack(expert_probs, dim=1)
        expert_masked_scores = torch.stack(expert_masked_scores, dim=1)

        # weighted by edge_gates
        gated_prob = torch.sum(edge_gates * expert_probs, dim=1)
        gated_masked_score = torch.sum(edge_gates * expert_masked_scores, dim=1)

        # --------------------------------------------------
        # 3) topology prior loss
        # --------------------------------------------------
        if self.topo_val is not None:
            topo_dim = self.topo_val.size(1)
            topo_targets = torch.stack(
                [self.topo_val[:, i % topo_dim] for i in range(self.num_experts)],
                dim=1,
            )

            weighted_topo_target = torch.sum(edge_gates * topo_targets, dim=1)  # [E]
            loss_topo = F.mse_loss(gated_prob, weighted_topo_target)
        else:
            weighted_topo_target = None
            loss_topo = torch.tensor(0.0, device=x.device)

        # --------------------------------------------------
        # 4) expression prior loss
        # --------------------------------------------------
        if self.expr_prior is not None:
            loss_expr = F.mse_loss(gated_prob, self.expr_prior)
        else:
            loss_expr = torch.tensor(0.0, device=x.device)

        # --------------------------------------------------
        # 5) selection score
        # --------------------------------------------------
        selection_score = gated_masked_score.clone()

        if weighted_topo_target is not None:
            selection_score = selection_score + self.lam * weighted_topo_target

        if self.expr_prior is not None and self.expr_aug_coef != 0.0:
            selection_score = selection_score + self.expr_aug_coef * self.expr_prior

        # --------------------------------------------------
        # 6) local MoG mask (per source node)
        # --------------------------------------------------
        src_sorted_perm = torch.argsort(src, stable=True)
        src_sorted = src[src_sorted_perm]
        unique_src, num_edges_per_node = torch.unique_consecutive(src_sorted, return_counts=True)

        k_per_node = torch.sum(node_gates * torch.unsqueeze(self.k_list, 0), dim=1)  # [num_nodes]
        k_edges_per_node = (k_per_node[unique_src] * num_edges_per_node.float()).round().long()
        k_edges_per_node = torch.clamp(k_edges_per_node, min=1)

        local_mask = self._group_topk_mask(
            scores=selection_score,
            src_index=src,
            keep_counts=k_edges_per_node,
        )

        # --------------------------------------------------
        # 7) hard retention_ratio
        # --------------------------------------------------
        final_mask = self._enforce_retention_ratio(
            scores=selection_score,
            mask=local_mask,
        )

        total_loss = loss_balance + self.topo_loss_coef * loss_topo + self.expr_loss_coef * loss_expr

        return {
            "mask": final_mask,
            "local_mask": local_mask,
            "edge_score": selection_score,
            "gated_prob": gated_prob,
            "loss": total_loss,
            "loss_balance": loss_balance.detach(),
            "loss_topo": loss_topo.detach(),
            "loss_expr": loss_expr.detach(),
        }


# ====================
# SpLearner
# ====================


class SpLearner(nn.Module):
    """Sparsification learner."""

    def __init__(self, nlayers, in_dim, hidden, activation, k, weight=True, metric=None, processors=None):
        super().__init__()

        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden))
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, 1))

        self.param_init()
        self.activation = activation
        self.k = float(k)
        self.weight = weight

    def param_init(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
        return x

    def sample_gumbel(self, shape, device):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, src_index, temperature):
        """
        logits: [num_edges]
        src_index: [num_edges], source node index for each edge
        """
        if self.training:
            logits = logits + self.sample_gumbel(logits.shape, logits.device)
        logits = logits / temperature
        return softmax(logits, src_index)

    def _group_topk_mask(self, scores, src_index, ratio_k):
        """
        scores: [num_edges]
        src_index: [num_edges]
        return mask aligned with original edge order
        """
        num_edges = scores.numel()
        if num_edges == 0:
            return torch.zeros_like(scores)

        perm = torch.argsort(src_index, stable=True)
        src_sorted = src_index[perm]
        scores_sorted = scores[perm]

        _, counts = torch.unique_consecutive(src_sorted, return_counts=True)

        mask_sorted = torch.zeros_like(scores_sorted)
        start = 0
        for cnt in counts.tolist():
            end = start + cnt
            seg = scores_sorted[start:end]
            k_keep = max(1, int(round(cnt * ratio_k)))
            k_keep = min(k_keep, cnt)
            top_idx = torch.topk(seg, k=k_keep, largest=True).indices
            mask_sorted[start + top_idx] = 1.0
            start = end

        mask = torch.zeros_like(mask_sorted)
        mask[perm] = mask_sorted
        return mask

    def forward(self, features, indices, values, temperature):
        """
        Returns:
            raw_scores:   [num_edges]
            edge_prob:    [num_edges]  grouped softmax over outgoing edges
            expert_mask:  [num_edges]  local top-k mask inside this expert
            masked_scores:[num_edges]  expert_mask * edge_prob
        """

        src = indices[0]
        dst = indices[1]

        f1_features = torch.index_select(features, 0, src)
        f2_features = torch.index_select(features, 0, dst)
        auv = torch.unsqueeze(values, -1)

        temp = torch.cat([f1_features, f2_features, auv], dim=-1)
        raw_scores = self.internal_forward(temp).view(-1)

        edge_prob = self.gumbel_softmax_sample(
            logits=raw_scores,
            src_index=src,
            temperature=temperature,
        )

        expert_mask = self._group_topk_mask(edge_prob, src, self.k)
        masked_scores = expert_mask * edge_prob

        return raw_scores, edge_prob, expert_mask, masked_scores

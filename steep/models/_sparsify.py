import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear, ReLU, Sequential
from torch.distributions.normal import Normal
from torch_geometric.nn import (
    global_mean_pool,
    graclus,
    max_pool,
    max_pool_x,
)
from torch_geometric.utils import normalized_cut

import numpy as np
import networkit as nk
import networkx as nx


# ------------------------------------------------------------
# Straight-Through Estimator for hard edge-selection (0/1 mask)
# ------------------------------------------------------------
class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save input for custom backward
        ctx.save_for_backward(input)
        # Hard thresholding to {0,1}
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Custom gradient shaping around the threshold region
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2 - 4 * torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input * additional


# -----------------------------------------
# Single sparsification learner (one expert)
# Learns edge scores per node, then top-k per node
# -----------------------------------------
class SpLearner(nn.Module):
    """Sparsification learner"""
    def __init__(self, nlayers, in_dim, hidden, activation, k, weight=True, metric=None, processors=None):
        super().__init__()

        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        # First layer maps concatenated node features (and optional edge features)
        self.layers.append(nn.Linear(in_dim, hidden))
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        # Output single scalar score per edge
        self.layers.append(nn.Linear(hidden, 1))

        self.param_init()
        self.activation = activation
        self.k = k              # target sparsity ratio (per node)
        self.weight = weight

    def param_init(self):
        # Xavier init for all linear layers
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def internal_forward(self, x):
        # MLP over edge feature representation
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
        return x

    def gumbel_softmax_sample(self, indices, values, temperature, training):
        """Draw a sample from the Gumbel-Softmax distribution on a sparse edge matrix"""
        r = self.sample_gumble(values.shape)
        if training:
            values = torch.log(values) + r.to(indices.device)
        else:
            values = torch.log(values)
        values /= temperature
        y = torch.sparse_coo_tensor(indices=indices, values=values, requires_grad=True)
        return torch.sparse.softmax(y, dim=1)

    def sample_gumble(self, shape, eps=1e-8):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, features, indices, values=None, temperature=0, training=False):
        # features: [num_nodes, in_dim_node]
        # indices: [2, num_edges] (source/target indices)
        # values: optional edge features

        # Gather endpoint features for each edge
        f1_features = torch.index_select(features, 0, indices[0, :])
        f2_features = torch.index_select(features, 0, indices[1, :])

        # Concatenate node features (and edge features if provided)
        if values is not None:
            auv = values
            temp = torch.cat([f1_features, f2_features, auv], -1)
        else:
            temp = torch.cat([f1_features, f2_features], -1)

        # Edge-wise score before normalization
        temp = self.internal_forward(temp)

        # Flatten to 1D scores per edge and normalize globally
        z = torch.reshape(temp, [-1])
        z = F.normalize(z, dim=0)

        # Build sparse matrix of scores per edge
        z_matrix = torch.sparse_coo_tensor(indices=indices, values=z, requires_grad=True)
        # Normalize per-row (per source node) into a distribution
        pi = torch.sparse.softmax(z_matrix, dim=1)
        pi_values = pi.coalesce().values()
        sparse_indices = pi.coalesce().indices()
        sparse_values = pi.coalesce().values()

        # Compute number of edges per node (row)
        node_idx, num_edges_per_node = sparse_indices[0].unique(return_counts=True)
        # Desired number of kept edges per node via ratio k
        k_edges_per_node = (num_edges_per_node.float() * self.k).round().long()
        k_edges_per_node = torch.where(
            k_edges_per_node > 0,
            k_edges_per_node,
            torch.ones_like(k_edges_per_node, device=k_edges_per_node.device)
        )

        # Sort probs globally but keep node grouping information
        sparse_values, val_sort_idx = sparse_values.sort(descending=True)
        sparse_idx0 = sparse_indices[0].index_select(dim=-1, index=val_sort_idx)
        idx_sort_idx = sparse_idx0.argsort(stable=True, dim=-1, descending=False)
        scores_sorted = sparse_values.index_select(dim=-1, index=idx_sort_idx)

        # For each node: compute index of threshold score
        edge_start_indices = torch.cat(
            (torch.tensor([0], device=pi.device), torch.cumsum(num_edges_per_node[:-1], dim=0))
        )
        edge_end_indices = torch.abs(torch.add(edge_start_indices, k_edges_per_node) - 1).long()
        node_keep_thre_cal = torch.index_select(scores_sorted, dim=-1, index=edge_end_indices)
        # Broadcast threshold to all edges
        node_keep_thre_augmented = node_keep_thre_cal.repeat_interleave(num_edges_per_node)

        # BinaryStep to get hard 0/1 mask of edges above threshold
        mask = BinaryStep.apply(scores_sorted - node_keep_thre_augmented + 1e-15)
        masked_scores = mask * scores_sorted

        # Restore original edge ordering
        idx_resort_idx = idx_sort_idx.argsort()
        val_resort_idx = val_sort_idx.argsort()
        masked_scores = masked_scores.index_select(dim=-1, index=idx_resort_idx)
        masked_scores = masked_scores.index_select(dim=-1, index=val_resort_idx)
        return masked_scores

    def write_tensor(self, x, msg):
        # Helper to dump tensor to disk for debugging
        with open('temp.txt', "w+") as log_file:
            log_file.write(msg)
            np.savetxt(log_file, x.cpu().detach().numpy())


# ---------------------------------------------------------
# Mixture-of-Experts over edges: per-node gating + experts
# Each expert is a SpLearner producing edge scores/masks
# ---------------------------------------------------------
class MoE(nn.Module):
    """Sparsely gated mixture of experts layer.

    Each expert is a SpLearner that outputs edge scores.
    The gating network assigns each node to k experts.
    """

    def __init__(self, in_dim, emb_dim, hidden_size, num_experts, nlayers,
                 activation, k_list, expert_select, noisy_gating=True,
                 coef=1e-2, edge_dim=1, lam=0.1):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = expert_select  # how many experts a node uses
        self.loss_coef = coef
        self.k_list = k_list
        self.emb_dim = emb_dim
        self.num_experts = num_experts

        # Instantiate experts; each expert sees [x_i, x_j, edge_attr]
        self.experts = nn.ModuleList([
            SpLearner(
                nlayers=nlayers,
                in_dim=in_dim * 2 + edge_dim,
                hidden=hidden_size,
                activation=activation,
                k=k
            ) for k in k_list
        ])

        # Gating network weights: node features -> logits over experts
        self.w_gate = nn.Parameter(torch.zeros(in_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(in_dim, num_experts), requires_grad=True)

        # Optional topological edge scores (LocalDegree, ForestFire, etc.)
        self.topo_val = None
        self.lam = lam

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """Squared coefficient of variation, used as load-balancing loss."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Load per expert = number of nodes that picked this expert."""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Backprop-friendly probability of being in top-k under noisy gating."""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, edge_index, train, noise_epsilon=1e-1):
        """Noisy top-k gating (Shazeer et al. 2017). Returns node->expert gates."""
        # Node-level logits over experts
        clean_logits = x @ self.w_gate  # [num_nodes, num_experts]

        if self.noisy_gating and train:
            # Add data-dependent Gaussian noise to logits
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Take top-(k+1) logits per node
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]       # [num_nodes, k]
        top_k_indices = top_indices[:, :self.k]     # [num_nodes, k]

        # Softmax over selected experts, then scatter back to full expert dim
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)  # [num_nodes, num_experts]

        # Expected load (for regularization)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, edge_index, temp, edge_attr=None, training=False):
        """Compute per-edge sparsification mask and MoE load-balancing loss."""
        # Node-level expert gates: [num_nodes, num_experts]
        node_gates, load = self.noisy_top_k_gating(x, edge_index, self.training)

        # Load-balancing loss: encourage uniform usage over experts
        importance = node_gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef
        self.importance = node_gates.mean(0)
        self.load = load

        # Convert node-level gates to edge-level gates using source node index
        edge_gates = torch.index_select(node_gates, dim=0, index=edge_index[0])  # [num_edges, num_experts]

        # Run each expert to get per-edge scores
        expert_outputs = []
        for i in range(self.num_experts):
            expert_i_output = self.experts[i](x, edge_index, edge_attr, temp, training)  # [num_edges]
            if self.topo_val is not None:
                # Optionally blend in topological scores
                expert_i_output = expert_i_output * self.lam + self.topo_val[:, i % 4]
            expert_outputs.append(expert_i_output)

        # Stack experts: [num_edges, num_experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # Mixture: weighted average over experts per edge
        gated_output = edge_gates * expert_outputs
        gated_output = gated_output.mean(dim=1)  # [num_edges]

        # Per-node top-k edges based on mixture scores
        node_idx, num_edges_per_node = edge_index[0].unique(return_counts=True)
        k_per_node = torch.sum(node_gates * torch.unsqueeze(self.k_list, 0), dim=1)
        k_edges_per_node = (k_per_node * num_edges_per_node).round().long()
        k_edges_per_node = torch.where(
            k_edges_per_node > 0,
            k_edges_per_node,
            torch.ones_like(k_edges_per_node, device=k_edges_per_node.device)
        )

        # Global sort and grouping by source node
        sparse_values, val_sort_idx = gated_output.sort(descending=True)
        sparse_idx0 = edge_index[0].index_select(dim=-1, index=val_sort_idx)
        idx_sort_idx = sparse_idx0.argsort(stable=True, dim=-1, descending=False)
        scores_sorted = sparse_values.index_select(dim=-1, index=idx_sort_idx)

        # Node-specific thresholds
        edge_start_indices = torch.cat(
            (torch.tensor([0], device=edge_index.device), torch.cumsum(num_edges_per_node[:-1], dim=0))
        )
        edge_end_indices = torch.abs(torch.add(edge_start_indices, k_edges_per_node) - 1).long()
        node_keep_thre_cal = torch.index_select(scores_sorted, dim=-1, index=edge_end_indices)
        node_keep_thre_augmented = node_keep_thre_cal.repeat_interleave(num_edges_per_node)

        # BinaryStep to get hard {0,1} edge mask
        mask = BinaryStep.apply(scores_sorted - node_keep_thre_augmented + 1e-12)

        # Restore original edge order
        idx_resort_idx = idx_sort_idx.argsort()
        val_resort_idx = val_sort_idx.argsort()
        mask = mask.index_select(dim=-1, index=idx_resort_idx)
        mask = mask.index_select(dim=-1, index=val_resort_idx)

        # Cache per-node stats
        self.num_edges_per_node = num_edges_per_node
        self.k_edges_per_node = k_edges_per_node
        self.k_per_node = k_per_node

        return mask, loss

    def get_topo_val(self, edge_index):
        """Compute 4 networkit-based topological scores for each edge."""
        G = nx.DiGraph()
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
        G = nk.nxadapter.nx2nk(G)
        G.indexEdges()

        # A set of sparsification scores per edge
        lds = nk.sparsification.LocalDegreeScore(G).run().scores()
        ffs = nk.sparsification.ForestFireScore(G, 0.6, 5.0).run().scores()
        triangles = nk.sparsification.TriangleEdgeScore(G).run().scores()
        lss = nk.sparsification.LocalSimilarityScore(G, triangles).run().scores()
        scan = nk.sparsification.SCANStructuralSimilarityScore(G, triangles).run().scores()

        topo_val = torch.tensor([lds, ffs, lss, scan], device=edge_index.device).t()
        normalized_features = F.normalize(topo_val, dim=0)
        self.topo_val = normalized_features


# -------------------------------------------------
# High-level model container:
#   - MoE learner produces edge mask
#   - Net is the downstream GNN
# -------------------------------------------------
class MoG(nn.Module):
    def __init__(self, in_dim, emb_dim, out_channels, edge_dim, args, device, params=None):
        super(MoG, self).__init__()
        self.args = args
        self.device = device
        self.k_list = torch.tensor(args["k_list"], device=device)

        # MoE-based sparsification learner
        self.learner = MoE(
            in_dim=in_dim,
            emb_dim=emb_dim,
            hidden_size=args["hidden_spl"],
            num_experts=self.k_list.size(0),
            nlayers=args["num_layers_spl"],
            activation=nn.ReLU(),
            k_list=self.k_list,
            expert_select=args['expert_select'],
            edge_dim=edge_dim,
            lam=args['lam']
        )

        # Task GNN (NNConv + pooling)
        self.gnn = Net(in_dim=in_dim, out_dim=out_channels)


# -------------------------------
# Utilities for GNN backbone
# -------------------------------
def normalized_cut_2d(edge_index, pos):
    # Edge weights = Euclidean distance between endpoints in pos
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


# ---------------------------
# GNN backbone using NNConv
# ---------------------------
class Net(torch.nn.Module):
    def __init__(self, in_dim=32, out_dim=32):
        super().__init__()
        # Edge network for first NNConv (2-dim edge_attr -> in_dim*32 weights)
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, in_dim * 32),
        )
        self.conv1 = NNConv(in_dim, 32, nn1, aggr='max')

        # Edge network for second NNConv (2-dim edge_attr -> 32*64 weights)
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 32 * 64),
        )
        self.conv2 = NNConv(32, 64, nn2, aggr='max')

        # Final MLP head
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, out_dim)

        # Cartesian coordinates as new edge_attr after pooling
        self.transform = T.Cartesian(cat=False)

    def forward(self, data, mask=None):
        # First NNConv + nonlinearity, optionally masked edges
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr, edge_mask=mask))

        # Graclus pooling with normalized-cut weights
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=self.transform)

        # Second NNConv + nonlinearity on pooled graph
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr, edge_mask=mask))

        # Second level of pooling
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        # Global pooling + MLP classifier
        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


# ============================================================
# Custom NNConv with optional edge_mask (used in Net)
# ============================================================
from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class NNConv(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.

    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, i.e. an MLP.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`.
        aggr (str, optional): Aggregation scheme (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`). (default: :obj:`"add"`)
        root_weight (bool, optional): If :obj:`False`, transformed root node
            features are not added. (default: :obj:`True`)
        bias (bool, optional): If :obj:`False`, no additive bias is learned.
            (default: :obj:`True`)
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, nn: Callable, aggr: str = 'max',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]

        if root_weight:
            self.lin = Linear(in_channels[1], out_channels, bias=False,
                              weight_initializer='uniform')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        if self.root_weight:
            self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, edge_mask=None) -> Tensor:
        # x: node features (or pair of source/target features)
        # edge_mask: optional per-edge scalar used to down-weight messages

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Store / register edge_mask
        if isinstance(edge_mask, Tensor):
            self.edge_mask = edge_mask
        else:
            self.register_parameter("edge_mask", None)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        # Add transformed root node features if enabled
        x_r = x[1]
        if x_r is not None and self.root_weight:
            out = out + self.lin(x_r)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # Compute edge-conditioned weight matrices
        weight = self.nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)

        # Standard NNConv message: x_j * W(e_ij)
        m = torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

        # If edge_mask matches number of edges, modulate messages
        if m.size(0) == self.edge_mask.size(0):
            m = m * self.edge_mask.unsqueeze(1)
        return m

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr}, nn={self.nn})')

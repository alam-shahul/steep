from typing import Optional, Tuple, Union

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

# from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

cudnn.deterministic = True
cudnn.benchmark = True


class STAGATE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        d_model: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_hidden = num_hidden
        self.d_model = d_model

        self._init_encoder()
        self._init_decoder()

    def _init_encoder(self):
        self.conv1 = GATConv(
            self.in_dim,
            self.num_hidden,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv2 = GATConv(
            self.num_hidden,
            self.d_model,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )

    def _init_decoder(self):
        self.conv3 = GATConv(
            self.d_model,
            self.num_hidden,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv4 = GATConv(
            self.num_hidden,
            self.in_dim,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )

    def encode(self, features, edge_index, edge_mask=None):
        h1 = F.elu(self.conv1(features, edge_index, edge_mask=edge_mask))
        h2 = self.conv2(h1, edge_index, attention=False, edge_mask=edge_mask)
        return h2

    def decode(self, h2, edge_index, edge_mask=None):
        h3 = F.elu(self.conv3(h2, edge_index, attention=True, edge_mask=edge_mask))
        h4 = self.conv4(h3, edge_index, attention=False, edge_mask=edge_mask)
        return h4

    def forward(self, data, edge_mask=None):
        features = data.x
        edge_index = data.edge_index
        h2 = self.encode(features, edge_index, edge_mask=edge_mask)
        h4 = self.decode(h2, edge_index, edge_mask=edge_mask)
        outputs = {
            "embedding": h2,
            "logits": h4,
        }
        return outputs


class STAGATEVAE(STAGATE):
    def _init_encoder(self):
        self.conv1 = GATConv(
            self.in_dim,
            self.num_hidden,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv2 = GATConv(
            self.num_hidden,
            self.d_model,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv2_5 = GATConv(
            self.num_hidden,
            self.d_model,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )

    def encode(self, features, edge_index, edge_mask=None):
        h1 = F.elu(self.conv1(features, edge_index, edge_mask=edge_mask))
        mean = self.conv2(h1, edge_index, attention=False, edge_mask=edge_mask)
        logvar = self.conv2_5(h1, edge_index, attention=False, edge_mask=edge_mask)
        return mean, logvar

    def reparametrize(self, mean: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, h2, edge_index, edge_mask=None):
        h3 = F.elu(self.conv3(h2, edge_index, attention=True, edge_mask=edge_mask))
        h4 = self.conv4(h3, edge_index, attention=False, edge_mask=edge_mask)
        return h4

    def forward(self, data, edge_mask=None):
        features = data.x
        edge_index = data.edge_index
        mean, logvar = self.encode(features, edge_index, edge_mask=edge_mask)
        h2 = self.reparametrize(mean, logvar)
        h4 = self.decode(h2, edge_index, edge_mask=edge_mask)
        outputs = {
            "mean": mean,
            "logvar": logvar,
            "embedding": h2,
            "logits": h4,
        }
        return outputs


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src

        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        self._alpha = None
        self.attentions = None
        self.edge_mask = None

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        return_attention_weights=None,
        attention=True,
        tied_attention=None,
        edge_mask: OptTensor = None,
    ):
        heads, channels = self.heads, self.out_channels

        self.edge_mask = edge_mask

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, heads, channels)
        else:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, heads, channels)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, heads, channels)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)

        if tied_attention is None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, torch_sparse.SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)

        if self.edge_mask is not None:
            if self.edge_mask.dim() == 1:
                alpha = alpha * self.edge_mask.view(-1, 1)
            else:
                alpha = alpha * self.edge_mask

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"

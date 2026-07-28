# =========================
# STAGATE + MoG (MoE) Wrapper
# =========================
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class STAGATE_MoG(nn.Module):
    """Wrap STAGATE with a MoG(MoE) edge-sparsification learner."""

    def __init__(
        self,
        stagate: nn.Module,
        learner: nn.Module,
        edge_dim: int = 1,
        use_pos_dist: bool = True,
        mask_encoder_only: bool = False,
    ):
        super().__init__()
        self.stagate = stagate
        self.learner = learner
        self.edge_dim = edge_dim
        self.use_pos_dist = use_pos_dist
        self.mask_encoder_only = mask_encoder_only

    @torch.no_grad()
    def _const_edge_attr(self, edge_index: Tensor) -> Tensor:
        E = edge_index.size(1)
        return torch.ones((E, self.edge_dim), device=edge_index.device)

    def _build_edge_attr(self, data) -> Tensor:
        """Build edge_attr with shape [E, edge_dim] for MoE experts."""
        edge_index = data.edge_index

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            ea = data.edge_attr
            if ea.dim() == 1:
                ea = ea.view(-1, 1)
            if ea.size(0) == edge_index.size(1) and ea.size(1) == self.edge_dim:
                return ea

        if self.use_pos_dist and hasattr(data, "pos") and data.pos is not None:
            src, dst = edge_index[0], edge_index[1]
            dist = torch.norm(data.pos[src] - data.pos[dst], p=2, dim=1, keepdim=True)  # [E,1]
            if self.edge_dim == 1:
                return dist
            return dist.repeat(1, self.edge_dim)

        return self._const_edge_attr(edge_index)

    def _encode_stagate(self, x: Tensor, edge_index: Tensor, edge_mask: Optional[Tensor]):
        h1 = F.elu(self.stagate.conv1(x, edge_index, attention=True, edge_mask=edge_mask))
        h2 = self.stagate.conv2(h1, edge_index, attention=False, edge_mask=edge_mask)
        return h1, h2

    def _decode_stagate(self, h2: Tensor, edge_index: Tensor, edge_mask: Optional[Tensor]):
        h3 = F.elu(self.stagate.conv3(h2, edge_index, attention=True, edge_mask=edge_mask))
        h4 = self.stagate.conv4(h3, edge_index, attention=False, edge_mask=edge_mask)
        return h3, h4

    def forward(self, data, temp: float = 1.0) -> Dict[str, Tensor]:
        x = data.x
        edge_index = data.edge_index
        edge_attr = self._build_edge_attr(data)
        mog_output = self.learner(
            x=x,
            edge_index=edge_index,
            temp=temp,
            edge_attr=edge_attr,
            training=self.training,
        )
        edge_mask = mog_output["edge_mask"]
        mog_loss = mog_output["loss"]

        enc_mask = edge_mask
        dec_mask = None if self.mask_encoder_only else edge_mask

        outputs: Dict[str, Tensor] = {"edge_mask": edge_mask, "mog_loss": mog_loss}

        is_vae = hasattr(self.stagate, "conv2_5")

        if not is_vae:
            _, h2 = self._encode_stagate(x, edge_index, enc_mask)
            _, h4 = self._decode_stagate(h2, edge_index, dec_mask)
            outputs.update({"embedding": h2, "logits": h4})
            return outputs

        h1 = F.elu(self.stagate.conv1(x, edge_index, attention=True, edge_mask=enc_mask))
        mean = self.stagate.conv2(h1, edge_index, attention=False, edge_mask=enc_mask)
        logvar = self.stagate.conv2_5(h1, edge_index, attention=False, edge_mask=enc_mask)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        h2 = mean + eps * std

        _, h4 = self._decode_stagate(h2, edge_index, dec_mask)

        outputs.update({"mean": mean, "logvar": logvar, "embedding": h2, "logits": h4})
        return outputs

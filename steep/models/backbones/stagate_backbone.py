from abc import ABC, abstractmethod
from typing import Dict

import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import Tensor

from ..stagate_model import STAGATE, STAGATEVAE

cudnn.deterministic = True
cudnn.benchmark = False


class BaseBackbone(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data, edge_mask=None):
        pass


class STAGATEBackbone(BaseBackbone):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        d_model: int,
        return_reconstruction: bool = True,
    ):
        super().__init__()
        self.model = STAGATE(
            in_dim=in_dim,
            num_hidden=num_hidden,
            d_model=d_model,
        )
        self.return_reconstruction = return_reconstruction

    def forward(self, data, edge_mask=None) -> Dict[str, Tensor]:
        out = self.model(data, edge_mask=edge_mask)
        if not self.return_reconstruction and "logits" in out:
            out = {k: v for k, v in out.items() if k != "logits"}
        return out


class STAGATEVAEBackbone(BaseBackbone):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        d_model: int,
    ):
        super().__init__()
        self.model = STAGATEVAE(
            in_dim=in_dim,
            num_hidden=num_hidden,
            d_model=d_model,
        )

    def forward(self, data, edge_mask=None) -> Dict[str, Tensor]:
        return self.model(data, edge_mask=edge_mask)

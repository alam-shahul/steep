from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch_geometric.data import Data


class Loss(nn.Module):
    @abstractmethod
    def forward(self, inputs: Data, outputs: dict[Tensor]):
        pass


class VAELoss(Loss):
    def __init__(self, total_epochs: int, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.kl_divergence = GaussianKLDivergence(reduction=reduction)
        self.total_epochs = total_epochs

    def forward(
        self,
        inputs: Data,
        outputs: dict[Tensor],
    ):
        mean = outputs["mean"]
        logvar = outputs["logvar"]
        labels = inputs.x
        logits = outputs["logits"]
        beta = 1  # TODO: Annealing

        return self.mse(logits, labels) + beta * self.kl_divergence(mean, logvar)


class GaussianKLDivergence(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, mean: Tensor, logvar: Tensor):
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl = kl.sum(dim=-1)  # sum over latent dim

        # Reduce over batch
        if self.reduction == "mean":
            return kl.mean()
        elif self.reduction == "sum":
            return kl.sum()
        else:
            return kl


class NodeMSELoss(Loss):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        inputs: Data,
        outputs: dict[Tensor],
    ):
        labels = inputs.x
        logits = outputs["logits"]

        return self.mse(logits, labels)


class NLLLoss(Loss):
    def __init__(self):
        super().__init__()
        self.nll == nn.NLLLoss()

    def forward(
        self,
        inputs: Data,
        outputs: dict[Tensor],
    ):
        labels = inputs.y
        logits = outputs["logits"]

        return self.mse(logits, labels)

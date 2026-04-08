from __future__ import annotations

import torch
from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    def __init__(self, name: str = "l1", beta: float = 1.0) -> None:
        super().__init__()
        self.name = name.lower()
        self.beta = beta

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        if self.name == "l1":
            return torch.nn.functional.l1_loss(outputs, targets)
        if self.name == "mse":
            return torch.nn.functional.mse_loss(outputs, targets)
        if self.name == "smooth_l1":
            return torch.nn.functional.smooth_l1_loss(outputs, targets, beta=self.beta)
        raise ValueError(f"Unsupported reconstruction loss: {self.name}")

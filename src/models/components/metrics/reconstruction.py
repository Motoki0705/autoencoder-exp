from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class _ScalarAccumulator:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int) -> None:
        self.total += float(value) * n
        self.count += int(n)

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


@dataclass
class AutoencoderMetricCollection:
    names: tuple[str, ...] = ("loss", "mae", "mse", "psnr")
    _accumulators: dict[str, _ScalarAccumulator] = field(init=False)

    def __post_init__(self) -> None:
        self._accumulators = {name: _ScalarAccumulator() for name in self.names}

    def update(self, outputs: Tensor, targets: Tensor, loss: Tensor) -> None:
        batch_size = int(targets.shape[0])
        detached_outputs = outputs.detach()
        detached_targets = targets.detach()
        mse = torch.mean((detached_outputs - detached_targets) ** 2)
        mae = torch.mean(torch.abs(detached_outputs - detached_targets))
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))

        values = {
            "loss": float(loss.detach().item()),
            "mae": float(mae.item()),
            "mse": float(mse.item()),
            "psnr": float(psnr.item()),
        }
        for name, value in values.items():
            if name in self._accumulators:
                self._accumulators[name].update(value, batch_size)

    def compute(self, prefix: str | None = None) -> dict[str, float]:
        prefix = prefix or ""
        return {f"{prefix}{name}": accumulator.compute() for name, accumulator in self._accumulators.items()}

    def reset(self) -> None:
        for accumulator in self._accumulators.values():
            accumulator.reset()

    def summary(self, prefix: str | None = None) -> dict[str, Any]:
        return self.compute(prefix=prefix)

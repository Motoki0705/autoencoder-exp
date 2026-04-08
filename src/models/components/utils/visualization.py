from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torchvision.utils import make_grid, save_image


def should_save_visualization(epoch: int, interval: int) -> bool:
    if interval <= 0:
        return False
    return epoch % interval == 0


def build_comparison_grid(inputs: Tensor, outputs: Tensor, targets: Tensor, max_images: int = 4) -> Tensor:
    count = max(1, min(max_images, inputs.shape[0]))
    stacked = torch.cat(
        [
            inputs[:count].detach().cpu(),
            outputs[:count].detach().cpu(),
            targets[:count].detach().cpu(),
        ],
        dim=0,
    ).clamp(0.0, 1.0)
    return make_grid(stacked, nrow=count)


def save_visualization_to_disk(grid: Tensor, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    save_image(grid, output_path)
    return output_path


def save_visualization_to_tensorboard(
    logger: TensorBoardLogger | None,
    tag: str,
    grid: Tensor,
    global_step: int,
) -> bool:
    if logger is None:
        return False
    experiment = logger.experiment
    experiment.add_image(tag, grid, global_step=global_step)
    return True

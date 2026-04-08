from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Any

import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch import Tensor, nn

from src.models.components.metrics import AutoencoderMetricCollection
from src.models.components.optim import build_optimizer, build_scheduler
from src.models.components.utils import (
    build_comparison_grid,
    save_visualization_to_disk,
    save_visualization_to_tensorboard,
    should_save_visualization,
)


class AutoencoderLitModule(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        optimizer: dict[str, Any],
        scheduler: dict[str, Any] | None = None,
        loss: dict[str, Any] | None = None,
        log_images_every_n_epochs: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["network", "logger"])
        self.app_logger = logger
        self.network = instantiate(network, logger=logger) if isinstance(network, Mapping | DictConfig) else network
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.loss_cfg = loss or {"_target_": "src.models.components.losses.reconstruction.ReconstructionLoss"}
        self.criterion = instantiate(self.loss_cfg)
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.train_metrics = AutoencoderMetricCollection()
        self.val_metrics = AutoencoderMetricCollection()
        self.test_metrics = AutoencoderMetricCollection()
        self.visualization_root: Path | None = None
        self._latest_batches: dict[str, dict[str, Tensor]] = {}
        if self.app_logger is not None:
            self.app_logger.info(
                "Initialized AutoencoderLitModule with network=%s loss=%s scheduler_enabled=%s log_images_every_n_epochs=%s",
                self.network.__class__.__name__,
                self.loss_cfg.get("_target_", self.criterion.__class__.__name__) if isinstance(self.loss_cfg, Mapping | DictConfig) else self.criterion.__class__.__name__,
                self.scheduler_cfg is not None,
                self.log_images_every_n_epochs,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, outputs = self._shared_step(batch, stage="train")
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch["targets"].shape[0],
        )
        self.train_metrics.update(outputs, batch["targets"], loss)
        self._remember_batch(stage="train", inputs=batch["inputs"], outputs=outputs, targets=batch["targets"])
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, outputs = self._shared_step(batch, stage="val")
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["targets"].shape[0],
        )
        self.val_metrics.update(outputs, batch["targets"], loss)
        self._remember_batch(stage="val", inputs=batch["inputs"], outputs=outputs, targets=batch["targets"])
        return loss

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, outputs = self._shared_step(batch, stage="test")
        self.test_metrics.update(outputs, batch["targets"], loss)
        self._remember_batch(stage="test", inputs=batch["inputs"], outputs=outputs, targets=batch["targets"])
        return loss

    def on_train_epoch_start(self) -> None:
        self._maybe_unfreeze_backbone()
        self.train_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()

    def on_test_epoch_start(self) -> None:
        self.test_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._finalize_stage(stage="train", metrics=self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        if self.trainer is not None and self.trainer.sanity_checking:
            self.val_metrics.reset()
            return
        self._finalize_stage(stage="val", metrics=self.val_metrics)

    def on_test_epoch_end(self) -> None:
        self._finalize_stage(stage="test", metrics=self.test_metrics)

    def configure_optimizers(self) -> Any:
        optimizer = build_optimizer(self.network, self.optimizer_cfg)
        if self.app_logger is not None:
            self.app_logger.info(
                "Configured optimizer: %s with param_groups=%s",
                optimizer.__class__.__name__,
                [
                    {
                        "name": group.get("name", f"group_{index}"),
                        "lr": group.get("lr"),
                        "weight_decay": group.get("weight_decay"),
                        "num_params": sum(parameter.numel() for parameter in group["params"]),
                    }
                    for index, group in enumerate(optimizer.param_groups)
                ],
            )
        if self.scheduler_cfg is None:
            return optimizer
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_cfg=self.scheduler_cfg,
            total_steps=total_steps,
        )
        if scheduler is None:
            return optimizer
        if self.app_logger is not None:
            self.app_logger.info(
                "Configured scheduler: %s total_steps=%s param_groups=%s",
                scheduler["scheduler"].__class__.__name__,
                total_steps,
                [
                    group.get("name", f"group_{index}")
                    for index, group in enumerate(optimizer.param_groups)
                ],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> tuple[Tensor, Tensor]:
        inputs = batch["inputs"]
        targets = batch["targets"]
        outputs = self(inputs)
        loss = self._compute_loss(outputs, targets)
        if torch.isnan(loss):
            raise RuntimeError(f"{stage} loss became NaN.")
        return loss, outputs

    def _compute_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(outputs, targets)

    def configure_runtime_paths(self, checkpoint_dir: str | Path | None = None) -> None:
        if checkpoint_dir is None:
            self.visualization_root = None
            return
        self.visualization_root = Path(checkpoint_dir).parent / "vis"
        if self.app_logger is not None:
            self.app_logger.info("Visualization root directory: %s", self.visualization_root)

    def _remember_batch(self, stage: str, inputs: Tensor, outputs: Tensor, targets: Tensor) -> None:
        self._latest_batches[stage] = {
            "inputs": inputs.detach(),
            "outputs": outputs.detach(),
            "targets": targets.detach(),
        }

    def _finalize_stage(self, stage: str, metrics: AutoencoderMetricCollection) -> None:
        summary = metrics.compute(prefix=f"{stage}/")
        if not summary:
            return
        summary_to_log = {
            key: value
            for key, value in summary.items()
            if key != f"{stage}/loss"
        }
        if summary_to_log:
            self.log_dict(summary_to_log, prog_bar=(stage != "train"), logger=True, on_step=False, on_epoch=True)
        if self.app_logger is not None:
            self.app_logger.info(
                "Epoch %s %s summary: %s",
                self.current_epoch,
                stage,
                {key: round(value, 6) for key, value in summary.items()},
                extra={"is_metric": True},
            )
        self._save_visualizations(stage)
        metrics.reset()

    def _save_visualizations(self, stage: str) -> None:
        if not should_save_visualization(self.current_epoch, self.log_images_every_n_epochs):
            return
        batch = self._latest_batches.get(stage)
        if batch is None:
            return
        grid = build_comparison_grid(
            inputs=batch["inputs"],
            outputs=batch["outputs"],
            targets=batch["targets"],
        )

        tensorboard_logger = self._get_tensorboard_logger()
        tag = f"vis/{stage}"
        saved_to_tb = save_visualization_to_tensorboard(
            logger=tensorboard_logger,
            tag=tag,
            grid=grid,
            global_step=self.global_step,
        )

        saved_path = None
        if self.visualization_root is not None:
            stage_dir = self.visualization_root / stage
            filename = f"epoch_{self.current_epoch:03d}_step_{self.global_step:06d}.png"
            saved_path = save_visualization_to_disk(grid, stage_dir, filename)

        if self.app_logger is not None:
            self.app_logger.info(
                "Saved %s visualization: tensorboard=%s path=%s",
                stage,
                saved_to_tb,
                saved_path,
            )

    def _get_tensorboard_logger(self) -> TensorBoardLogger | None:
        if isinstance(self.logger, TensorBoardLogger):
            return self.logger
        if isinstance(self.logger, list):
            for item in self.logger:
                if isinstance(item, TensorBoardLogger):
                    return item
        return None

    def _maybe_unfreeze_backbone(self) -> None:
        if not hasattr(self.network, "maybe_unfreeze_backbone"):
            return
        if not self.network.maybe_unfreeze_backbone(self.current_epoch):
            return
        if self.app_logger is not None:
            self.app_logger.info(
                "Unfroze backbone at epoch=%s",
                self.current_epoch,
            )

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src.models.lightning_module import AutoencoderLitModule


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, float | str]:
    summary: dict[str, float | str] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            try:
                summary[key] = float(value.item())
                continue
            except (TypeError, ValueError):
                pass
        summary[key] = str(value)
    return summary


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> Any:
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    app_logger = instantiate(
        cfg.loggers.setup,
        name=str(cfg.experiment.name),
        log_file=str(hydra_output_dir / "eval.log"),
    )

    if cfg.get("seed") is not None:
        seed_everything(int(cfg.seed), workers=True)
        app_logger.info("Seeded all frameworks with seed=%s", cfg.seed)

    checkpoint_path = cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("`checkpoint_path` must be provided for evaluation.")
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())

    app_logger.info("Hydra output directory: %s", hydra_output_dir)
    app_logger.info("Experiment: name=%s version=%s", cfg.experiment.name, cfg.experiment.version)
    app_logger.info("Evaluation checkpoint: %s", checkpoint_path)
    if cfg.get("print_config", False):
        app_logger.info("Full config:\n%s", OmegaConf.to_yaml(cfg))
    else:
        app_logger.info(
            "Config summary: data_target=%s model_target=%s trainer_target=%s",
            cfg.data.get("_target_"),
            cfg.model.get("_target_"),
            cfg.trainer.get("_target_"),
        )

    datamodule = instantiate(cfg.data, logger=app_logger)
    loggers = [instantiate(logger_cfg) for name, logger_cfg in cfg.get("loggers", {}).items() if name != "setup"]
    tensorboard_logger = next((item for item in loggers if isinstance(item, TensorBoardLogger)), None)
    checkpoint_dir = None
    if tensorboard_logger is not None:
        tensorboard_log_dir = Path(tensorboard_logger.log_dir).resolve()
        checkpoint_dir = tensorboard_log_dir / "checkpoints"
        app_logger.info("TensorBoard log directory: %s", tensorboard_log_dir)
        app_logger.info("Evaluation artifact directory: %s", checkpoint_dir.parent)

    module = AutoencoderLitModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        network=cfg.model.network,
        optimizer=cfg.model.optimizer,
        scheduler=cfg.model.get("scheduler"),
        loss=cfg.model.get("loss"),
        log_images_every_n_epochs=cfg.model.get("log_images_every_n_epochs", 0),
        logger=app_logger,
        map_location="cpu",
        weights_only=False,
    )
    if hasattr(module, "configure_runtime_paths"):
        module.configure_runtime_paths(checkpoint_dir=checkpoint_dir)
    app_logger.info("Loaded module from checkpoint: %s", checkpoint_path)

    trainer: Trainer = instantiate(cfg.trainer, logger=loggers)
    app_logger.info(
        "Trainer initialized: accelerator=%s devices=%s precision=%s",
        trainer.accelerator.__class__.__name__,
        trainer.num_devices,
        trainer.precision,
    )

    app_logger.info("Starting trainer.test()")
    results = trainer.test(model=module, datamodule=datamodule)
    app_logger.info("Finished trainer.test()")

    metric_summary = _summarize_metrics(trainer.callback_metrics)
    app_logger.info("Evaluation metrics: %s", metric_summary, extra={"is_metric": True})
    app_logger.info("trainer.test() returned: %s", results, extra={"is_metric": True})
    app_logger.info("Hydra run directory: %s", hydra_output_dir)
    return results


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf


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


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> Any:
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    app_logger = instantiate(cfg.loggers.setup, log_file=str(hydra_output_dir / "train.log"))
    if cfg.get("seed") is not None:
        seed_everything(int(cfg.seed), workers=True)
        app_logger.info("Seeded all frameworks with seed=%s", cfg.seed)

    app_logger.info("Hydra output directory: %s", hydra_output_dir)
    app_logger.info("Experiment: name=%s version=%s", cfg.experiment.name, cfg.experiment.version)
    if cfg.get("print_config", False):
        app_logger.info("Full config:\n%s", OmegaConf.to_yaml(cfg))
    else:
        app_logger.info(
            "Config summary: run_mode=%s, data_target=%s, model_target=%s, trainer_target=%s",
            cfg.get("run_mode", "fit"),
            cfg.data.get("_target_"),
            cfg.model.get("_target_"),
            cfg.trainer.get("_target_"),
        )

    datamodule = instantiate(cfg.data, logger=app_logger)
    module = instantiate(cfg.model, logger=app_logger)
    loggers = [instantiate(logger_cfg) for name, logger_cfg in cfg.get("loggers", {}).items() if name != "setup"]
    tensorboard_logger = next((item for item in loggers if isinstance(item, TensorBoardLogger)), None)
    checkpoint_dir = None
    if tensorboard_logger is not None:
        tensorboard_log_dir = Path(tensorboard_logger.log_dir).resolve()
        checkpoint_dir = tensorboard_log_dir / "checkpoints"
        app_logger.info("TensorBoard log directory: %s", tensorboard_log_dir)
        app_logger.info("Checkpoint directory: %s", checkpoint_dir)

    callbacks = []
    for callback_name, callback_cfg in cfg.get("callbacks", {}).items():
        callback_target = str(callback_cfg.get("_target_", ""))
        if "ModelCheckpoint" in callback_target and checkpoint_dir is not None:
            callback = instantiate(callback_cfg, dirpath=str(checkpoint_dir))
            app_logger.info("Configured callback '%s' with dirpath=%s", callback_name, checkpoint_dir)
        else:
            callback = instantiate(callback_cfg)
            app_logger.info("Configured callback '%s' (%s)", callback_name, callback_target)
        callbacks.append(callback)

    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    if hasattr(module, "configure_runtime_paths"):
        module.configure_runtime_paths(checkpoint_dir=checkpoint_dir)
    app_logger.info(
        "Trainer initialized: accelerator=%s devices=%s max_epochs=%s precision=%s",
        trainer.accelerator.__class__.__name__,
        trainer.num_devices,
        trainer.max_epochs,
        trainer.precision,
    )

    run_mode = str(cfg.get("run_mode", "fit"))
    app_logger.info("Starting trainer.%s()", run_mode)
    if run_mode == "fit":
        trainer.fit(model=module, datamodule=datamodule)
    elif run_mode == "validate":
        trainer.validate(model=module, datamodule=datamodule)
    elif run_mode == "test":
        trainer.test(model=module, datamodule=datamodule)
    else:
        raise ValueError(f"Unsupported run_mode: {run_mode}")

    output_dir = hydra_output_dir
    app_logger.info("Finished trainer.%s()", run_mode)
    app_logger.info("Callback metrics: %s", _summarize_metrics(trainer.callback_metrics), extra={"is_metric": True})
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            app_logger.info(
                "Checkpoint summary: best_model_path=%s best_model_score=%s last_model_path=%s",
                callback.best_model_path,
                callback.best_model_score.item() if callback.best_model_score is not None else None,
                callback.last_model_path,
            )
    app_logger.info("Hydra run directory: %s", output_dir)
    return trainer.callback_metrics


if __name__ == "__main__":
    main()

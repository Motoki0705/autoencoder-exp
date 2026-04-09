from __future__ import annotations

from collections.abc import Sequence
import importlib
import logging
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn

from src.models.components.layers.decoder import build_decoder

_DINOV3_BACKBONES = {
    "vits16": "dinov3_vits16",
    "vitl16": "dinov3_vitl16",
}


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        repo_path: str | None = None,
        variant: str = "vits16",
        feature_layers: Sequence[int] = (2, 5, 8, 11),
        strict_checkpoint: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        if not checkpoint_path:
            raise ValueError("DINOv3Encoder requires a checkpoint_path.")
        if not feature_layers:
            raise ValueError("feature_layers must not be empty.")

        self.app_logger = logger
        self.variant = variant
        self.feature_layers = tuple(int(layer_index) for layer_index in feature_layers)
        self.repo_path = _resolve_dinov3_repo_path(checkpoint_path=checkpoint_path, repo_path=repo_path)
        self.backbone = _build_dinov3_backbone(
            variant=variant,
            repo_path=self.repo_path,
            checkpoint_path=checkpoint_path,
        )

        max_layer_index = self.backbone.n_blocks - 1
        invalid_layers = [layer_index for layer_index in self.feature_layers if layer_index < 0 or layer_index > max_layer_index]
        if invalid_layers:
            raise ValueError(f"feature_layers must be within [0, {max_layer_index}], got {invalid_layers}")

        self.embed_dim = int(self.backbone.embed_dim)
        self.out_channels = [self.embed_dim]
        self.load_pretrained_weights(checkpoint_path=checkpoint_path, strict=strict_checkpoint)

    def forward(self, x: Tensor) -> list[Tensor]:
        feature_maps = self.backbone.get_intermediate_layers(
            x,
            n=list(self.feature_layers),
            reshape=True,
        )
        encoded = torch.stack(feature_maps, dim=0).mean(dim=0)
        return [encoded]

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = True) -> dict[str, list[str]]:
        if self.app_logger is not None:
            self.app_logger.info(
                "Loading DINOv3 %s encoder weights from %s (repo_path=%s)",
                self.variant,
                checkpoint_path,
                self.repo_path,
            )
        checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(checkpoint)
        load_result = self.backbone.load_state_dict(state_dict, strict=False)

        if strict and (load_result.missing_keys or load_result.unexpected_keys):
            raise RuntimeError(
                f"Checkpoint load was not strict. Missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        if self.app_logger is not None:
            self.app_logger.info(
                "Loaded DINOv3 encoder checkpoint. missing=%s unexpected=%s strict=%s",
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
                strict,
            )
        return {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }


class DINOv3Autoencoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        repo_path: str | None = None,
        variant: str = "vits16",
        feature_layers: Sequence[int] = (2, 5, 8, 11),
        decoder_name: str = "simple",
        decoder_channels: Sequence[int] = (1024, 512, 256, 128),
        input_channels: int = 3,
        output_channels: int = 3,
        strict_checkpoint: bool = True,
        freeze_backbone: bool = False,
        unfreeze_backbone_epoch: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("The current DINOv3 encoder expects 3-channel input.")

        self.app_logger = logger
        self.freeze_backbone_at_init = freeze_backbone
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch
        self._backbone_frozen = False

        if self.app_logger is not None:
            self.app_logger.info(
                "Initializing DINOv3Autoencoder variant=%s decoder=%s decoder_channels=%s feature_layers=%s checkpoint_path=%s freeze_backbone=%s unfreeze_backbone_epoch=%s",
                variant,
                decoder_name,
                list(decoder_channels),
                list(feature_layers),
                checkpoint_path,
                freeze_backbone,
                unfreeze_backbone_epoch,
            )

        self.encoder = DINOv3Encoder(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            variant=variant,
            feature_layers=feature_layers,
            strict_checkpoint=strict_checkpoint,
            logger=logger,
        )
        self.decoder = build_decoder(
            name=decoder_name,
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            output_channels=output_channels,
        )
        self.output_activation = nn.Sigmoid()

        if self.freeze_backbone_at_init:
            self.freeze_backbone()

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        reconstructed = self.decoder(features)
        if reconstructed.shape[-2:] != x.shape[-2:]:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.output_activation(reconstructed)

    def get_backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def get_param_group_selectors(self) -> dict[str, dict[str, tuple[str, ...]]]:
        return {"backbone": {"prefixes": ("encoder.",)}}

    def freeze_backbone(self) -> None:
        self._set_backbone_trainable(False)

    def unfreeze_backbone(self) -> None:
        self._set_backbone_trainable(True)

    def maybe_unfreeze_backbone(self, epoch: int) -> bool:
        if self.unfreeze_backbone_epoch is None:
            return False
        if not self._backbone_frozen:
            return False
        if epoch < self.unfreeze_backbone_epoch:
            return False
        self.unfreeze_backbone()
        return True

    def is_backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def _set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        self._backbone_frozen = not trainable
        if self.app_logger is not None:
            self.app_logger.info("Backbone trainable=%s", trainable)


def _build_dinov3_backbone(variant: str, repo_path: Path | None, checkpoint_path: str) -> nn.Module:
    builder_name = _DINOV3_BACKBONES.get(variant)
    if builder_name is None:
        raise ValueError(f"Unsupported DINOv3 variant: {variant}")

    backbones_module = _import_dinov3_backbones_module(repo_path=repo_path)
    builder = getattr(backbones_module, builder_name, None)
    if builder is None:
        raise AttributeError(f"dinov3.hub.backbones does not expose {builder_name}")
    return builder(pretrained=False, weights=str(Path(checkpoint_path).expanduser().resolve()))


def _import_dinov3_backbones_module(repo_path: Path | None):
    try:
        return importlib.import_module("dinov3.hub.backbones")
    except ModuleNotFoundError as import_error:
        if repo_path is None:
            raise ModuleNotFoundError(
                "Could not import dinov3. Install the package or provide repo_path, or place the checkpoint inside the dinov3 repository."
            ) from import_error
        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
        return importlib.import_module("dinov3.hub.backbones")


def _resolve_dinov3_repo_path(checkpoint_path: str, repo_path: str | None) -> Path | None:
    candidates: list[Path] = []
    if repo_path:
        candidates.append(Path(repo_path).expanduser().resolve())

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    candidates.extend(parent for parent in checkpoint.parents if parent not in candidates)

    for candidate in candidates:
        if (candidate / "dinov3" / "hub" / "backbones.py").exists():
            return candidate
    return None


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    candidates: list[dict[str, Any]] = []
    if isinstance(checkpoint, dict):
        candidates.append(checkpoint)
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                candidates.append(value)

    for candidate in candidates:
        normalized = {key: value for key, value in candidate.items() if isinstance(value, torch.Tensor)}
        if normalized:
            return normalized
    raise RuntimeError("Could not extract a compatible DINOv3 state dict from the checkpoint.")

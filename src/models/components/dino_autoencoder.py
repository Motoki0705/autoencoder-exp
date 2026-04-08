from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import resnet50

from src.models.components.layers.decoder import build_decoder


class ResNet50Encoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        strict_checkpoint: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self.app_logger = logger
        backbone = resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = [64, 256, 512, 1024, 2048]

        if checkpoint_path:
            self.load_pretrained_weights(checkpoint_path=checkpoint_path, strict=strict_checkpoint)

    def forward(self, x: Tensor) -> list[Tensor]:
        stem = self.stem(x)
        x = self.maxpool(stem)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return [stem, layer1, layer2, layer3, layer4]

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> dict[str, list[str]]:
        if self.app_logger is not None:
            self.app_logger.info("Loading pretrained ResNet50 encoder weights from %s", checkpoint_path)
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(checkpoint)

        model_state = self.state_dict()
        filtered_state = {key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape}
        load_result = self.load_state_dict(filtered_state, strict=False)

        if strict and (load_result.missing_keys or load_result.unexpected_keys):
            raise RuntimeError(
                f"Checkpoint load was not strict. Missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        if self.app_logger is not None:
            self.app_logger.info(
                "Loaded encoder checkpoint. matched=%s missing=%s unexpected=%s strict=%s",
                len(filtered_state),
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
                strict,
            )
        return {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }


class DINOAutoencoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        decoder_name: str = "unet",
        decoder_channels: Sequence[int] = (1024, 512, 256, 128, 64),
        input_channels: int = 3,
        output_channels: int = 3,
        strict_checkpoint: bool = False,
        freeze_backbone: bool = False,
        unfreeze_backbone_epoch: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self.app_logger = logger
        self.freeze_backbone_at_init = freeze_backbone
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch
        self._backbone_frozen = False
        if input_channels != 3:
            raise ValueError("The current ResNet50 encoder expects 3-channel input.")
        if self.app_logger is not None:
            self.app_logger.info(
                "Initializing DINOAutoencoder with decoder=%s decoder_channels=%s output_channels=%s checkpoint_path=%s freeze_backbone=%s unfreeze_backbone_epoch=%s",
                decoder_name,
                list(decoder_channels),
                output_channels,
                checkpoint_path,
                freeze_backbone,
                unfreeze_backbone_epoch,
            )
        self.encoder = ResNet50Encoder(checkpoint_path=checkpoint_path, strict_checkpoint=strict_checkpoint, logger=logger)
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
            self.app_logger.info(
                "Backbone trainable=%s",
                trainable,
            )


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    candidates: list[dict[str, Any]] = []
    if isinstance(checkpoint, dict):
        candidates.append(checkpoint)
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                candidates.append(value)

    for candidate in candidates:
        normalized = _normalize_state_dict(candidate)
        if normalized:
            return normalized
    raise RuntimeError("Could not extract a compatible ResNet50 state dict from the checkpoint.")


def _normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    prefixes = (
        "",
        "module.",
        "encoder.",
        "backbone.",
        "backbone.body.",
        "backbone.0.body.",
    )
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        stripped_key = None
        for prefix in prefixes:
            if prefix and key.startswith(prefix):
                stripped_key = key[len(prefix) :]
                break
            if not prefix:
                stripped_key = key
        if stripped_key is None:
            continue
        if stripped_key.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")):
            normalized[stripped_key] = value
    return normalized

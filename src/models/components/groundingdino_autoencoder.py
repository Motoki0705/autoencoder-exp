from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
import warnings

import torch
from torch import Tensor, nn

from src.models.components.layers.decoder import build_decoder

_SWIN_MODULE_CACHE: dict[str, ModuleType] = {}
_SUPPORTED_BACKBONES = {
    "swin_T_224_1k": "groundingdino_swint_ogc.pth",
    "swin_B_224_22k": "groundingdino_swinb_cogcoor.pth",
    "swin_B_384_22k": "groundingdino_swinb_cogcoor.pth",
    "swin_L_224_22k": None,
    "swin_L_384_22k": None,
}


class GroundingDINOSwinEncoder(nn.Module):
    def __init__(
        self,
        groundingdino_repo_path: str,
        backbone_name: str = "swin_T_224_1k",
        checkpoint_path: str | None = None,
        return_interm_indices: Sequence[int] = (0, 1, 2, 3),
        strict_checkpoint: bool = False,
        allow_missing_checkpoint: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        if backbone_name not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported GroundingDINO backbone_name={backbone_name!r}. "
                f"Expected one of {sorted(_SUPPORTED_BACKBONES)}."
            )

        self.app_logger = logger
        self.backbone_name = backbone_name
        self.return_interm_indices = tuple(return_interm_indices)
        self.allow_missing_checkpoint = allow_missing_checkpoint
        self.backbone = _build_groundingdino_swin_backbone(
            repo_path=groundingdino_repo_path,
            backbone_name=backbone_name,
            return_interm_indices=self.return_interm_indices,
        )
        self.out_channels = [int(self.backbone.num_features[index]) for index in self.return_interm_indices]

        if checkpoint_path:
            self.load_pretrained_weights(checkpoint_path=checkpoint_path, strict=strict_checkpoint)
        else:
            self._log(
                logging.WARNING,
                "GroundingDINO encoder is initialized without checkpoint_path. "
                "Backbone weights will stay random until a checkpoint is configured.",
            )

    def forward(self, x: Tensor) -> list[Tensor]:
        return list(self.backbone.forward_raw(x))

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> dict[str, list[str]]:
        checkpoint_file = Path(checkpoint_path).expanduser()
        if not checkpoint_file.is_file():
            message = (
                f"GroundingDINO checkpoint not found: {checkpoint_file}. "
                "Download an official checkpoint and set network.checkpoint_path. "
                f"Expected filename for backbone {self.backbone_name}: {_SUPPORTED_BACKBONES[self.backbone_name]!r}."
            )
            if self.allow_missing_checkpoint:
                self._log(logging.WARNING, message)
                return {"missing_keys": [], "unexpected_keys": []}
            raise FileNotFoundError(message)

        self._log(logging.INFO, "Loading GroundingDINO encoder weights from %s", checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        model_state = self.backbone.state_dict()
        filtered_state = _extract_compatible_state_dict(checkpoint=checkpoint, model_state=model_state)
        if not filtered_state:
            raise RuntimeError(
                f"Checkpoint {checkpoint_file} does not contain compatible GroundingDINO backbone weights "
                f"for backbone_name={self.backbone_name}."
            )

        load_result = self.backbone.load_state_dict(filtered_state, strict=False)
        if strict and (load_result.missing_keys or load_result.unexpected_keys):
            raise RuntimeError(
                "Checkpoint load was not strict. "
                f"Missing={list(load_result.missing_keys)}, unexpected={list(load_result.unexpected_keys)}"
            )

        self._log(
            logging.INFO,
            "Loaded GroundingDINO backbone checkpoint. matched=%s missing=%s unexpected=%s strict=%s",
            len(filtered_state),
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
            strict,
        )
        return {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }

    def _log(self, level: int, message: str, *args: object) -> None:
        if self.app_logger is not None:
            self.app_logger.log(level, message, *args)
            return
        formatted = message % args if args else message
        warnings.warn(formatted, stacklevel=2)


class GroundingDINOAutoencoder(nn.Module):
    def __init__(
        self,
        groundingdino_repo_path: str,
        backbone_name: str = "swin_T_224_1k",
        checkpoint_path: str | None = None,
        decoder_name: str = "unet",
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
        return_interm_indices: Sequence[int] = (0, 1, 2, 3),
        input_channels: int = 3,
        output_channels: int = 3,
        strict_checkpoint: bool = False,
        allow_missing_checkpoint: bool = True,
        freeze_backbone: bool = False,
        unfreeze_backbone_epoch: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("The current GroundingDINO encoder expects 3-channel input.")
        if len(decoder_channels) != len(return_interm_indices):
            raise ValueError("decoder_channels must have the same length as return_interm_indices.")

        self.app_logger = logger
        self.freeze_backbone_at_init = freeze_backbone
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch
        self._backbone_frozen = False
        if self.app_logger is not None:
            self.app_logger.info(
                "Initializing GroundingDINOAutoencoder with backbone=%s decoder=%s decoder_channels=%s "
                "checkpoint_path=%s allow_missing_checkpoint=%s freeze_backbone=%s unfreeze_backbone_epoch=%s",
                backbone_name,
                decoder_name,
                list(decoder_channels),
                checkpoint_path,
                allow_missing_checkpoint,
                freeze_backbone,
                unfreeze_backbone_epoch,
            )

        self.encoder = GroundingDINOSwinEncoder(
            groundingdino_repo_path=groundingdino_repo_path,
            backbone_name=backbone_name,
            checkpoint_path=checkpoint_path,
            return_interm_indices=return_interm_indices,
            strict_checkpoint=strict_checkpoint,
            allow_missing_checkpoint=allow_missing_checkpoint,
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


def _build_groundingdino_swin_backbone(
    repo_path: str,
    backbone_name: str,
    return_interm_indices: Sequence[int],
) -> nn.Module:
    if tuple(sorted(return_interm_indices)) != tuple(return_interm_indices):
        raise ValueError("return_interm_indices must be sorted in ascending order.")
    module = _load_groundingdino_swin_module(repo_path)
    pretrain_img_size = int(backbone_name.split("_")[-2])
    return module.build_swin_transformer(
        backbone_name,
        pretrain_img_size=pretrain_img_size,
        out_indices=tuple(return_interm_indices),
        dilation=False,
        use_checkpoint=False,
    )


def _load_groundingdino_swin_module(repo_path: str) -> ModuleType:
    repo_root = Path(repo_path).expanduser().resolve()
    module_path = repo_root / "groundingdino" / "models" / "GroundingDINO" / "backbone" / "swin_transformer.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"GroundingDINO repo path is invalid: {repo_root}. Expected file {module_path}."
        )

    cache_key = str(module_path)
    cached = _SWIN_MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module_name = f"_groundingdino_swin_transformer_{hashlib.md5(cache_key.encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create an import spec for {module_path}.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Failed to import GroundingDINO Swin backbone from {module_path}. "
            f"Missing dependency: {error.name}. Install the dependency and retry."
        ) from error

    _SWIN_MODULE_CACHE[cache_key] = module
    return module


def _extract_compatible_state_dict(
    checkpoint: Any,
    model_state: dict[str, Tensor],
) -> dict[str, Tensor]:
    candidates: list[dict[str, Any]] = []
    if isinstance(checkpoint, dict):
        candidates.append(checkpoint)
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                candidates.append(value)

    prefixes = (
        "module.backbone.0.",
        "model.backbone.0.",
        "module.backbone.",
        "model.backbone.",
        "backbone.0.",
        "module.encoder.",
        "model.encoder.",
        "backbone.",
        "encoder.",
        "module.",
        "model.",
        "",
    )

    best_match: dict[str, Tensor] = {}
    for candidate in candidates:
        matched: dict[str, Tensor] = {}
        for key, value in candidate.items():
            if not isinstance(value, torch.Tensor):
                continue
            for prefix in prefixes:
                if prefix and not key.startswith(prefix):
                    continue
                normalized_key = key[len(prefix) :] if prefix else key
                if normalized_key not in model_state:
                    continue
                if model_state[normalized_key].shape != value.shape:
                    continue
                matched[normalized_key] = value
                break
        if len(matched) > len(best_match):
            best_match = matched

    return best_match

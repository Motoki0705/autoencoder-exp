from __future__ import annotations

from collections.abc import Sequence
import importlib
import importlib.machinery
import logging
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn

from src.models.components.layers.decoder import build_decoder


class SAM31Encoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        repo_path: str | None = None,
        strict_checkpoint: bool = False,
        drop_lowest_resolution_feature: bool = True,
        compile_backbone: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        if not checkpoint_path:
            raise ValueError("SAM31Encoder requires a checkpoint_path.")

        self.app_logger = logger
        self.drop_lowest_resolution_feature = drop_lowest_resolution_feature
        self.repo_path = _resolve_sam3_repo_path(checkpoint_path=checkpoint_path, repo_path=repo_path)
        self.scale_factors = [4.0, 2.0, 1.0] if drop_lowest_resolution_feature else [4.0, 2.0, 1.0, 0.5]
        self.backbone = _build_sam31_backbone(
            repo_path=self.repo_path,
            compile_backbone=compile_backbone,
            scale_factors=self.scale_factors,
        )
        self.out_channels = [256] * len(self.scale_factors)
        self.load_pretrained_weights(checkpoint_path=checkpoint_path, strict=strict_checkpoint)

    def forward(self, x: Tensor) -> list[Tensor]:
        sam3_features, _, _, _ = self.backbone(x)
        return list(sam3_features)

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> dict[str, list[str]]:
        checkpoint_file = Path(checkpoint_path).expanduser().resolve()
        if self.app_logger is not None:
            self.app_logger.info(
                "Loading SAM3.1 encoder weights from %s (repo_path=%s)",
                checkpoint_file,
                self.repo_path,
            )
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        model_state = self.backbone.state_dict()
        filtered_state = _extract_compatible_state_dict(checkpoint=checkpoint, model_state=model_state)
        if not filtered_state:
            raise RuntimeError(
                f"Checkpoint {checkpoint_file} does not contain compatible SAM3.1 backbone weights."
            )

        load_result = self.backbone.load_state_dict(filtered_state, strict=False)
        if strict and (load_result.missing_keys or load_result.unexpected_keys):
            raise RuntimeError(
                f"Checkpoint load was not strict. Missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        if self.app_logger is not None:
            self.app_logger.info(
                "Loaded SAM3.1 encoder checkpoint. matched=%s missing=%s unexpected=%s strict=%s",
                len(filtered_state),
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
                strict,
            )
        return {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }


class SAM31Autoencoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        repo_path: str | None = None,
        decoder_name: str = "unet",
        decoder_channels: Sequence[int] = (256, 128, 64),
        input_channels: int = 3,
        output_channels: int = 3,
        strict_checkpoint: bool = False,
        freeze_backbone: bool = False,
        unfreeze_backbone_epoch: int | None = None,
        drop_lowest_resolution_feature: bool = True,
        compile_backbone: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("The current SAM3.1 encoder expects 3-channel input.")

        self.app_logger = logger
        self.freeze_backbone_at_init = freeze_backbone
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch
        self._backbone_frozen = False

        if self.app_logger is not None:
            self.app_logger.info(
                "Initializing SAM31Autoencoder decoder=%s decoder_channels=%s checkpoint_path=%s repo_path=%s freeze_backbone=%s unfreeze_backbone_epoch=%s drop_lowest_resolution_feature=%s compile_backbone=%s",
                decoder_name,
                list(decoder_channels),
                checkpoint_path,
                repo_path,
                freeze_backbone,
                unfreeze_backbone_epoch,
                drop_lowest_resolution_feature,
                compile_backbone,
            )

        self.encoder = SAM31Encoder(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            strict_checkpoint=strict_checkpoint,
            drop_lowest_resolution_feature=drop_lowest_resolution_feature,
            compile_backbone=compile_backbone,
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


def _build_sam31_backbone(
    repo_path: Path | None,
    compile_backbone: bool,
    scale_factors: Sequence[float],
) -> nn.Module:
    if repo_path is None:
        raise ModuleNotFoundError(
            "Could not locate the sam3 repository. Provide repo_path or place the checkpoint inside the sam3 repository."
        )

    _ensure_sam3_package_paths(repo_path)
    necks_module = importlib.import_module("sam3.model.necks")
    position_encoding_module = importlib.import_module("sam3.model.position_encoding")
    vitdet_module = importlib.import_module("sam3.model.vitdet")

    position_encoding = position_encoding_module.PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008 if torch.cuda.is_available() else None,
    )
    compile_mode = "default" if compile_backbone else None
    vit_backbone = vitdet_module.ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        use_fa3=False,
        use_rope_real=False,
    )
    return necks_module.Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=tuple(scale_factors),
        trunk=vit_backbone,
        add_sam2_neck=False,
    )


def _ensure_sam3_package_paths(repo_path: Path) -> None:
    package_dirs = {
        "sam3": repo_path / "sam3",
        "sam3.model": repo_path / "sam3" / "model",
        "sam3.sam": repo_path / "sam3" / "sam",
        "sam3.perflib": repo_path / "sam3" / "perflib",
    }
    for package_name, package_dir in package_dirs.items():
        _ensure_namespace_package(package_name, package_dir)


def _ensure_namespace_package(package_name: str, package_dir: Path) -> None:
    existing = sys.modules.get(package_name)
    package_dir_str = str(package_dir)
    if existing is not None:
        module_paths = list(getattr(existing, "__path__", []))
        if package_dir_str not in module_paths:
            module_paths.append(package_dir_str)
            existing.__path__ = module_paths
        return

    module = ModuleType(package_name)
    module.__path__ = [package_dir_str]
    module.__package__ = package_name
    module.__file__ = str(package_dir / "__init__.py")
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=package_name,
        loader=None,
        is_package=True,
    )
    if module.__spec__.submodule_search_locations is not None:
        module.__spec__.submodule_search_locations.append(package_dir_str)
    sys.modules[package_name] = module


def _resolve_sam3_repo_path(checkpoint_path: str, repo_path: str | None) -> Path | None:
    candidates: list[Path] = []
    if repo_path:
        candidates.append(Path(repo_path).expanduser().resolve())

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    candidates.extend(parent for parent in checkpoint.parents if parent not in candidates)

    for candidate in candidates:
        if (candidate / "sam3" / "model_builder.py").exists():
            return candidate
    return None


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
        "",
        "module.",
        "detector.backbone.vision_backbone.",
        "module.detector.backbone.vision_backbone.",
        "backbone.vision_backbone.",
        "module.backbone.vision_backbone.",
        "vision_backbone.",
        "module.vision_backbone.",
        "detector.backbone.",
        "module.detector.backbone.",
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

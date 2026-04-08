from __future__ import annotations

from collections.abc import Sequence
import copy
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_
from torchvision.models import resnet50

from src.models.components.layers.decoder import build_decoder

try:
    import MultiScaleDeformableAttention as _msda  # type: ignore[import-not-found]
except ImportError:
    _msda = None


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + self.eps).rsqrt()
        bias_term = bias - running_mean * scale
        return x * scale + bias_term


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, scale: float = 2 * torch.pi) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = float(scale)

    def forward(self, mask: Tensor, dtype: torch.dtype) -> Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(dtype=dtype)


def _ms_deform_attn_core_pytorch(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, num_heads, head_dim = value.shape
    _, query_length, _, num_levels, num_points, _ = sampling_locations.shape
    split_sizes = [int(height * width) for height, width in value_spatial_shapes.tolist()]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list: list[Tensor] = []
    for level_index, (height, width) in enumerate(value_spatial_shapes.tolist()):
        value_level = value_list[level_index].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, head_dim, height, width)
        sampling_grid = sampling_grids[:, :, :, level_index].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(
            value_level,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampled)
    attention = attention_weights.transpose(1, 2).reshape(batch_size * num_heads, 1, query_length, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention).sum(-1).view(batch_size, num_heads * head_dim, query_length)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}.")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.im2col_step = 64

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * torch.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for index in range(self.n_points):
            grid_init[:, :, index, :] *= index + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, query_length, _ = query.shape
        _, input_length, _ = input_flatten.shape
        if int((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum().item()) != input_length:
            raise ValueError("Input spatial shapes do not match the flattened input length.")

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(batch_size, input_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size,
            query_length,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            batch_size,
            query_length,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            batch_size,
            query_length,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], dim=-1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + (
                sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}.")

        if _msda is None:
            output = _ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        else:
            output = _msda.ms_deform_attn_forward(
                value,
                input_spatial_shapes,
                input_level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        return self.output_proj(output)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ) -> None:
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: DeformableTransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(encoder_layer) for _ in range(num_layers))

    @staticmethod
    def get_reference_points(spatial_shapes: Tensor, valid_ratios: Tensor, device: torch.device) -> Tensor:
        reference_points_list: list[Tensor] = []
        for level_index, (height, width) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level_index, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level_index, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), dim=-1))
        reference_points = torch.cat(reference_points_list, dim=1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        pos: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_feature_levels: int,
        enc_layers: int,
        dim_feedforward: int,
        dropout: float,
        nheads: int,
        enc_n_points: int,
    ) -> None:
        super().__init__()
        self.encoder = DeformableTransformerEncoder(
            encoder_layer=DeformableTransformerEncoderLayer(
                d_model=hidden_dim,
                d_ffn=dim_feedforward,
                dropout=dropout,
                n_levels=num_feature_levels,
                n_heads=nheads,
                n_points=enc_n_points,
            ),
            num_layers=enc_layers,
        )
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, hidden_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.level_embed)

    @staticmethod
    def _get_valid_ratio(mask: Tensor) -> Tensor:
        _, height, width = mask.shape
        valid_height = (~mask[:, :, 0]).sum(1)
        valid_width = (~mask[:, 0, :]).sum(1)
        return torch.stack((valid_width.float() / width, valid_height.float() / height), dim=-1)

    def forward(self, srcs: Sequence[Tensor], masks: Sequence[Tensor], pos_embeds: Sequence[Tensor]) -> list[Tensor]:
        src_flatten: list[Tensor] = []
        mask_flatten: list[Tensor] = []
        pos_flatten: list[Tensor] = []
        spatial_shapes: list[tuple[int, int]] = []
        for level_index, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds, strict=True)):
            _, _, height, width = src.shape
            spatial_shapes.append((height, width))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos = pos_embed.flatten(2).transpose(1, 2)
            pos_flatten.append(pos + self.level_embed[level_index].view(1, 1, -1))

        src_flatten_tensor = torch.cat(src_flatten, dim=1)
        mask_flatten_tensor = torch.cat(mask_flatten, dim=1)
        pos_flatten_tensor = torch.cat(pos_flatten, dim=1)
        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten_tensor.device)
        level_start_index = torch.cat(
            (
                spatial_shapes_tensor.new_zeros((1,)),
                spatial_shapes_tensor.prod(1).cumsum(0)[:-1],
            )
        )
        valid_ratios = torch.stack([self._get_valid_ratio(mask) for mask in masks], dim=1)
        memory = self.encoder(
            src=src_flatten_tensor,
            spatial_shapes=spatial_shapes_tensor,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            pos=pos_flatten_tensor,
            padding_mask=mask_flatten_tensor,
        )

        outputs: list[Tensor] = []
        start_index = 0
        batch_size, _, hidden_dim = memory.shape
        for height, width in spatial_shapes:
            spatial_size = height * width
            level_memory = memory[:, start_index : start_index + spatial_size]
            outputs.append(level_memory.transpose(1, 2).reshape(batch_size, hidden_dim, height, width))
            start_index += spatial_size
        return outputs


class DeformableDETREncoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        strict_checkpoint: bool = False,
        hidden_dim: int = 256,
        num_feature_levels: int = 4,
        enc_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        nheads: int = 8,
        enc_n_points: int = 4,
        dilation: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        if num_feature_levels < 1:
            raise ValueError("num_feature_levels must be at least 1.")
        self.app_logger = logger
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        backbone = resnet50(weights=None, norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, dilation])
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)

        if num_feature_levels == 1:
            input_proj_in_channels = [2048]
        else:
            input_proj_in_channels = [512, 1024, 2048][: min(3, num_feature_levels)]
        self.num_backbone_feature_levels = len(input_proj_in_channels)
        self.input_proj = nn.ModuleList()
        for in_channels in input_proj_in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
        extra_levels = num_feature_levels - len(input_proj_in_channels)
        extra_in_channels = input_proj_in_channels[-1]
        for _ in range(extra_levels):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(extra_in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
            extra_in_channels = hidden_dim
        for projection in self.input_proj:
            nn.init.xavier_uniform_(projection[0].weight, gain=1.0)
            nn.init.constant_(projection[0].bias, 0.0)

        self.transformer = DeformableTransformerEncoderOnly(
            hidden_dim=hidden_dim,
            num_feature_levels=num_feature_levels,
            enc_layers=enc_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            nheads=nheads,
            enc_n_points=enc_n_points,
        )
        self.out_channels = [64, 256, *([hidden_dim] * num_feature_levels)]

        if self.app_logger is not None and _msda is None:
            self.app_logger.warning(
                "MultiScaleDeformableAttention extension was not found. Falling back to the PyTorch implementation."
            )

        if checkpoint_path:
            self.load_pretrained_weights(checkpoint_path=checkpoint_path, strict=strict_checkpoint)

    def forward(self, x: Tensor) -> list[Tensor]:
        stem = self.stem(x)
        x = self.maxpool(stem)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        if self.num_feature_levels == 1:
            base_features = [layer4]
        else:
            base_features = [layer2, layer3, layer4][: self.num_backbone_feature_levels]

        projected_srcs: list[Tensor] = []
        masks: list[Tensor] = []
        pos_embeds: list[Tensor] = []
        for level_index, feature in enumerate(base_features):
            projected = self.input_proj[level_index](feature)
            projected_srcs.append(projected)
            mask = torch.zeros(
                (projected.shape[0], projected.shape[2], projected.shape[3]),
                dtype=torch.bool,
                device=projected.device,
            )
            masks.append(mask)
            pos_embeds.append(self.position_embedding(mask=mask, dtype=projected.dtype))

        for level_index in range(len(base_features), self.num_feature_levels):
            if level_index == len(base_features):
                projected = self.input_proj[level_index](base_features[-1])
            else:
                projected = self.input_proj[level_index](projected_srcs[-1])
            projected_srcs.append(projected)
            mask = torch.zeros(
                (projected.shape[0], projected.shape[2], projected.shape[3]),
                dtype=torch.bool,
                device=projected.device,
            )
            masks.append(mask)
            pos_embeds.append(self.position_embedding(mask=mask, dtype=projected.dtype))

        encoded_features = self.transformer(projected_srcs, masks, pos_embeds)
        return [stem, layer1, *encoded_features]

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> dict[str, list[str] | int | str]:
        resolved_path = Path(checkpoint_path).expanduser()
        if not resolved_path.exists():
            message = f"Checkpoint file was not found: {resolved_path}"
            if strict:
                raise FileNotFoundError(message)
            if self.app_logger is not None:
                self.app_logger.warning(message)
            return {"missing_keys": [], "unexpected_keys": [], "matched_keys": 0, "checkpoint_path": str(resolved_path)}

        if self.app_logger is not None:
            self.app_logger.info("Loading Deformable-DETR encoder weights from %s", resolved_path)
        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
        try:
            state_dict = _extract_deformable_detr_state_dict(checkpoint)
        except RuntimeError as error:
            if strict:
                raise
            if self.app_logger is not None:
                self.app_logger.warning("Skipping checkpoint load: %s", error)
            return {"missing_keys": [], "unexpected_keys": [], "matched_keys": 0, "checkpoint_path": str(resolved_path)}

        model_state = self.state_dict()
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        load_result = self.load_state_dict(filtered_state, strict=False)

        if strict and (load_result.missing_keys or load_result.unexpected_keys):
            raise RuntimeError(
                f"Checkpoint load was not strict. Missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        if self.app_logger is not None:
            self.app_logger.info(
                "Loaded Deformable-DETR encoder checkpoint. matched=%s missing=%s unexpected=%s strict=%s",
                len(filtered_state),
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
                strict,
            )
        return {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "matched_keys": len(filtered_state),
            "checkpoint_path": str(resolved_path),
        }


class DeformableDETRAutoencoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        decoder_name: str = "unet",
        decoder_channels: Sequence[int] = (256, 256, 256, 128, 64, 32),
        input_channels: int = 3,
        output_channels: int = 3,
        strict_checkpoint: bool = False,
        freeze_backbone: bool = False,
        unfreeze_backbone_epoch: int | None = None,
        hidden_dim: int = 256,
        num_feature_levels: int = 4,
        enc_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        nheads: int = 8,
        enc_n_points: int = 4,
        dilation: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("The current Deformable-DETR encoder expects 3-channel input.")
        self.app_logger = logger
        self.freeze_backbone_at_init = freeze_backbone
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch
        self._backbone_frozen = False
        if self.app_logger is not None:
            self.app_logger.info(
                "Initializing DeformableDETRAutoencoder with decoder=%s decoder_channels=%s output_channels=%s checkpoint_path=%s freeze_backbone=%s unfreeze_backbone_epoch=%s hidden_dim=%s num_feature_levels=%s",
                decoder_name,
                list(decoder_channels),
                output_channels,
                checkpoint_path,
                freeze_backbone,
                unfreeze_backbone_epoch,
                hidden_dim,
                num_feature_levels,
            )
        self.encoder = DeformableDETREncoder(
            checkpoint_path=checkpoint_path,
            strict_checkpoint=strict_checkpoint,
            hidden_dim=hidden_dim,
            num_feature_levels=num_feature_levels,
            enc_layers=enc_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            nheads=nheads,
            enc_n_points=enc_n_points,
            dilation=dilation,
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
            reconstructed = F.interpolate(reconstructed, size=x.shape[-2:], mode="bilinear", align_corners=False)
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


def _extract_deformable_detr_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    candidates: list[dict[str, Any]] = []
    if isinstance(checkpoint, dict):
        candidates.append(checkpoint)
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                candidates.append(value)

    for candidate in candidates:
        normalized = _normalize_deformable_detr_state_dict(candidate)
        if normalized:
            return normalized
    raise RuntimeError("Could not extract a compatible Deformable-DETR state dict from the checkpoint.")


def _normalize_deformable_detr_state_dict(state_dict: dict[str, Any]) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    for raw_key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = raw_key
        for prefix in ("module.", "model.", "network."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith("encoder."):
            key = key[len("encoder.") :]

        backbone_prefixes = (
            ("backbone.0.body.conv1.", "stem.0."),
            ("backbone.0.body.bn1.", "stem.1."),
            ("backbone.0.body.layer1.", "layer1."),
            ("backbone.0.body.layer2.", "layer2."),
            ("backbone.0.body.layer3.", "layer3."),
            ("backbone.0.body.layer4.", "layer4."),
            ("backbone.body.conv1.", "stem.0."),
            ("backbone.body.bn1.", "stem.1."),
            ("backbone.body.layer1.", "layer1."),
            ("backbone.body.layer2.", "layer2."),
            ("backbone.body.layer3.", "layer3."),
            ("backbone.body.layer4.", "layer4."),
        )
        for source_prefix, target_prefix in backbone_prefixes:
            if key.startswith(source_prefix):
                key = target_prefix + key[len(source_prefix) :]
                break

        if key.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4.", "input_proj.", "transformer.")):
            normalized[key] = value
    return normalized

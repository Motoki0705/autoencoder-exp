from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.models.components.layers.transformer import TransformerDecoder, TransformerDecoderLayer


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels: Sequence[int], decoder_channels: Sequence[int], output_channels: int) -> None:
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("UNetDecoder expects at least two encoder stages.")
        if len(decoder_channels) != len(encoder_channels):
            raise ValueError("decoder_channels must have the same length as encoder_channels for the current implementation.")

        skip_channels = list(encoder_channels[:-1])[::-1]
        in_channels = encoder_channels[-1]
        out_channels = list(decoder_channels[:-1])
        self.blocks = nn.ModuleList(
            UpBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip([in_channels, *out_channels[:-1]], skip_channels, out_channels, strict=True)
        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(out_channels[-1], decoder_channels[-1]),
            nn.Conv2d(decoder_channels[-1], output_channels, kernel_size=1),
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        x = features[-1]
        skips = list(features[:-1])[::-1]
        for block, skip in zip(self.blocks, skips, strict=True):
            x = block(x, skip)
        return self.final(x)


class SimpleDecoder(nn.Module):
    def __init__(self, encoder_channels: Sequence[int], decoder_channels: Sequence[int], output_channels: int) -> None:
        super().__init__()
        channels = [encoder_channels[-1], *decoder_channels]
        layers: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:], strict=True):
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ConvBlock(in_channels, out_channels),
                ]
            )
        layers.append(nn.Conv2d(decoder_channels[-1], output_channels, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        return self.decoder(features[-1])


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int, temperature: int = 10000, scale: float = 2 * torch.pi) -> None:
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


class TransformerSimpleDecoder(nn.Module):
    def __init__(self, encoder_channels: Sequence[int], decoder_channels: Sequence[int], output_channels: int) -> None:
        super().__init__()
        if not decoder_channels:
            raise ValueError("TransformerSimpleDecoder expects decoder_channels to be non-empty.")

        hidden_dim = int(decoder_channels[0])
        if hidden_dim % 2 != 0:
            raise ValueError(f"TransformerSimpleDecoder requires an even hidden_dim, got {hidden_dim}.")

        input_channels = int(encoder_channels[-1])
        nhead = _resolve_attention_heads(hidden_dim)
        num_groups = _resolve_group_norm_groups(hidden_dim)

        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.GELU(),
        )
        self.query_score_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            normalize_before=False,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(hidden_dim),
            return_intermediate=False,
        )
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)

        transformer_decoder_channels = [*decoder_channels, 64]
        self.simple_decoder = SimpleDecoder(
            encoder_channels=[hidden_dim],
            decoder_channels=transformer_decoder_channels,
            output_channels=output_channels,
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        feature_map = self.input_projection(features[-1])
        batch_size, channels, height, width = feature_map.shape

        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "TransformerSimpleDecoder expects even height and width "
                f"to reshape top-k=N/4 tokens into a 2D map, got {(height, width)}."
            )

        num_tokens = height * width
        topk = max(1, num_tokens // 4)
        coarse_height = height // 2
        coarse_width = width // 2

        if coarse_height * coarse_width != topk:
            raise ValueError(
                "coarse_height * coarse_width must equal topk. "
                f"Got coarse={(coarse_height, coarse_width)}, topk={topk}."
            )

        mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=feature_map.device)
        pos_embed = self.position_embedding(mask=mask, dtype=feature_map.dtype)

        memory_bnc = feature_map.flatten(2).transpose(1, 2)
        pos_bnc = pos_embed.flatten(2).transpose(1, 2)

        query_scores = self.query_score_head(feature_map).flatten(1)
        topk_indices = torch.topk(query_scores, k=topk, dim=1, sorted=True).indices

        query_tokens_bkc = _gather_tokens(memory_bnc, topk_indices)
        query_pos_bkc = _gather_tokens(pos_bnc, topk_indices)

        memory = memory_bnc.transpose(0, 1)
        pos_tokens = pos_bnc.transpose(0, 1)
        query_tokens = query_tokens_bkc.permute(1, 0, 2)
        query_pos = query_pos_bkc.permute(1, 0, 2)

        decoded_queries = self.transformer_decoder(
            tgt=query_tokens,
            memory=memory,
            memory_key_padding_mask=mask.flatten(1),
            pos=pos_tokens,
            query_pos=query_pos,
        ).squeeze(0)

        decoded_queries_bkc = decoded_queries.permute(1, 0, 2)
        decoded_map = decoded_queries_bkc.transpose(1, 2).reshape(batch_size, channels, coarse_height, coarse_width)
        return self.simple_decoder([decoded_map])


def _gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
    gather_index = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    return torch.gather(tokens, dim=1, index=gather_index)


def _resolve_attention_heads(hidden_dim: int) -> int:
    for candidate in (16, 12, 8, 6, 4, 3, 2, 1):
        if hidden_dim % candidate == 0:
            return candidate
    raise ValueError(f"Could not infer a valid number of attention heads for hidden_dim={hidden_dim}.")


def _resolve_group_norm_groups(hidden_dim: int) -> int:
    for candidate in (32, 16, 8, 4, 2, 1):
        if hidden_dim % candidate == 0:
            return candidate
    raise ValueError(f"Could not infer a valid number of groups for hidden_dim={hidden_dim}.")


def build_decoder(
    name: str,
    encoder_channels: Sequence[int],
    decoder_channels: Sequence[int],
    output_channels: int,
) -> nn.Module:
    if name == "unet":
        return UNetDecoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, output_channels=output_channels)
    if name == "simple":
        return SimpleDecoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, output_channels=output_channels)
    if name == "transformer":
        return TransformerSimpleDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            output_channels=output_channels,
        )
    raise ValueError(f"Unknown decoder name: {name}")

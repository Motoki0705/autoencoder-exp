from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


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
    raise ValueError(f"Unknown decoder name: {name}")

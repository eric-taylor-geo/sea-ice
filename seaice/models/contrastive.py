from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            skip_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock(skip_channels * 2, skip_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 2,
        feature_channels: Sequence[int] = (64, 128, 256, 512),
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_channels = tuple(feature_channels)

        self.encoders = nn.ModuleList()
        c_in = in_channels
        for c_out in self.feature_channels:
            self.encoders.append(ConvBlock(c_in, c_out))
            c_in = c_out

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(
            self.feature_channels[-1], self.feature_channels[-1] * 2
        )
        self.embedding_dim = self.feature_channels[-1] * 2

        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, projection_dim),
        )

    def forward_encoder(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips: list[torch.Tensor] = []
        h = x
        for block in self.encoders:
            h = block(h)
            skips.append(h)
            h = self.pool(h)
        bottleneck = self.bottleneck(h)
        return skips, bottleneck

    def forward_representation(self, x: torch.Tensor) -> torch.Tensor:
        _, bottleneck = self.forward_encoder(x)
        pooled = F.adaptive_avg_pool2d(bottleneck, output_size=(1, 1)).flatten(1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep = self.forward_representation(x)
        z = self.projection_head(rep)
        return F.normalize(z, p=2, dim=1)


class ContrastiveSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: ContrastiveEncoder,
        *,
        out_channels: int = 1,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.apply_sigmoid = apply_sigmoid

        reversed_channels = list(reversed(self.encoder.feature_channels))
        decoder_blocks: list[DecoderBlock] = []

        in_channels = self.encoder.embedding_dim
        for skip_channels in reversed_channels:
            decoder_blocks.append(DecoderBlock(in_channels, skip_channels))
            in_channels = skip_channels

        self.decoders = nn.ModuleList(decoder_blocks)
        self.output = nn.Conv2d(
            self.encoder.feature_channels[0], out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips, bottleneck = self.encoder.forward_encoder(x)
        h = bottleneck
        for block, skip in zip(self.decoders, reversed(skips)):
            h = block(h, skip)

        out = self.output(h)
        if self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out


def freeze_encoder(model: ContrastiveSegmentationModel) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: ContrastiveSegmentationModel) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = True

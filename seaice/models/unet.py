"""
UNet model for sea ice concentration prediction.
Based on the original UNet architecture by Ronneberger et al. (2015).
"""

import torch
import torch.nn as nn

# ================================================
# Convolutional, Encoder, and Decoder blocks
# ================================================


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn = nn.GroupNorm(16, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv_block1 = ConvBlock(in_c, out_c)
        self.conv_block2 = ConvBlock(out_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        p = self.pool(h)
        return h, p


class DecBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, padding=0
        )
        self.conv_block1 = ConvBlock(2 * out_c, out_c)
        self.conv_block2 = ConvBlock(out_c, out_c)

    def forward(self, x, s):
        h = self.up(x)
        h = torch.cat([h, s], axis=1)  # concatenate skip connection
        h = self.conv_block1(h)
        h = self.conv_block2(h)
        return h


# ================================================
# Full UNet
# ================================================


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # encoder
        self.e1 = EncBlock(in_channels, 64)
        self.e2 = EncBlock(64, 128)
        self.e3 = EncBlock(128, 256)
        self.e4 = EncBlock(256, 512)

        # bottleneck
        self.b1 = ConvBlock(512, 1024)
        self.b2 = ConvBlock(1024, 1024)

        # decoder
        self.d4 = DecBlock(1024, 512)
        self.d3 = DecBlock(512, 256)
        self.d2 = DecBlock(256, 128)
        self.d1 = DecBlock(128, 64)

        # 	output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        self.output_sig = nn.Sigmoid()

    def forward(self, x):

        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # bottleneck
        b = self.b1(p4)
        b = self.b2(b)

        # decoder
        d4 = self.d4(b, s4)
        d3 = self.d3(d4, s3)
        d2 = self.d2(d3, s2)
        d1 = self.d1(d2, s1)

        # output
        output = self.output(d1)
        output = self.output_sig(output)

        # if in eval mode, round to nearest 0.1
        if not self.training:
            output = torch.round(output * 10) / 10

        return output

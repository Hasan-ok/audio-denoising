import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            128, 64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64, 32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(
            32, 16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(32, 16)

        # Final output
        self.output = nn.Conv2d(
            16, 1,
            kernel_size=1
        )

    def forward(self, x):

        # -------------------------
        # Encoder
        # -------------------------

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # -------------------------
        # Bottleneck
        # -------------------------

        b = self.bottleneck(
            self.pool(e3)
        )

        # -------------------------
        # Decoder
        # -------------------------

        d3 = self.up3(b)

        # Make dimensions match e3
        d3 = F.interpolate(
            d3,
            size=e3.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        # Make dimensions match e2
        d2 = F.interpolate(
            d2,
            size=e2.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        # Make dimensions match e1
        d1 = F.interpolate(
            d1,
            size=e1.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Residual Double Convolution Block
# ==========================================================

class ResidualDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm1 = nn.BatchNorm2d(
            out_channels
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.BatchNorm2d(
            out_channels
        )

        # --------------------------------------------------
        # Residual connection
        # --------------------------------------------------

        if in_channels != out_channels:

            self.residual = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )

        else:

            self.residual = nn.Identity()


    def forward(self, x):

        residual = self.residual(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual

        x = self.activation(x)

        return x


# ==========================================================
# U-Net
# ==========================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ==================================================
        # Encoder
        # ==================================================

        self.enc1 = ResidualDoubleConv(
            1,
            32
        )

        self.enc2 = ResidualDoubleConv(
            32,
            64
        )

        self.enc3 = ResidualDoubleConv(
            64,
            128
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        # ==================================================
        # Bottleneck
        # ==================================================

        self.bottleneck = ResidualDoubleConv(
            128,
            256
        )

        # ==================================================
        # Decoder
        # ==================================================

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ResidualDoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ResidualDoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ResidualDoubleConv(
            64,
            32
        )

        # ==================================================
        # Output
        # ==================================================

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )


    def forward(self, x):

        # ==================================================
        # Encoder
        # ==================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # ==================================================
        # Bottleneck
        # ==================================================

        b = self.bottleneck(
            self.pool(e3)
        )

        # ==================================================
        # Decoder - Level 3
        # ==================================================

        d3 = self.up3(b)

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

        # ==================================================
        # Decoder - Level 2
        # ==================================================

        d2 = self.up2(d3)

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

        # ==================================================
        # Decoder - Level 1
        # ==================================================

        d1 = self.up1(d2)

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

        # ==================================================
        # Output
        # ==================================================

        return self.output(d1)
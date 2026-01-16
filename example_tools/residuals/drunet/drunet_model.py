"""
DRUNet (Denoising Residual UNet) model implementation.

Based on: "Plug-and-Play Image Restoration with Deep Denoiser Prior"
https://github.com/cszn/DPIR

This implementation matches the pretrained weights structure exactly.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with two 3x3 convolutions and ReLU (no bias)."""

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.res(x)


class UNetRes(nn.Module):
    """
    UNet with residual blocks - matches DPIR/KAIR pretrained weights.

    Architecture:
    - m_head: Conv 3x3, in_nc -> 64
    - m_down1: 4x ResBlock(64) + StrideConv(64->128)
    - m_down2: 4x ResBlock(128) + StrideConv(128->256)
    - m_down3: 4x ResBlock(256) + StrideConv(256->512)
    - m_body: 4x ResBlock(512)
    - m_up3: ConvTranspose(512->256) + 4x ResBlock(256)
    - m_up2: ConvTranspose(256->128) + 4x ResBlock(128)
    - m_up1: ConvTranspose(128->64) + 4x ResBlock(64)
    - m_tail: Conv 3x3, 64 -> out_nc

    Note: All convolutions use bias=False to match pretrained weights.
    """

    def __init__(self, in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4):
        super(UNetRes, self).__init__()

        # Head
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, padding=1, bias=False)

        # Encoder
        self.m_down1 = nn.Sequential(
            *[ResBlock(nc[0]) for _ in range(nb)],
            nn.Conv2d(nc[0], nc[1], 2, stride=2, bias=False)  # Downsample
        )
        self.m_down2 = nn.Sequential(
            *[ResBlock(nc[1]) for _ in range(nb)],
            nn.Conv2d(nc[1], nc[2], 2, stride=2, bias=False)
        )
        self.m_down3 = nn.Sequential(
            *[ResBlock(nc[2]) for _ in range(nb)],
            nn.Conv2d(nc[2], nc[3], 2, stride=2, bias=False)
        )

        # Bottleneck
        self.m_body = nn.Sequential(
            *[ResBlock(nc[3]) for _ in range(nb)]
        )

        # Decoder
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(nc[3], nc[2], 2, stride=2, bias=False),
            *[ResBlock(nc[2]) for _ in range(nb)]
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(nc[2], nc[1], 2, stride=2, bias=False),
            *[ResBlock(nc[1]) for _ in range(nb)]
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(nc[1], nc[0], 2, stride=2, bias=False),
            *[ResBlock(nc[0]) for _ in range(nb)]
        )

        # Tail
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, padding=1, bias=False)

    def forward(self, x):
        # Pad to multiple of 8
        h, w = x.shape[2], x.shape[3]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        x = self.m_body(x4)

        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]

        return x


class DRUNet(nn.Module):
    """
    DRUNet wrapper that handles noise level map concatenation.

    For forensic residual extraction, we use a fixed noise level
    since we want consistent denoising behavior.
    """

    def __init__(self, in_nc=1, out_nc=1, noise_level=15):
        """
        Args:
            in_nc: Input channels (1 for grayscale)
            out_nc: Output channels (1 for grayscale)
            noise_level: Fixed noise level (0-255 scale)
        """
        super(DRUNet, self).__init__()

        # UNet takes image + noise map
        self.unet = UNetRes(in_nc=in_nc + 1, out_nc=out_nc)
        self.noise_level = noise_level / 255.0

    def forward(self, x, noise_level=None):
        """
        Args:
            x: Input image tensor (B, C, H, W) in [0, 1]
            noise_level: Optional noise level override (0-1 scale)

        Returns:
            Denoised image tensor (B, C, H, W) in [0, 1]
        """
        if noise_level is None:
            noise_level = self.noise_level

        b, c, h, w = x.shape
        noise_map = torch.full((b, 1, h, w), noise_level, device=x.device, dtype=x.dtype)
        x_in = torch.cat([x, noise_map], dim=1)

        return self.unet(x_in)


def load_drunet_gray(weights_path, noise_level=15, device=None):
    """
    Load pretrained DRUNet grayscale model.

    Args:
        weights_path: Path to drunet_gray.pth
        noise_level: Noise level for denoising (0-255 scale)
        device: torch device

    Returns:
        Loaded DRUNet model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DRUNet(in_nc=1, out_nc=1, noise_level=noise_level)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.unet.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model

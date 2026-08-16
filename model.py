"""
model.py
--------
A lightweight U-Net that takes a noisy, low-resolution image and outputs
a clean image at 2x the input resolution (e.g. 256x256 -> 512x512).

Why this architecture:
- U-Net's skip connections preserve fine detail (edges, texture) that
  would otherwise be lost/blurred by the encoder-decoder bottleneck.
- Kept intentionally small (32-256 channels) so inference stays fast
  on the H100 benchmark -- KLA explicitly penalizes unnecessarily large
  models on throughput.
- Final upsampling uses PixelShuffle (not naive interpolation) because
  it learns the upsampling instead of just smoothing pixels, giving
  sharper output.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two conv layers + BatchNorm + ReLU. The basic building block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscale by 2 (maxpool) then DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upscale by 2 (bilinear), concat skip connection, then DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle any off-by-one size mismatch from odd input sizes
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                   diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FinalUpsample(nn.Module):
    """
    Learned 2x upsampling using PixelShuffle.
    Input: (B, C, H, W) -> Output: (B, 1, 2H, 2W), values in [0,1].
    (1 output channel because the KLA dataset is grayscale.)
    """

    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)  # doubles H and W, divides channels by 4
        self.out_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)  # force output into [0,1] to match GT range


class RestorationUNet(nn.Module):
    """
    Full model: encoder -> bottleneck -> decoder -> 2x learned upsample.

    base_ch controls model size/speed. 32 is a good speed/quality tradeoff
    for a hackathon timeline. Increase to 48-64 only if you have time to
    retrain and inference time budget allows it.
    """

    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        self.up1 = Up(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up3 = Up(base_ch * 2 + base_ch, base_ch)

        self.final_up = FinalUpsample(base_ch)

    def forward(self, x):
        x1 = self.inc(x)       # base_ch
        x2 = self.down1(x1)    # base_ch*2
        x3 = self.down2(x2)    # base_ch*4
        x4 = self.down3(x3)    # base_ch*8

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        out = self.final_up(x)  # 2x resolution, [0,1]
        return out


if __name__ == "__main__":
    # Quick sanity check: run a dummy batch through the model
    model = RestorationUNet(in_ch=1, base_ch=32)
    dummy = torch.randn(2, 1, 128, 128)
    out = model(dummy)
    print("Input:", dummy.shape, "-> Output:", out.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

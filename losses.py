"""
losses.py
---------
Combined loss = L1 + SSIM + LPIPS.

Why combine three:
- L1 (pixel-wise absolute difference): drives raw pixel accuracy,
  directly related to the PSNR metric you're scored on.
- SSIM loss: directly related to the SSIM metric you're scored on;
  pushes the model to preserve structure/contrast, not just average
  brightness.
- LPIPS: a perceptual loss using a pretrained network's features;
  directly related to the LPIPS metric you're scored on, and helps
  avoid the classic "blurry safe average" output that pure L1 causes.

Weights (l1_w, ssim_w, lpips_w) are a reasonable starting point -- feel
free to tune them if you have time, and report whatever you land on in
your PPT (Slide 7 wants this justified).
"""

import torch
import torch.nn as nn
from pytorch_msssim import ssim
import lpips


class CombinedLoss(nn.Module):
    def __init__(self, l1_w=1.0, ssim_w=1.0, lpips_w=0.1, device="cuda"):
        super().__init__()
        self.l1_w = l1_w
        self.ssim_w = ssim_w
        self.lpips_w = lpips_w
        self.l1 = nn.L1Loss()
        # 'alex' backbone is lightweight and standard for LPIPS
        self.lpips_fn = lpips.LPIPS(net="alex").to(device)
        for p in self.lpips_fn.parameters():
            p.requires_grad = False  # frozen, only used to compute loss

    def forward(self, pred, target):
        pred_c = torch.clamp(pred, 0, 1)
        target_c = torch.clamp(target, 0, 1)

        l1_loss = self.l1(pred_c, target_c)
        ssim_loss = 1 - ssim(pred_c, target_c, data_range=1.0, size_average=True)
        # LPIPS expects inputs roughly in [-1,1]
        lpips_loss = self.lpips_fn(pred_c * 2 - 1, target_c * 2 - 1).mean()

        total = (self.l1_w * l1_loss
                 + self.ssim_w * ssim_loss
                 + self.lpips_w * lpips_loss)

        return total, {
            "l1": l1_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "lpips": lpips_loss.item(),
            "total": total.item(),
        }

"""
train.py
--------
Trains the RestorationUNet on the KLA dataset.

USAGE (run this in a Colab cell, or terminal if local):
    python train.py --gt_dir dataset/train/GT --noisylr_dir dataset/train/NoisyLR \
                     --val_gt_dir dataset/val/GT --val_noisylr_dir dataset/val/NoisyLR \
                     --epochs 30 --batch_size 8 --out_dir weights

EDIT the paths below (or pass as --args) to match wherever you put the
dataset. This script:
  1. Loads train + val data
  2. Trains the model with the combined loss
  3. Tracks PSNR/SSIM on validation each epoch
  4. Saves the best checkpoint to weights/best_model.pth
  5. Logs everything to a CSV so you can screenshot a plot for your PPT
"""

import os
import csv
import argparse
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from pytorch_msssim import ssim as ssim_fn

from model import RestorationUNet
from dataset import KLARestorationDataset
from losses import CombinedLoss


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def evaluate(model, loader, device):
    model.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    with torch.no_grad():
        for noisylr, gt in loader:
            noisylr, gt = noisylr.to(device), gt.to(device)
            pred = torch.clamp(model(noisylr), 0, 1)
            total_psnr += psnr(pred, gt).item() * noisylr.size(0)
            total_ssim += ssim_fn(pred, gt, data_range=1.0, size_average=True).item() * noisylr.size(0)
            n += noisylr.size(0)
    model.train()
    return total_psnr / n, total_ssim / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="dataset/train/GT")
    parser.add_argument("--noisylr_dir", type=str, default="dataset/train/NoisyLR")
    parser.add_argument("--val_gt_dir", type=str, default="dataset/val/GT")
    parser.add_argument("--val_noisylr_dir", type=str, default="dataset/val/NoisyLR")
    parser.add_argument("--out_dir", type=str, default="weights")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--use_synthetic_aug", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = KLARestorationDataset(
        args.gt_dir, args.noisylr_dir,
        patch_size=args.patch_size,
        use_synthetic_aug=args.use_synthetic_aug,
        synthetic_aug_prob=0.3,
    )
    val_ds = KLARestorationDataset(
        args.val_gt_dir, args.val_noisylr_dir,
        patch_size=None, use_synthetic_aug=False, augment_flips=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = RestorationUNet(base_ch=args.base_ch).to(device)
    criterion = CombinedLoss(device=device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "time_sec"])

    best_psnr = -1
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        for noisylr, gt in train_loader:
            noisylr, gt = noisylr.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(noisylr)
            loss, parts = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * noisylr.size(0)
        epoch_loss /= len(train_ds)
        scheduler.step()

        val_psnr, val_ssim = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | train_loss={epoch_loss:.4f} "
              f"| val_PSNR={val_psnr:.2f} | val_SSIM={val_ssim:.4f} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, epoch_loss, val_psnr, val_ssim, elapsed])

        # Save best checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "model_state_dict": model.state_dict(),
                "base_ch": args.base_ch,
                "epoch": epoch,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            }, os.path.join(args.out_dir, "best_model.pth"))
            print(f"  -> New best model saved (PSNR={val_psnr:.2f})")

    print("Training complete. Best PSNR:", best_psnr)


if __name__ == "__main__":
    main()

"""
dataset.py
----------
Loads paired (NoisyLR, GT) images from the KLA dataset.

CONFIRMED REAL FORMAT (checked directly against the dataset):
  - Files are .npy (NumPy arrays), NOT .png
  - Grayscale, single channel (shape: H x W, no channel dimension)
  - GT: 256x256, float32, values in [0, 1]
  - NoisyLR: 128x128, float32, values can go slightly below 0 and above 1
             (this is expected -- due to noise, not a bug)

FOLDER STRUCTURE (matches what you unzipped):
  train_data/train/GT/000000.npy
  train_data/train/NoisyLR/000000.npy
  (filenames match between the two folders)

Also includes an optional SYNTHETIC augmentation: takes a clean GT array,
downsamples it and adds speckle + Gaussian noise at a RANDOMLY sampled
strength. This is explicitly permitted by the problem statement and
helps the model generalize across noise severities it wasn't shown in
the fixed original training pairs.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def add_synthetic_degradation(gt_arr: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Given a clean GT array (H,W) float32 in [0,1], produce a synthetic
    NoisyLR version: downsample by `scale`, then add speckle + Gaussian
    noise at a randomly sampled strength each call.
    """
    h, w = gt_arr.shape
    lr_img = Image.fromarray((gt_arr * 255).astype(np.uint8))
    lr_img = lr_img.resize((w // scale, h // scale), Image.BICUBIC)
    lr = np.array(lr_img).astype(np.float32) / 255.0

    speckle_sigma = random.uniform(0.02, 0.15)
    gaussian_sigma = random.uniform(0.005, 0.05)

    speckle_noise = np.random.randn(*lr.shape).astype(np.float32) * speckle_sigma
    lr_speckled = lr + lr * speckle_noise

    gaussian_noise = np.random.randn(*lr.shape).astype(np.float32) * gaussian_sigma
    lr_final = lr_speckled + gaussian_noise

    return lr_final.astype(np.float32)


class KLARestorationDataset(Dataset):
    def __init__(self, gt_dir, noisylr_dir, patch_size=None,
                 use_synthetic_aug=False, synthetic_aug_prob=0.3,
                 augment_flips=True):
        self.gt_dir = gt_dir
        self.noisylr_dir = noisylr_dir
        self.filenames = sorted(
            f for f in os.listdir(gt_dir) if f.endswith(".npy")
        )
        self.patch_size = patch_size
        self.use_synthetic_aug = use_synthetic_aug
        self.synthetic_aug_prob = synthetic_aug_prob
        self.augment_flips = augment_flips

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)

        if self.use_synthetic_aug and random.random() < self.synthetic_aug_prob:
            noisylr = add_synthetic_degradation(gt, scale=2)
        else:
            noisylr = np.load(os.path.join(self.noisylr_dir, fname)).astype(np.float32)

        if self.patch_size is not None:
            lr_h, lr_w = noisylr.shape
            lr_patch = self.patch_size // 2
            if lr_h > lr_patch and lr_w > lr_patch:
                top = random.randint(0, lr_h - lr_patch)
                left = random.randint(0, lr_w - lr_patch)
                noisylr = noisylr[top:top + lr_patch, left:left + lr_patch]
                gt = gt[top * 2:(top + lr_patch) * 2, left * 2:(left + lr_patch) * 2]

        if self.augment_flips:
            if random.random() < 0.5:
                noisylr = np.ascontiguousarray(noisylr[:, ::-1])
                gt = np.ascontiguousarray(gt[:, ::-1])
            if random.random() < 0.5:
                noisylr = np.ascontiguousarray(noisylr[::-1, :])
                gt = np.ascontiguousarray(gt[::-1, :])

        noisylr_t = torch.from_numpy(noisylr).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt).unsqueeze(0).float()
        return noisylr_t, gt_t

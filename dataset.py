"""
dataset.py
----------
Loads paired (NoisyLR, GT) images from the KLA dataset.

Also includes an optional SYNTHETIC augmentation: takes a clean GT image,
downsamples it and adds speckle + Gaussian noise at a RANDOMLY sampled
strength. This is explicitly permitted by the problem statement
("You may create extra synthetic degraded pairs from the provided GT
images") and helps the model generalize across noise severities it
wasn't shown in the fixed original training pairs.

EXPECTED FOLDER STRUCTURE (adjust paths in train.py if yours differs):

dataset/
  train/
    GT/        <- clean ground-truth images (512x512 or 256x256)
    NoisyLR/   <- degraded images, same filenames as GT
  val/
    GT/
    NoisyLR/
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def add_synthetic_degradation(gt_img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Given a clean GT image (H,W,3) float32 in [0,1], produce a synthetic
    NoisyLR version: downsample by `scale`, then add speckle + Gaussian
    noise at a randomly sampled strength each call.
    """
    h, w = gt_img.shape[:2]
    lr = np.array(
        Image.fromarray((gt_img * 255).astype(np.uint8))
        .resize((w // scale, h // scale), Image.BICUBIC)
    ).astype(np.float32) / 255.0

    # Randomly sampled noise strengths -> teaches the model to handle a
    # RANGE of degradation severity, not just one fixed level.
    speckle_sigma = random.uniform(0.02, 0.15)
    gaussian_sigma = random.uniform(0.005, 0.05)

    speckle_noise = np.random.randn(*lr.shape).astype(np.float32) * speckle_sigma
    lr_speckled = lr + lr * speckle_noise  # multiplicative (speckle) noise

    gaussian_noise = np.random.randn(*lr.shape).astype(np.float32) * gaussian_sigma
    lr_final = lr_speckled + gaussian_noise  # additive Gaussian noise

    # NOTE: we intentionally do NOT clip here -- the real NoisyLR data
    # also exceeds [0,1] sometimes, so the model must learn to handle that.
    return lr_final.astype(np.float32)


class KLARestorationDataset(Dataset):
    def __init__(self, gt_dir, noisylr_dir, patch_size=None,
                 use_synthetic_aug=False, synthetic_aug_prob=0.3,
                 augment_flips=True):
        """
        gt_dir, noisylr_dir: paths to the paired image folders
        patch_size: if set (e.g. 128), randomly crops GT to this size
                    (and NoisyLR to patch_size//2) for faster training.
                    Set to None to use full images.
        use_synthetic_aug: if True, sometimes replace the loaded NoisyLR
                    with a freshly-generated synthetic one (see above).
        synthetic_aug_prob: probability of using synthetic aug per sample.
        augment_flips: random horizontal/vertical flips for more data.
        """
        self.gt_dir = gt_dir
        self.noisylr_dir = noisylr_dir
        self.filenames = sorted(os.listdir(gt_dir))
        self.patch_size = patch_size
        self.use_synthetic_aug = use_synthetic_aug
        self.synthetic_aug_prob = synthetic_aug_prob
        self.augment_flips = augment_flips

    def __len__(self):
        return len(self.filenames)

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        return np.array(img).astype(np.float32) / 255.0

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        gt = self._load_img(os.path.join(self.gt_dir, fname))

        if self.use_synthetic_aug and random.random() < self.synthetic_aug_prob:
            noisylr = add_synthetic_degradation(gt, scale=2)
        else:
            noisylr = self._load_img(os.path.join(self.noisylr_dir, fname))

        # Random crop (helps train faster + acts as augmentation)
        if self.patch_size is not None:
            lr_h, lr_w = noisylr.shape[:2]
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

        # HWC -> CHW tensors
        noisylr_t = torch.from_numpy(noisylr.transpose(2, 0, 1)).float()
        gt_t = torch.from_numpy(gt.transpose(2, 0, 1)).float()
        return noisylr_t, gt_t

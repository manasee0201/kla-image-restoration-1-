# AI-Based Restoration of Degraded Images — KLA Hackathon Solution

## What this is
A U-Net that takes noisy, low-resolution semiconductor inspection images
and restores them to clean, full-resolution images (2x upsampling +
denoising in a single forward pass).

## Repository structure
```
kla_restoration/
  src/
    model.py      -> U-Net architecture
    dataset.py     -> data loading + synthetic noise augmentation
    losses.py       -> combined L1 + SSIM + LPIPS loss
  train.py           -> trains the model
  inference.py        -> standalone inference (this is what evaluators run)
  requirements.txt
  README.md
  weights/            -> trained checkpoint goes here (best_model.pth)
  results/            -> put metric summaries / example images here
```

## Environment setup (Google Colab)
1. Go to https://colab.research.google.com and start a new notebook.
2. Runtime -> Change runtime type -> select GPU (T4 is fine).
3. In the first cell, install dependencies:
```
!pip install pytorch_msssim lpips
```
(torch/torchvision/numpy/Pillow already come pre-installed on Colab.)

## Getting the dataset onto Colab
1. Download the official KLA dataset from the hackathon portal (link is
   in the "Official Resources & Links" section of the student help
   document — check the portal for the current working link).
2. Upload the zip to your Google Drive.
3. In Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
!unzip "/content/drive/MyDrive/YOUR_DATASET.zip" -d /content/dataset
```
4. Confirm the folder structure matches:
```
/content/dataset/train/GT/
/content/dataset/train/NoisyLR/
/content/dataset/val/GT/
/content/dataset/val/NoisyLR/
```
   If the official dataset doesn't have a val split, manually move ~10%
   of the training pairs into a `val/` folder (same filenames in both
   GT and NoisyLR) before training. This is your held-out validation
   set and should NOT be used for training itself.

## Getting this code onto Colab
Easiest: push this whole `kla_restoration/` folder to a GitHub repo,
then in Colab:
```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

## Training
```python
!python train.py \
  --gt_dir /content/dataset/train/GT \
  --noisylr_dir /content/dataset/train/NoisyLR \
  --val_gt_dir /content/dataset/val/GT \
  --val_noisylr_dir /content/dataset/val/NoisyLR \
  --epochs 30 \
  --batch_size 8 \
  --out_dir weights
```
This saves the best checkpoint to `weights/best_model.pth` and a
`weights/training_log.csv` with per-epoch loss/PSNR/SSIM you can plot
for the PPT.

## Running inference (exactly what KLA will run)
```python
!python inference.py \
  --input_dir /content/dataset/val/NoisyLR \
  --output_dir /content/restored_outputs \
  --weights weights/best_model.pth
```

## Reproducing metrics for your report
After inference, compare `restored_outputs/` against the corresponding
GT folder using PSNR/SSIM/LPIPS (a short evaluation snippet can be
added to `src/` if needed — ask if you want this written too).

## Notes on design choices (for your PPT)
- **Architecture**: Lightweight U-Net (base_ch=32) with PixelShuffle
  final upsampling — chosen for a strong speed/quality balance given
  the H100 inference-time scoring axis.
- **Loss**: Combined L1 + SSIM + LPIPS — directly targets all three
  scored metrics instead of optimizing pixel accuracy alone.
- **Augmentation**: Random flips + synthetic noise-level augmentation
  (varying speckle/Gaussian strength on GT-derived synthetic pairs) to
  improve robustness to the out-of-distribution test samples KLA
  mentioned will be included.
- **Inference**: Batched, mixed-precision (fp16 autocast) GPU inference
  to minimize end-to-end runtime.

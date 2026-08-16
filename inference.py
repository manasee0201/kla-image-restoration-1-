"""
inference.py
------------
Standalone inference script -- THIS is what KLA will run to score you.

CONFIRMED REAL FORMAT: inputs and outputs are .npy files (NumPy arrays),
grayscale (H,W), NOT .png images. NoisyLR is 128x128, output/GT is
256x256.

Requirements it satisfies (from the problem statement):
  - Accepts input directory (degraded images) and output directory as args
  - Loads the trained model, runs inference on every image
  - Writes restored .npy outputs to the output directory, same filenames
  - Runs end-to-end with no manual edits needed
  - Uses batching + GPU + mixed precision for speed (scored on H100 runtime)

USAGE:
    python inference.py --input_dir path/to/test/NoisyLR \
                         --output_dir path/to/restored_outputs \
                         --weights weights/best_model.pth
"""

import os
import argparse
import time
import numpy as np
import torch

from model import RestorationUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory of degraded (NoisyLR) .npy input files")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Directory to write restored .npy files to")
    parser.add_argument("--weights", type=str, default="weights/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    t_start = time.time()

    # --- Load model ---
    checkpoint = torch.load(args.weights, map_location=device)
    base_ch = checkpoint.get("base_ch", 32)
    model = RestorationUNet(in_ch=1, base_ch=base_ch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Gather input files ---
    filenames = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    print(f"Found {len(filenames)} files to restore.")

    # --- Batched inference ---
    use_amp = device == "cuda"
    with torch.no_grad():
        for i in range(0, len(filenames), args.batch_size):
            batch_names = filenames[i:i + args.batch_size]
            arrs = [np.load(os.path.join(args.input_dir, name)).astype(np.float32)
                    for name in batch_names]

            # All KLA NoisyLR files are the same size (128x128), so we can
            # always batch them together for a fast single forward pass.
            batch_tensor = torch.from_numpy(
                np.stack(arrs)
            ).unsqueeze(1).float().to(device)  # (B, 1, H, W)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(batch_tensor)
            out = out.squeeze(1).cpu().numpy()  # (B, H, W)

            for j, name in enumerate(batch_names):
                np.save(os.path.join(args.output_dir, name), out[j].astype(np.float32))

            print(f"  Processed {min(i + args.batch_size, len(filenames))}/{len(filenames)}")

    elapsed = time.time() - t_start
    print(f"Done. Restored {len(filenames)} files in {elapsed:.2f}s "
          f"({elapsed / max(len(filenames),1):.3f}s/file).")


if __name__ == "__main__":
    main()

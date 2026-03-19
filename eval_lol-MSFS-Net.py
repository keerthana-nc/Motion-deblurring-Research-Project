"""
Runs MSFS-Net pretrained model on LOL-Blur dataset.
Place this file in the root of the MSFS-Net folder.

Run:
    python eval_lol.py
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.freNet import make_model as build_net
from lol_dataloader import test_dataloader

# ── CONFIG ────────────────────────────────────────────────────────────────────
LOL_BLUR_PATH    = "/content/drive/MyDrive/MSFS-Net/LOL_BLUR"
PRETRAINED_MODEL = "/content/drive/MyDrive/MSFS-Net/model.pkl"
RESULT_DIR       = "results/MSFS_LOLBlur/"

# Exactly which 4 images to visualize (must match filenames in your dataset)
VISUALIZE_NAMES  = ["0012_0032.png", "0017_0039.png", "0042_0060.png", "0045_0083.png"]
# ─────────────────────────────────────────────────────────────────────────────


def save_comparison(input_img, output_img, label_img, name, save_dir):
    """
    Saves a side-by-side figure:
      Input (low_blur_noise) | MSFS-Net Output | Ground Truth
    """
    def to_np(tensor):
        img = tensor.squeeze(0).cpu().clamp(0, 1)
        return img.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"MSFS-Net Result — {name}", fontsize=13, fontweight='bold')

    axes[0].imshow(to_np(input_img))
    axes[0].set_title("Input\n(low_blur_noise)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(to_np(output_img))
    axes[1].set_title("MSFS-Net Output\n(deblurred)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(to_np(label_img))
    axes[2].set_title("Ground Truth\n(high_sharp_original)", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"visual_{name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved: {save_path}")


def eval_lol():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_net()
    state_dict = torch.load(PRETRAINED_MODEL, map_location=device)
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.\n")

    # ── Setup output folders ──────────────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    dataloader = test_dataloader(LOL_BLUR_PATH, batch_size=1, num_workers=0)

    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for idx, (input_img, label_img, name) in enumerate(dataloader):
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            # Run MSFS-Net — pass blur twice (sharp not available at inference)
            output = model([input_img, input_img])[0]
            pred_clip = torch.clamp(output, 0, 1)

            # ── Compute metrics ───────────────────────────────────────────────
            pred_np  = pred_clip.squeeze(0).cpu().numpy()
            label_np = label_img.squeeze(0).cpu().numpy()

            psnr_val = peak_signal_noise_ratio(label_np, pred_np, data_range=1.0)
            ssim_val = structural_similarity(label_np, pred_np,
                                             data_range=1.0, channel_axis=0)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            print(f"[{idx+1:04d}/{len(dataloader)}] {name[0]:30s}  "
                  f"PSNR: {psnr_val:.3f}  SSIM: {ssim_val:.4f}")

            # ── Save visualization for selected 4 images ──────────────────────
            if name[0] in VISUALIZE_NAMES:
                print(f"  → Saving visualization for {name[0]}")
                save_comparison(input_img, pred_clip, label_img,
                                name[0], RESULT_DIR)

    # ── Final average results ─────────────────────────────────────────────────
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n" + "="*60)
    print(f"MSFS-Net Evaluation on LOL-Blur Dataset")
    print(f"Total images evaluated : {len(psnr_list)}")
    print(f"Average PSNR           : {avg_psnr:.3f} dB")
    print(f"Average SSIM           : {avg_ssim:.4f}")
    print("="*60)

    # ── Save results to txt ───────────────────────────────────────────────────
    results_path = os.path.join(RESULT_DIR, "results_lol_blur.txt")
    with open(results_path, "w") as f:
        f.write("MSFS-Net Evaluation on LOL-Blur Dataset\n")
        f.write("="*50 + "\n")
        f.write(f"Total images : {len(psnr_list)}\n")
        f.write(f"Average PSNR : {avg_psnr:.3f} dB\n")
        f.write(f"Average SSIM : {avg_ssim:.4f}\n\n")
        f.write("Per-image results:\n")
        for n, p, s in zip(range(len(psnr_list)), psnr_list, ssim_list):
            f.write(f"  Image {n+1:04d}: PSNR={p:.3f}  SSIM={s:.4f}\n")

    print(f"\nFull results saved to: {results_path}")
    print(f"Visualizations saved to: {RESULT_DIR}")


if __name__ == "__main__":
    eval_lol()
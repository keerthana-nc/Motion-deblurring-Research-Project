import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

from lol_blur_dataloader import LOLBlurDataset
from fsm_pipeline import FSMPipeline


def tensor_to_image(tensor):
    img = tensor.cpu().detach().clone()
    img = img * 0.5 + 0.5
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0)
    return img.numpy()


def freq_to_image(tensor):
    img = tensor.cpu().detach().clone()
    img = img.mean(dim=0)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.numpy()


def get_hf_from_gaussian(image_tensor):
    """
    Compute HF directly from the RAW image using Gaussian blur subtraction.
    This is independent of FSM weights — it's a signal processing approach.
    HF = original - low_pass_gaussian
    This gives a meaningful HF result even without a trained model.
    """
    # Gaussian kernel
    kernel_size = 15
    sigma = 3.0
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
    kernel_2d = kernel_2d.repeat(3, 1, 1, 1)         # (3,1,k,k)

    img = image_tensor * 0.5 + 0.5   # denormalize to [0,1]

    # Apply gaussian blur per channel
    low_freq = F.conv2d(img, kernel_2d.to(image_tensor.device),
                        padding=kernel_size//2, groups=3)
    high_freq = img - low_freq        # HF = original - smooth

    return high_freq, low_freq


def visualize_specific(dataset_path, target_filenames,
                        output_dir="frequency_visualizations",
                        image_size=256, channels=64):
    """
    Visualizes specific images by filename.

    Args:
        target_filenames : list of filenames e.g. ['0032.png', '0039.png']
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = LOLBlurDataset(root=dataset_path, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = FSMPipeline(channels=channels).to(device)
    model.eval()

    found = 0
    with torch.no_grad():
        for batch in dataloader:
            fname = batch["filename"][0]
            if fname not in target_filenames:
                continue

            found += 1
            input_img = batch["input"].to(device)
            gt_img    = batch["gt"].to(device)

            # ── Gaussian-based HF/LF (signal processing, no training needed) ──
            hf_raw, lf_raw = get_hf_from_gaussian(input_img[0].unsqueeze(0))
            hf_gt,  lf_gt  = get_hf_from_gaussian(gt_img[0].unsqueeze(0))

            # Convert to display images
            img_input  = tensor_to_image(input_img[0])
            img_gt     = tensor_to_image(gt_img[0])

            # HF: boost brightness for visualization (values are small)
            hf_input_vis = hf_raw[0].mean(dim=0).cpu().numpy()
            hf_gt_vis    = hf_gt[0].mean(dim=0).cpu().numpy()
            lf_input_vis = lf_raw[0].permute(1,2,0).cpu().clamp(0,1).numpy()
            lf_gt_vis    = lf_gt[0].permute(1,2,0).cpu().clamp(0,1).numpy()

            # Normalize HF for display
            hf_input_vis = (hf_input_vis - hf_input_vis.min()) / (hf_input_vis.max() - hf_input_vis.min() + 1e-8)
            hf_gt_vis    = (hf_gt_vis    - hf_gt_vis.min())    / (hf_gt_vis.max()    - hf_gt_vis.min()    + 1e-8)

            # ── Plot: 2 rows (input row, GT row), 3 cols (original, HF, LF) ──
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            fig.suptitle(f"Frequency Separation — {fname}", fontsize=14, fontweight='bold')

            # Row 0: input (low_blur_noise)
            axes[0,0].imshow(img_input)
            axes[0,0].set_title("Input (low_blur_noise)", fontsize=10)
            axes[0,0].axis("off")

            axes[0,1].imshow(hf_input_vis, cmap="hot")
            axes[0,1].set_title("Input — High Freq\n(edges + noise)", fontsize=10)
            axes[0,1].axis("off")

            axes[0,2].imshow(lf_input_vis)
            axes[0,2].set_title("Input — Low Freq\n(smooth structure)", fontsize=10)
            axes[0,2].axis("off")

            # Row 1: ground truth
            axes[1,0].imshow(img_gt)
            axes[1,0].set_title("Ground Truth (high_sharp_original)", fontsize=10)
            axes[1,0].axis("off")

            axes[1,1].imshow(hf_gt_vis, cmap="hot")
            axes[1,1].set_title("GT — High Freq\n(clean edges)", fontsize=10)
            axes[1,1].axis("off")

            axes[1,2].imshow(lf_gt_vis)
            axes[1,2].set_title("GT — Low Freq\n(smooth structure)", fontsize=10)
            axes[1,2].axis("off")

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"vis_{fname}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {save_path}")

            if found == len(target_filenames):
                break

    if found == 0:
        print("WARNING: None of the requested filenames were found in the dataset.")
    else:
        print(f"\nDone. {found} images saved to '{output_dir}/'")


if __name__ == "__main__":

    DATASET_PATH = "C:/Users/nckee/OneDrive/Documents/CV Project track - Motion deblurring/LOL_BLUR"

    # ── Pick exactly which images you want ──
    TARGET_IMAGES = [
        "0032.png",
        "0039.png",
        "0060.png",
        "0083.png"
    ]

    visualize_specific(
        dataset_path     = DATASET_PATH,
        target_filenames = TARGET_IMAGES,
        output_dir       = "frequency_visualizations_2_rows",
        image_size       = 256,
        channels         = 64
    )
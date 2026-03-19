import os
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


class LOLBlurTestDataset(Dataset):
    """
    Adapts LOL-Blur dataset to MSFS-Net's expected format.
    MSFS-Net's eval expects: (input_img, label_img, filename)
    """

    def __init__(self, root: str):
        self.input_dir = os.path.join(root, "low_blur_noise")
        self.gt_dir    = os.path.join(root, "high_sharp_original")

        if not os.path.exists(self.input_dir):
            raise RuntimeError(f"Input folder not found: {self.input_dir}")
        if not os.path.exists(self.gt_dir):
            raise RuntimeError(f"GT folder not found: {self.gt_dir}")

        self.input_paths = []
        self.gt_paths    = []
        self.filenames   = []

        for subfolder in sorted(os.listdir(self.input_dir)):
            sub_input = os.path.join(self.input_dir, subfolder)
            sub_gt    = os.path.join(self.gt_dir,    subfolder)

            if not os.path.isdir(sub_input):
                continue

            for fname in sorted(os.listdir(sub_input)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                ip = os.path.join(sub_input, fname)
                gp = os.path.join(sub_gt,    fname)
                if os.path.exists(gp):
                    self.input_paths.append(ip)
                    self.gt_paths.append(gp)
                    self.filenames.append(f"{subfolder}_{fname}")

        print(f"LOL-Blur test set: {len(self.input_paths)} pairs found.")

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        # Load images — NO resizing, use original resolution
        input_img = Image.open(self.input_paths[idx]).convert("RGB")
        gt_img    = Image.open(self.gt_paths[idx]).convert("RGB")

        # Convert to tensor [0, 1] — same as MSFS-Net's original loader
        input_tensor = F.to_tensor(input_img)
        gt_tensor    = F.to_tensor(gt_img)

        return input_tensor, gt_tensor, self.filenames[idx]


def test_dataloader(path, batch_size=1, num_workers=0):
    dataset = LOLBlurTestDataset(root=path)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers,
                      pin_memory=True)
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms




class LOLBlurDataset(Dataset):
    """
    Loads image pairs from LOL-Blur dataset.

    Folder structure expected:
        root/
          low_blur_noise/
            subfolder1/
              img1.png
            subfolder2/
              ...
          high_sharp_original/
            subfolder1/
              img1.png
            ...
    """

    def __init__(self, root: str, image_size: int = 256):

        self.input_dir = os.path.join(root, "low_blur_noise")
        self.gt_dir    = os.path.join(root, "high_sharp_original")

        if not os.path.exists(self.input_dir):
            raise RuntimeError(f"Input folder not found: {self.input_dir}")
        if not os.path.exists(self.gt_dir):
            raise RuntimeError(f"Ground truth folder not found: {self.gt_dir}")

        # Collect all image pairs recursively across subfolders
        self.input_paths = []
        self.gt_paths    = []

        for subfolder in sorted(os.listdir(self.input_dir)):

            sub_input = os.path.join(self.input_dir, subfolder)
            sub_gt    = os.path.join(self.gt_dir,    subfolder)

            if not os.path.isdir(sub_input):
                continue

            for fname in sorted(os.listdir(sub_input)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                input_path = os.path.join(sub_input, fname)
                gt_path    = os.path.join(sub_gt,    fname)

                if os.path.exists(gt_path):
                    self.input_paths.append(input_path)
                    self.gt_paths.append(gt_path)
                else:
                    print(f"WARNING: No GT match for {input_path} — skipping.")

        if len(self.input_paths) == 0:
            raise RuntimeError("No image pairs found. Check subfolder structure.")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
        ])

        print(f"Dataset root      : {root}")
        print(f"Total image pairs : {len(self.input_paths)}")
        print(f"Image size        : {image_size} x {image_size}")

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_img = Image.open(self.input_paths[idx]).convert("RGB")
        gt_img    = Image.open(self.gt_paths[idx]).convert("RGB")

        return {
            "input"     : self.transform(input_img),
            "gt"        : self.transform(gt_img),
            "filename"  : os.path.basename(self.input_paths[idx]),
            "subfolder" : os.path.basename(os.path.dirname(self.input_paths[idx])),
        }


if __name__ == "__main__":

    DATASET_PATH = "C:/Users/nckee/OneDrive/Documents/CV Project track - Motion deblurring/LOL_BLUR"

    dataset    = LOLBlurDataset(root=DATASET_PATH, image_size=256)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
        print(f"\nBatch {i+1}")
        print(f"  Input  shape : {batch['input'].shape}")
        print(f"  GT     shape : {batch['gt'].shape}")
        print(f"  Files        : {batch['filename']}")

    print("\nDataloader working correctly.")
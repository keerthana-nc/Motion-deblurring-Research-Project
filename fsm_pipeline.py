import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lol_blur_dataloader import LOLBlurDataset   # ← imported from separate file


# ─────────────────────────────────────────────
# OctConv 
# ─────────────────────────────────────────────
class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 alpha_in=0.5, alpha_out=0.5, padding=0):
        super().__init__()
        self.alpha_in  = alpha_in
        self.alpha_out = alpha_out
        self.lf_in_ch  = int(alpha_in  * in_channels)
        self.hf_in_ch  = in_channels  - self.lf_in_ch
        self.lf_out_ch = int(alpha_out * out_channels)
        self.hf_out_ch = out_channels - self.lf_out_ch

        self.conv_hh = nn.Conv2d(self.hf_in_ch, self.hf_out_ch, kernel_size, padding=padding) \
                       if self.hf_in_ch > 0 and self.hf_out_ch > 0 else None
        self.conv_lh = nn.Conv2d(self.lf_in_ch, self.hf_out_ch, kernel_size, padding=padding) \
                       if self.lf_in_ch > 0 and self.hf_out_ch > 0 else None
        self.conv_ll = nn.Conv2d(self.lf_in_ch, self.lf_out_ch, kernel_size, padding=padding) \
                       if self.lf_in_ch > 0 and self.lf_out_ch > 0 else None
        self.conv_hl = nn.Conv2d(self.hf_in_ch, self.lf_out_ch, kernel_size, padding=padding) \
                       if self.hf_in_ch > 0 and self.lf_out_ch > 0 else None

    def forward(self, x):
        x_h, x_l = (x, None) if self.alpha_in == 0 else x
        y_h = None
        if self.hf_out_ch > 0:
            y_h = 0
            if self.conv_hh is not None:
                y_h = y_h + self.conv_hh(x_h)
            if self.conv_lh is not None:
                y_h = y_h + self.conv_lh(F.interpolate(x_l, scale_factor=2, mode='nearest'))
        y_l = None
        if self.lf_out_ch > 0:
            y_l = 0
            if self.conv_ll is not None:
                y_l = y_l + self.conv_ll(x_l)
            if self.conv_hl is not None:
                y_l = y_l + self.conv_hl(F.avg_pool2d(x_h, kernel_size=2, stride=2))
        return y_h if self.alpha_out == 0 else (y_h, y_l)


# ─────────────────────────────────────────────
# FSM
# ─────────────────────────────────────────────
class FSM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.split   = OctConv(channels, channels, kernel_size=1, padding=0, alpha_in=0.0, alpha_out=0.5)
        self.process = OctConv(channels, channels, kernel_size=3, padding=1, alpha_in=0.5, alpha_out=0.5)
        self.merge   = OctConv(channels, channels, kernel_size=1, padding=0, alpha_in=0.5, alpha_out=0.0)
        self.relu    = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual    = x
        x_h, x_l   = self.split(x)
        x_h, x_l   = self.relu(x_h), self.relu(x_l)
        x_h, x_l   = self.process((x_h, x_l))
        x_h, x_l   = self.relu(x_h), self.relu(x_l)
        out         = self.merge((x_h, x_l)) + residual
        return out, x_h, x_l


# ─────────────────────────────────────────────
# Shallow Feature Extractor
# ─────────────────────────────────────────────
class ShallowFeatureExtractor(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


# ─────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────
class FSMPipeline(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.extractor = ShallowFeatureExtractor(channels)
        self.fsm       = FSM(channels)

    def forward(self, x):
        features             = self.extractor(x)
        fused, high_f, low_f = self.fsm(features)
        return fused, high_f, low_f


# ─────────────────────────────────────────────
# Deblurring wrapper used for training FSM
# ─────────────────────────────────────────────
class DeblurWithFSM(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.fsm_pipeline = FSMPipeline(channels)
        self.recon_head   = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        fused, high_f, low_f = self.fsm_pipeline(x)
        restored = torch.tanh(self.recon_head(fused))
        return restored, fused, high_f, low_f


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":

    DATASET_PATH = "C:/Users/nckee/OneDrive/Documents/CV Project track - Motion deblurring/LOL_BLUR"
    CHANNELS     = 64
    IMAGE_SIZE   = 256
    BATCH_SIZE   = 4
    NUM_EPOCHS   = 2
    LR           = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dataset    = LOLBlurDataset(root=DATASET_PATH, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = DeblurWithFSM(channels=CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    print("Starting FSM training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            input_imgs = batch["input"].to(device)
            gt_imgs    = batch["gt"].to(device)

            restored, fused, high_freq, low_freq = model(input_imgs)

            loss = criterion(restored, gt_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 50 == 0:
                avg_loss = running_loss / 50
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Step [{step}]  L1 loss: {avg_loss:.4f}")
                running_loss = 0.0

        # End of epoch summary
        if step % 50 != 0:
            avg_loss = running_loss / max(step % 50, 1)
            print(f"Epoch [{epoch}/{NUM_EPOCHS}]  Final epoch loss: {avg_loss:.4f}")

    # Save only the FSM pipeline so it can be reused for visualization / downstream models
    torch.save(model.fsm_pipeline.state_dict(), "fsm_pipeline_trained.pth")
    print("\nTraining finished. Saved trained FSM pipeline weights to 'fsm_pipeline_trained.pth'.")
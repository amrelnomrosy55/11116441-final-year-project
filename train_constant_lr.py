# ABLATION (constant_lr)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Ensure your dataset_modified.py is in the same folder
from dataset_modified import NoisyCleanFrameDataset

# --- Configuration ---
CLEAN_DIR = "data/clean"
NOISE_DIR = "data/noise"
CLEAN_VAL_DIR = "data/clean_val"   #  folder for validation clean files
NOISE_VAL_DIR = "data/noise_val"   #  folder for validation noise files
BATCH = 32
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ABLATION (constant_lr): redirected save path to ablation/outputs/ tree.
MODEL_SAVE_PATH = "ablation/outputs/constant_lr.pth"  # New save path to preserve original


# -----------------------
class UNetMasker(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 24, 3, stride=1, padding=1)       # Full resolution
        self.enc2 = nn.Conv2d(24, 48, 3, stride=2, padding=1)      # /2 resolution
        self.enc3 = nn.Conv2d(48, 96, 3, stride=2, padding=1)      # /4 resolution
        self.enc4 = nn.Conv2d(96, 192, 3, stride=2, padding=1)     # /8 resolution 
        # BatchNorm layers for encoder (stabilises training, speeds convergence)
        self.bn_e1 = nn.BatchNorm2d(24)
        self.bn_e2 = nn.BatchNorm2d(48)
        self.bn_e3 = nn.BatchNorm2d(96)
        self.bn_e4 = nn.BatchNorm2d(192)                            # BatchNorm for enc4

        # Decoder
        self.dec4 = nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1)    
        self.dec3 = nn.ConvTranspose2d(192, 48, 4, stride=2, padding=1)    # Input: cat[dec4, e3] = 192
        self.dec2 = nn.ConvTranspose2d(96, 24, 4, stride=2, padding=1)     # Input: cat[dec3, e2] = 96
        self.dec1 = nn.Conv2d(48, 1, 3, stride=1, padding=1)               # Input: cat[dec2, e1] = 48
        # BatchNorm layers for decoder
        self.bn_d4 = nn.BatchNorm2d(96)                            
        self.bn_d3 = nn.BatchNorm2d(48)
        self.bn_d2 = nn.BatchNorm2d(24)

    def forward(self, x):
        #  Log-Scaling to help model "see" the noise floor (-80dB)
        x_log = torch.log1p(x)

        # Encoder passes
        #  BatchNorm inserted between conv and activation in each encoder stage
        e1 = F.leaky_relu(self.bn_e1(self.enc1(x_log)), 0.2)
        e2 = F.leaky_relu(self.bn_e2(self.enc2(e1)), 0.2)
        e3 = F.leaky_relu(self.bn_e3(self.enc3(e2)), 0.2)
        e4 = F.leaky_relu(self.bn_e4(self.enc4(e3)), 0.2)          

        # Decoder 4 -> Resize to match e3 (Skip Connection 1) 
        #  BatchNorm inserted between convtranspose and activation in each decoder stage
        d4 = F.leaky_relu(self.bn_d4(self.dec4(e4)), 0.2)
        if d4.shape[-2:] != e3.shape[-2:]:
            d4 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)

        # Decoder 3 -> Resize to match e2 (Skip Connection 2)
        d3 = F.leaky_relu(self.bn_d3(self.dec3(torch.cat([d4, e3], dim=1))), 0.2)
        if d3.shape[-2:] != e2.shape[-2:]:
            d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)

        # Decoder 2 -> Resize to match e1 (Skip Connection 3)
        d2 = F.leaky_relu(self.bn_d2(self.dec2(torch.cat([d3, e2], dim=1))), 0.2)
        if d2.shape[-2:] != e1.shape[-2:]:
            d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)

        # Final Mask Generation (Sigmoid forces values between 0.0 and 1.0)
        # clamp(min=0.05) prevents mask going to near-zero which causes musical noise artifacts
        mask = torch.sigmoid(self.dec1(torch.cat([d2, e1], dim=1))).clamp(min=0.05)

        # Ensure mask matches input size exactly
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Output = Input * Mask. (0.0 in mask = PURE BLACK in spectrogram)
        return x * mask


# -----------------------
# 2. Training Logic
# -----------------------
def main():
    os.makedirs("outputs", exist_ok=True)
    # ABLATION (constant_lr): also ensure the ablation output dir exists.
    os.makedirs("ablation/outputs", exist_ok=True)

    # Print parameter count before training
    model_check = UNetMasker()
    total_params = sum(p.numel() for p in model_check.parameters())
    print(f"--- Upgraded UNetMasker v2 ---")
    # ABLATION (constant_lr): variant banner for log readability.
    print(f"[ABLATION] variant: constant_lr")
    print(f"Total parameters: {total_params:,}")
    del model_check

    # Initialize Dataset
    ds = NoisyCleanFrameDataset(
        CLEAN_DIR, NOISE_DIR,
        snr_min=0, snr_max=10,
        samples_per_epoch=8000,
        sr=16000,
        n_fft=512,
        hop_length=128,
        patch_frames=64 # Increased context for better speech/noise separation
    )
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)

    # Validation dataset and loader (points to held-out folder)
    val_ds = NoisyCleanFrameDataset(
        CLEAN_VAL_DIR, NOISE_VAL_DIR,
        snr_min=0, snr_max=10,
        samples_per_epoch=1000,
        sr=16000,
        n_fft=512,
        hop_length=128,
        patch_frames=64
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, drop_last=False)

    model = UNetMasker().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # CosineAnnealingLR scheduler gradually reduces LR to help converge cleanly
    # ABLATION (constant_lr): scheduler disabled; LR stays fixed at 1e-3.
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # MSE Loss on Log-Magnitude creates the highest contrast for "black" silence
    loss_fn = nn.MSELoss()

    print(f"--- Starting Training on {DEVICE} ---")
    best_loss = float("inf")

    # Store loss history for plotting
    train_losses = []
    val_losses   = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for noisy_mag, clean_mag in dl:
            noisy_mag = noisy_mag.to(DEVICE)
            clean_mag = clean_mag.to(DEVICE)

            # Pred is the denoised Magnitude Spectrogram
            pred_mag = model(noisy_mag)

            # Calculate loss in the Log Domain (emphasizes quiet background areas)
            loss = loss_fn(torch.log1p(pred_mag), torch.log1p(clean_mag))

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents exploding gradients (safe upper bound = 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {avg_loss:.6f}", end="")

        #  Validation loop evaluates on held-out data to detect overfitting
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for noisy_mag, clean_mag in val_dl:
                noisy_mag = noisy_mag.to(DEVICE)
                clean_mag = clean_mag.to(DEVICE)
                pred_mag  = model(noisy_mag)
                val_loss  = loss_fn(torch.log1p(pred_mag), torch.log1p(clean_mag))
                val_total += val_loss.item()
        avg_val_loss = val_total / len(val_dl)
        val_losses.append(avg_val_loss)
        print(f" | Val Loss: {avg_val_loss:.6f}")

        # Step the scheduler at the end of each epoch
        # ABLATION (constant_lr): scheduler.step() disabled since no scheduler is used.
        # scheduler.step()

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   --> New Best Loss! Model saved to {MODEL_SAVE_PATH}")

    print(f"\nTraining Complete. Best observed loss: {best_loss:.6f}")

    # Save loss history for loss curve plot
    import json
    # ABLATION (constant_lr): per-variant loss history file so baselines are not overwritten.
    loss_path = "ablation/outputs/loss_history_constant_lr.json"
    with open(loss_path, "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)
    print(f"Loss history saved to {loss_path}")


if __name__ == "__main__":
    main()

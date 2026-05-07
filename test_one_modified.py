import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------
# Configuration (Must match Training)
# -----------------------
SR = 16000
N_FFT = 512
HOP = 128
WIN = 512
PATCH_FRAMES = 64  # Updated to 64 to match your training context
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "outputs/unet_mask_denoiser_v2.pth"

# -----------------------
# 1. The Correct Model Architecture (UNetMasker)
# -----------------------
class UNetMasker(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 24, 3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(24, 48, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(48, 96, 3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(96, 192, 3, stride=2, padding=1)
        # Added: BatchNorm layers for encoder (stabilises training, speeds convergence)
        self.bn_e1 = nn.BatchNorm2d(24)
        self.bn_e2 = nn.BatchNorm2d(48)
        self.bn_e3 = nn.BatchNorm2d(96)
        self.bn_e4 = nn.BatchNorm2d(192)
        # Decoder
        self.dec4 = nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(192, 48, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(96, 24, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(48, 1, 3, stride=1, padding=1)
        # Added: BatchNorm layers for decoder
        self.bn_d4 = nn.BatchNorm2d(96)
        self.bn_d3 = nn.BatchNorm2d(48)
        self.bn_d2 = nn.BatchNorm2d(24)

    def forward(self, x):
        # Apply Log-Scaling to help model "see" the noise floor (-80dB)
        x_log = torch.log1p(x)
        # Encoder passes
        # Added: BatchNorm inserted between conv and activation in each encoder stage
        e1 = F.leaky_relu(self.bn_e1(self.enc1(x_log)), 0.2)
        e2 = F.leaky_relu(self.bn_e2(self.enc2(e1)), 0.2)
        e3 = F.leaky_relu(self.bn_e3(self.enc3(e2)), 0.2)
        e4 = F.leaky_relu(self.bn_e4(self.enc4(e3)), 0.2)
        # Decoder 4 -> Resize to match e3 (Skip Connection 1)
        # Added: BatchNorm inserted between convtranspose and activation in each decoder stage
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
        # Added: .clamp(min=0.05) prevents mask going to near-zero which causes musical noise artifacts
        mask = torch.sigmoid(self.dec1(torch.cat([d2, e1], dim=1))).clamp(min=0.05)
        # Ensure mask matches input size exactly
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # Output = Input * Mask. (0.0 in mask = PURE BLACK in spectrogram)
        return x * mask

# -----------------------
# 2. Helpers & Denoising Logic
# -----------------------
def mix_at_snr(clean, noise, snr_db):
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]
    
    clean_rms = np.sqrt(np.mean(clean**2) + 1e-8)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-8)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / (noise_rms + 1e-8))
    return clean + noise

# Added: SNR metric to objectively measure denoising quality (higher dB = better)
def compute_snr(clean, denoised):
    noise_residual = clean - denoised
    snr = 10 * np.log10(
        np.mean(clean ** 2) / (np.mean(noise_residual ** 2) + 1e-8)
    )
    return snr

@torch.no_grad()
def denoise_with_unet(model, noisy_signal):
    S = librosa.stft(noisy_signal, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    mag, phase = librosa.magphase(S)
    mag = mag.astype(np.float32)

    Fbins, Ttotal = mag.shape
    step = PATCH_FRAMES // 2 

    # Padding logic for sliding window
    pad_T = (PATCH_FRAMES - (Ttotal % step)) % step
    mag_pad = np.pad(mag, ((0, 0), (0, pad_T + PATCH_FRAMES)), mode="edge")
    
    Tpad = mag_pad.shape[1]
    out_mag_pad = np.zeros_like(mag_pad)
    weight_pad = np.zeros_like(mag_pad)

    # Added: Hann window for weighted overlap-add - reduces boundary artifacts between patches
    hann = np.hanning(PATCH_FRAMES).astype(np.float32)

    for t0 in range(0, Tpad - PATCH_FRAMES + 1, step):
        patch = mag_pad[:, t0:t0 + PATCH_FRAMES]
        xb = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # UNetMasker returns (Input * Mask)
        pred = model(xb).squeeze().cpu().numpy()

        # Added: multiply by hann window so patch edges contribute less than patch centres
        out_mag_pad[:, t0:t0 + PATCH_FRAMES] += pred * hann
        weight_pad[:, t0:t0 + PATCH_FRAMES] += hann

    out_mag = (out_mag_pad / np.maximum(weight_pad, 1e-8))[:, :Ttotal]
    
    S_hat = out_mag * phase
    return librosa.istft(S_hat, hop_length=HOP, win_length=WIN, length=len(noisy_signal))

def show_spec(y, title):
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S_db, sr=SR, hop_length=HOP, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.show()

# -----------------------
# 3. Main Test Loop
# -----------------------
def main():
    # Load Model
    model = UNetMasker().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load Files (Update these paths to your test files!)
    clean_path = "data/clean/bigtips_16.wav"
    noise_path = "data/noise/1-88409-A-45.wav"

    clean, _ = librosa.load(clean_path, sr=SR)
    noise, _ = librosa.load(noise_path, sr=SR)
    noisy = mix_at_snr(clean, noise, snr_db=5)

    # Denoise
    print("Denoising...")
    denoised = denoise_with_unet(model, noisy)

    # Added: Print SNR improvement to objectively compare noisy vs denoised quality
    snr_noisy = compute_snr(clean, noisy)
    snr_denoised = compute_snr(clean, denoised)
    print(f"SNR (noisy):    {snr_noisy:.2f} dB")
    print(f"SNR (denoised): {snr_denoised:.2f} dB")
    print(f"Improvement:    {snr_denoised - snr_noisy:+.2f} dB")

    # Visualize
    show_spec(noisy, "Noisy Signal (5dB SNR)")
    show_spec(denoised, "Denoised Signal (U-Net)")

   # Save
    os.makedirs("outputs", exist_ok=True)
    sf.write("outputs/test_noisy.wav", noisy, SR)
    sf.write("outputs/test_denoised.wav", denoised, SR)
    print("Done! Check outputs/test_denoised.wav")

if __name__ == "__main__":
    main()
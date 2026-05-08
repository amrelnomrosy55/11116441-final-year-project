import os
import sys
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, send_from_directory, send_file
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# -----------------------
# Configuration 
# -----------------------
SR = 16000
N_FFT = 512
HOP = 128
WIN = 512
PATCH_FRAMES = 64
DEVICE = "cpu"  
MODEL_PATH = "outputs/unet_mask_denoiser.pth"
OUTPUT_DIR = "outputs"
RECORD_SECONDS = 5  # How many seconds to record each time
RECORD_SR = 44100  # Actual mic sample rate / resampled to SR for the model

# -----------------------
# Global state shared between threads
# -----------------------
state = {
    "status": "idle",        # idle | recording | denoising | done | error
    "session": 0,
    "snr_noisy": None,
    "snr_denoised": None,
    "improvement": None,
    "processing_time": None,
    "noisy_path": None,
    "denoised_path": None,
    "spectrogram_noisy": None,   # list of dB values for visualisation
    "spectrogram_denoised": None,
    "error": None,
}

# -----------------------
# 1. Model Architecture 
# -----------------------
class UNetMasker(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        # BatchNorm layers for encoder (stabilises training, speeds convergence)
        self.bn_e1 = nn.BatchNorm2d(16)
        self.bn_e2 = nn.BatchNorm2d(32)
        self.bn_e3 = nn.BatchNorm2d(64)
        # Decoder
        self.dec3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        # BatchNorm layers for decoder
        self.bn_d3 = nn.BatchNorm2d(32)
        self.bn_d2 = nn.BatchNorm2d(16)

    def forward(self, x):
        # Log-Scaling to help model see the noise floor (-80dB)
        x_log = torch.log1p(x)
        # Encoder passes
        # BatchNorm inserted between conv and activation in each encoder stage
        e1 = F.leaky_relu(self.bn_e1(self.enc1(x_log)), 0.2)
        e2 = F.leaky_relu(self.bn_e2(self.enc2(e1)), 0.2)
        e3 = F.leaky_relu(self.bn_e3(self.enc3(e2)), 0.2)
        # Decoder 3 -> Resize to match e2 (Skip Connection 1)
        # BatchNorm inserted between convtranspose and activation in each decoder stage
        d3 = F.leaky_relu(self.bn_d3(self.dec3(e3)), 0.2)
        if d3.shape[-2:] != e2.shape[-2:]:
            d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        # Decoder 2 -> Resize to match e1 (Skip Connection 2)
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
# 2. Denoising Logic 
# -----------------------
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
    # Hann window for weighted overlap-add and reduces boundary artifacts between patches
    hann = np.hanning(PATCH_FRAMES).astype(np.float32)
    for t0 in range(0, Tpad - PATCH_FRAMES + 1, step):
        patch = mag_pad[:, t0:t0 + PATCH_FRAMES]
        xb = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
        # UNetMasker returns (Input * Mask)
        pred = model(xb).squeeze().cpu().numpy()
        # Weighted overlap-add using Hann window
        out_mag_pad[:, t0:t0 + PATCH_FRAMES] += pred * hann
        weight_pad[:, t0:t0 + PATCH_FRAMES] += hann
    out_mag = (out_mag_pad / np.maximum(weight_pad, 1e-8))[:, :Ttotal]
    S_hat = out_mag * phase
    return librosa.istft(S_hat, hop_length=HOP, win_length=WIN, length=len(noisy_signal))


# -----------------------
# 3. SNR metric
# -----------------------
def compute_snr(signal):
    # Estimates SNR of a single signal using noise floor estimation
    S = np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP))
    frame_power = S.mean(axis=0)
    noise_floor = np.percentile(frame_power, 10)
    signal_power = np.mean(frame_power)
    if noise_floor < 1e-8:
        return 0.0
    return float(10 * np.log10(signal_power / noise_floor))


# -----------------------
# 4. Spectrogram helper for visualisation
# -----------------------
def get_spectrogram_columns(signal, n_cols=80):
    # Returns a 1D list of average dB values across frequency for each time column
    S = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    # Downsample time axis to n_cols columns
    T = S_db.shape[1]
    indices = np.linspace(0, T - 1, n_cols, dtype=int)
    cols = S_db[:, indices].mean(axis=0).tolist()
    # Normalise to 0-1 range for the frontend
    min_val = min(cols)
    max_val = max(cols)
    if max_val - min_val < 1e-6:
        return [0.5] * n_cols
    return [(v - min_val) / (max_val - min_val) for v in cols]


# -----------------------
# 4b. Spectrogram PNG generator — exact show_spec() logic from test script
# -----------------------
def save_spectrogram_png(signal, path, title):
   
    # librosa.stft → amplitude_to_db(ref=np.max) → magma colormap
    S = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="black")
    img = librosa.display.specshow(
        S_db,
        sr=SR,
        hop_length=HOP,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Hz", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")  # Set each spine colour individually for Pi compatibility
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")  # Colorbar tick labels in white
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)  # Free memory after saving / important on Pi with limited RAM


# -----------------------
# 5. Recording and processing thread
# -----------------------
def run_session(model):
    try:
        state["status"] = "recording"
        state["error"] = None

        # Record audio at mic sample rate
        audio_raw = sd.rec(
            int(RECORD_SECONDS * RECORD_SR),
            samplerate=RECORD_SR,
            channels=1,
            dtype="float32"
        )
        sd.wait()  # Block until recording is finished

        # Resample from mic sample rate down to model sample rate
        noisy_audio = librosa.resample(audio_raw.flatten(), orig_sr=RECORD_SR, target_sr=SR)

        # Always overwrite the same fixed filenames so play always serves latest recording
        noisy_path = os.path.join(OUTPUT_DIR, "noisy_latest.wav")
        sf.write(noisy_path, noisy_audio, SR)
        state["noisy_path"] = noisy_path
        state["session"] += 1  # Increment session counter for cache-busting only

        # Compute spectrogram for noisy audio visualisation
        state["spectrogram_noisy"] = get_spectrogram_columns(noisy_audio)

        # Save full-colour spectrogram PNG using exact show_spec() pipeline
        save_spectrogram_png(noisy_audio, os.path.join(OUTPUT_DIR, "spec_noisy_latest.png"), "Noisy Signal")

        # Denoise
        state["status"] = "denoising"
        t_start = time.time()
        denoised_audio = denoise_with_unet(model, noisy_audio)
        t_elapsed = time.time() - t_start

        # Always overwrite same fixed filename
        denoised_path = os.path.join(OUTPUT_DIR, "denoised_latest.wav")
        sf.write(denoised_path, denoised_audio, SR)
        state["denoised_path"] = denoised_path

        # Compute spectrogram for denoised audio visualisation
        state["spectrogram_denoised"] = get_spectrogram_columns(denoised_audio)

        # Save full-colour spectrogram PNG using exact show_spec() pipeline
        save_spectrogram_png(denoised_audio, os.path.join(OUTPUT_DIR, "spec_denoised_latest.png"), "Denoised Signal (U-Net)")

        # Compute SNR metrics
        snr_noisy    = compute_snr(noisy_audio)
        snr_denoised = compute_snr(denoised_audio)

        state["snr_noisy"]       = round(snr_noisy, 1)
        state["snr_denoised"]    = round(snr_denoised, 1)
        state["improvement"]     = round(snr_denoised - snr_noisy, 1)
        state["processing_time"] = round(t_elapsed, 2)
        state["status"]          = "done"

    except Exception as e:
        state["status"] = "error"
        state["error"]  = str(e)


# -----------------------
# 6. Flask web server
# -----------------------
app = Flask(__name__, static_folder="static")
model = None


@app.route("/")
def index():
    # Serve the main HTML interface
    return send_from_directory("static", "index.html")


@app.route("/api/record", methods=["POST"])
def api_record():
    # Start a new recording session in a background thread
    if state["status"] in ("recording", "denoising"):
        return jsonify({"error": "Already processing"}), 400
    thread = threading.Thread(target=run_session, args=(model,), daemon=True)
    thread.start()
    return jsonify({"ok": True, "session": state["session"]})


@app.route("/api/status")
def api_status():
    # Return current state to the frontend for live updates
    return jsonify({
        "status":               state["status"],
        "session":              state["session"],
        "snr_noisy":            state["snr_noisy"],
        "snr_denoised":         state["snr_denoised"],
        "improvement":          state["improvement"],
        "processing_time":      state["processing_time"],
        "spectrogram_noisy":    state["spectrogram_noisy"],
        "spectrogram_denoised": state["spectrogram_denoised"],
        "error":                state["error"],
    })


@app.route("/api/audio/noisy")
def api_audio_noisy():
    # Serve the latest noisy audio file and always the most recent recording
    path = os.path.join(OUTPUT_DIR, "noisy_latest.wav")
    if not os.path.exists(path):
        return jsonify({"error": "No noisy audio available"}), 404
    return send_file(path, mimetype="audio/wav")


@app.route("/api/audio/denoised")
def api_audio_denoised():
    # Serve the latest denoised audio file and always the most recent recording
    path = os.path.join(OUTPUT_DIR, "denoised_latest.wav")
    if not os.path.exists(path):
        return jsonify({"error": "No denoised audio available"}), 404
    return send_file(path, mimetype="audio/wav")


@app.route("/api/spectrogram/noisy")
def api_spectrogram_noisy():
    # Serve the latest noisy spectrogram PNG
    path = os.path.join(OUTPUT_DIR, "spec_noisy_latest.png")
    if not os.path.exists(path):
        return jsonify({"error": "No noisy spectrogram available"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/spectrogram/denoised")
def api_spectrogram_denoised():
    # Serve the latest denoised spectrogram PNG
    path = os.path.join(OUTPUT_DIR, "spec_denoised_latest.png")
    if not os.path.exists(path):
        return jsonify({"error": "No denoised spectrogram available"}), 404
    return send_file(path, mimetype="image/png")


# -----------------------
# 7. Entry point
# -----------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model once at startup
    print("\n  Loading model...")
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    model = UNetMasker().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("  Model loaded successfully.")
    print("  Starting web server...")
    print("  Open http://localhost:5000 in the browser on the Pi.\n")

    # Run Flask on all interfaces so it's accessible on the local network too
    app.run(host="0.0.0.0", port=5000, debug=False)

# ABLATION driver: trains every variant sequentially, then evaluates each

import os
import sys
import argparse
import numpy as np
import librosa
import torch

from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
for p in (REPO_ROOT, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Per-variant module imports. Each module defines its own `UNetMasker` class
# with a possibly-different architecture, so we alias them to avoid collisions.
from train_no_bn          import UNetMasker as UNetMasker_no_bn,          main as train_no_bn_main
from train_no_clamp       import UNetMasker as UNetMasker_no_clamp,       main as train_no_clamp_main
from train_no_skip        import UNetMasker as UNetMasker_no_skip,        main as train_no_skip_main
from train_linear_scaling import UNetMasker as UNetMasker_linear_scaling, main as train_linear_scaling_main
from train_constant_lr    import UNetMasker as UNetMasker_constant_lr,    main as train_constant_lr_main

from train_v2             import UNetMasker as UNetMasker_v2

# --- STFT / patch config 
SR = 16000
N_FFT = 512
HOP = 128
WIN = 512
PATCH_FRAMES = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 9-condition test grid 
CLEAN_FILE = "data/clean/1221-135767-0013.wav"
NOISE_FILES = {
    "Dog bark (natural)":       "data/noise/1-40730-A-1.wav",
    "Crackling fire (machine)": "data/noise/1-4211-A-12.wav",
    "Crowd (human noise)":      "data/noise/1-40621-A-28.wav",
}
SNR_CONDITIONS = [0, 5, 10]

# Variants to train + evaluate. Each entry is (display_name, checkpoint_path,
# UNetMasker class, training entry-point).
VARIANTS = [
    
    ("full_v2",        "outputs/unet_mask_denoiser_v2.pth",    UNetMasker_v2,             None),
    ("no_bn",          "ablation/outputs/no_bn.pth",          UNetMasker_no_bn,          train_no_bn_main),
    ("no_clamp",       "ablation/outputs/no_clamp.pth",       UNetMasker_no_clamp,       train_no_clamp_main),
    ("no_skip",        "ablation/outputs/no_skip.pth",        UNetMasker_no_skip,        train_no_skip_main),
    ("linear_scaling", "ablation/outputs/linear_scaling.pth", UNetMasker_linear_scaling, train_linear_scaling_main),
    ("constant_lr",    "ablation/outputs/constant_lr.pth",    UNetMasker_constant_lr,    train_constant_lr_main),
]


V2_BASELINE_AVG_DB = 5.2
V2_BASELINE_AVG_SNR  = 5.23
V2_BASELINE_AVG_PESQ = 2.364
V2_BASELINE_AVG_STOI = 0.895


# -----------------------
# Helpers 
# -----------------------
def mix_at_snr(clean, noise, snr_db):
    # Tile noise if shorter than clean signal
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]
    clean_rms = np.sqrt(np.mean(clean**2) + 1e-8)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-8)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / (noise_rms + 1e-8))
    return clean + noise


def compute_snr(clean, signal):
    # SNR using clean reference
    noise = clean - signal
    signal_power = np.mean(clean**2)
    noise_power  = np.mean(noise**2) + 1e-8
    return 10 * np.log10(signal_power / noise_power)


def compute_pesq(clean, signal, sr=16000):
    # Wideband PESQ (ITU-T P.862.2). Returns None if PESQ raises on an edge
    # case (silence, NaN, length mismatch) so one bad row doesn't abort eval.
    try:
        return float(pesq_fn(sr, clean.astype(np.float32), signal.astype(np.float32), 'wb'))
    except Exception as exc:
        print(f"   [pesq] skipped: {exc}")
        return None


def compute_stoi(clean, signal, sr=16000):
    # Short-Time Objective Intelligibility in [0, 1]. Same defensive pattern
    # as compute_pesq. (The user's request labelled this compute_pesq_score;
    # renamed to compute_stoi since the body computes STOI.)
    try:
        return float(stoi_fn(clean, signal, sr, extended=False))
    except Exception as exc:
        print(f"   [stoi] skipped: {exc}")
        return None


@torch.no_grad()
def denoise_with_unet(model, noisy_signal):
    # Sliding-window patch inference with Hann-weighted overlap-add.
    S = librosa.stft(noisy_signal, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    mag, phase = librosa.magphase(S)
    mag = mag.astype(np.float32)
    Fbins, Ttotal = mag.shape
    step = PATCH_FRAMES // 2
    pad_T   = (PATCH_FRAMES - (Ttotal % step)) % step
    mag_pad = np.pad(mag, ((0, 0), (0, pad_T + PATCH_FRAMES)), mode="edge")
    Tpad    = mag_pad.shape[1]
    out_mag_pad = np.zeros_like(mag_pad)
    weight_pad  = np.zeros_like(mag_pad)
    hann = np.hanning(PATCH_FRAMES).astype(np.float32)
    for t0 in range(0, Tpad - PATCH_FRAMES + 1, step):
        patch = mag_pad[:, t0:t0 + PATCH_FRAMES]
        xb    = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
        pred  = model(xb).squeeze().cpu().numpy()
        out_mag_pad[:, t0:t0 + PATCH_FRAMES] += pred * hann
        weight_pad[:, t0:t0 + PATCH_FRAMES]  += hann
    out_mag = (out_mag_pad / np.maximum(weight_pad, 1e-8))[:, :Ttotal]
    S_hat   = out_mag * phase
    return librosa.istft(S_hat, hop_length=HOP, win_length=WIN, length=len(noisy_signal))


def evaluate_variant(name, ckpt_path, model_cls, clean):
    # Loads one variant's checkpoint and runs the full 9-condition sweep,
    # returning the mean SNR improvement (output SNR - input SNR) across
    # all 9 conditions.
    # also computes PESQ (wideband) and STOI on both noisy and
    # denoised signals per condition, and returns the mean output-side PESQ
    # and STOI alongside the mean SNR improvement.
    print(f"\n--- Evaluating variant: {name} ({ckpt_path}) ---")
    model = model_cls().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    improvements = []
    pesq_in_vals,  pesq_out_vals  = [], []
    stoi_in_vals,  stoi_out_vals  = [], []
    print(f"{'Noise Type':<25} {'Input SNR':>10} {'Output SNR':>11} {'Improvement':>12}"
          f" {'PESQ in':>8} {'PESQ out':>9} {'STOI in':>8} {'STOI out':>9}")
    print("-" * 102)
    for noise_name, noise_file in NOISE_FILES.items():
        noise, _ = librosa.load(noise_file, sr=SR)
        for snr_db in SNR_CONDITIONS:
            noisy = mix_at_snr(clean, noise, snr_db)
            snr_in = compute_snr(clean, noisy)
            denoised = denoise_with_unet(model, noisy)
            snr_out = compute_snr(clean, denoised)
            improvement = snr_out - snr_in
            improvements.append(improvement)
            # Perceptual metrics on both input (noisy) and output (denoised).
            p_in  = compute_pesq(clean, noisy.astype(np.float32), SR)
            p_out = compute_pesq(clean, denoised.astype(np.float32), SR)
            s_in  = compute_stoi(clean, noisy.astype(np.float32), SR)
            s_out = compute_stoi(clean, denoised.astype(np.float32), SR)
            if p_in  is not None: pesq_in_vals.append(p_in)
            if p_out is not None: pesq_out_vals.append(p_out)
            if s_in  is not None: stoi_in_vals.append(s_in)
            if s_out is not None: stoi_out_vals.append(s_out)
            p_in_s  = f"{p_in:>8.3f}"  if p_in  is not None else f"{'n/a':>8}"
            p_out_s = f"{p_out:>9.3f}" if p_out is not None else f"{'n/a':>9}"
            s_in_s  = f"{s_in:>8.3f}"  if s_in  is not None else f"{'n/a':>8}"
            s_out_s = f"{s_out:>9.3f}" if s_out is not None else f"{'n/a':>9}"
            print(f"{noise_name:<25} {snr_in:>9.1f}dB {snr_out:>10.1f}dB {improvement:>+11.1f}dB"
                  f" {p_in_s} {p_out_s} {s_in_s} {s_out_s}")
        print("-" * 102)
    avg = float(np.mean(improvements))
    mean_pesq_out = float(np.mean(pesq_out_vals)) if pesq_out_vals else None
    mean_stoi_out = float(np.mean(stoi_out_vals)) if stoi_out_vals else None
    pesq_s = f"{mean_pesq_out:.3f}" if mean_pesq_out is not None else "n/a"
    stoi_s = f"{mean_stoi_out:.3f}" if mean_stoi_out is not None else "n/a"
    print(f"{name}: mean improvement = {avg:+.2f} dB | PESQ = {pesq_s} | STOI = {stoi_s}")
    return avg, mean_pesq_out, mean_stoi_out


# -----------------------
# Main driver
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Ablation training + evaluation driver.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip Phase 1 training; evaluate existing checkpoints only.")
    args = parser.parse_args()

    os.makedirs("ablation/outputs", exist_ok=True)

    # Phase 1: Train every variant in sequence.
    if not args.eval_only:
        print("\n" + "=" * 72)
        print("  PHASE 1 — Training all ablation variants")
        print("=" * 72)
        for name, ckpt_path, _, train_main in VARIANTS:
            if train_main is None:
                print(f"\n>>> Skipping training for {name} (eval-only variant)")
                continue
            print(f"\n>>> Training variant: {name}")
            train_main()
            if not os.path.exists(ckpt_path):
                print(f"  WARNING: expected checkpoint {ckpt_path} was not created.")
    else:
        print("\n[eval-only mode] skipping Phase 1 training.")

    # Phase 2: Load each trained variant and evaluate.
    print("\n" + "=" * 72)
    print("  PHASE 2 — Evaluating all variants on the 9-condition grid")
    print("=" * 72)
    clean, _ = librosa.load(CLEAN_FILE, sr=SR)
    results = {}
    for name, ckpt_path, model_cls, _ in VARIANTS:
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {name}: checkpoint missing at {ckpt_path}")
            results[name] = None
            continue
        results[name] = evaluate_variant(name, ckpt_path, model_cls, clean)

    # Phase 3: Comparison table.
    # Three-metric table: ΔSNR (dB, improvement), PESQ (wb, absolute score of
    # denoised output), STOI (absolute score of denoised output).
    print("\n" + "=" * 72)
    print("  ABLATION COMPARISON — average metrics across 9 conditions")
    print("=" * 72)
    print(f"  {'Variant':<22} {'ΔSNR (dB)':>11} {'PESQ':>9} {'STOI':>8}")
    print("  " + "-" * 51)
    for name, _, _, _ in VARIANTS:
        row = results.get(name)
        if row is None:
            print(f"  {name:<22} {'n/a':>11} {'n/a':>9} {'n/a':>8}")
            continue
        snr_v, pesq_v, stoi_v = row
        snr_s  = f"{snr_v:>+11.2f}"                if snr_v  is not None else f"{'n/a':>11}"
        pesq_s = f"{pesq_v:>9.3f}"                 if pesq_v is not None else f"{'n/a':>9}"
        stoi_s = f"{stoi_v:>8.3f}"                 if stoi_v is not None else f"{'n/a':>8}"
        print(f"  {name:<22} {snr_s} {pesq_s} {stoi_s}")
    print("=" * 72)


if __name__ == "__main__":
    main()

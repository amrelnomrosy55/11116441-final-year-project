import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

class NoisyCleanFrameDataset(Dataset):
    """
    Returns spectrogram patches for Conv2D:

      xb: noisy magnitude patch  (1, F, T)
      yb: clean magnitude patch  (1, F, T)

    Notes:
    - Mixes clean + noise at random SNR (dB) in [snr_min, snr_max]
    - Computes STFT magnitude and samples a random time window of patch_frames
    """
    def __init__(
        self,
        clean_dir,
        noise_dir,
        snr_min=5,
        snr_max=10,
        samples_per_epoch=4000,
        sr=16000,
        n_fft=512,
        hop_length=128,
        patch_frames=16
    ):
        self.clean_paths = [
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir)
            if f.lower().endswith((".wav", ".flac", ".mp3"))
        ]
        self.noise_paths = [
            os.path.join(noise_dir, f) for f in os.listdir(noise_dir)
            if f.lower().endswith((".wav", ".flac", ".mp3"))
        ]

        if len(self.clean_paths) == 0:
            raise RuntimeError(f"No clean audio files found in: {clean_dir}")
        if len(self.noise_paths) == 0:
            raise RuntimeError(f"No noise audio files found in: {noise_dir}")

        self.snr_min = snr_min
        self.snr_max = snr_max
        self.samples_per_epoch = samples_per_epoch

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_frames = patch_frames

        self.freq_bins = n_fft // 2 + 1  # e.g. 257 for n_fft=512

    def __len__(self):
        return self.samples_per_epoch

    def _load_audio_segment(self, path, length):
        x, _ = librosa.load(path, sr=self.sr, mono=True)
        if len(x) < length:
            x = np.pad(x, (0, length - len(x)))
        else:
            start = random.randint(0, len(x) - length)
            x = x[start:start + length]
        return x.astype(np.float32)

    def _mix_at_snr(self, clean, noise, snr_db):
        # match length
        if len(noise) < len(clean):
            noise = np.pad(noise, (0, len(clean) - len(noise)))
        else:
            noise = noise[:len(clean)]

        clean_rms = np.sqrt(np.mean(clean**2) + 1e-8)
        noise_rms = np.sqrt(np.mean(noise**2) + 1e-8)

        # target_noise_rms = clean_rms / 10^(snr/20)
        target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
        noise = noise * (target_noise_rms / (noise_rms + 1e-8))

        return clean + noise

    def __getitem__(self, idx):
        clean_path = random.choice(self.clean_paths)
        noise_path = random.choice(self.noise_paths)
        snr_db = random.uniform(self.snr_min, self.snr_max)

        # enough samples to yield >= patch_frames STFT frames
        min_len = self.patch_frames * self.hop_length + self.n_fft

        clean = self._load_audio_segment(clean_path, min_len)
        noise = self._load_audio_segment(noise_path, min_len)
        noisy = self._mix_at_snr(clean, noise, snr_db)

        S_clean = librosa.stft(clean, n_fft=self.n_fft, hop_length=self.hop_length)
        S_noisy = librosa.stft(noisy, n_fft=self.n_fft, hop_length=self.hop_length)

        mag_clean = np.abs(S_clean).astype(np.float32)  # (F, Ttotal)
        mag_noisy = np.abs(S_noisy).astype(np.float32)

        Ttotal = mag_clean.shape[1]
        if Ttotal <= self.patch_frames:
            pad = (self.patch_frames - Ttotal) + 1
            mag_clean = np.pad(mag_clean, ((0, 0), (0, pad)))
            mag_noisy = np.pad(mag_noisy, ((0, 0), (0, pad)))
            Ttotal = mag_clean.shape[1]

        t0 = random.randint(0, Ttotal - self.patch_frames)
        x_patch = mag_noisy[:, t0:t0 + self.patch_frames]  # (F, T)
        y_patch = mag_clean[:, t0:t0 + self.patch_frames]  # (F, T)

        # (1, F, T) for Conv2D
        x_patch = torch.from_numpy(x_patch).unsqueeze(0)
        y_patch = torch.from_numpy(y_patch).unsqueeze(0)

        return x_patch, y_patch

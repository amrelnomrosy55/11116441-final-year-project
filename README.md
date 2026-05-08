# 11116441-final-year-project

# U-Net Speech Denoiser on Raspberry Pi 5

Final-year individual project (EEEN30330) at the University of Manchester.
A lightweight U-Net masking architecture for single-channel speech enhancement, deployed standalone on a Raspberry Pi 5.

**Author:** Amr Elnomrosy (11116441)
**Supervisor:** Dr. Zhirun Hu
**Submission:** 1 May 2026

---

## Project summary

This repository contains the source code for a 699,193-parameter U-Net that performs real-time speech enhancement on a Raspberry Pi 5. The system was trained on LibriSpeech clean speech mixed with ESC-50 environmental noise at SNRs from 0 to 10 dB, evaluated against classical spectral subtraction and Wiener filtering baselines, and deployed as a fully standalone embedded unit.

**Headline results on the Pi 5:**

| Metric | Value |
|---|---|
| Average SNR improvement | +5.23 dB |
| PESQ improvement | +0.84 |
| STOI improvement | +0.030 |
| Real-time factor (RTF) | 0.24 |
| End-to-end latency | 1193 ms |
| Peak CPU utilisation | 90.6% |
| Peak RAM | 456 MB |

The U-Net is the only method evaluated that improves SNR, PESQ, and STOI simultaneously on every one of the nine test conditions.

---

## Repository contents

```
.
├── app.py                         # Live demo GUI: runs on Raspberry Pi 5
├── dataset_modified.py            # LibriSpeech + ESC-50 mixing pipeline
├── train_v2.py                    # Main training script (final model)
├── test_one_modified.py           # Single-utterance evaluation
├── evaluate_all.py                # Multi-condition evaluation across all noise types and SNRs
├── train_no_skip.py               # Ablation: no skip connections
├── train_no_bn.py                 # Ablation: no Batch Normalisation
├── train_no_clamp.py              # Ablation: no mask floor clamp
├── train_linear_scaling.py        # Ablation: linear input scaling instead of log1p
├── train_constant_lr.py           # Ablation: constant learning rate instead of cosine annealing
├── .gitignore
└── README.md
```

---

## How to navigate this repository (for the marker)

If you have ten minutes, look at these in order:

1. **`app.py`** — the deployed application that runs on the Raspberry Pi. Contains the model architecture, the inference pipeline (STFT, U-Net forward pass, overlap-add reconstruction), the live demonstration GUI, and the audio capture/playback logic. This is what was demonstrated live during the project assessment.

2. **`train_v2.py`** — the main training script for the final model. Shows the U-Net architecture, the training loop, the cosine annealing schedule, gradient clipping, log-domain MSE loss, and best-checkpoint selection.

3. **`dataset_modified.py`** — the data preparation pipeline. Shows how clean LibriSpeech utterances and ESC-50 noise clips are mixed at random SNRs in [0, 10] dB, and how training patches are extracted.

4. **`evaluate_all.py`** — the multi-condition evaluation that produced Tables 4, 4b, 4c, and 5 of the report. Computes SNR, PESQ, and STOI across three noise types (dog bark, crackling fire, crowd) at three input SNRs (0, 5, 10 dB) for the U-Net and the classical baselines.

5. **The five `train_*.py` ablation scripts** — each retrains the model with one architectural component removed or replaced, producing the results in Section 4.5 of the report. The naming corresponds directly to the configurations in Table 7:
   - `train_no_skip.py` — skip connections removed
   - `train_no_bn.py` — BatchNorm layers removed
   - `train_no_clamp.py` — mask floor clamp removed
   - `train_linear_scaling.py` — log1p replaced with linear input scaling
   - `train_constant_lr.py` — cosine annealing replaced with constant learning rate

6. **`test_one_modified.py`** — single-utterance evaluation used during development for quick spot-checks of the trained model on a chosen test sample.

If you only have two minutes, **read `app.py`** — it contains the deployed model architecture and shows how the system runs end-to-end on the Pi.

---

## Architecture overview

The model is a U-Net masking network operating on STFT magnitude spectrograms.

- **Input:** STFT magnitude (n_fft=512, hop=128, sample rate 16 kHz), passed through `log1p` compression
- **Encoder:** Four convolutional stages (1→24→48→96→192 channels), each with stride-2 downsampling, BatchNorm, and LeakyReLU(0.2)
- **Bottleneck:** 192-channel feature map at 1/8 spatial resolution
- **Decoder:** Three transposed-convolution upsampling stages with skip connections from the encoder, followed by a final 1×1 projection
- **Output:** Sigmoid-activated soft mask in (0.05, 1) applied element-wise to the noisy magnitude
- **Reconstruction:** iSTFT using the original noisy phase, with Hann-windowed overlap-add for patch boundaries

Total parameters: **699,193**.

---

## Reproduction instructions

### Dependencies

```bash
pip install torch numpy librosa soundfile sounddevice pesq pystoi
```

### Datasets

Download the required datasets and place them in a `data/` folder:

- **Clean speech:** [LibriSpeech train-clean-100](https://www.openslr.org/12/) (251 speakers, ~100 hours of read English speech at 16 kHz)
- **Noise:** [ESC-50](https://github.com/karolpiczak/ESC-50) (2,000 environmental sound clips across 50 categories)
- **Held-out evaluation:** [LibriSpeech test-clean](https://www.openslr.org/12/) for the multi-utterance robustness check

### Training

```bash
python train_v2.py
```

Training takes approximately 12 hours on CPU for 30 epochs. The best checkpoint is saved to a local `outputs/` directory.

### Ablation studies

Each ablation can be run individually:

```bash
python train_no_skip.py
python train_no_bn.py
python train_no_clamp.py
python train_linear_scaling.py
python train_constant_lr.py
```

Each script retrains the model from scratch with the relevant component changed.

### Evaluation

```bash
# Multi-condition evaluation (all noise types and SNRs)
python evaluate_all.py

# Single-utterance evaluation (quick test)
python test_one_modified.py
```

### Live demo on Raspberry Pi 5

On the Pi (with USB microphone connected and the trained model present):

```bash
python app.py
```

This launches the demonstration GUI. Pressing the record button captures audio, runs U-Net inference, and displays the noisy and denoised spectrograms side by side along with quantitative metrics.

---

## Hardware setup

The deployed system uses:

- Raspberry Pi 5 (BCM2712 ARM CPU, 4 GB RAM)
- Trust All-round USB microphone (16 kHz mono input)
- HDMI monitor for the GUI
- USB-powered desktop speaker

No GPU, no cloud connection, and no external computing resources are required at inference time.

---

## Notes on iterative development

Several components were added during development in response to specific observations:

- **BatchNorm layers** were added after observing oscillatory training convergence in an earlier BN-free version.
- **The mask floor clamp at 0.05** was added after observing musical-noise artefacts in early listening tests with an unclamped mask.
- **The Hann-windowed overlap-add procedure** was introduced after observing audible clicks at patch boundaries in an earlier non-overlapping version.
- **`log1p` input scaling** was adopted to better weight noise-floor regions in the loss, replacing an earlier linear-scaling version.

The five `train_*.py` ablation scripts test the contribution of each of these components systematically. Results are reported in Section 4.5 of the dissertation.

---

## Citation



> Elnomrosy, A. (2026). *Deep Learning-Based Speech Enhancement Using a U-Net Masking Architecture with Hardware Deployment on Raspberry Pi.* Final-year individual project, Department of Electrical and Electronic Engineering, University of Manchester.
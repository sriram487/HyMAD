"""
Generate all dataset section figures for the HyMAD paper.

Outputs (saved to seismic_final_tex/figures/):
  waveform_examples.png         — Single vs merged waveform (normalised, same y-axis)
  energy_distribution.png       — Per-class RMS energy box plot
  psd_single_vs_merged.png      — PSD: Human, Vehicle, and merged Human+Vehicle
  spectrogram_single_vs_merged.png — Spectrogram: single Human vs merged Human+Vehicle

Usage:
  conda run -n torch_env python scripts/generate_dataset_figures.py
"""

import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/sriram/Desktop/seismic/Superimposed_Data"
OUT_DIR    = "/home/sriram/Desktop/seismic_final_tex/figures"
SPLIT      = "test"
SR         = 8000

os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_signals(prefix, split=SPLIT, n=None):
    pattern = os.path.join(DATA_DIR, split, f"{prefix}_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files for prefix '{prefix}' in split '{split}'")
    if n is not None:
        files = files[:n]
    sigs = [torch.load(f, weights_only=True)["signal"].numpy() for f in files]
    return np.array(sigs)

def normalise(sig):
    """Zero-mean unit-variance (matches model pre-processing)."""
    return (sig - sig.mean()) / (sig.std() + 1e-8)

def rms(sig):
    return float(np.sqrt(np.mean(sig ** 2)))

STYLE = dict(dpi=150, bbox_inches="tight")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Waveform comparison (same y-axis, normalised signals)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating waveform_examples.png …")

single_human  = normalise(load_signals("Human_Movement",  n=1)[0])
merged_hv     = normalise(load_signals("merged_Human_Movement_Vehicle_Movement", n=1)[0])

t = np.arange(len(single_human)) / SR

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.35)

axes[0].plot(t, single_human, lw=0.7, color="#2c7bb6")
axes[0].set_title("Single-Activity: Human Movement", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Amplitude (normalised)")

axes[1].plot(t, merged_hv, lw=0.7, color="#d7191c")
axes[1].set_title("Merged Multi-Activity: Human Movement + Vehicle Movement",
                   fontsize=12, fontweight="bold")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude (normalised)")

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.savefig(os.path.join(OUT_DIR, "waveform_examples.png"), **STYLE)
plt.close()
print("  ✓ waveform_examples.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — RMS energy box plot per class
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating energy_distribution.png …")

CLASS_MAP = {
    "Animal Movement":   "Animal_Movement",
    "Human Movement":    "Human_Movement",
    "Vehicle Movement":  "Vehicle_Movement",
    "No Event":          "No_Movement",
}

rms_by_class = {}
for label, prefix in CLASS_MAP.items():
    sigs = load_signals(prefix)
    rms_by_class[label] = np.array([rms(s) for s in sigs])

labels_ordered = ["No Event", "Animal Movement", "Human Movement", "Vehicle Movement"]
colors = ["#9e9e9e", "#4dac26", "#0571b0", "#ca0020"]

fig, ax = plt.subplots(figsize=(8, 5))

bp = ax.boxplot(
    [rms_by_class[l] for l in labels_ordered],
    tick_labels=labels_ordered,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=2, alpha=0.4, linestyle="none"),
    widths=0.5,
)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_ylabel("RMS Energy", fontsize=11)
ax.set_title("RMS Energy Distribution per Activity Class", fontsize=12, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate medians
for i, label in enumerate(labels_ordered):
    med = np.median(rms_by_class[label])
    ax.text(i + 1, med * 1.08, f"{med:.4f}", ha="center", va="bottom",
            fontsize=8, color="black")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "energy_distribution.png"), **STYLE)
plt.close()
print("  ✓ energy_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — PSD: Human, Vehicle, and merged Human+Vehicle
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating psd_single_vs_merged.png …")

sig_human   = load_signals("Human_Movement",   n=1)[0]
sig_vehicle = load_signals("Vehicle_Movement",  n=1)[0]
sig_merged  = load_signals("merged_Human_Movement_Vehicle_Movement", n=1)[0]

def compute_psd(sig):
    f, Pxx = welch(sig, fs=SR, nperseg=512)
    return f, Pxx

f_h, pxx_h = compute_psd(sig_human)
f_v, pxx_v = compute_psd(sig_vehicle)
f_m, pxx_m = compute_psd(sig_merged)

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(f_h, pxx_h, color="#0571b0", lw=1.5, label="Single: Human Movement")
ax.semilogy(f_v, pxx_v, color="#4dac26", lw=1.5, label="Single: Vehicle Movement", linestyle="--")
ax.semilogy(f_m, pxx_m, color="#d7191c", lw=1.5, label="Merged: Human + Vehicle")

ax.set_xlabel("Frequency (Hz)", fontsize=11)
ax.set_ylabel("PSD (V²/Hz)", fontsize=11)
ax.set_title("Power Spectral Density: Single vs Merged Activities", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(0, SR / 2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "psd_single_vs_merged.png"), **STYLE)
plt.close()
print("  ✓ psd_single_vs_merged.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Spectrogram: single Human vs merged Human+Vehicle
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating spectrogram_single_vs_merged.png …")

from scipy.signal import spectrogram as scipy_spectrogram

NFFT    = 2048   # 3.9 Hz resolution → ~38 bins below 150 Hz
HOP     = 128
FMAX_HZ = 150    # seismic energy concentrated below 150 Hz

def compute_spectrogram(sig):
    f, t, Sxx = scipy_spectrogram(sig, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP,
                                   window="hann", scaling="density")
    mask = f <= FMAX_HZ
    return f[mask], t, Sxx[mask, :]

f_h, t_h, S_human  = compute_spectrogram(sig_human)
f_m, t_m, S_merged = compute_spectrogram(sig_merged)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

panels = [
    (axes[0], S_human,  t_h, f_h, "Single: Human Movement"),
    (axes[1], S_merged, t_m, f_m, "Merged: Human Movement + Vehicle Movement"),
]

ims = []
for ax, S, t, f, title in panels:
    # Per-panel colour range at 2nd–99th percentile for visibility
    log_S = np.log10(S + 1e-30)
    vmin_p, vmax_p = np.percentile(log_S, 2), np.percentile(log_S, 99)
    im = ax.pcolormesh(t, f, log_S, vmin=vmin_p, vmax=vmax_p,
                       cmap="inferno", shading="gouraud")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, FMAX_HZ)
    ims.append(im)

axes[0].set_ylabel("Frequency (Hz)", fontsize=11)

# Individual colourbars
for ax, im in zip(axes, ims):
    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cb.set_label("log₁₀ PSD", fontsize=9)

fig.suptitle("Spectrogram Comparison (0–150 Hz)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spectrogram_single_vs_merged.png"), **STYLE)
plt.close()
print("  ✓ spectrogram_single_vs_merged.png")

print("\nAll figures saved to", OUT_DIR)

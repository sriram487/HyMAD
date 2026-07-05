"""
Generate a two-panel frequency analysis figure that motivates the
SincNet filter initialization range (20–150 Hz).

Panel A: Mean PSD per class (0–500 Hz), with 150 Hz threshold marked.
Panel B: Between-class Fisher discriminability per frequency bin.

Output: seismic_final_tex/figures/discriminative_freq_analysis.png

Usage:
  conda run -n torch_env python scripts/plot_discriminative_freq.py
"""

import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/sriram/Desktop/seismic/Superimposed_Data"
OUT_PATH = "/home/sriram/Desktop/seismic_final_tex/figures/discriminative_freq_analysis.png"
SR       = 8000
NPERSEG  = 1024      # frequency resolution ~7.8 Hz
FMAX_HZ  = 500       # plot up to 500 Hz to show context beyond the 150 Hz band
THRESH   = 150.0     # proposed initialization upper bound
N_MAX    = 200       # max samples per class (for speed; all classes have >200)

CLASSES = {
    "No Event":        "No_Movement",
    "Animal Movement": "Animal_Movement",
    "Human Movement":  "Human_Movement",
    "Vehicle Movement":"Vehicle_Movement",
}
COLORS = {
    "No Event":         "#9e9e9e",
    "Animal Movement":  "#4dac26",
    "Human Movement":   "#0571b0",
    "Vehicle Movement": "#ca0020",
}

# ── Load signals ───────────────────────────────────────────────────────────────
def load_signals(prefix, split="test", n=N_MAX):
    files = sorted(glob.glob(os.path.join(DATA_DIR, split, f"{prefix}_*.pt")))[:n]
    return [torch.load(f, weights_only=True)["signal"].numpy() for f in files]

# ── Compute mean PSD per class ─────────────────────────────────────────────────
print("Computing per-class mean PSDs …")
class_psds = {}
freqs = None

for label, prefix in CLASSES.items():
    signals = load_signals(prefix)
    psds = []
    for sig in signals:
        f, pxx = welch(sig, fs=SR, nperseg=NPERSEG)
        psds.append(pxx)
    class_psds[label] = np.array(psds)   # (N, F)
    if freqs is None:
        freqs = f

# Frequency mask for plot
mask = freqs <= FMAX_HZ
f_plot = freqs[mask]

# ── Fisher discriminability per bin ───────────────────────────────────────────
# Fisher = between-class variance / within-class variance (per freq bin)
all_means  = np.array([class_psds[l][:, mask].mean(0) for l in CLASSES])  # (4, F)
grand_mean = all_means.mean(0)                                              # (F,)

between_var = np.mean((all_means - grand_mean) ** 2, axis=0)               # (F,)
within_var  = np.mean([class_psds[l][:, mask].var(0) for l in CLASSES], axis=0)  # (F,)
fisher      = between_var / (within_var + 1e-30)

# Smooth with a 5-bin rolling average
fisher_smooth = np.convolve(fisher, np.ones(5) / 5, mode="same")
fisher_norm   = fisher_smooth / fisher_smooth.max()   # normalise 0–1

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.subplots_adjust(hspace=0.10)

# ── Panel A: Mean PSD per class ───────────────────────────────────────────────
ax = axes[0]
for label in CLASSES:
    mean_psd = class_psds[label][:, mask].mean(0)
    ax.semilogy(f_plot, mean_psd, color=COLORS[label], lw=1.6, label=label)

ax.axvline(THRESH, color="#333333", lw=1.2, ls="--", label=f"{int(THRESH)} Hz threshold")
ax.axvspan(0, THRESH, alpha=0.06, color="#333333")
ax.set_ylabel("PSD (V²/Hz)", fontsize=11)
ax.set_title("Per-Class Mean Power Spectral Density (0–500 Hz)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, ncol=1, framealpha=0.9, loc="upper right")
ax.grid(True, which="both", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Panel B: Fisher discriminability ─────────────────────────────────────────
ax = axes[1]
ax.fill_between(f_plot, fisher_norm, alpha=0.25, color="#5e5e5e")
ax.plot(f_plot, fisher_norm, color="#222222", lw=1.4)
ax.axvline(THRESH, color="#333333", lw=1.2, ls="--")
ax.axvspan(0, THRESH, alpha=0.06, color="#333333")

# Annotate peak
peak_idx = np.argmax(fisher_norm)
ax.annotate(f"Peak: {f_plot[peak_idx]:.0f} Hz",
            xy=(f_plot[peak_idx], fisher_norm[peak_idx]),
            xytext=(f_plot[peak_idx] + 30, 0.85),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=1.0))

ax.set_xlabel("Frequency (Hz)", fontsize=11)
ax.set_ylabel("Normalised Fisher\nDiscriminability", fontsize=11)
ax.set_title("Between-Class Discriminability per Frequency Bin", fontsize=12, fontweight="bold")
ax.set_xlim(0, FMAX_HZ)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Energy fraction below 150 Hz per class — annotate on panel B
fractions = {}
for label in CLASSES:
    mean_psd = class_psds[label][:, freqs <= FMAX_HZ].mean(0)
    total    = mean_psd.sum()
    below    = mean_psd[f_plot <= THRESH].sum()
    fractions[label] = below / total * 100

short = {"No Event": "No Event", "Animal Movement": "Animal",
         "Human Movement": "Human", "Vehicle Movement": "Vehicle"}
frac_lines = "\n".join(f"  {short[l]}: {v:.0f}%" for l, v in fractions.items())
axes[1].text(
    0.98, 0.97,
    f"Energy below 150 Hz\n{frac_lines}",
    transform=axes[1].transAxes,
    fontsize=8, va="top", ha="right",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9),
)

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {OUT_PATH}")

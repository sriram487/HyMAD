"""
Visualise the learned SincNet bandpass filters from the trained HyMAD model.

Two-panel figure:
  Top   : Frequency response of all 40 learned filters (0–500 Hz),
           overlaid with the Fisher discriminability curve for reference.
  Bottom: Scatter of each filter's (centre frequency, bandwidth),
           coloured by centre frequency.

Output: seismic_final_tex/figures/sincnet_learned_filters.png

Usage:
  conda run -n torch_env python scripts/plot_sincnet_filters.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import welch, freqz
import glob

from model.model import SincNetRNN

# ── Config ─────────────────────────────────────────────────────────────────────
WEIGHTS  = "/home/sriram/Desktop/HyMAD/runs/HyMAD/best_model.pth"
DATA_DIR = "/home/sriram/Desktop/seismic/Superimposed_Data"
OUT_PATH = "/home/sriram/Desktop/seismic_final_tex/figures/sincnet_learned_filters.png"
SR       = 8000
FMAX     = 500       # plot up to 500 Hz
N_FISHER = 200       # samples per class for Fisher curve

CLASSES = {
    "No Event":         "No_Movement",
    "Animal Movement":  "Animal_Movement",
    "Human Movement":   "Human_Movement",
    "Vehicle Movement": "Vehicle_Movement",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load model and extract filter parameters ───────────────────────────────────
model = SincNetRNN().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
model.eval()

with torch.no_grad():
    low_hz_  = model.sinc.low_hz_.cpu()
    band_hz_ = model.sinc.band_hz_.cpu()
    min_low  = model.sinc.min_low_hz     # 20
    min_band = model.sinc.min_band_hz    # 5
    kernel_size = model.sinc.kernel_size # 251

    f1 = min_low + torch.abs(low_hz_)                          # lower cutoffs
    f2 = torch.clamp(f1 + min_band + torch.abs(band_hz_),
                     min_low, SR / 2)                           # upper cutoffs
    fc = (f1 + f2) / 2                                         # centre freqs
    bw = f2 - f1                                               # bandwidths

f1_np = f1.numpy()
f2_np = f2.numpy()
fc_np = fc.numpy()
bw_np = bw.numpy()

print(f"\nLearned filter statistics:")
print(f"  Centre freq  : min={fc_np.min():.1f}  max={fc_np.max():.1f}  mean={fc_np.mean():.1f} Hz")
print(f"  Bandwidth    : min={bw_np.min():.1f}  max={bw_np.max():.1f}  mean={bw_np.mean():.1f} Hz")
print(f"  Filters with fc < 150 Hz : {(fc_np < 150).sum()} / {len(fc_np)}")
print(f"  Filters with fc < 300 Hz : {(fc_np < 300).sum()} / {len(fc_np)}")

# ── Compute frequency responses ────────────────────────────────────────────────
def sinc_filter_response(f1_hz, f2_hz, kernel_size, sr, n_freqs=4096):
    """Compute magnitude response of one SincNet bandpass filter."""
    n = np.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(kernel_size) / kernel_size)

    f_times_t_low  = 2 * np.pi * f1_hz * n / sr
    f_times_t_high = 2 * np.pi * f2_hz * n / sr

    h = (np.sinc(f_times_t_high / np.pi) * 2 * f2_hz / sr
       - np.sinc(f_times_t_low  / np.pi) * 2 * f1_hz / sr)
    h *= window

    _, H = freqz(h, worN=n_freqs, fs=sr)
    return np.abs(H)

freqs_resp = np.linspace(0, SR / 2, 4096)

# ── Fisher discriminability (recomputed quickly) ───────────────────────────────
print("\nComputing Fisher discriminability …")
class_psds = {}
for label, prefix in CLASSES.items():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "test", f"{prefix}_*.pt")))[:N_FISHER]
    psds  = [welch(torch.load(f, weights_only=True)["signal"].numpy(),
                   fs=SR, nperseg=1024)[1] for f in files]
    class_psds[label] = np.array(psds)

f_psd = welch(torch.load(files[0], weights_only=True)["signal"].numpy(),
              fs=SR, nperseg=1024)[0]

mask_f   = f_psd <= FMAX
f_plot   = f_psd[mask_f]
all_means = np.array([class_psds[l][:, mask_f].mean(0) for l in CLASSES])
grand_mean = all_means.mean(0)
between_var = np.mean((all_means - grand_mean) ** 2, axis=0)
within_var  = np.mean([class_psds[l][:, mask_f].var(0) for l in CLASSES], axis=0)
fisher      = between_var / (within_var + 1e-30)
fisher_smooth = np.convolve(fisher, np.ones(5) / 5, mode="same")
fisher_norm   = fisher_smooth / fisher_smooth.max()

# ── Plot ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
fig.subplots_adjust(hspace=0.35)

# colour map: blue (low freq) → red (high freq)
norm = plt.Normalize(fc_np.min(), fc_np.max())
cmap = cm.get_cmap("coolwarm")

# ── Panel A: filter frequency responses ───────────────────────────────────────
ax = axes[0]
mask_resp = freqs_resp <= FMAX

# Fisher on twin axis
ax2 = ax.twinx()
ax2.fill_between(f_plot, fisher_norm, alpha=0.10, color="#555555")
ax2.plot(f_plot, fisher_norm, color="#555555", lw=1.0, ls="--", label="Fisher discriminability")
ax2.set_ylabel("Normalised Fisher Discriminability", fontsize=9, color="#555555")
ax2.set_ylim(0, 1.6)
ax2.tick_params(axis="y", colors="#555555")

# Filter responses
for i in range(len(f1_np)):
    H = sinc_filter_response(f1_np[i], f2_np[i], kernel_size, SR)
    H_norm = H[mask_resp] / (H[mask_resp].max() + 1e-12)
    ax.plot(freqs_resp[mask_resp], H_norm,
            color=cmap(norm(fc_np[i])), alpha=0.45, lw=0.8)

ax.axvline(150, color="#222222", lw=1.2, ls=":", label="150 Hz")
ax.set_xlim(0, FMAX)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Normalised Filter Response", fontsize=10)
ax.set_title("Learned SincNet Bandpass Filters — Frequency Responses (0–500 Hz)",
             fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)

# Colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.01)
cbar.set_label("Centre Frequency (Hz)", fontsize=8)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

# ── Panel B: centre freq vs bandwidth scatter ─────────────────────────────────
ax = axes[1]
sc = ax.scatter(fc_np, bw_np, c=fc_np, cmap="coolwarm", s=60,
                edgecolors="white", linewidths=0.5, zorder=3)

ax.axvline(150, color="#222222", lw=1.2, ls=":", label="150 Hz boundary")
ax.axvspan(0, 150, alpha=0.06, color="#0571b0", label="Discriminative band")

# Annotate counts
n_below = (fc_np < 150).sum()
ax.text(0.02, 0.95, f"{n_below}/40 filters with $f_c$ < 150 Hz",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

ax.set_xlabel("Centre Frequency (Hz)", fontsize=10)
ax.set_ylabel("Bandwidth (Hz)", fontsize=10)
ax.set_title("Learned Filter Bank: Centre Frequency vs. Bandwidth",
             fontsize=11, fontweight="bold")
ax.set_xlim(0, FMAX)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.03, pad=0.01,
             label="Centre Frequency (Hz)")

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\n✓ Saved: {OUT_PATH}")

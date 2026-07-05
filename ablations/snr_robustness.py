"""
SNR robustness evaluation for all HyMAD ablation variants.

Adds AWGN at increasing noise levels to the test set at inference time
(no re-training) and measures how each variant degrades.

Usage:
    python ablations/snr_robustness.py
    python ablations/snr_robustness.py --snr_min -5 --snr_max 30 --snr_step 5

Outputs:
    ablations/results/snr_robustness.json   per-variant per-SNR metrics
    ablations/results/snr_robustness.png    publication-quality F1 + AUROC plot
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_ABL_DIR)
for p in (_ROOT, _ABL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset.multi_label import SeismicMergedDataset
from models import (
    FullHyMAD, HyMAD_NoRNN, HyMAD_Conv1d,
    HyMAD_NoSelfAttn, HyMAD_UniCrossAttn, HyMAD_NaiveFusion,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.normpath(os.path.join(_ROOT, '..', 'seismic', 'Superimposed_Data'))
RESULTS_DIR = os.path.join(_ABL_DIR, 'results')

# ── Variant registry ──────────────────────────────────────────────────────────
ABLATIONS = {
    "full_hymad":     FullHyMAD,
    "no_rnn":         HyMAD_NoRNN,
    "conv1d_no_sinc": HyMAD_Conv1d,
    "no_self_attn":   HyMAD_NoSelfAttn,
    "uni_cross_attn": HyMAD_UniCrossAttn,
    "naive_fusion":   HyMAD_NaiveFusion,
}

DISPLAY_NAMES = {
    "full_hymad":     "HyMAD (Proposed)",
    "no_rnn":         "w/o Temporal Modeling (RNN)",
    "conv1d_no_sinc": "w/o SincNet Filterbank (Conv1d)",
    "no_self_attn":   "w/o Self-Attention",
    "uni_cross_attn": "Unidirectional Cross-Attention",
    "naive_fusion":   "Naive Fusion (Concat)",
}

# ── Validated categorical palette (reference palette slots 6,1,2,3,5,8) ──────
# HyMAD gets red (slot 6) for maximum contrast; ablations get remaining slots.
SERIES_STYLE = {
    "full_hymad":     dict(color="#e34948", lw=2.5, ls="-",  marker="o", ms=6, zorder=6),
    "no_rnn":         dict(color="#2a78d6", lw=1.5, ls="--", marker="s", ms=4, zorder=4),
    "conv1d_no_sinc": dict(color="#1baf7a", lw=1.5, ls="--", marker="^", ms=4, zorder=4),
    "no_self_attn":   dict(color="#eda100", lw=1.5, ls=":",  marker="D", ms=4, zorder=4),
    "uni_cross_attn": dict(color="#4a3aa7", lw=1.5, ls=":",  marker="v", ms=4, zorder=4),
    "naive_fusion":   dict(color="#eb6834", lw=1.5, ls="-.", marker="P", ms=4, zorder=4),
}


# ── Noise injection ───────────────────────────────────────────────────────────
def add_awgn(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add AWGN to batch x at the specified SNR relative to per-sample signal power.

    x: (B, L) — raw waveforms (before the model's internal z-normalisation).
    The test signals already contain noise from dataset generation (SNR 10–25 dB);
    this adds *additional* noise to stress-test at lower effective SNRs.
    """
    sig_power   = x.pow(2).mean(dim=-1, keepdim=True)            # (B, 1)
    noise_var   = sig_power / (10.0 ** (snr_db / 10.0))
    noise       = torch.randn_like(x) * noise_var.clamp(min=1e-12).sqrt()
    return x + noise


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_at_snr(model, loader, snr_db: float, device) -> dict:
    model.eval()
    preds_l, probs_l, labels_l = [], [], []
    for x, y in loader:
        x = add_awgn(x, snr_db).to(device)
        p = torch.sigmoid(model(x)).cpu()
        probs_l.append(p)
        preds_l.append((p >= 0.5).int())
        labels_l.append(y.cpu())
    preds  = torch.cat(preds_l).numpy()
    probs  = torch.cat(probs_l).numpy()
    labels = torch.cat(labels_l).numpy().astype(int)
    return {
        "f1":        float(f1_score(labels, preds,     average="macro", zero_division=0)),
        "auroc":     float(roc_auc_score(labels, probs, average="macro")),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(labels, preds,    average="macro", zero_division=0)),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(results: dict, snr_levels: list, out_path: str) -> None:
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif"],
        "font.size":        10,
        "axes.linewidth":   0.8,
        "axes.edgecolor":   "#c3c2b7",
        "xtick.color":      "#52514e",
        "ytick.color":      "#52514e",
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "grid.color":       "#e1e0d9",
        "grid.linewidth":   0.6,
        "legend.frameon":   False,
        "legend.fontsize":  8,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor":   "#fcfcfb",
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    metrics_cfg = [
        ("f1",    axes[0], "Macro F1",    (0.88, 1.002)),
        ("auroc", axes[1], "Macro AUROC", (0.90, 1.001)),
    ]

    for metric, ax, ylabel, ylim in metrics_cfg:
        # Training SNR shading
        ax.axvspan(10, 25, alpha=0.08, color="#0b0b0b",
                   label="Training SNR range (10–25 dB)")

        for name, snr_data in results.items():
            vals  = [snr_data[str(snr)][metric] for snr in snr_levels]
            style = SERIES_STYLE.get(name, {})
            ax.plot(
                snr_levels, vals,
                label=DISPLAY_NAMES.get(name, name),
                markeredgewidth=0.5,
                markeredgecolor="#fcfcfb",
                **style,
            )

        ax.set_xlabel("Added SNR (dB)", fontsize=10, color="#52514e")
        ax.set_ylabel(ylabel, fontsize=10, color="#52514e")
        ax.set_xlim(snr_levels[0] - 1, snr_levels[-1] + 1)
        ax.set_ylim(*ylim)
        ax.set_xticks(snr_levels)
        ax.grid(True, axis="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3, width=0.6)

    # Single shared legend below both plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.14),
        handlelength=2.2,
    )

    fig.suptitle(
        "SNR Robustness: All Ablation Variants",
        fontsize=11, fontweight="bold", color="#0b0b0b", y=1.01,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="#fcfcfb", edgecolor="none")
    plt.close()
    print(f"[✓] Plot saved to {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyMAD SNR robustness evaluation")
    parser.add_argument("--snr_min",  type=float, default=-5,  help="Lowest  SNR to test (dB)")
    parser.add_argument("--snr_max",  type=float, default=30,  help="Highest SNR to test (dB)")
    parser.add_argument("--snr_step", type=float, default=5,   help="SNR step size (dB)")
    parser.add_argument("--batch",    type=int,   default=512)
    args = parser.parse_args()

    snr_levels = [
        round(args.snr_min + i * args.snr_step, 1)
        for i in range(int((args.snr_max - args.snr_min) / args.snr_step) + 1)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"SNR levels : {snr_levels} dB")
    print(f"Data       : {DATA_DIR}\n")

    test_loader = DataLoader(
        SeismicMergedDataset(DATA_DIR, split="test"),
        batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Load any partial results so interrupted runs can resume
    out_json = os.path.join(RESULTS_DIR, "snr_robustness.json")
    results  = json.load(open(out_json)) if os.path.exists(out_json) else {}

    for name, cls in ABLATIONS.items():
        ckpt = os.path.join(RESULTS_DIR, name, "best_model.pth")
        if not os.path.exists(ckpt):
            print(f"[!] No checkpoint for '{name}' — run run_ablations.py first. Skipping.")
            continue

        if name in results:
            already = set(results[name].keys())
            remaining = [s for s in snr_levels if str(s) not in already]
            if not remaining:
                print(f"[=] '{name}' already complete, skipping.")
                continue
        else:
            results[name] = {}
            remaining = snr_levels

        model = cls().to(device)
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True)
        )
        model.eval()

        print(f"[{DISPLAY_NAMES.get(name, name)}]")
        for snr in remaining:
            m = evaluate_at_snr(model, test_loader, snr, device)
            results[name][str(snr)] = m
            print(f"  SNR = {snr:+5.1f} dB  →  "
                  f"F1={m['f1']:.4f}  AUROC={m['auroc']:.4f}")

        # Write after each model so partial results survive interruption
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[✓] Results saved to {out_json}")
    plot_results(results, snr_levels, os.path.join(RESULTS_DIR, "snr_robustness.png"))

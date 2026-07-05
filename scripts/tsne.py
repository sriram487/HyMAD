"""
Generate t-SNE visualisation of HyMAD penultimate-layer embeddings.

Usage (from repo root):
  conda run -n torch_env python scripts/tsne.py

Output:
  plots/tsne_combo_plot.png
"""

import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from collections import defaultdict, Counter

from model.model import SincNetRNN
from dataset.multi_label import SeismicMergedDataset

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "/home/sriram/Desktop/seismic/Superimposed_Data"
WEIGHTS     = os.path.join(REPO_ROOT, "runs", "HyMAD", "best_model.pth")
OUT_PATH    = os.path.join(REPO_ROOT, "plots", "tsne_combo_plot.png")

# Internal keys from the dataset label order: [Animal, Human, No Event, Vehicle]
CLASS_KEYS  = ["Animal_Movement", "Human_Movement", "No_Movement", "Vehicle_Movement"]

# Display names (used in legend)
DISPLAY = {
    "Animal_Movement":                          "Animal",
    "Human_Movement":                           "Human",
    "No_Movement":                              "No Event",
    "Vehicle_Movement":                         "Vehicle",
    "Animal_Movement+Human_Movement":           "Animal + Human",
    "Human_Movement+Vehicle_Movement":          "Human + Vehicle",
    "Animal_Movement+Vehicle_Movement":         "Animal + Vehicle",
}

# Fixed colours: single-class first, then multi-label
PALETTE = {
    "Animal":           "#4dac26",   # green
    "Human":            "#0571b0",   # blue
    "No Event":         "#9e9e9e",   # grey
    "Vehicle":          "#ca0020",   # red
    "Animal + Human":   "#a6d96a",   # light green
    "Human + Vehicle":  "#fdae61",   # orange
    "Animal + Vehicle": "#abd9e9",   # light blue
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def label_key(vec):
    return "+".join(CLASS_KEYS[i] for i, v in enumerate(vec) if v == 1) or "None"


def extract_features(model, loader, device):
    model.eval()
    feats, labs, captured = [], [], []

    handle = model.classifier.register_forward_hook(
        lambda m, inp, out: captured.append(inp[0].detach().cpu())
    )
    try:
        with torch.no_grad():
            for x, y in loader:
                captured.clear()
                model(x.to(device))
                feats.append(captured[0])
                labs.append(y.cpu())
    finally:
        handle.remove()

    return np.vstack(feats), np.vstack(labs)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = SincNetRNN().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    print(f"Loaded : {WEIGHTS}")

    loader = DataLoader(
        SeismicMergedDataset(DATA_DIR, split="test"),
        batch_size=256, shuffle=False, num_workers=4, pin_memory=True,
    )

    print("Extracting embeddings…")
    features, labels = extract_features(model, loader, device)

    keys = [label_key(l) for l in labels]
    counts = Counter(keys)
    print("\nLabel combination counts:")
    for k, v in sorted(counts.items()):
        disp = DISPLAY.get(k, k)
        print(f"  {disp:<22}: {v}")

    print("\nRunning t-SNE…")
    reduced = TSNE(n_components=2, perplexity=40, init="pca",
                   random_state=42, n_jobs=-1).fit_transform(features)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    combo_map = defaultdict(list)
    for i, k in enumerate(keys):
        combo_map[k].append(i)

    for raw_key, indices in combo_map.items():
        disp  = DISPLAY.get(raw_key, raw_key)
        color = PALETTE.get(disp, "#333333")
        idx   = np.array(indices)
        ax.scatter(reduced[idx, 0], reduced[idx, 1],
                   c=color, alpha=0.65, s=18, linewidths=0,
                   label=disp)

    # Legend — single-class first, then multi-label
    order = ["Animal", "Human", "No Event", "Vehicle",
             "Animal + Human", "Human + Vehicle", "Animal + Vehicle"]
    handles = [mpatches.Patch(color=PALETTE[d], label=d) for d in order if d in PALETTE]
    ax.legend(handles=handles, fontsize=10, framealpha=0.9,
              loc="upper left", title="Activity Class", title_fontsize=10)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("t-SNE Visualisation of HyMAD Penultimate Embeddings",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {OUT_PATH}")

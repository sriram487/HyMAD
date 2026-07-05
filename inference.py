import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
from tqdm import tqdm

from sklearn.metrics import (
    classification_report, roc_auc_score,
    multilabel_confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, hamming_loss,
)

from model.model import SincNetRNN
from dataset.multi_label import SeismicMergedDataset

CLASSES = ['Animal_Movement', 'Human_Movement', 'No_Movement', 'Vehicle_Movement']


# ---------- Attention Visualisation ----------

def overlay_attention_on_raw_input(waveform, attn_weights, query_index=0, save_path=None):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze().detach().cpu().numpy()
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights[0] if attn_weights.dim() == 3 else attn_weights
        attn_weights = attn_weights.detach().cpu().numpy()

    L = waveform.shape[0]
    attention = attn_weights[query_index]
    interp_attn = zoom(attention, L / len(attention))

    mean, std = np.mean(interp_attn), np.std(interp_attn) + 1e-9
    z_scores = (interp_attn - mean) / std
    mask  = z_scores > 1.0
    alpha = np.clip(z_scores, 0, 3) / 3

    t = np.arange(L)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, waveform, color='black', linewidth=1, label='Raw Input')

    masked_idx = np.where(mask[:-1])[0]
    if len(masked_idx) > 0:
        segs = np.stack([
            np.column_stack([t[masked_idx],     waveform[masked_idx]]),
            np.column_stack([t[masked_idx + 1], waveform[masked_idx + 1]]),
        ], axis=1)
        colors = np.zeros((len(masked_idx), 4))
        colors[:, 0] = 1.0
        colors[:, 3] = np.clip(alpha[masked_idx], 0, 1)
        ax.add_collection(LineCollection(segs, colors=colors, linewidths=2))

    ax.set_title(f"Attention Overlay on Raw Input (Query {query_index})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def visualize_attention_heatmap(attn_weights, itr, save_path=None):
    if attn_weights.dim() != 3:
        raise ValueError(f"Expected 3D attention weights, got shape {attn_weights.shape}")
    heatmap = attn_weights[0].detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap,
                xticklabels=[f"c{i}" for i in range(heatmap.shape[1])],
                yticklabels=[f"q{i}" for i in range(heatmap.shape[0])],
                cmap="viridis")
    plt.xlabel("Context (Keys from RNN output)")
    plt.ylabel("Query (from Sinc branch)")
    plt.title("Cross-Attention Heatmap")
    plt.savefig(f"{save_path}/heatmap_{itr}.png")
    plt.close()


# ---------- Inference ----------

def run_inference(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="[Inference]"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    return (
        torch.cat(all_preds).numpy().astype(int),
        torch.cat(all_labels).numpy().astype(int),
        torch.cat(all_probs).numpy().astype(float),
    )


# ---------- Evaluation ----------

def evaluate_and_save_confusion_matrix(all_labels, all_preds, class_names,
                                        save_path="confusion_matrix.png"):
    strict_acc  = accuracy_score(all_labels, all_preds)
    hamming_acc = 1.0 - hamming_loss(all_labels, all_preds)
    print(f"[Metrics] Exact Match Accuracy : {strict_acc:.4f}")
    print(f"[Metrics] Hamming Accuracy     : {hamming_acc:.4f}")

    cm = multilabel_confusion_matrix(all_labels, all_preds)
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 3))
    if n_classes == 1:
        axes = [axes]

    for i, (class_name, ax) in enumerate(zip(class_names, axes)):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i],
                                      display_labels=["Not " + class_name, class_name])
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d', colorbar=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right",
                           fontsize=8, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, fontweight="bold")
        ax.set_title(class_name, fontsize=11, fontweight="bold")
        ax.grid(False)

    plt.tight_layout(pad=2.0, w_pad=2.5)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[✓] Confusion matrices saved to {save_path}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


def plot_multilabel_roc(labels, probs, class_names, save_path=None):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "font.size": 12})
    plt.figure(figsize=(7, 5))
    aucs = []
    for i in range(labels.shape[1]):
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.plot([], [], " ", label=f"Macro-AUC = {np.mean(aucs):.2f}")
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title("ROC Curve (Multi-Label)", fontweight="bold")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(loc="lower right", frameon=False, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[✓] ROC saved to {save_path}")
    plt.show()


def plot_multilabel_prc(labels, probs, class_names, save_path=None):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "font.size": 12})
    plt.figure(figsize=(7, 5))
    aps = []
    for i in range(labels.shape[1]):
        precision, recall, _ = precision_recall_curve(labels[:, i], probs[:, i])
        ap = average_precision_score(labels[:, i], probs[:, i])
        aps.append(ap)
        plt.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP={ap:.2f})")
    plt.plot([], [], " ", label=f"Macro-AP = {np.mean(aps):.2f}")
    plt.xlabel("Recall", fontweight="bold")
    plt.ylabel("Precision", fontweight="bold")
    plt.title("Precision–Recall Curve (Multi-Label)", fontweight="bold")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(loc="lower left", frameon=False, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[✓] PRC saved to {save_path}")
    plt.show()


# ---------- Main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     type=str, default="HyMAD",
                        help="Experiment name (under runs/)")
    parser.add_argument("--weights", type=str, default="best_model.pth",
                        help="Checkpoint filename")
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_set    = SeismicMergedDataset('/home/sriram/Desktop/seismic/Superimposed_Data', split='test')
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = SincNetRNN().to(device)
    weights_path = os.path.join("runs", args.exp, args.weights)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"[✓] Loaded model from {weights_path}")

    preds, labels, probs = run_inference(model, test_loader, device)

    out_dir = os.path.join("runs", args.exp)
    evaluate_and_save_confusion_matrix(
        labels, preds, CLASSES,
        save_path=os.path.join(out_dir, "confusion_matrix.png"),
    )

    macro_auroc = roc_auc_score(labels, probs, average='macro')
    print(f"[Metrics] Macro AUROC: {macro_auroc:.4f}")

    plot_multilabel_roc(labels, probs, CLASSES,
                        save_path=os.path.join(out_dir, "roc_curve.png"))
    plot_multilabel_prc(labels, probs, CLASSES,
                        save_path=os.path.join(out_dir, "prc_curve.png"))

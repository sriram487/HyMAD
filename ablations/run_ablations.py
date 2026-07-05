"""
Run HyMAD ablation studies.

Usage:
    python ablations/run_ablations.py                        # run all variants fresh
    python ablations/run_ablations.py --resume               # skip done, resume interrupted
    python ablations/run_ablations.py --ablation no_rnn
    python ablations/run_ablations.py --ablation no_rnn --resume

Results are written to:
    ablations/results/<variant>/best_model.pth       ← model weights (inference-compatible)
    ablations/results/<variant>/checkpoint_best.pth  ← full checkpoint at best epoch
    ablations/results/<variant>/checkpoint_latest.pth← full checkpoint at last completed epoch
    ablations/results/<variant>/results.json
    ablations/results/summary.json                   ← all variants merged
"""

import os
import sys
import json
import math
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_ABL_DIR)
for p in (_ROOT, _ABL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset.multi_label import SeismicMergedDataset
from models import (
    FullHyMAD,
    HyMAD_NoRNN,
    HyMAD_Conv1d,
    HyMAD_NoSelfAttn,
    HyMAD_UniCrossAttn,
    HyMAD_NaiveFusion,
)
from sklearn.metrics import (
    accuracy_score, hamming_loss,
    precision_score, recall_score, f1_score, roc_auc_score,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.normpath(os.path.join(_ROOT, '..', 'seismic', 'Superimposed_Data'))
RESULTS_DIR = os.path.join(_ABL_DIR, 'results')

NUM_EPOCHS  = 400
BATCH_SIZE  = 512
LR_PEAK     = 5e-3
WARMUP_FRAC = 0.15
SEED        = 42

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
    "conv1d_no_sinc": "w/o SincNet (Conv1d)",
    "no_self_attn":   "w/o Self-Attention",
    "uni_cross_attn": "Unidirectional Cross-Attention",
    "naive_fusion":   "Naive Fusion (Concat)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        total_loss += criterion(model(x), y).item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_metrics(model, loader, device) -> dict:
    model.eval()
    preds_list, probs_list, labels_list = [], [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device))).cpu()
        probs_list.append(p)
        preds_list.append((p >= 0.5).int())
        labels_list.append(y.cpu())
    preds  = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy().astype(int)
    probs  = torch.cat(probs_list).numpy()
    return {
        "exact_match_acc": float(accuracy_score(labels, preds)),
        "hamming_acc":     float(1.0 - hamming_loss(labels, preds)),
        "precision":       float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall":          float(recall_score(labels, preds,    average="macro", zero_division=0)),
        "f1":              float(f1_score(labels, preds,        average="macro", zero_division=0)),
        "auroc":           float(roc_auc_score(labels, probs,   average="macro")),
    }


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_loss, metrics):
    torch.save({
        "epoch":         epoch,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "metrics":       metrics,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_val_loss"]


# ── Single variant training run ───────────────────────────────────────────────
def run_one(name: str, model_cls, loaders, device, resume: bool = False) -> dict:
    train_ld, val_ld, test_ld = loaders
    out_dir     = os.path.join(RESULTS_DIR, name)
    ckpt_latest = os.path.join(out_dir, "checkpoint_latest.pth")

    if resume and os.path.exists(ckpt_latest):
        raw = torch.load(ckpt_latest, map_location="cpu", weights_only=True)
        print(f"  [{name}] Resuming from epoch {raw['epoch']+1} "
              f"(best_val_loss={raw['best_val_loss']:.4f})")
    else:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    set_seed()
    model     = model_cls().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PEAK)
    warmup    = int(WARMUP_FRAC * NUM_EPOCHS)
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda s: warmup_cosine(s, warmup, NUM_EPOCHS))

    start_epoch   = 0
    best_val_loss = float('inf')

    if resume and os.path.exists(ckpt_latest):
        start_epoch, best_val_loss = load_checkpoint(
            ckpt_latest, model, optimizer, scheduler, device)
        start_epoch += 1

    pbar = tqdm(range(start_epoch, NUM_EPOCHS), desc=f"  [{name}]")
    for epoch in pbar:
        train_loss = train_epoch(model, train_ld, optimizer, criterion, device)
        scheduler.step()
        val_loss   = val_epoch(model, val_ld, criterion, device)
        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        metrics = {"train_loss": train_loss, "val_loss": val_loss}

        save_checkpoint(ckpt_latest, epoch, model, optimizer, scheduler,
                        best_val_loss, metrics)

        if is_best:
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            save_checkpoint(os.path.join(out_dir, "checkpoint_best.pth"),
                            epoch, model, optimizer, scheduler, best_val_loss, metrics)

    # Evaluate best checkpoint on test set
    model.load_state_dict(
        torch.load(os.path.join(out_dir, "best_model.pth"),
                   map_location=device, weights_only=True)
    )
    metrics = compute_metrics(model, test_ld, device)
    metrics["best_val_loss"] = float(best_val_loss)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  [{name}] F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}"
          f"  Recall={metrics['recall']:.4f}  AUROC={metrics['auroc']:.4f}")
    return metrics


# ── Summary helpers ───────────────────────────────────────────────────────────
def _load_summary() -> dict:
    path = os.path.join(RESULTS_DIR, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_summary(data: dict) -> None:
    path = os.path.join(RESULTS_DIR, "summary.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _print_summary(data: dict) -> None:
    cols = ("F1", "Prec", "Recall", "AUROC", "ExactAcc")
    keys = ("f1", "precision", "recall", "auroc", "exact_match_acc")
    header = f"{'Variant':<34} " + "  ".join(f"{c:>8}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for key, m in data.items():
        label = DISPLAY_NAMES.get(key, key)
        vals  = "  ".join(f"{m[k]:>8.4f}" for k in keys)
        print(f"{label:<34} {vals}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyMAD ablation study runner")
    parser.add_argument(
        "--ablation", default="all",
        choices=list(ABLATIONS.keys()) + ["all"],
        help="Variant to run (default: all)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Compute device, e.g. cuda, cuda:0, cuda:1 (default: cuda)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip completed variants and resume any interrupted run",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available on this machine. "
                "Install the CUDA-enabled PyTorch build or run inside the correct conda env:\n"
                "  conda run -n torch_env python ablations/run_ablations.py"
            )
        device = torch.device(args.device)
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem  = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"Device : {device}  ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        device = torch.device(args.device)
        print(f"Device : {device}")

    print(f"Data   : {DATA_DIR}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nLoading datasets...")
    loaders = (
        DataLoader(SeismicMergedDataset(DATA_DIR, split="train"),
                   batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True),
        DataLoader(SeismicMergedDataset(DATA_DIR, split="val"),
                   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        DataLoader(SeismicMergedDataset(DATA_DIR, split="test"),
                   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
    )

    to_run  = (ABLATIONS if args.ablation == "all"
               else {args.ablation: ABLATIONS[args.ablation]})
    summary = _load_summary()

    for name, cls in to_run.items():
        result_path = os.path.join(RESULTS_DIR, name, "results.json")
        if args.resume and os.path.exists(result_path):
            print(f"\n  [{name}] Already completed — skipping.")
            continue

        print(f"\n{'='*60}\n  Ablation: {DISPLAY_NAMES.get(name, name)}\n{'='*60}")
        summary[name] = run_one(name, cls, loaders, device, resume=args.resume)
        _save_summary(summary)

    print(f"\n  Summary saved to {os.path.join(RESULTS_DIR, 'summary.json')}")
    _print_summary(summary)

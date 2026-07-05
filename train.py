import os
import json
import math
import random
import shutil
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model.model import SincNetRNN
from dataset.multi_label import SeismicMergedDataset

from tensorboardX import SummaryWriter

from sklearn.metrics import (
    classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, hamming_loss, precision_score, recall_score, f1_score, roc_auc_score,
)
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def warmup_cosine_lr_lambda(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1. + math.cos(math.pi * progress))

def evaluate_and_plot_confusion_matrix(model, test_loader, device, class_names, criterion, save_dir="."):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    test_loss = 0.0

    test_bar = tqdm(test_loader, desc="[Test]", leave=False)
    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            test_loss += loss.item() * inputs.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            test_bar.set_postfix(loss=loss.item())

    test_loss /= len(test_loader.dataset)
    all_preds  = torch.cat(all_preds).numpy().astype(int)
    all_labels = torch.cat(all_labels).numpy().astype(int)
    all_probs  = torch.cat(all_probs).numpy()

    metrics = {
        "exact_match_acc": float(accuracy_score(all_labels, all_preds)),
        "hamming_acc":     float(1.0 - hamming_loss(all_labels, all_preds)),
        "precision":       float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall":          float(recall_score(all_labels, all_preds,    average="macro", zero_division=0)),
        "f1":              float(f1_score(all_labels, all_preds,        average="macro", zero_division=0)),
        "auroc":           float(roc_auc_score(all_labels, all_probs,   average="macro")),
        "test_loss":       float(test_loss),
    }

    print(f"\n[Test] Loss: {test_loss:.4f}")
    for k, v in metrics.items():
        if k != "test_loss":
            print(f"  {k}: {v:.4f}")

    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved to {results_path}")

    cm = multilabel_confusion_matrix(all_labels, all_preds)
    for i, class_name in enumerate(class_names):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i], display_labels=["Not " + class_name, class_name])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f"Confusion Matrix for '{class_name}'")
        plt.grid(False)
        plt.savefig(f"{save_dir}/cm_{class_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(classification_report(all_labels, all_preds, target_names=class_names))


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_loss, metrics):
    torch.save({
        "epoch":          epoch,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
        "best_val_loss":  best_val_loss,
        "metrics":        metrics,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_val_loss"]


if __name__ == "__main__":

    classes = ['Animal_Movement', 'Human_Movement', 'No_Movement', 'Vehicle_Movement']

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",    type=str, default='HyMAD')
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint_latest.pth in the exp dir")
    args = parser.parse_args()

    exp_dir   = f"./runs/{args.exp}"
    ckpt_latest = os.path.join(exp_dir, "checkpoint_latest.pth")

    if args.resume and os.path.exists(ckpt_latest):
        print(f"[Resume] Found checkpoint in {exp_dir}")
    else:
        if args.resume:
            print(f"[Resume] No checkpoint found in {exp_dir}, starting fresh.")
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device", device)
    set_seed(42)

    DATA_DIR = '../seismic/Superimposed_Data'
    train_set = SeismicMergedDataset(DATA_DIR, split='train')
    val_set   = SeismicMergedDataset(DATA_DIR, split='val')
    test_set  = SeismicMergedDataset(DATA_DIR, split='test')

    batch_size = 256
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SincNetRNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)

    num_epochs    = 400
    warmup_steps  = int(0.15 * num_epochs)
    scheduler     = LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_lr_lambda(step, warmup_steps, num_epochs),
    )

    start_epoch   = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(ckpt_latest):
        start_epoch, best_val_loss = load_checkpoint(
            ckpt_latest, model, optimizer, scheduler, device)
        start_epoch += 1   # resume from the next epoch
        print(f"[Resume] Resuming from epoch {start_epoch+1}  (best_val_loss={best_val_loss:.4f})")

    writer = SummaryWriter(log_dir=f'runs/{args.exp}')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss    = 0.0
        correct_train = 0.0
        total_labels  = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device).float()

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            preds          = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (preds == labels).float().sum().item()
            total_labels  += labels.numel()

            train_bar.set_postfix(loss=loss.item())

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_acc   = 100.0 * correct_train / total_labels

        writer.add_scalar('Loss/train',    train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        model.eval()
        with torch.no_grad():
            val_loss     = 0.0
            correct_val  = 0.0
            total_labels = 0

            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs  = model(inputs)
                loss     = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs    = torch.sigmoid(outputs)
                preds    = (probs >= 0.5).float()
                correct_val  += (preds == labels).float().sum().item()
                total_labels += labels.numel()
                val_bar.set_postfix(loss=loss.item())

            val_loss /= len(val_loader.dataset)
            val_acc   = 100.0 * correct_val / total_labels

            writer.add_scalar('Loss/val',    val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        metrics = {
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        }

        # Always save latest checkpoint so training can be resumed
        save_checkpoint(ckpt_latest, epoch, model, optimizer, scheduler, best_val_loss, metrics)

        if is_best:
            # model-weights-only file for inference.py compatibility
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            # full checkpoint at best epoch (optimizer + scheduler state included)
            save_checkpoint(os.path.join(exp_dir, "checkpoint_best.pth"),
                            epoch, model, optimizer, scheduler, best_val_loss, metrics)
            print(f"[✓] New best model at epoch {epoch+1} (val_loss={val_loss:.4f})")

        print(f"[Validation @ Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Load best checkpoint before final evaluation
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth"),
                                     map_location=device, weights_only=True))
    evaluate_and_plot_confusion_matrix(model, test_loader, device, classes, criterion, save_dir=exp_dir)

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, hamming_loss, classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Allow importing the root-level dataset loader for the raw-waveform CNN baseline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.multi_label import SeismicMergedDataset


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Feature loading ───────────────────────────────────────────────────────────

def load_features(feature_root: str, ftype: str, split: str):
    """Load pre-extracted features for one feature type and one split."""
    split_dir = os.path.join(feature_root, ftype, split)
    files = sorted(
        [f for f in os.listdir(split_dir) if f.endswith(".pt")],
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    feats, labels = [], []
    for fname in tqdm(files, desc=f"  {ftype}/{split}", leave=False):
        d = torch.load(os.path.join(split_dir, fname), weights_only=True)
        feats.append(d["features"])
        labels.append(d["labels"])
    return torch.stack(feats), torch.stack(labels)   # (N, D, T), (N, C)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    m = {
        "exact_match_acc": accuracy_score(y_true, y_pred),
        "hamming_acc":     1.0 - hamming_loss(y_true, y_pred),
        "precision":       precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":          recall_score(y_true, y_pred,    average="macro", zero_division=0),
        "f1":              f1_score(y_true, y_pred,        average="macro", zero_division=0),
    }
    try:
        m["auroc"] = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        m["auroc"] = float("nan")
    return m


# ── PyTorch model definitions ─────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Single-direction LSTM baseline (matches paper Table III)."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):             # (B, T, D) → (B, C) logits
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM — stronger recurrent baseline."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):             # (B, T, D) → (B, C) logits
        _, (hn, _) = self.lstm(x)
        return self.fc(torch.cat([hn[-2], hn[-1]], dim=-1))


class MLPClassifier(nn.Module):
    """3-layer MLP on flattened features (matches paper Table III)."""
    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):             # (B, D) → (B, C) logits
        return self.net(x)


class TransformerClassifier(nn.Module):
    """Transformer encoder on spectral feature sequences."""
    def __init__(self, input_dim: int, d_model: int = 64, num_heads: int = 4,
                 num_layers: int = 2, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(input_dim, d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout,
            batch_first=True, dim_feedforward=d_model * 4,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, num_classes)

    def forward(self, x):             # (B, T, D) → (B, C) logits
        return self.fc(self.encoder(self.proj(x)).mean(dim=1))


class CNN1DClassifier(nn.Module):
    """
    End-to-end 1-D CNN operating directly on raw waveforms.
    Key ablation for TGRS: shows what a plain CNN achieves without
    the learnable SincNet frequency filters and temporal fusion.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):             # (B, L) → (B, C) logits
        x = x.unsqueeze(1).float()
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-9)
        return self.fc(self.encoder(x).squeeze(-1))


# ── Training / evaluation helpers ────────────────────────────────────────────

def train_torch(model: nn.Module, tr_loader: DataLoader,
                val_loader: DataLoader | None = None,
                epochs: int = 200, lr: float = 1e-3,
                patience: int = 20,
                device: torch.device = torch.device("cpu")) -> nn.Module:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss   = float("inf")
    best_state      = None
    patience_count  = 0

    for _ in tqdm(range(epochs), desc="    epochs", leave=False):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device).float()
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_torch(model: nn.Module, loader: DataLoader,
               device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_list, probs_list = [], []
    for xb, _ in loader:
        probs = torch.sigmoid(model(xb.to(device))).cpu()
        preds_list.append((probs >= 0.5).int())
        probs_list.append(probs)
    return torch.cat(preds_list).numpy(), torch.cat(probs_list).numpy()


def collect_labels(loader: DataLoader) -> np.ndarray:
    return torch.cat([yb for _, yb in loader]).numpy()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CLASSES       = ["Animal_Movement", "Human_Movement", "No_Movement", "Vehicle_Movement"]
    FEATURE_ROOT  = "features"
    FEATURE_TYPES = ["MFCC", "LogMel", "LFSCC"]
    DATA_DIR      = "/home/sriram/Desktop/seismic/Superimposed_Data"
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR      = "trained_models"
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_seed(42)

    print(f"Using device: {DEVICE}")
    all_results: list[dict] = []

    # ── 1. Feature-based baselines ────────────────────────────────────────────
    for ftype in FEATURE_TYPES:
        print(f"\n{'='*60}\n  Feature type: {ftype}\n{'='*60}")

        feat_tr, lbl_tr   = load_features(FEATURE_ROOT, ftype, "train")
        feat_val, lbl_val = load_features(FEATURE_ROOT, ftype, "val")
        feat_te, lbl_te   = load_features(FEATURE_ROOT, ftype, "test")

        def to_flat(t): return t.reshape(len(t), -1).numpy()
        def to_seq(t):  return t.permute(0, 2, 1).numpy()

        # Normalise: fit scaler on train, apply to val/test
        flat_scaler = StandardScaler()
        Xtr_flat  = flat_scaler.fit_transform(to_flat(feat_tr))
        Xval_flat = flat_scaler.transform(to_flat(feat_val))
        Xte_flat  = flat_scaler.transform(to_flat(feat_te))

        T_, D_ = to_seq(feat_tr).shape[1], to_seq(feat_tr).shape[2]
        seq_scaler = StandardScaler()
        Xtr_seq  = seq_scaler.fit_transform(to_seq(feat_tr).reshape(-1, D_)).reshape(-1, T_, D_)
        Xval_seq = seq_scaler.transform(to_seq(feat_val).reshape(-1, D_)).reshape(-1, T_, D_)
        Xte_seq  = seq_scaler.transform(to_seq(feat_te).reshape(-1, D_)).reshape(-1, T_, D_)

        ytr  = lbl_tr.numpy()
        yval = lbl_val.numpy()
        yte  = lbl_te.numpy()

        # ── sklearn classifiers ───────────────────────────────────────────────
        sklearn_cfgs = {
            "RandomForest": OneVsRestClassifier(
                RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
            "ExtraTrees": OneVsRestClassifier(
                ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
            "XGBoost": OneVsRestClassifier(
                xgb.XGBClassifier(n_estimators=200, tree_method="hist", device="cpu",
                                   eval_metric="logloss", n_jobs=-1, random_state=42)),
            "LightGBM": OneVsRestClassifier(
                lgb.LGBMClassifier(n_estimators=200, device="cpu", n_jobs=-1,
                                   random_state=42, verbose=-1)),
            "CatBoost": OneVsRestClassifier(
                CatBoostClassifier(iterations=200, verbose=0,
                                   task_type="CPU", thread_count=-1, random_seed=42)),
            "GaussianNB": OneVsRestClassifier(
                GaussianNB()),
        }

        for name, clf in sklearn_cfgs.items():
            print(f"  [{name}]")
            # LightGBM requires DataFrame column names to avoid warnings
            Xtr_ = pd.DataFrame(Xtr_flat) if name == "LightGBM" else Xtr_flat
            Xte_ = pd.DataFrame(Xte_flat) if name == "LightGBM" else Xte_flat
            clf.fit(Xtr_, ytr)
            preds = clf.predict(Xte_)
            probs = (clf.predict_proba(Xte_)
                     if hasattr(clf, "predict_proba")
                     else preds.astype(float))
            m = compute_metrics(yte, preds, probs)
            m.update({"model": name, "feature": ftype})
            all_results.append(m)
            print(classification_report(yte, preds, target_names=CLASSES,
                                        digits=4, zero_division=0))
            joblib.dump(clf, os.path.join(SAVE_DIR, f"{name}_{ftype}.pkl"))

        # ── Sequential DL: LSTM, BiLSTM, Transformer ─────────────────────────
        D = Xtr_seq.shape[2]
        seq_models = {
            "LSTM":        LSTMClassifier(input_dim=D),
            "BiLSTM":      BiLSTMClassifier(input_dim=D),
            "Transformer": TransformerClassifier(input_dim=D),
        }
        for name, mdl in seq_models.items():
            print(f"  [{name}]")
            tr_ds  = TensorDataset(torch.tensor(Xtr_seq,  dtype=torch.float32),
                                   torch.tensor(ytr,      dtype=torch.float32))
            val_ds = TensorDataset(torch.tensor(Xval_seq, dtype=torch.float32),
                                   torch.tensor(yval,     dtype=torch.float32))
            te_ds  = TensorDataset(torch.tensor(Xte_seq,  dtype=torch.float32),
                                   torch.tensor(yte,      dtype=torch.float32))
            tr_ld  = DataLoader(tr_ds,  batch_size=128, shuffle=True,
                                num_workers=4, pin_memory=True)
            val_ld = DataLoader(val_ds, batch_size=128, shuffle=False,
                                num_workers=4, pin_memory=True)
            te_ld  = DataLoader(te_ds,  batch_size=128, shuffle=False,
                                num_workers=4, pin_memory=True)
            mdl = train_torch(mdl, tr_ld, val_loader=val_ld, epochs=200, device=DEVICE)
            preds, probs = eval_torch(mdl, te_ld, DEVICE)
            m = compute_metrics(yte, preds, probs)
            m.update({"model": name, "feature": ftype})
            all_results.append(m)
            print(classification_report(yte, preds, target_names=CLASSES,
                                        digits=4, zero_division=0))
            torch.save(mdl.state_dict(), os.path.join(SAVE_DIR, f"{name}_{ftype}.pt"))

        # ── MLP on flat features ──────────────────────────────────────────────
        print(f"  [MLP]")
        tr_ds  = TensorDataset(torch.tensor(Xtr_flat,  dtype=torch.float32),
                               torch.tensor(ytr,       dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(Xval_flat, dtype=torch.float32),
                               torch.tensor(yval,      dtype=torch.float32))
        te_ds  = TensorDataset(torch.tensor(Xte_flat,  dtype=torch.float32),
                               torch.tensor(yte,       dtype=torch.float32))
        tr_ld  = DataLoader(tr_ds,  batch_size=128, shuffle=True,
                            num_workers=4, pin_memory=True)
        val_ld = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)
        te_ld  = DataLoader(te_ds,  batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)
        mlp = MLPClassifier(input_dim=Xtr_flat.shape[1])
        mlp = train_torch(mlp, tr_ld, val_loader=val_ld, epochs=200, device=DEVICE)
        preds, probs = eval_torch(mlp, te_ld, DEVICE)
        m = compute_metrics(yte, preds, probs)
        m.update({"model": "MLP", "feature": ftype})
        all_results.append(m)
        print(classification_report(yte, preds, target_names=CLASSES,
                                    digits=4, zero_division=0))
        torch.save(mlp.state_dict(), os.path.join(SAVE_DIR, f"MLP_{ftype}.pt"))

    # ── 2. End-to-end raw waveform baseline: CNN-1D ───────────────────────────
    print(f"\n{'='*60}\n  Raw waveform baseline: CNN-1D\n{'='*60}")
    tr_ld  = DataLoader(SeismicMergedDataset(DATA_DIR, split="train"),
                        batch_size=256, shuffle=True,  num_workers=4, pin_memory=True)
    val_ld = DataLoader(SeismicMergedDataset(DATA_DIR, split="val"),
                        batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    te_ld  = DataLoader(SeismicMergedDataset(DATA_DIR, split="test"),
                        batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    cnn = CNN1DClassifier(num_classes=len(CLASSES))
    cnn = train_torch(cnn, tr_ld, val_loader=val_ld, epochs=400, patience=30, device=DEVICE)
    yte_raw      = collect_labels(te_ld)
    preds, probs = eval_torch(cnn, te_ld, DEVICE)
    m = compute_metrics(yte_raw, preds, probs)
    m.update({"model": "CNN-1D (Raw)", "feature": "Raw"})
    all_results.append(m)
    print(classification_report(yte_raw, preds, target_names=CLASSES,
                                digits=4, zero_division=0))
    torch.save(cnn.state_dict(), os.path.join(SAVE_DIR, "CNN1D_Raw.pt"))

    # ── 3. Save all results ───────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    col_order = ["model", "feature", "exact_match_acc", "hamming_acc",
                 "precision", "recall", "f1", "auroc"]
    df = df[col_order]
    df.to_csv("baseline_results.csv", index=False)
    with open("baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n" + df.to_string(index=False))
    print("\n✅ Saved to baseline_results.csv and baseline_results.json")

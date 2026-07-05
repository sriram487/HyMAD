import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scipy.signal import welch

# =========================================================
# CONFIG
# =========================================================

DATA_DIR   = "Seismic_data-4000"
OUTPUT_DIR = "Superimposed_Data"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR   = os.path.join(OUTPUT_DIR, "val")
TEST_DIR  = os.path.join(OUTPUT_DIR, "test")

SEED         = 42
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
# test = remainder

SR_FS  = 8000   # actual DAQ sample rate (Hz)
SR_LEN = 7999   # target waveform length  (samples)

# =========================================================
# LABEL MAP
# =========================================================

dict_label = {
    'Animal_Movement': 0,
    'Human_Movement':  1,
    'No_Movement':     2,
    'Vehicle_Movement': 3,
}
index_to_label = {v: k for k, v in dict_label.items()}

PAIRS = [(0, 1), (1, 3), (0, 3)]   # Animal+Human, Human+Vehicle, Animal+Vehicle

# =========================================================
# UTILITIES
# =========================================================

def rms(x):
    return np.sqrt(np.mean(x ** 2))


def enforce_length(signal, target=SR_LEN):
    if len(signal) >= target:
        return signal[:target]
    return np.pad(signal, (0, target - len(signal)), mode='constant')


def compute_psd(signal):
    f, Pxx = welch(signal, fs=SR_FS, nperseg=2048)
    return f, Pxx



def estimate_class_energy(signals):
    return np.median([rms(s) for s in signals])


def apply_random_delay(signal, max_delay=SR_LEN // 2):
    delay = np.random.randint(0, max_delay)
    return np.pad(signal, (delay, 0), mode='constant')


def add_seismic_noise(signal, snr_range=(10, 25)):
    snr_db = np.random.uniform(*snr_range)
    sig_pwr = np.mean(signal ** 2)
    noise   = np.random.normal(0, np.sqrt(sig_pwr / (10 ** (snr_db / 10))), signal.shape)
    return signal + noise, snr_db


def merge_signals(sig1, sig2, energy1, energy2):
    sig1 = sig1 / (rms(sig1) + 1e-8) * energy1
    sig2 = sig2 / (rms(sig2) + 1e-8) * energy2
    merged = sig1 + sig2
    p99 = np.percentile(np.abs(merged), 99)
    if p99 > 0:
        merged /= p99
    return merged


def per_class_split(signals, seed=SEED):
    """Split a list of signals into (train, val, test) lists — no leakage."""
    idx = list(range(len(signals)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(TRAIN_RATIO * len(idx))
    n_val   = int(VAL_RATIO   * len(idx))
    tr  = [signals[i] for i in idx[:n_train]]
    val = [signals[i] for i in idx[n_train:n_train + n_val]]
    te  = [signals[i] for i in idx[n_train + n_val:]]
    return tr, val, te


# =========================================================
# SAVE HELPERS
# =========================================================

def save_individual(signals, class_idx, out_dir):
    for i, sig in enumerate(signals):
        sig = enforce_length(sig)
        label = [0] * 4
        label[class_idx] = 1
        torch.save(
            {"signal": torch.tensor(sig, dtype=torch.float32),
             "label":  torch.tensor(label, dtype=torch.float32)},
            os.path.join(out_dir, f"{index_to_label[class_idx]}_{i}.pt"),
        )


def save_merged(sigs1, sigs2, class1, class2, energy, out_dir):
    rng = random.Random(SEED)
    s1 = sigs1[:]
    s2 = sigs2[:]
    rng.shuffle(s1)
    rng.shuffle(s2)
    count = min(len(s1), len(s2))

    for idx in tqdm(range(count), desc=f"  {index_to_label[class1]}+{index_to_label[class2]}", leave=False):
        a = apply_random_delay(s1[idx])
        b = apply_random_delay(s2[idx])
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        merged = merge_signals(a, b, energy[class1], energy[class2])
        merged, snr_db = add_seismic_noise(merged)
        merged = enforce_length(merged)

        label = [0] * 4
        label[class1] = 1
        label[class2] = 1

        torch.save(
            {"signal": torch.tensor(merged, dtype=torch.float32),
             "label":  torch.tensor(label, dtype=torch.float32),
             "meta":   {"classes": [class1, class2], "snr_db": snr_db}},
            os.path.join(
                out_dir,
                f"merged_{index_to_label[class1]}_{index_to_label[class2]}_{idx}.pt",
            ),
        )


# =========================================================
# MAIN
# =========================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load all raw source signals
    # ------------------------------------------------------------------
    print("Loading source signals...")
    class_signals = {i: [] for i in range(4)}
    file_paths = []

    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            if name.endswith(".csv"):
                parts = root.split(os.sep)
                cls, ch = parts[-2], parts[-1]
                if cls not in dict_label:
                    continue
                ch_index = int(''.join(filter(str.isdigit, ch))) + 1
                file_paths.append((os.path.join(root, name), cls, ch_index))

    for file_path, cls, ch_index in tqdm(file_paths):
        data = pd.read_csv(file_path, skiprows=22, header=None).iloc[:, ch_index]
        class_signals[dict_label[cls]].append(data.to_numpy())

    for k, v in class_signals.items():
        print(f"  {index_to_label[k]}: {len(v)} source signals")

    # ------------------------------------------------------------------
    # 2. Split source signals per-class BEFORE any merging
    # ------------------------------------------------------------------
    print("\nSplitting source signals per class...")
    splits = {}   # {class_idx: {'train': [...], 'val': [...], 'test': [...]}}
    for class_idx, signals in class_signals.items():
        tr, val, te = per_class_split(signals)
        splits[class_idx] = {'train': tr, 'val': val, 'test': te}
        print(f"  {index_to_label[class_idx]}: {len(tr)} train | {len(val)} val | {len(te)} test")

    # ------------------------------------------------------------------
    # 3. Estimate class energy from TRAINING signals only
    # ------------------------------------------------------------------
    print("\nEstimating class energy from training split...")
    energy = {}
    for class_idx in range(4):
        energy[class_idx] = estimate_class_energy(splits[class_idx]['train'])
        print(f"  {index_to_label[class_idx]}: {energy[class_idx]:.4e}")

    # ------------------------------------------------------------------
    # 4. Generate individual + merged signals within each split
    # ------------------------------------------------------------------
    for split_name, out_dir in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:
        print(f"\n--- {split_name.upper()} ---")

        print("  Saving individual signals...")
        for class_idx in range(4):
            save_individual(splits[class_idx][split_name], class_idx, out_dir)

        print("  Generating merged signals...")
        for class1, class2 in PAIRS:
            save_merged(
                splits[class1][split_name],
                splits[class2][split_name],
                class1, class2, energy, out_dir,
            )

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n✅  Dataset summary:")
    total = 0
    for split_name, out_dir in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:
        n = len([f for f in os.listdir(out_dir) if f.endswith('.pt')])
        print(f"  {split_name:6s}: {n} samples")
        total += n
    print(f"  {'total':6s}: {total} samples")


if __name__ == "__main__":
    main()

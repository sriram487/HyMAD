import os
import sys
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from scipy.fftpack import dct
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.multi_label import SeismicMergedDataset

DATA_DIR   = '/home/sriram/Desktop/seismic/Superimposed_Data'
OUTPUT_DIR = 'features'

SAMPLE_RATE = 8000
N_FFT       = 256
WIN_LENGTH  = 256
HOP_LENGTH  = 128
N_MELS      = 40
N_MFCC      = 13
N_LFSCC     = 13

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": N_FFT, "n_mels": N_MELS, "hop_length": HOP_LENGTH},
)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)


def compute_lfscc_batch(waveforms: torch.Tensor, n_lfscc: int = N_LFSCC) -> torch.Tensor:
    """Input: (B, L) → Output: (B, n_lfscc, T). Low-frequency DCT of the STFT power."""
    window = torch.hann_window(WIN_LENGTH, device=waveforms.device)
    stft = torch.stft(
        waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window, return_complex=True,
    )                                                          # (B, F, T)
    mag      = stft.abs() ** 2
    low      = mag[:, : mag.shape[1] // 4, :].cpu().numpy()  # lower quarter of spectrum
    lfscc_np = dct(low, type=2, axis=1, norm="ortho")
    return torch.tensor(lfscc_np[:, :n_lfscc, :], dtype=torch.float32)


def extract_split(split: str):
    dataset = SeismicMergedDataset(DATA_DIR, split=split)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=False)

    out_mfcc   = os.path.join(OUTPUT_DIR, "MFCC",   split)
    out_logmel = os.path.join(OUTPUT_DIR, "LogMel", split)
    out_lfscc  = os.path.join(OUTPUT_DIR, "LFSCC",  split)
    for d in [out_mfcc, out_logmel, out_lfscc]:
        os.makedirs(d, exist_ok=True)

    idx = 0
    for waveforms, labels in tqdm(loader, desc=f"[{split}]"):
        mfccs   = mfcc_transform(waveforms)
        logmels = torch.log(mel_transform(waveforms) + 1e-6)
        lfsccs  = compute_lfscc_batch(waveforms)

        for i in range(waveforms.shape[0]):
            lbl = labels[i].float()
            torch.save({"features": mfccs[i],   "labels": lbl}, os.path.join(out_mfcc,   f"{idx}.pt"))
            torch.save({"features": logmels[i], "labels": lbl}, os.path.join(out_logmel, f"{idx}.pt"))
            torch.save({"features": lfsccs[i],  "labels": lbl}, os.path.join(out_lfscc,  f"{idx}.pt"))
            idx += 1

    print(f"  [{split}] {idx} samples saved to {OUTPUT_DIR}/*/{{MFCC,LogMel,LFSCC}}/{split}/")


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        extract_split(split)

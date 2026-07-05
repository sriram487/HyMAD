import os
import glob
import torch
from torch.utils.data import Dataset


class SeismicMergedDataset(Dataset):
    """
    Loads pre-split seismic data produced by data_realism.py (v4+).

    Directory layout expected:
        data_dir/
            train/   *.pt
            val/     *.pt
            test/    *.pt

    Each .pt file contains {"signal": Tensor[7999], "label": Tensor[4]}.
    """

    def __init__(self, data_dir, split: str):
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split!r}"
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Run data_realism.py to regenerate the dataset."
            )
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        if not self.files:
            raise RuntimeError(f"No .pt files found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        return data["signal"], data["label"]

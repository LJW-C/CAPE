"""Panoradio/HF-Digital loading and fixed known-only fitting/validation pools."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .io import ROOT


class IQDataset(Dataset):
    def __init__(self, config, data_root=None, indices=None):
        self.config = config
        self.root = Path(data_root or config["data_root"]).expanduser()
        if not self.root.is_absolute():
            self.root = ROOT / self.root
        self.x = np.load(self.root / config["x_file"], mmap_mode="r")
        if config["dataset_type"] == "panoradio":
            tags = pd.read_csv(self.root / config["tags_file"])
            tags.columns = [str(c).strip() for c in tags.columns]
            modes = sorted(tags["mode"].unique().tolist())
            self.class_names = modes
            self.labels = tags["mode"].map({name: i for i, name in enumerate(modes)}).to_numpy(np.int64)
            self.snr = tags["snr"].to_numpy(dtype=np.float64)
        elif config["dataset_type"] == "npy_iq":
            self.labels = np.asarray(np.load(self.root / config["y_file"]), dtype=np.int64).reshape(-1)
            self.snr = np.asarray(np.load(self.root / config["snr_file"]), dtype=np.float64).reshape(-1)
            self.class_names = [str(i) for i in sorted(np.unique(self.labels))]
        else:
            raise ValueError("Supported dataset types: panoradio, npy_iq")
        if len(self.x) != len(self.labels) or len(self.snr) != len(self.labels):
            raise ValueError("I/Q, labels, and SNR must have the same number of rows")
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= len(self.labels)):
            raise ValueError("Invalid sample IDs")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        value = np.asarray(self.x[index])
        if np.iscomplexobj(value):
            value = np.stack([value.real.astype(np.float32), value.imag.astype(np.float32)], axis=0)
        else:
            value = np.array(value, dtype=np.float32, copy=True)
            if value.ndim == 2 and value.shape[0] != 2 and value.shape[-1] == 2:
                value = value.T
        if value.shape != (2, self.config["input_length"]):
            raise ValueError(f"Unexpected I/Q shape: {value.shape}")
        x = torch.from_numpy(np.ascontiguousarray(value))
        if self.config.get("normalize", True):
            x = x / x.pow(2).mean().sqrt().clamp_min(1e-8)
        return x, torch.tensor(int(self.labels[index])), torch.tensor(index)


def generate_split(labels, config):
    """Match the historical class-stratified two-stage split order."""
    rng = np.random.default_rng(config["data_seed"])
    train_ids, test_ids = [], []
    for cls in np.unique(labels):
        ids = np.flatnonzero(labels == cls)
        rng.shuffle(ids)
        cut = int(round(len(ids) * config["train_ratio"]))
        train_ids.extend(ids[:cut].tolist())
        test_ids.extend(ids[cut:].tolist())
    train_ids = np.asarray(train_ids, dtype=np.int64)
    train_labels = labels[train_ids]
    rng = np.random.default_rng(config["data_seed"])
    fitting, validation, unused_known, unused_unknown = [], [], [], []
    known = set(config["known_classes"])
    for cls in sorted(np.unique(train_labels).tolist()):
        relative = np.flatnonzero(train_labels == cls)
        rng.shuffle(relative)
        if cls in known:
            budget = min(max(1, int(round(len(relative) * config["label_ratio"]))), len(relative))
            selected = relative[:budget]
            n_val = int(round(len(selected) * config["validation_ratio"]))
            if config["validation_ratio"] > 0 and len(selected) > 1:
                n_val = max(1, min(n_val, len(selected) - 1))
            validation.extend(selected[:n_val].tolist())
            fitting.extend(selected[n_val:].tolist())
            unused_known.extend(relative[budget:].tolist())
        else:
            unused_unknown.extend(relative.tolist())
    # Preserve historical RNG consumption even though no unlabeled pool is trained here.
    unused = unused_known + unused_unknown
    rng.shuffle(fitting)
    rng.shuffle(unused)
    rng.shuffle(validation)
    return {"fitting_ids": train_ids[fitting], "validation_ids": train_ids[validation],
            "test_ids": np.asarray(test_ids, dtype=np.int64)}


def load_split(config, labels):
    path = ROOT / config["split_file"]
    if path.is_file():
        with np.load(path, allow_pickle=False) as arrays:
            split = {name: arrays[name].copy() for name in ("fitting_ids", "validation_ids", "test_ids")}
    else:
        split = generate_split(labels, config)
    fitting, validation, test = [split[name] for name in ("fitting_ids", "validation_ids", "test_ids")]
    if len(np.unique(fitting)) != len(fitting) or len(np.unique(validation)) != len(validation):
        raise ValueError("Fitting/validation contains duplicate sample IDs")
    if (np.intersect1d(fitting, validation).size or np.intersect1d(fitting, test).size
            or np.intersect1d(validation, test).size):
        raise ValueError("Fitting, validation, and test must be disjoint")
    if not np.isin(labels[fitting], config["known_classes"]).all() or not np.isin(
            labels[validation], config["known_classes"]).all():
        raise ValueError("Unknown classes entered fitting/calibration")
    return split


def datasets(config, data_root=None):
    base = IQDataset(config, data_root)
    split = load_split(config, base.labels)
    pools = {name: IQDataset(config, data_root, split[name + "_ids"])
             for name in ("fitting", "validation", "test")}
    return pools, split


def complete_loader(dataset, batch_size=128):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())

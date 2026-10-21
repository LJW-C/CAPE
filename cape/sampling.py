"""Class/SNR-stratified batches over occupied metadata cells."""

import math

import numpy as np
from torch.utils.data import Sampler


class ClassSNRBatchSampler(Sampler):
    def __init__(self, labels, snr, batch_size, seed,
                 cutpoints=(-7.5, -2.5, 2.5, 7.5, 12.5, 17.5, 22.5)):
        self.labels = np.asarray(labels, dtype=np.int64)
        values, cuts = np.asarray(snr, dtype=np.float64), np.asarray(cutpoints, dtype=np.float64)
        if not np.isfinite(values).all() or not np.isfinite(cuts).all() or np.any(np.diff(cuts) <= 0):
            raise ValueError("Finite SNR metadata and increasing cutpoints are required")
        self.bins = np.searchsorted(cuts, values, side="right")
        self.batch_size, self.seed, self.iteration = int(batch_size), int(seed), 0
        if len(self.labels) != len(self.bins) or len(self.labels) < batch_size or batch_size < 1:
            raise ValueError("Invalid pool length or batch size")
        self.classes = np.unique(self.labels).tolist()
        quota = math.ceil(batch_size / len(self.classes))
        if any(np.sum(self.labels == cls) < quota for cls in self.classes):
            raise ValueError("A class is too small for duplicate-free balanced batches")
        self.cells = {(cls, int(index)): np.flatnonzero((self.labels == cls) & (self.bins == index))
                      for cls in self.classes for index in np.unique(self.bins[self.labels == cls])}
        self.class_bins = {cls: [index for c, index in self.cells if c == cls] for cls in self.classes}

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.iteration * 1_000_003)
        self.iteration += 1
        queues = {key: rng.permutation(values).tolist() for key, values in self.cells.items()}
        cursor = {key: 0 for key in queues}
        bin_rings = {cls: [] for cls in self.classes}
        for _ in range(len(self)):
            class_ring, batch, used, used_counts = [], [], set(), {}
            for _ in range(self.batch_size):
                if not class_ring:
                    class_ring = rng.permutation(self.classes).tolist()
                cls = class_ring.pop()
                if not bin_rings[cls]:
                    bin_rings[cls] = rng.permutation(self.class_bins[cls]).tolist()
                preferred = bin_rings[cls].pop()
                candidates = [preferred, *[b for b in self.class_bins[cls] if b != preferred]]
                key = next(((cls, b) for b in candidates
                            if used_counts.get((cls, b), 0) < len(self.cells[(cls, b)])), None)
                if key is None:
                    raise RuntimeError("Class quota exhausted unexpectedly")
                while True:
                    if cursor[key] == len(queues[key]):
                        queues[key] = rng.permutation(self.cells[key]).tolist()
                        cursor[key] = 0
                    sample = queues[key][cursor[key]]
                    cursor[key] += 1
                    if sample not in used:
                        break
                used.add(sample)
                used_counts[key] = used_counts.get(key, 0) + 1
                batch.append(sample)
            yield batch

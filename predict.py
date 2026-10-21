"""Predict original class IDs or reject (-1) for a NumPy I/Q batch."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path

import numpy as np
import torch

from cape.io import choose_device, deterministic, load_checkpoint
from cape.scoring import prototype_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.npz"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    deterministic()
    device = choose_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    raw = np.load(args.input, mmap_mode="r")
    features = []
    with torch.no_grad():
        for start in range(0, len(raw), args.batch_size):
            value = np.asarray(raw[start:start + args.batch_size])
            if np.iscomplexobj(value):
                value = np.stack([value.real, value.imag], axis=1)
            elif value.ndim == 3 and value.shape[1] != 2 and value.shape[-1] == 2:
                value = value.transpose(0, 2, 1)
            if value.ndim != 3 or tuple(value.shape[1:]) != (2, payload["config"]["input_length"]):
                raise ValueError("Expected [N,L] complex or [N,2,L]/[N,L,2] real I/Q")
            x = torch.from_numpy(np.array(value, dtype=np.float32, copy=True)).to(device)
            x = x / x.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
            features.append(model.embed(x).cpu().numpy())
    scores, predicted = prototype_energy(np.concatenate(features), payload["prototypes"].numpy(),
                                         payload["config"]["energy_temperature"])
    labels = np.asarray(payload["config"]["known_classes"])[predicted]
    rejected = scores > payload["threshold"]
    labels[rejected] = -1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, predicted_class=labels, rejected=rejected, scores=scores,
                        threshold=payload["threshold"])
    print(args.output)


if __name__ == "__main__":
    main()

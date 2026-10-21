"""Reevaluate the three released CAPE seeds using their stored calibrated PE rule."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path

from cape.evaluation import evaluate, percentage_row, aggregate
from cape.io import ROOT, choose_device, deterministic, load_checkpoint, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["panoradio", "hfdigital"], default="panoradio")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    deterministic()
    output = args.output_dir or ROOT / "evaluation" / args.dataset
    device, rows = choose_device(args.device), []
    for seed in args.seeds:
        path = ROOT / "checkpoints" / args.dataset / f"seed{seed}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not included: {path}")
        model, payload = load_checkpoint(path, device)
        measured = evaluate(model, payload, output / f"seed{seed}", args.data_root, device)
        row = percentage_row(measured)
        rows.append(row)
        write_csv(output / f"results_seed{seed}.csv", [row])
    write_csv(output / "results_mean_std.csv", [aggregate(rows)])
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()

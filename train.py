"""Train the released CE-only CAPE architecture on labeled known classes."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cape import CAPE
from cape.data import datasets, complete_loader
from cape.evaluation import collect
from cape.io import ROOT, choose_device, deterministic, read_json, set_seed, write_csv, write_json
from cape.sampling import ClassSNRBatchSampler
from cape.scoring import empirical_prototypes, prototype_energy, calibrate_known


def known_validation_accuracy(model, fitting_loader, validation_loader, config, device):
    """Preserve the historical known-validation selector, separate from final PE."""
    fitting = collect(model, fitting_loader, device, with_logits=True)
    validation = collect(model, validation_loader, device, with_logits=True)
    prototypes = empirical_prototypes(fitting["z"], fitting["labels"], config["known_classes"])
    def scores(pack):
        sims = pack["z"] @ prototypes.T
        prob = torch.softmax(torch.tensor(pack["logits"], dtype=torch.float32), dim=1)
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=1) / math.log(len(config["known_classes"]))
        return 1.0 - sims.max(axis=1) + .25 * entropy.numpy(), sims.argmax(axis=1)
    fit_scores, _ = scores(fitting)
    val_scores, prediction = scores(validation)
    cutoff = float(np.quantile(fit_scores, .95))
    correct = np.asarray(config["known_classes"])[prediction] == validation["labels"]
    return float((correct & (val_scores <= cutoff)).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["panoradio", "hfdigital"], default="panoradio")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config or ROOT / "configs" / f"{args.dataset}.json")
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if config["epochs"] < 1:
        parser.error("--epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "cape.pt"
    if checkpoint.exists():
        raise FileExistsError("Use a new output directory; trained checkpoints are not overwritten")
    deterministic()
    set_seed(args.seed)
    device = choose_device(args.device)
    pools, _ = datasets(config, args.data_root)
    fit = pools["fitting"]
    stream_seed = 2_000_000 + args.seed * 10_000 + 11
    sampler = ClassSNRBatchSampler(fit.labels[fit.indices], fit.snr[fit.indices],
                                   config["batch_size"], stream_seed, config["snr_cutpoints"])
    training = DataLoader(fit, batch_sampler=sampler, num_workers=0,
                          generator=torch.Generator().manual_seed(stream_seed))
    # The original selector used a shuffled fitting pass with drop_last=True.
    selector_fitting = DataLoader(fit, batch_size=config["batch_size"], shuffle=True,
                                 drop_last=True, num_workers=0,
                                 generator=torch.Generator().manual_seed(stream_seed))
    validation_loader = complete_loader(pools["validation"], config["batch_size"])
    model = CAPE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    known_to_local = {cls: i for i, cls in enumerate(config["known_classes"])}
    best, history = -1.0, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses, correct, count = [], 0, 0
        for x, y, _ in training:
            x = x.to(device)
            labels = torch.tensor([known_to_local[int(v)] for v in y], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(x)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
        model.eval()
        selected = known_validation_accuracy(model, selector_fitting, validation_loader, config, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "train_accuracy": correct / count,
               "known_validation_accuracy": selected}
        history.append(row)
        if selected > best:
            best = selected
            torch.save({"method": "CAPE", "format_version": 1, "seed": args.seed,
                        "config": config, "epoch": epoch, "initialization": "random",
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}},
                       checkpoint)
        print(f"epoch={epoch:03d} loss={row['loss']:.6f} train={100*row['train_accuracy']:.2f} "
              f"known_val={100*selected:.2f} best={100*best:.2f}", flush=True)
        write_csv(args.output_dir / "train_history.csv", history)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    fitting = collect(model, complete_loader(fit, config["batch_size"]), device)
    validation = collect(model, validation_loader, device)
    prototypes = empirical_prototypes(fitting["z"], fitting["labels"], config["known_classes"])
    scores, _ = prototype_energy(validation["z"], prototypes, config["energy_temperature"])
    threshold, calibration = calibrate_known(scores, config["target_fpr"])
    payload.update(prototypes=torch.from_numpy(prototypes), threshold=threshold, calibration=calibration)
    torch.save(payload, checkpoint)
    write_json(args.output_dir / "objective_audit.json", {"method": "CAPE", "completed": True,
               "loss": "CE(x,y)", "training_inputs": "original labeled known samples",
               "test_used_for_selection": False, "initialization": "random"})
    print(f"Saved: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()

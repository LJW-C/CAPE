"""Evaluate CAPE with a frozen prototype bank and known-validation threshold."""

from pathlib import Path
import statistics

import numpy as np
import torch
import torch.nn.functional as F

from .data import datasets, complete_loader
from .io import set_seed, write_csv, write_json
from .perturbations import apply
from .scoring import METRICS, condition_metrics, four_group, prototype_energy, summarize_conditions


@torch.no_grad()
def collect(model, loader, device, perturbation=None, with_logits=False):
    features, labels, indices, logits = [], [], [], []
    for x, y, index in loader:
        x = x.to(device)
        if perturbation is not None:
            x = apply(x, *perturbation)
        output, semantic, _ = model(x)
        features.append(F.normalize(semantic, dim=1).cpu().numpy())
        labels.append(y.numpy())
        indices.append(index.numpy())
        if with_logits:
            logits.append(output.cpu().numpy())
    result = {"z": np.concatenate(features), "labels": np.concatenate(labels),
              "indices": np.concatenate(indices)}
    if with_logits:
        result["logits"] = np.concatenate(logits)
    return result


def evaluate(model, payload, output_dir, data_root=None, device="cpu"):
    config = payload["config"]
    pools, _ = datasets(config, data_root)
    loader = complete_loader(pools["test"], config["batch_size"])
    prototypes = payload["prototypes"].cpu().numpy()
    threshold = float(payload["threshold"])
    known, unknown = config["known_classes"], config["unknown_classes"]
    temperature, rng_seed = config["energy_temperature"], config["evaluation_seed"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def score(pack, filename):
        scores, prediction = prototype_energy(pack["z"], prototypes, temperature)
        np.savez_compressed(output_dir / filename, indices=pack["indices"], labels=pack["labels"],
                            scores=scores, predicted_local=prediction)
        return condition_metrics(pack["labels"], scores, prediction, threshold, known, unknown)

    model.eval()
    set_seed(rng_seed)
    print(f"CAPE seed {payload['seed']}: clean test", flush=True)
    clean_pack = collect(model, loader, device)
    clean = score(clean_pack, "clean_scores.npz")
    rows = []
    for index, item in enumerate(config["perturbations"]):
        print(f"CAPE seed {payload['seed']}: {item['name']}={item['level']}", flush=True)
        set_seed(rng_seed + index)
        shifted = collect(model, loader, device, (item["name"], item["level"]))
        if not np.array_equal(shifted["indices"], clean_pack["indices"]):
            raise RuntimeError("Clean/degraded test order differs")
        current = score(shifted, f"scores_condition{index}.npz")
        rows.append({"method": "CAPE", "dataset": config["dataset"], "seed": payload["seed"],
                     "perturbation": item["name"], "level": str(item["level"]),
                     **four_group(clean, current)})
    result = {"method": "CAPE", "dataset": config["dataset"], "seed": payload["seed"],
              "threshold": threshold, "metric_units": "fraction", **summarize_conditions(clean, rows)}
    write_json(output_dir / "summary.json", result)
    write_csv(output_dir / "perturbations.csv", rows)
    return result


def percentage_row(row):
    return {"dataset": row["dataset"], "method": "CAPE", "seed": int(row["seed"]),
            "metric_units": "percent", **{key: 100 * float(row[key]) for key in METRICS}}


def aggregate(rows):
    if len({int(row["seed"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate seeds")
    output = {"dataset": rows[0]["dataset"], "method": "CAPE",
              "seeds": " ".join(str(row["seed"]) for row in rows), "n_seeds": len(rows),
              "std_ddof": 1, "metric_units": "percent"}
    for key in METRICS:
        values = [float(row[key]) for row in rows]
        output[key + "_mean"] = statistics.mean(values)
        output[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
    return output

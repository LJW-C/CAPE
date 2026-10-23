"""Verify weights, reported seed sets, CSV aggregation, and available raw scores."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import json
import sys

import numpy as np
import torch

from cape.evaluation import aggregate, percentage_row
from cape.io import ROOT, read_csv, read_json, load_checkpoint, sha256
from cape.scoring import (METRICS, calibrate_known, condition_metrics, four_group,
                          prototype_energy, summarize_conditions)


def main():
    torch.set_num_threads(1)
    manifest = read_json(ROOT / "protocols/export_audit.json")
    weight_datasets = manifest.get("weight_datasets", manifest["datasets"])
    expected_weights = {f"checkpoints/{name}/seed{seed}.pt"
                        for name in weight_datasets for seed in (1, 2, 3)}
    actual_weights = {p.relative_to(ROOT).as_posix() for p in (ROOT / "checkpoints").rglob("*.pt")}
    assert actual_weights == expected_weights, "Unexpected or missing model weights"
    has_results = (ROOT / "results/results_all_seeds.csv").is_file()
    all_reported = read_csv(ROOT / "results/results_all_seeds.csv") if has_results else []
    summaries = read_csv(ROOT / "results/results_mean_std.csv") if has_results else []
    selection_path = ROOT / "results/selection_manifest.json"
    selection = read_json(selection_path) if selection_path.is_file() else {}
    reported_runs = selection.get("reported_run_ids", {name: [1, 2, 3] for name in manifest["datasets"]})
    training_seeds = selection.get("training_seeds", reported_runs)
    has_raw_scores = (ROOT / "results/raw").is_dir()
    if has_results:
        expected_pairs = {(name, run_id) for name, runs in reported_runs.items() for run_id in runs}
        actual_pairs = [(row["dataset"], int(row["run_id"])) for row in all_reported]
        assert len(actual_pairs) == len(set(actual_pairs)) and set(actual_pairs) == expected_pairs
        assert len(summaries) == len(reported_runs)
        assert {row["dataset"] for row in summaries} == set(reported_runs)
    all_recomputed = []
    for dataset in manifest["datasets"]:
        if not has_results and dataset not in weight_datasets:
            continue
        recomputed = []
        for run_id in reported_runs.get(dataset, [1, 2, 3]):
            training_seed = training_seeds[dataset][reported_runs[dataset].index(run_id)]
            path = ROOT / f"checkpoints/{dataset}/seed{training_seed}.pt"
            if dataset in weight_datasets:
                model, payload = load_checkpoint(path)
                audit = next(a for a in manifest["audits"] if a["dataset"] == dataset and a["seed"] == training_seed)
                assert sha256(path) == audit["release_checkpoint_sha256"]
                assert payload["seed"] == training_seed
                config = payload["config"]
                with torch.no_grad():
                    output, semantic, channel = model(torch.randn(2, 2, config["input_length"]))
                assert output.shape == (2, len(config["known_classes"]))
                assert semantic.shape == (2, config["semantic_dim"])
                assert channel.shape == (2, config["channel_dim"])
                assert not model.classifier.weight.requires_grad
            else:
                config = read_json(ROOT / f"configs/{dataset}.json")
            if not has_results:
                print(f"PASS {dataset} training seed {training_seed}: checkpoint integrity and model forward pass")
                continue
            archived = next(r for r in all_reported if r["dataset"] == dataset and int(r["run_id"]) == run_id)
            assert int(archived["training_seed"]) == training_seed
            assert archived["method"] == "CAPE" and archived["metric_units"] == "percent"
            assert all(np.isfinite(float(archived[key])) and 0 <= float(archived[key]) <= 100 for key in METRICS)
            if not has_raw_scores:
                individual = read_csv(ROOT / f"results/{dataset}/results_seed{run_id}.csv")
                assert len(individual) == 1 and individual[0] == archived
                conditions = read_csv(ROOT / f"results/{dataset}/perturbations_seed{run_id}.csv")
                assert [row["perturbation"] for row in conditions] == [item["name"] for item in config["perturbations"]]
                for condition in conditions:
                    assert condition["dataset"] == dataset and int(condition["run_id"]) == run_id
                    assert int(condition["training_seed"]) == training_seed
                    assert condition["metric_units"] == "fraction"
                    rates = [float(condition[key]) for key in ("KAR_clean", "KAR_deg", "URR_clean", "URR_deg")]
                    assert all(np.isfinite(rate) and 0 <= rate <= 1 for rate in rates)
                    assert abs(4 / sum(1 / (rate + 1e-8) for rate in rates) - float(condition["H4"])) < 1e-9
                for key in ("KAR_clean", "KAR_deg", "URR_clean", "URR_deg", "H4"):
                    average = 100 * np.mean([float(row[key]) for row in conditions])
                    assert abs(average - float(archived[key + "_avg"])) < 1e-9
                recomputed.append({"dataset": dataset, "seed": run_id,
                                   **{key: float(archived[key]) for key in METRICS}})
                print(f"PASS {dataset} run {run_id} (training seed {training_seed}): CSV and four-group aggregation")
                continue
            directory = ROOT / f"results/raw/{dataset}/seed{training_seed}"
            with np.load(directory / "validation_scores.npz", allow_pickle=False) as pack:
                assert np.isin(pack["labels"], config["known_classes"]).all()
                threshold, _ = calibrate_known(pack["scores"], config["target_fpr"])
                if dataset in weight_datasets:
                    assert threshold == payload["threshold"]
            def measured(filename):
                with np.load(directory / filename, allow_pickle=False) as pack:
                    return condition_metrics(pack["labels"], pack["scores"], pack["predicted_local"],
                                             threshold, config["known_classes"], config["unknown_classes"])
            clean = measured("clean_scores.npz")
            conditions = [four_group(clean, measured(f"scores_condition{i}.npz"))
                          for i in range(len(config["perturbations"]))]
            row = percentage_row({"dataset": dataset, "seed": run_id, **summarize_conditions(clean, conditions)})
            assert all(abs(row[k] - float(archived[k])) < 1e-9 for k in METRICS)
            recomputed.append(row)
            scope = "weights and archived scores" if dataset in weight_datasets else "archived scores (weights not published)"
            print(f"PASS {dataset} run {run_id} (training seed {training_seed}): {scope}, calibration, clean metrics, six-condition H4")
        if not has_results:
            continue
        computed = aggregate(recomputed)
        reference = next(r for r in summaries if r["dataset"] == dataset)
        assert reference["metric_units"] == "percent" and int(reference["std_ddof"]) == 1
        assert int(reference["n_seeds"]) == len(reported_runs[dataset])
        assert [int(value) for value in reference["run_ids"].split()] == reported_runs[dataset]
        assert [int(value) for value in reference["training_seeds"].split()] == training_seeds[dataset]
        for key in METRICS:
            for suffix in ("_mean", "_std"):
                assert abs(computed[key + suffix] - float(reference[key + suffix])) < 1e-9
        all_recomputed.extend(recomputed)
    checksum_file = ROOT / "checksums.sha256"
    if checksum_file.exists():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split("  ", 1)
            assert sha256(ROOT / filename) == digest, f"Changed release file: {filename}"
    if has_results and has_raw_scores:
        print("PASS: all exported seed tables and sample standard deviations (ddof=1) match raw scores.")
    elif has_results:
        print("PASS: reported CSVs, per-perturbation H4, and sample standard deviations (ddof=1) are internally consistent.")
        print("SKIP: raw scores are not published; AUROC and Unknown F1 were not independently recomputed here.")
        if selection.get("selection", {}).get("hfdigital", {}).get("timing") == "post_hoc_requested_subset":
            print(f"NOTE: HF-Digital runs {reported_runs['hfdigital']} use post hoc selected training seeds {training_seeds['hfdigital']}.")
    else:
        print("SKIP: result CSVs and raw scores are not published; metrics were not verified.")
    assert not any(name == "src" or name.startswith("src.") or name.startswith("scripts.")
                   for name in sys.modules), "Unexpected original-project import"
    print("PASS: complete checkpoint states load without any external initializer or original project imports.")


if __name__ == "__main__":
    main()

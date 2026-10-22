"""Plot CAPE seed results and six-condition robustness using the included CSVs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cape.io import ROOT, read_csv, read_json
from cape.scoring import condition_metrics, four_group


PALETTE = {"panoradio": "#0072B2", "hfdigital": "#E64B35"}
NAMES = {"panoradio": "Panoradio HF", "hfdigital": "HF-Digital"}


def perturbation_rows(dataset, seed):
    path = ROOT / f"results/{dataset}/perturbations_seed{seed}.csv"
    if path.exists():
        return read_csv(path)
    directory = ROOT / f"results/raw/{dataset}/seed{seed}"
    config = read_json(ROOT / f"configs/{dataset}.json")
    threshold = read_json(directory / "summary.json")["threshold"]

    def measured(filename):
        with np.load(directory / filename, allow_pickle=False) as scores:
            return condition_metrics(
                scores["labels"], scores["scores"], scores["predicted_local"],
                threshold, config["known_classes"], config["unknown_classes"])

    clean = measured("clean_scores.npz")
    return [{"perturbation": condition["name"],
             **four_group(clean, measured(f"scores_condition{index}.npz"))}
            for index, condition in enumerate(config["perturbations"])]


def style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 13,
        "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix"})


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "panoradio", "hfdigital"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    if not (ROOT / "results/results_all_seeds.csv").is_file():
        parser.error("Result CSVs are not included in this repository. Supply your own local results before plotting.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    rows = read_csv(ROOT / "results/results_all_seeds.csv")
    names = [name for name in NAMES if args.dataset in ("all", name)
             and any(row["dataset"] == name for row in rows)]
    fields = [("known_acc", "Known Acc."), ("unknown_auroc", "AUROC"),
              ("unknown_f1", "Unknown F1"), ("h_score", "H-score"), ("H4_avg", "Avg. H4")]
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.2), squeeze=False,
                             constrained_layout=True)
    for axis, name in zip(axes[0], names):
        selected = [r for r in rows if r["dataset"] == name]
        if sorted(int(r["seed"]) for r in selected) != [1, 2, 3]:
            raise ValueError("Expected exactly seeds 1, 2, and 3")
        values = np.asarray([[float(row[key]) for key, _ in fields] for row in selected])
        x = np.arange(len(fields))
        for index, marker in enumerate(("o", "s", "^")):
            axis.scatter(x + (index - 1) * .13, values[index], marker=marker, s=42,
                         facecolors="white", edgecolors=PALETTE[name], linewidths=1.3,
                         label=f"Seed {selected[index]['seed']}", zorder=3)
        axis.errorbar(x + .28, values.mean(axis=0), yerr=values.std(axis=0, ddof=1),
                      fmt="o", color=PALETTE[name], markersize=7, capsize=4,
                      linewidth=1.8, label="Mean +/- SD", zorder=4)
        axis.set_xticks(x, [label for _, label in fields], rotation=18, ha="right")
        axis.set_ylim(0, 105)
        axis.set_ylabel("Performance (%)")
        axis.set_title(NAMES[name], fontsize=17)
        axis.grid(axis="y", color="#E2E5E8", linewidth=.7)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=10, ncol=2, loc="lower left")
    save(fig, args.output_dir / "fig_cape_seed_results")

    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.2), squeeze=False,
                             constrained_layout=True)
    metrics = [("KAR_deg", "KAR-deg", "#0072B2", "o"),
               ("URR_deg", "URR-deg", "#E64B35", "s"), ("H4", "H4", "#00A087", "^")]
    for axis, name in zip(axes[0], names):
        groups = [perturbation_rows(name, seed) for seed in (1, 2, 3)]
        labels = [r["perturbation"] for r in groups[0]]
        if any([r["perturbation"] for r in group] != labels for group in groups):
            raise ValueError("Perturbation order differs across seeds")
        x = np.arange(len(labels))
        for key, label, color, marker in metrics:
            values = 100 * np.asarray([[float(row[key]) for row in group] for group in groups])
            axis.errorbar(x, values.mean(axis=0), yerr=values.std(axis=0, ddof=1),
                          color=color, marker=marker, linewidth=2.0, markersize=6,
                          capsize=3, label=label)
        axis.set_xticks(x, [p.upper() if p in ("awgn", "cfo") else p.title() for p in labels],
                        rotation=22, ha="right")
        axis.set_ylim(0, 105)
        axis.set_ylabel("Performance (%)")
        axis.set_title(NAMES[name], fontsize=17)
        axis.grid(axis="y", color="#E2E5E8", linewidth=.7)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=12, loc="lower left")
    save(fig, args.output_dir / "fig_cape_channel_robustness")
    print(args.output_dir)


if __name__ == "__main__":
    main()

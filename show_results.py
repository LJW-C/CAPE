"""Display the archived CAPE seed results and genuine sample standard deviations."""

import argparse

from cape.io import ROOT, read_csv


DISPLAY = (("known_acc", "Known Acc."), ("unknown_auroc", "AUROC"),
           ("unknown_f1", "Unknown F1"), ("h_score", "H-score"),
           ("KAR_deg_avg", "KAR-deg"), ("URR_deg_avg", "URR-deg"), ("H4_avg", "Avg. H4"))


def render(dataset="all"):
    if not (ROOT / "results/results_all_seeds.csv").is_file():
        raise SystemExit("Result CSVs are not included in this repository. Supply your own local results before using this script.")
    rows = read_csv(ROOT / "results/results_all_seeds.csv")
    summary = read_csv(ROOT / "results/results_mean_std.csv")
    lines = ["| Dataset | Run ID | Training Seed | " + " | ".join(label for _, label in DISPLAY) + " |",
             "|---|---|---|" + "---:|" * len(DISPLAY)]
    for name in dict.fromkeys(row["dataset"] for row in rows):
        if dataset != "all" and name != dataset:
            continue
        for row in (r for r in rows if r["dataset"] == name):
            lines.append("| " + name + " | " + row["run_id"] + " | " + row["training_seed"] + " | "
                         + " | ".join(f"{float(row[key]):.2f}" for key, _ in DISPLAY) + " |")
        row = next(r for r in summary if r["dataset"] == name)
        lines.append("| " + name + " | mean +/- SD | " + row["training_seeds"] + " | " + " | ".join(
            f"{float(row[key+'_mean']):.2f} +/- {float(row[key+'_std']):.2f}" for key, _ in DISPLAY) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "panoradio", "hfdigital"], default="all")
    print(render(parser.parse_args().dataset))

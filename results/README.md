# CAPE reported results

- Panoradio HF: seeds 1, 2, 3 (existing reported subset).
- HF-Digital: seeds 1, 3, 6 (post hoc requested subset).
- All rows use the known-only, CE-only CAPE method with CFDA, class/SNR sampling,
  fixed ETF classification, and empirical Prototype-Energy rejection. CI and CISC are disabled.

## Seed selection

HF-Digital seeds 1, 3, and 6 were requested after results for seeds 1 through 6 had
been viewed. Seeds 2, 4, and 5 were also completed but are not part of this report.
This is a descriptive post hoc subset, not a preregistered or unbiased
random-seed stability estimate. Do not describe these values as results across
all completed seeds or as the same training seeds used for Panoradio HF.

Panoradio HF seeds 1, 2, and 3 retain the existing reporting subset; no
preregistration claim is made for that subset either.

## Summary

Values are percentages. Uncertainty is the sample standard deviation (ddof=1)
over the three listed seeds within each dataset.

| Dataset | Seeds | Known Acc. | AUROC | Unknown F1 | H-score | KAR-deg | URR-deg | Avg. H4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Panoradio HF | 1, 2, 3 | 89.76 +/- 0.20 | 82.56 +/- 0.72 | 44.14 +/- 1.99 | 46.57 +/- 1.93 | 91.09 +/- 0.04 | 35.18 +/- 1.35 | 48.71 +/- 1.68 |
| HF-Digital | 1, 3, 6 | 66.49 +/- 0.31 | 80.78 +/- 0.43 | 34.10 +/- 3.63 | 33.55 +/- 3.21 | 92.00 +/- 0.49 | 24.83 +/- 2.10 | 37.50 +/- 3.22 |

## Files and units

- `results_all_seeds.csv`: all six reported dataset/seed records, with actual seed IDs.
- `results_mean_std.csv`: one mean/SD row per dataset; `seeds` records its exact seed set.
- `panoradio/results_seed*.csv` and `hfdigital/results_seed*.csv`: individual records.
- `*/perturbations_seed*.csv`: six perturbations per seed; rates use fractions in [0,1].
- `selection_manifest.json`: seed selection, protocol, and verification provenance.

The `selection_timing` column distinguishes the two reporting subsets.
CSV precision is retained; the table above is rounded for display only.

## Protocol and verification

Data seed=0, ETF seed=0, evaluation seed=100200. Training uses 10% known-class
labels, a 20% known-only validation holdout, 40 epochs, batch size=128, classifier
temperature=0.2, and PE temperature=0.2. Thresholds reject exactly
floor(0.05 * N) known-validation samples. Unknown test data is not used for
training, checkpoint selection, or threshold calibration.

Within each dataset, all reported seeds share the same SSL encoder
initialization, data split, ETF direction, and evaluation randomness.
SD therefore describes second-stage training variation conditional on this
shared setup and, for HF-Digital, the selected subset.

H4 is computed for each perturbation from KAR-clean, KAR-deg, URR-clean, and
URR-deg, then averaged over all six perturbations. It is not the harmonic mean
of already averaged rates.

Each row was independently recomputed from saved per-sample scores before
publication and checked against the source summary. Scores, source training logs,
and HF-Digital weights are not included in this publication. `verify_release.py`
checks released weights, CSV consistency, per-perturbation H4 aggregation, and
sample SD. Without the original scores it cannot independently recompute AUROC
or Unknown F1; provenance hashes do not replace those scores.

The existing PDFs/PNGs in `figures/` were not regenerated for this seed subset.

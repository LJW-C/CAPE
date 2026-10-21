# CAPE: three training seeds

All metrics below are percentages. SD is the sample standard deviation
across seeds 1, 2, and 3 (ddof=1), recomputed from the archived raw scores.

| Dataset | Seed | Known Acc. | AUROC | Unknown F1 | H-score | KAR-deg | URR-deg | Avg. H4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Panoradio HF | 1 | 89.70 | 82.16 | 45.90 | 48.29 | 91.13 | 35.72 | 49.98 |
| Panoradio HF | 2 | 89.98 | 83.39 | 44.54 | 46.95 | 91.05 | 36.18 | 49.36 |
| Panoradio HF | 3 | 89.58 | 82.11 | 41.98 | 44.48 | 91.09 | 33.64 | 46.81 |
| Panoradio HF | Mean +/- SD | 89.76 +/- 0.20 | 82.56 +/- 0.72 | 44.14 +/- 1.99 | 46.57 +/- 1.93 | 91.09 +/- 0.04 | 35.18 +/- 1.35 | 48.71 +/- 1.68 |
| HF-Digital | 1 | 66.74 | 80.93 | 33.77 | 33.29 | 91.56 | 24.84 | 37.32 |
| HF-Digital | 2 | 66.39 | 77.28 | 24.13 | 24.39 | 92.20 | 16.99 | 27.03 |
| HF-Digital | 3 | 66.59 | 81.12 | 37.89 | 36.88 | 92.53 | 26.92 | 40.80 |
| HF-Digital | Mean +/- SD | 66.57 +/- 0.18 | 79.78 +/- 2.16 | 31.93 +/- 7.06 | 31.52 +/- 6.43 | 92.10 +/- 0.49 | 22.91 +/- 5.24 | 35.05 +/- 7.16 |

The machine-readable CSVs also contain clean known/unknown acceptance or
rejection rates and closed-set known accuracy. Clean and degraded raw score
files are included under `raw/` for independent verification.

The reported variability is conditional on one shared known-only encoder
initializer per dataset, a fixed data split, fixed ETF orientation, and fixed
evaluation random numbers. It measures second-stage training randomness.

HF-Digital's H4 SD is 7.16 in the original seed results. A previously written
value of 2.16 is not supported by those three observations and is not used here.

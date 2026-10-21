# Data Files

Large original datasets are not copied into this code release. The data loader
and the fixed fitting/validation/test sample IDs are included.

## Panoradio HF

Put these files under `data/panoradio/`, or use `--data-root`:

- `dataset_panoradio_hf.npy`: complex `[N, 2048]` signals.
- `dataset_panoradio_hf_tags.csv`: one row per signal, including `mode` and `snr`.

Class IDs are assigned by sorting the unique `mode` strings. Known IDs are
0--11 and unknown IDs are 12--17. Do not reorder the data or metadata rows.

## HF-Digital

Put these files under `data/hfdigital/`, or use `--data-root`:

- `X.npy`: `[N, 2, 2048]`, `[N, 2048, 2]`, or complex `[N, 2048]`.
- `y.npy`: integer class IDs, one label per signal.
- `snr_db.npy`: measured/supplied SNR metadata, one value per signal.

Known IDs are 0--10 and unknown IDs are 11--16. SNR is required for the
training sampler; it is not an input to prediction or the rejection score.

The stored sample IDs refer to the original dataset order. A different data
release, ordering, or cropping pipeline cannot reproduce the archived results
merely by reusing those IDs. Both datasets use the original sample-level split;
this does not establish source-recording-disjoint or fading-realization-disjoint
evaluation.

# B2 Hard-case Oversampling Specification

## Selected primary rule

`small_low_2x`: within each training partition only, an image is listed one additional time when it contains at least one GT object satisfying both:

- `bbox_min_side_px <= 8`
- `absolute_median_difference <= 4`

The definitions are pre-existing diagnostic definitions from Stage 1/2. The rule is image-level duplication, not object cropping or weighting. Original images and labels are unchanged.

## Why this rule

The small-plus-low-contrast interaction is the strongest pre-specified B0 failure concentration (101 objects, 10 FNs, recall 0.90099). Duplicating only this intersection is simpler and more targeted than duplicating every image containing a small object. It avoids a 3x/4x sweep and limits the optimizer-budget shift.

## Leakage and integrity controls

- Eligibility is computed from `outputs/tables/01b_object_xray_features.csv` only after the frozen held-out fold is identified.
- Only stems assigned to the training partition can receive an extra occurrence.
- Validation paths are copied once and are never oversampled.
- Fold assignment, image pixels, and TXT labels are not modified.
- `outputs/tables/08_hardcase_oversampling_manifest_fold<fold>.csv` records every train/validation stem, eligibility, and occurrence count.
- Runtime lists are generated under `outputs/cache/b2_final_splits/`; they are execution cache, not a new dataset split.

## Comparison limitation

Oversampling increases train entries and optimizer updates per epoch. Therefore it tests a practical exposure strategy, not a perfectly step-matched causal ablation. Epochs and batch remain fixed for operational comparability, and the changed training budget must be reported.

## Fold1 deterministic preflight

- Frozen train images: 376
- Eligible training images duplicated once: 63
- Effective train-list entries: 439
- Frozen validation images: 124
- Validation extra occurrences: 0
- Unique source stems represented in the manifest: 500

The generated audit table is `outputs/tables/08_hardcase_oversampling_manifest_fold1.csv`.

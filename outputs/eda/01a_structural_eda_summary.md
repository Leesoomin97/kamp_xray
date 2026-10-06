# Stage 1A — Pre-model Structural EDA Summary

## 1. Stage 1 sample scope

- **VERIFIED** — allowlist integrity passed: 500 eligible rows, unique stems/raw paths/TXT paths, and no missing files.
- Only canonical raw BMP and primary official TXT were used. No excluded or held-out data was accessed.

## 2. Manufacturing-data unit interpretation

- **VERIFIED** — one logical observation is one X-ray BMP frame with its same-stem TXT annotation.
- **VERIFIED** — paths expose machine, SN, folder date; filenames expose prefix, date, time, and an opaque parenthetical value.
- **UNRESOLVED** — product/lot identity and parenthetical-token meaning are unavailable. A frame must not be claimed to equal a unique physical product without documentation.
- Temporal coverage: 2020-06-22T20:30:53 to 2020-09-22T16:33:35; folder-date vs filename-date conflicts: 6.

## 3. Image structure

- **VERIFIED** — resolutions: 316x332=209, 352x332=124, 576x444=167. All are uncompressed 8-bit indexed BMP with one shared palette and 12 colored palette entries.
- **VERIFIED** — basic intensity statistics use palette-decoded BT.601 luma without artifact removal.
- **EXPLORATORY** — machine-level mean-luma averages span 183.672 to 193.936; descriptive differences may reflect acquisition or sample-mix confounding.

## 4. Label/object structure

- **VERIFIED** — total objects: 1147; zero-object images: 0; one-object: 176; multi-object: 324; structural label-review rows: 0.
- **VERIFIED** — class distribution: class 0=1147.
- **VERIFIED** — bbox minimum-side quantiles (px): q01=6.000, q05=7.000, q50=10.000, q95=14.000, q99=15.000.
- **VERIFIED** — bbox area-ratio quantiles: q01=0.000344094, q50=0.000762544, q99=0.00173479.
- Bbox aspect ratio is a shape proxy; image-edge distance is not product-edge distance.

## 5. Dataset imbalance

- **VERIFIED** — machine counts: 1호기(2020.09.22)=156, 2호기(2020.09.22)=177, 3호기(2020.09.22)=167.
- **VERIFIED** — 24 dates are represented; largest date is 20200623 with 108 samples.
- **EXPLORATORY** — bbox size, position, aspect proxy, edge distance, and object-count concentration are documented with continuous values and quantile bins; bins are not operational thresholds.

## 6. Machine/date/sequence structure

- **VERIFIED** — 3 machines, 3 SN identifiers, 24 filename dates.
- **VERIFIED** — adjacent within-group gaps: 463; <=2s 7, <=4s 280, <=6s 353, <=10s 374, <=60s 386.
- **VERIFIED** — most frequent exact gaps: 4s=175, 3s=98, 5s=67, 9s=9, 8s=6, 6s=6, 2s=5, 7s=4.
- **STRONGLY SUPPORTED** — seconds-apart runs indicate dependency risk, but do not prove the same physical item.

## 7. Exact/near-duplicate findings

- **VERIFIED** — cross-stem exact duplicate pairs: 0.
- **EXPLORATORY** — dHash threshold Hamming≤8: near pairs=3; 3 near pairs; same-machine 100.0%, same-date 0.0%, <=10 s 0.0%.
- Near pairs were not collapsed; they are split-leakage candidates only.
- **LIMITATION** — dHash includes global palette-decoded structure, so colored markings may influence similarity; they were not treated as object features.

## 8. Structural feature relationships

- **EXPLORATORY** — strongest tested absolute Spearman association: bbox_min_side_px vs image_edge_distance_px, rho=0.435.
- Machine/date group comparisons are descriptive and potentially confounded. Correlation does not imply causation.

## 9. Resize/small-object representation risk

- **EXPLORATORY** — target 320: projected minimum side below 2/4/8/16 px = 0.0%/0.2%/53.9%/100.0%.
- **EXPLORATORY** — target 1024: projected minimum side below 2/4/8/16 px = 0.0%/0.0%/0.0%/1.6%.
- These are diagnostic letterbox projections, not small/medium/large definitions or an input-size selection.

## 10. Validation leakage risks

- Random image split: high risk of separating adjacent/near-similar frames.
- Machine split: useful stress test but only 3 groups and strong confounding.
- Date split: reduces same-day leakage but can introduce date/machine/object-distribution shift.
- Sequence/similarity grouping: best aligned with observed dependency structure, but grouping thresholds need sensitivity tests.

## 11. Candidate validation strategies

1. Sequence/similarity-group-aware grouped validation with 6/10/60-second sensitivity checks.
2. Date-grouped temporal validation as a stress test.
3. Leave-one-machine-out as a secondary equipment-transfer stress test.

No final split was selected or created.

## 12. Chapter 1 findings

- Verified analysis unit, equipment/date/time structure, file and label integrity, duplicate risk, structural imbalance, and resize risk are mapped in `01a_chapter1_evidence.csv`.
- Explicit product identity is unavailable and was not invented.

## 13. Questions deferred to Stage 1B

- Local object/background contrast, normalized contrast, CNR-like metrics, bbox-region heterogeneity, background complexity, product-edge feasibility, and colored-artifact relationships.
- No preprocessing representation was evaluated.

## 14. Remaining unresolved issues

- Physical meaning of the parenthetical filename value.
- Physical product/lot identity and whether seconds-apart frames are repeat views or successive items.
- Origin and semantics of colored palette indices.
- External official held-out test availability.
- Sensitivity of sequence and dHash grouping thresholds before validation design.

# Stage 1B — X-ray Detection-Difficulty EDA Summary

## 1. Analysis scope

- **VERIFIED** — 500 approved canonical BMP/TXT samples and 1147 stable `(stem, object_id)` GT objects were analyzed. Input integrity is unchanged.
- No excluded, legacy-only, derived JPG, XML-as-GT, prediction, weight, external, or held-out data was used.

## 2. Colored-artifact findings

- **VERIFIED** — strong chromatic pixels (channel range≥200) occur in 500/500 images; 0 images contain none.
- **VERIFIED** — used chromatic pixels comprise five palette colors, all with channel range 255; no color was assigned class/detection semantics.
- **VERIFIED** — median image fraction=0.1658%, maximum=0.4118%; 139/1147 GT bboxes contain at least one masked pixel.
- **VERIFIED** — 1124/1147 objects lie within 3px of a strong chromatic component, and thin-rectangle component count equals TXT object count in 493/500 images. This is a critical shortcut/leakage risk even though no color semantics are assigned.
- **VERIFIED** — median chromatic fractions differ descriptively by acquisition group: machine 1/2/3 = 0.3286%/0.3660%/0.1564%; resolution 316×332/352×332/576×444 = 0.1373%/0.3286%/0.1564%. Machine and resolution are confounded.
- **STRONGLY SUPPORTED** — connected components frequently form thin outline-like structures, supporting a conservative strong-chromatic analysis mask. TXT remains the only GT.

## 3. Artifact-mask reliability

- **VERIFIED** — 1147/1147 objects retain ≥30 bbox pixels with ≤35% masked; 0 are flagged as artifact-contaminated for affected local statistics.
- Strategy: exclude strongly chromatic pixels only. Rectangle interiors and full GT boxes are never removed; source BMP files are unchanged.
- **EXPLORATORY** — the mask is suitable for EDA contamination control, not a selected production preprocessing method.
- A detector must not receive the embedded outlines as predictive input. The marked RAW BMP is an analysis reference, not yet an approved modeling representation.

## 4. Local object/background region definition

- 1.5×: reliable=100.0%, median valid pixels=103, median |background median − 2×|=3.
- 2.0×: reliable=100.0%, median valid pixels=222, median |background median − 2×|=0.
- 3.0×: reliable=100.0%, median valid pixels=726, median |background median − 2×|=13.

- **STRONGLY SUPPORTED** — selected 2.0× as the primary ring using predeclared valid-pixel/stability criteria. It excludes the target bbox, other GT boxes, image-outside pixels, and artifact pixels.
- The GT bbox and ring are proxy regions, not pure foreign-object/product masks.

## 5. Intensity/contrast findings

- **VERIFIED** — absolute median contrast q25/q50/q75=4/8/11.5 luma units.
- **VERIFIED** — signed median contrast is negative for 1106/1147, positive for 20/1147, and zero for 21 objects; both polarities are retained.
- **EXPLORATORY** — local contrast is a candidate visibility factor, not evidence of model difficulty.

## 6. CNR-like findings

- **VERIFIED** — robust image-derived CNR-like q25/q50/q75=0.518/0.8244/1.214; near-zero background-variance cases=0.
- This is not a calibrated physical detector CNR and does not measure physical density.

## 7. Bbox-region heterogeneity

- **VERIFIED** — object IQR q25/q50/q75=8/11/14.25.
- **STRONGLY SUPPORTED** — object IQR is retained as the compact robust proxy; entropy/gradient/CV remain diagnostic because they overlap conceptually or are less stable.

## 8. Background complexity

- **VERIFIED** — background gradient mean q25/q50/q75=3.111/4.354/5.305.
- **STRONGLY SUPPORTED** — background gradient mean plus background IQR retain spatial-gradient and intensity-spread concepts without claiming a physical food property.

## 9. Product-boundary feasibility

- **EXPLORATORY** — two simple methods on 36 diverse images yield heuristic-plausible rows=100.0%, median pair IoU=0.973.
- **STRONGLY SUPPORTED** — 12 representative overlays were reviewed and the selected Otsu-dark largest-component rule passed 500/500 approved images (100.0%). Product-edge distance was therefore created as an analysis-only candidate.
- **UNRESOLVED** — no physical product-boundary GT exists, so the inferred distance is not universally validated and must remain separate from image-edge distance.

## 10. Relationships among candidate difficulty variables

- **EXPLORATORY** — strongest non-redundancy difficulty association: bbox_min_side_px vs background_gradient_mean, rho=0.548.
- **VERIFIED** — redundancy audit peaks at object_iqr vs object_robust_range_p90_p10, rho=0.846; this supports retaining compact representative features rather than near-duplicate metrics.
- Machine/resolution summaries are descriptive and confounded by acquisition and object distributions. Correlation does not imply causation or model failure.

## 11. Joint-condition coverage

- small + low_contrast: 101 objects (8.8%); later-analysis coverage=adequate.
- small + near_image_edge: 104 objects (9.1%); later-analysis coverage=adequate.
- small + complex_background: 12 objects (1.0%); later-analysis coverage=limited.
- low_contrast + complex_background: 64 objects (5.6%); later-analysis coverage=adequate.
- low_cnr + high_heterogeneity: 47 objects (4.1%); later-analysis coverage=adequate.
- near_product_edge + low_contrast: 64 objects (5.6%); later-analysis coverage=adequate.

- Quartiles are exploratory coverage bins, not permanent hard/easy thresholds.

## 12. Representation-comparison findings

- RAW: median absolute separation=8, robust CNR-like=0.8244, background-std ratio vs RAW=1.000.
- WINDOW_P01_P99: median absolute separation=8, robust CNR-like=0.6745, background-std ratio vs RAW=1.648.
- CLAHE_8X8_CLIP2_NO_INTERP: median absolute separation=14, robust CNR-like=0.6909, background-std ratio vs RAW=1.985.
- LOCAL_RESIDUAL_ABS_15PX: median absolute separation=1.662, robust CNR-like=0.3125, background-std ratio vs RAW=0.475.
- GRADIENT_MAGNITUDE: median absolute separation=0.7605, robust CNR-like=0.2904, background-std ratio vs RAW=0.261.

- **EXPLORATORY** — separation gains accompanied by background amplification are not automatically improvements. CLAHE is an 8×8 clipped tile implementation without interpolation and is reported with that limitation.
- **EXPLORATORY** — percentile windowing is unfavorable under these proxies (no median-separation gain, lower robust CNR-like value, 1.648× background std). CLAHE-like processing is inconclusive because larger absolute separation accompanies 1.985× background std and lower robust CNR-like separation.
- **EXPLORATORY** — residual and gradient views reduce the selected standalone separation/CNR-like proxies; they are not supported as replacements for RAW, though later models could test them only as separately justified auxiliary channels.
- Representation montages use the same sample in left-to-right order: RAW, window, CLAHE-like, absolute local residual, gradient magnitude; magenta marks analysis-excluded chromatic pixels.
- No production representation was selected.

## 13. Compact candidate feature set for Stage 2

- Size: bbox minimum side/area ratio.
- Position: normalized center and image-edge distance.
- Shape proxy: bbox aspect ratio.
- Visibility: signed/absolute median contrast and robust CNR-like proxy.
- Heterogeneity: bbox-region IQR.
- Background complexity: ring gradient mean and IQR.
- Product-edge distance: included with a sensitivity flag because the simple rule was consistent across the approved scope, but no physical boundary GT exists.

## 14. Mapping to competition factors

- **VERIFIED** — size/position/bbox-shape are annotation-derived proxies.
- **VERIFIED** — physical density is unavailable; only image-derived intensity/contrast/CNR-like proxies are mapped.
- **STRONGLY SUPPORTED** — background contrast is represented by artifact-excluded bbox-versus-ring statistics with explicit proxy limitations.

## 15. Preprocessing hypotheses generated by EDA

- Windowing, clipped local equalization, residual, and gradient representations remain testable hypotheses only where separation gains are not offset by background amplification.
- Artifact exclusion is an EDA measurement safeguard, not a production preprocessing choice.
- Because artifact components nearly reproduce GT positions/counts, artifact-control ablation is required before any fair baseline; training directly on marked RAW pixels is not acceptable.
- Any later comparison must use identical leakage-aware folds and raw input as the reference.

## 16. Remaining unresolved issues

- Residual product-boundary accuracy uncertainty without segmentation GT, despite strong within-dataset consistency.
- Physical meaning of intensity and acquisition settings; density/material/thickness are unavailable.
- Whether strong chromatic marks are present at deployment and how a production pipeline should handle them.
- Whether conservative exclusion/inpainting can remove outline shortcuts without damaging underlying X-ray signal.
- Stability of representation rankings under leakage-aware validation and model outcomes.
- Local rings may include mixed product structures and GT bbox regions include background.

## 17. Recommended next step

- First finalize leakage-aware candidate folds using the Stage 1A.5 6s/10s results and reviewed similarity components. Then run a separately authorized artifact-control ablation to define a fair modeling input before any baseline. Stage 2 should test whether errors actually concentrate under these candidate factors.

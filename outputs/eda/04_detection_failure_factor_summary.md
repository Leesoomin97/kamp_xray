# Baseline Detection Failure Factor Comparison

## Scope

- Primary model: **B0 baseline OOF only** — YOLOv8n, Conservative representation, 640 px, frozen group-aware 4-fold.
- Objects: **1,147 GT objects; TP 1,112; FN 35** at reporting confidence 0.25 and IoU 0.50.
- Localization failures: **16**, retained separately from low-confidence and no-detection failures.
- No training, inference, split generation, threshold tuning, or B2 result was used.
- Bootstrap unit: frozen 10-second/similarity `final_group_component_id`; 2,000 replicates, seed 42.

## Main findings

### Detection miss

The largest exploratory high-risk-versus-low-risk quartile gaps were:

1. **bbox area ratio**: 6.49 percentage points (95% component-bootstrap CI 3.28–10.34); corrected miss RR 8.72 (CI 3.50–49.99).
2. **inferred product-edge distance**: 5.57 pp (CI 2.88–8.52); RR 4.56 (CI 2.22–14.82).
3. **background gradient mean**: 4.53 pp (CI 1.50–7.63); RR 6.20 (CI 1.92–37.61).
4. **bbox minimum side**: 3.83 pp (CI 0.84–7.99); RR 3.88 (CI 1.46–11.90).
5. **robust image-derived CNR-like proxy**: 3.83 pp (CI 0.68–7.30); RR 3.44 (CI 1.27–26.66).

These ratios are conditional on the observed maximum/minimum event-rate quartiles and use a 0.5 continuity correction. Several comparison bins contain only 2–4 FN, so ratio CIs are wide. Size and background patterns are also non-monotonic: the largest-size quartile and some intermediate-complexity bins retain substantial FN counts.

### Localization failure

Localization failure shares part, but not all, of the detection-miss structure:

- **Size**: area-ratio Q1 contained 15/16 localization failures; min-side Q1 localization-failure rate was 3.73%.
- **Visibility**: absolute-contrast Q1 contained 11 failures versus 0 in Q4; robust-CNR Q1 also contained 11.
- **Inferred product edge**: nearest quartile contained 9 failures versus 1–2 in farther quartiles.
- **Background and image position**: some bins differed, but the patterns were weaker or non-monotonic.

Thus miss and localization are related but not interchangeable outcomes. Size, visibility, and inferred product-edge proximity overlap; background/position evidence is less consistent.

## Multivariable exploratory analysis

An L2-regularized logistic model used six preselected representatives, one per factor group. Coefficients are standardized associations, not causal effects. With only 35 FN and 16 localization failures, all multivariable results remain exploratory.

For **miss**, two coefficient intervals excluded zero:

- inferred product-edge distance: OR per SD 0.38; coefficient CI −1.51 to −0.60;
- image-edge distance: OR 1.44; coefficient CI 0.08 to 0.71.

The image-edge direction does not support a simple “near image edge is worse” rule and is likely capturing non-linear layout/acquisition structure. Size, shape, absolute contrast, and background-gradient coefficient intervals crossed zero after adjustment.

For **localization failure**, three intervals excluded zero:

- bbox minimum side: OR per SD 0.23; coefficient CI −2.37 to −0.43;
- absolute median contrast: OR 0.38; coefficient CI −1.67 to −0.25;
- inferred product-edge distance: OR 0.41; coefficient CI −1.55 to −0.51.

Machine/date/resolution categories were not added to the sparse-event regression because they are mutually confounded and would over-expand the model. The component-aware bootstrap partially respects observed sequence dependence but does not remove acquisition confounding.

## Factor synthesis

| Factor | Evidence strength | B0 interpretation |
|---|---|---|
| Size | MODERATE | Small area/min-side is associated particularly with localization failure, but size-only miss evidence is non-monotonic and adjusted min-side support is weak. |
| Position | EXPLORATORY | x/y regions differ, but image-edge effects are non-monotonic and do not support a universal near-edge rule. |
| Shape | INSUFFICIENT | Aspect-ratio gaps and adjusted coefficients are unstable; it is only a bbox-shape proxy. |
| Visibility/contrast | MODERATE | Low CNR-like/absolute contrast is associated most consistently with localization failure; no physical density claim is made. |
| Background/heterogeneity | EXPLORATORY | Some intermediate-gradient/IQR bins concentrate errors, but no monotonic complexity rule survives adjustment. |
| Product-edge proxy | STRONG | Strongest consistent within-dataset association for miss and localization, but the boundary is inferred and lacks segmentation GT. |

## Model hypotheses generated from B0 evidence

1. **Small-object/localization hypothesis** — test scale-preserving or small-object-sensitive localization strategies, while recognizing that size alone is not monotonic and simple global upscaling is not guaranteed to help.
2. **Photometric-robustness hypothesis** — test controlled intensity/contrast robustness because low image-derived CNR-like and absolute contrast are linked especially to localization failure; preserve artifact control and monitor background amplification.
3. **Edge/context hypothesis** — investigate context-aware localization near the inferred product boundary, but validate the boundary proxy before product-edge-aware training or operating rules.

## Limitations and additional EDA

- Product-edge distance is an unsupervised inferred proxy, not GT boundary distance. Expert segmentation annotations are needed before physical product-edge claims.
- FN and localization counts are small; repeatability on another grouped split or independent development corpus is needed for stable ranking.
- Non-monotonic size/background/image-edge patterns warrant spline or other predeclared non-linear sensitivity analysis rather than permanent quartile thresholds.
- Acquisition confounding should be assessed with more events or external/field data; current machine/date/resolution categories cannot establish causes.
- Actual physical density, material, thickness, and 3D properties are unavailable and were not analyzed.

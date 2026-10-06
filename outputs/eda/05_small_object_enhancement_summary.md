# Stage 5 — Small-object observability and crop/enhancement feasibility

## Scope and evidence boundary

- **VERIFIED:** Analysis used the frozen B0 4-fold OOF object table (1,147 GT objects: 1,112 TP and 35 FN), Stage 1 object features, the approved 500-sample corpus, and Conservative artifact-controlled images.
- **VERIFIED:** No training, detector inference, fold change, or original-data modification was performed.
- All contrast, CNR-like, edge, texture, and product-edge quantities are image-derived proxies. Product edge is an inferred boundary, not GT segmentation. No physical density, material, or thickness is inferred.
- Crop results assume deterministic grid crops resized to 640. They establish representation feasibility, not detector performance or an object-proposal solution.

## 1. Original-pixel observability by B0 outcome

| Outcome | n | Min-side median [IQR], px | Area median [IQR], px² | Absolute contrast median | Interpretation |
|---|---:|---:|---:|---:|---|
| TP | 1,112 | 10 [8, 11] | 108 [81, 143] | 8 | Reference |
| All FN | 35 | 9 [8, 13] | 96 [74.5, 182] | 7 | Aggregate FN differences are heterogeneous |
| Low-confidence FN | 16 | 13 [10, 15] | 175.5 [131.5, 252] | 20 | Not primarily a small/low-contrast subgroup |
| Localization failure | 16 | 8 [7, 9] | 78.5 [62, 88.5] | 3 | Small and low-contrast profile is concentrated here |
| No-detection FN | 3 | 10 [7.5, 11.5] | 100 [72.5, 141] | 5 | Too sparse for a stable conclusion |

- **SUPPORTED:** Localization failures are the clearest scale-related outcome. Their min-side median was 2 px below TP (component-bootstrap 95% CI: -2.5 to 0), and area was 29.5 px² lower (95% CI: -44.0 to -11.0).
- **SUPPORTED:** Localization failures also showed lower visibility: absolute median contrast was 5 lower than TP (95% CI: -5.25 to -1).
- **EXPLORATORY:** All FN combined were only 1 px smaller in median min-side than TP; the 95% CI crossed zero (-2 to 3). Size is therefore not a uniform explanation for every FN type.
- **UNRESOLVED:** The three no-detection FNs are too few to define a distinct observability profile.

The pre-specified Stage 1B interaction was stronger than either factor alone: small+low-contrast had 91 TP / 10 FN among 101 objects (Recall 0.9010), versus small-only 0.9742, low-contrast-only 0.9903, and neither 0.9721. These are associations, not causal effects.

## 2. Crop-to-640 scale simulation

| Representation | Overall median min-side | FN median | Small+low-contrast median | Overall ≥24 px | FN ≥24 px | Small+low-contrast ≥24 px |
|---|---:|---:|---:|---:|---:|---:|
| Full image → 640 | 15.42 | 12.22 | 13.49 | — | — | — |
| Crop 128 → 640 | 50.0 | 45.0 | 40.0 | 100.0% | 100.0% | 100.0% |
| Crop 192 → 640 | 33.33 | 30.0 | 26.67 | 90.3% | 77.1% | 57.4% |
| Crop 256 → 640 | 25.0 | 22.5 | 20.0 | 50.6% | 48.6% | 0.0% |
| Crop 320 → 640 | 20.0 | 18.0 | 16.0 | 19.9% | 31.4% | 0.0% |

- **VERIFIED:** Every simulated crop size represented all objects at ≥8 px. The 128 and 192 crops represented every object at ≥16 px.
- **SUPPORTED:** 192 px offers a meaningful scale gain without the 128 px configuration's highest patch count. This remains a pre-model design candidate, not a selected production strategy.

## 3. Patch-boundary and computation trade-off

| Patch / nominal overlap | Fully contained | Boundary crossing | Duplicate coverage | Mean patches/image |
|---|---:|---:|---:|---:|
| 128 / 0% | 95.38% | 4.62% | 11.25% | 12.67 |
| 128 / 25% | 100% | 0% | 42.28% | 19.00 |
| 128 / 50% | 100% | 0% | 99.13% | 30.59 |
| 192 / 0% | 95.12% | 4.88% | 53.10% | 5.67 |
| 192 / 25% | 100% | 0% | 80.38% | 7.17 |
| 192 / 50% | 100% | 0% | 99.91% | 12.67 |
| 256 / 0% | 100% | 0% | 98.69% | 4.67 |
| 320 / 0% | 100% | 0% | 99.65% | 3.16 |

The last tile is anchored to the image edge, so nominal 0% grids can overlap near image boundaries and duplicate coverage is not equivalent to the stated nominal overlap. Padding is assumed when a patch exceeds an image dimension. **SUPPORTED:** 192/25% is the most plausible first scale experiment among the requested candidates: no simulated GT boundary cuts, about 7.17 patches/image, and stronger scale gain than 256/320. A deployable pipeline would still require patch-level prediction merging and duplicate suppression.

## 4. Enhancement feasibility and signal preservation

| Method | Overall median Δ absolute contrast | Overall median Δ robust CNR-like | Background gradient ratio | Median bbox intensity correlation | Median edge displacement |
|---|---:|---:|---:|---:|---:|
| CLAHE mild | +5.0 | -0.045 | 1.748× | 0.976 | 0.166 px |
| Median 3×3 | 0.0 | 0.000 | 0.931× | 0.808 | 0.466 px |
| Bilateral mild | 0.0 | +0.006 | 0.917× | 0.972 | 0.171 px |
| Unsharp mild | 0.0 | 0.000 | 1.020× | 0.999 | 0.035 px |
| Gaussian reference | 0.0 | +0.031 | 0.961× | 0.986 | 0.084 px |

- **VERIFIED:** Mild CLAHE increased absolute contrast in 96.4% of objects but also amplified background gradients to a median 1.748× and did not improve median robust CNR-like proxy. It is not automatically preferable.
- **SUPPORTED:** Mild bilateral filtering was the most balanced edge-preserving candidate: small overall CNR-like gain, background gradient reduction, high bbox intensity correlation, and low edge displacement. Its median contrast gain was zero, so any detector benefit remains uncertain.
- **SUPPORTED:** Median filtering changed local structure more strongly (bbox correlation 0.808 and edge shift 0.466 px) and is not a leading small-object candidate.
- **SUPPORTED:** Mild unsharp preserved rank/structure well but supplied little measured separation benefit.
- **CONFLICTING EVIDENCE:** Gaussian reference produced some feature-level CNR gains, but the completed B6 detector experiment was unfavorable. It is retained only as a reference, not promoted for GPU testing.

Artifact-trace gradient ratios followed the same broad direction as background gradients: CLAHE amplified them (median 1.740× overall), whereas bilateral was close to or below identity (0.968× overall). This is a repair-trace proxy, not proof of residual shortcut use.

## 5. Small + low-contrast subgroup

For the 101 pre-specified small+low-contrast objects:

- Identity median absolute contrast was 3.0 and robust CNR-like proxy was 0.450.
- CLAHE raised median contrast by 2.0 and improved contrast in 84.2%, but median robust CNR-like changed by -0.045 and background gradient rose to 1.759×.
- Bilateral filtering produced median CNR-like change +0.067, background gradient 0.905×, bbox intensity correlation 0.978, and 0.123 px median edge displacement; contrast improved in 46.5%.
- A 192 crop represented 57.4% at ≥24 px. Joint feature-level coverage of 192-crop ≥24 px plus contrast improvement was 46.5% for CLAHE and 26.7% for bilateral.
- These joint percentages are feasibility indicators only. They do not estimate future Recall or establish that enhancement will improve a detector.

## 6. Hypothesis comparison and next experiment candidates

| Hypothesis | Evidence | Main conflict/risk | GPU experiment status |
|---|---|---|---|
| SCALE | Localization failures are smaller; 192/25% removes simulated boundary cuts with moderate patch count | Not all FN are small; patch proposal/merge and compute overhead remain | **JUSTIFIED** as a single-axis patch experiment |
| VISIBILITY | Localization failures have lower contrast; bilateral offers conservative CNR/background trade-off | CLAHE amplifies background and low-confidence FNs are not low contrast | **JUSTIFIED** as one mild bilateral ablation; CLAHE secondary |
| SCALE + VISIBILITY | Small+low-contrast is the weakest pre-specified interaction | Highest complexity; feature gains are not detector gains | **CONDITIONAL**, only after separate scale and visibility tests |

No final model or preprocessing method is selected. The most defensible GPU sequence is: (1) 192 px patches with 25% overlap and identity intensity processing; (2) a separate full-image mild bilateral ablation; (3) only if both show relevant OOF benefit, a combined experiment. Frozen folds, B0 evaluation rules, and artifact-controlled Conservative inputs must remain fixed.

## Evidence files

- `outputs/tables/05_object_observability.csv`
- `outputs/tables/05_crop_scale_simulation.csv`
- `outputs/tables/05_patch_boundary_tradeoff.csv`
- `outputs/tables/05_enhancement_comparison.csv`
- `outputs/tables/05_signal_preservation.csv`
- `outputs/tables/05_hypothesis_comparison.csv`
- `outputs/figures/stage5_feasibility/`

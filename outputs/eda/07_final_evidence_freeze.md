# Stage 6 Final Freeze and Competition Evidence Package

Freeze date: 2026-10-06  
Scope: existing Stage 0-6 development evidence only. No new training or inference was performed for this freeze.

## 1. Problem definition

The task is single-class foreign-object detection in X-ray images plus analysis of conditions associated with missed detection. Official TXT annotations are the ground truth. Colored rectangles embedded in the BMP pixels are acquisition/annotation artifacts, not foreign-object features or labels.

The approved development corpus contains 500 logical images and 1,147 GT objects. All 500 images are GT-positive, so the project can measure detection behavior on positive images but cannot estimate specificity, true-negative PASS behavior, or production reinspection workload.

## 2. Data integrity and leakage control

- Stage 0 separated primary labeled data from derivative, legacy, practice, and result artifacts. `test1/yolov3` is a 15-sample OpenLabeling/YOLOv3 practice workspace (train 12 / validation 3), not an official held-out set and not part of Stage 1-6 development.
- Stage 1 confirmed 500 eligible primary-GT samples and 1,147 objects. The corpus spans three machines, three resolutions, and 24 dates.
- Chromatic marks occur in 500/500 images and have strong spatial association with GT boxes. The deterministic Conservative repair was therefore frozen as the primary artifact-controlled representation.
- The development folds were frozen before model evaluation: 10-second temporal components plus three manually approved similarity links, assigned to four group-aware folds of 124/125/126/125 images.
- No external official held-out dataset was identified or used. The same frozen four folds were used for development comparison, so all reported 4-fold results are development OOF evidence, not untouched final-test performance.

Evidence: `outputs/eda/00_03_eda_master_summary_verified.md`, `outputs/eda/01c_validation_design_summary.md`, `outputs/eda/01d_artifact_control_summary.md`.

## 3. Baseline and model comparison

### Stage 2: four-fold OOF selection evidence

| Experiment | Single changed axis | TP / FP / FN | Precision | Recall | F1 | AP50 | mAP50-95 | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 | Reference | 1112 / 83 / 35 | 0.93054 | 0.96949 | 0.94962 | 0.95058 | 0.37403 | Stage 6 reference |
| B1 | Mosaic off | 1114 / 69 / 33 | 0.94167 | 0.97123 | 0.95622 | 0.95107 | 0.37884 | Below B2 |
| B2 | Training-only `hsv_v=0.05` | 1121 / 51 / 26 | 0.95648 | 0.97733 | **0.96680** | **0.96039** | **0.38077** | **Selected detector** |
| B3 | B1 + B2 | 1105 / 66 / 42 | 0.94364 | 0.96338 | 0.95341 | 0.94605 | 0.37079 | Rejected |
| B4 | Local-interpolation representation | 1100 / 80 / 47 | 0.93220 | 0.95902 | 0.94542 | 0.93501 | 0.35937 | Rejected |
| B5 | YOLOv8s | 1063 / 204 / 84 | 0.83899 | 0.92677 | 0.88070 | 0.89412 | 0.31318 | Rejected |
| B6 | Weak Gaussian blur | 1106 / 91 / 41 | 0.92398 | 0.96425 | 0.94369 | 0.94997 | 0.37388 | Rejected |

These rows are directly traceable to the seven OOF summary JSON files listed in `outputs/tables/07_final_model_comparison.csv`.

### Stage 6 experiment lineage

| Step | Why / changed factor | Key result | Freeze decision |
|---|---|---|---|
| B0 | Fixed full-image reference | Fold1 281 TP / 15 FP / 9 FN, F1 0.9590 | Keep as Stage 6 global primary/reference |
| P192 | Increase model-space object size with 192 px patches | Rescued 5/9 B0 FNs but lost 20 B0 TPs; F1 0.9094 | Reject as replacement; diagnostic specialist only |
| P256 | Test whether less magnification preserves context | Rescued 2/9 and lost 7; F1 0.9293 | Reject |
| M2 P2 | Add a high-resolution P2 head without cropping | 277/20/13, F1 0.9438; small recall worsened | Reject |
| M3 bilateral | Test mild edge-preserving smoothing | Same TP/FP/FN/F1 as B0 but two rescues and two losses; AP lower | Reject; no global benefit |
| M4 CLAHE | Test local contrast enhancement | 276/29/14, F1 0.9277 | Reject |
| M5 bilateral to CLAHE | Test preprocessing order | 275/14/15, F1 0.9499; one unique rescue | Reject |
| M6 CLAHE to bilateral | Reverse preprocessing order | 277/21/13, F1 0.9422; localization failures 2, unique rescues 0 | Reject |
| Selective routing | Route candidates/images using observable prediction-time signals | Rescue coverage required high burden; no practical rule | Reject |
| Prediction fusion | Union-NMS of complementary B0/M3/P192 predictions | Recall increased in places but FP surged and F1 fell | Reject |
| Candidate-guided local re-detection | Local second pass around B0 candidates | Best fixed-scale result tied B0; observable routing rescued at most 1/9 with F1 loss | Reject |
| Policy invariant audit | Verify local matching is independent of policy | 5,385 candidate-scale rows; zero selection/transformation/metric mismatches | Audit passed; negative conclusion retained |

Stage 6 uses only Fold1 and is exploratory screening. It cannot replace the four-fold Stage 2 selection evidence.

## 4. Final model selection

**Final selected detector for the competition report: B2** — YOLOv8n, Conservative artifact-controlled images, 640 input, 30 epochs, batch 8, seed 42, with mild training-only `hsv_v=0.05` visibility augmentation. It is the strongest same-fold four-fold OOF candidate: F1 0.96680 with 26 FNs and 51 FPs.

**Stage 6 reference/global primary: B0.** Stage 6 deliberately froze B0 as a controlled reference so that patch scale, P2 architecture, preprocessing, routing, fusion, and local redetection could each be interpreted relative to one stable model. Calling B0 the Stage 6 global primary does not supersede B2's four-fold development selection.

P192 is not a final model. It demonstrates that geometric magnification can rescue some B0 misses, but its TP loss, FP increase, context trade-off, and inference burden make it unsuitable as a full replacement. Selective routing, fusion, and candidate-local redetection are also not frozen as operational components.

## 5. Why small and low-visibility objects fail

The following statements separate observation, derived association, and hypothesis.

### Observed facts

- B0 OOF has 35 FNs. The pre-specified small-plus-low-contrast group has 101 objects, 10 FNs, and recall 0.90099—the clearest interaction-level concentration.
- Low-CNR-like plus high heterogeneity has 47 objects, 3 FNs, and recall 0.93617.
- Small plus near-image-edge has 104 objects, 5 FNs, and recall 0.95192.
- Small plus complex background has only 12 objects and one FN; it is too sparse for a strong claim.

### Derived associations

- Bbox area ratio: 6.49 percentage-point FN-rate gap; corrected RR 8.72 (95% CI 3.50-49.99).
- Inferred product-edge distance: 5.57 percentage-point gap; RR 4.56 (2.22-14.82).
- Background gradient: 4.53 percentage-point gap; RR 6.20 (1.92-37.61).
- Bbox minimum side: 3.83 percentage-point gap; RR 3.88 (1.46-11.90).
- Robust image-derived CNR-like proxy: 3.83 percentage-point gap; RR 3.44 (1.27-26.66).

Only 35 misses support these estimates, so several confidence intervals are wide. Size and image-derived visibility have moderate evidence; position and background remain exploratory; bbox-shape evidence is insufficient. Product-edge distance uses an inferred boundary, not GT product segmentation. None of the intensity, contrast, or CNR-like variables measures physical density, material, thickness, or 3D properties.

### Hypotheses

Small pixel support and weak object/background separation plausibly make detection and localization less stable. These are model-development hypotheses, not causal or physical claims. Machine, date, and resolution patterns are also confounded and cannot be interpreted as equipment causes.

## 6. Why magnification helped only partially

The original bbox minimum-side median is 10 px, with the 5th-95th percentile spanning 7-14 px. Projected size diagnostics show:

- 320 input: 53.9% of objects project below 8 px minimum side.
- 640 input: 0.2% project below 8 px and 53.9% below 16 px.
- 1024 input: 0% project below 8 px and 1.6% below 16 px.

These are geometric projections. Upscaling cannot create sensor detail absent from the source image.

A 192-to-640 crop projection raises the median minimum side to 33.33 px overall, 30 px for B0 FNs, and 26.67 px for the small-plus-low-contrast group. At least 24 px is reached by 90.3% overall, 77.1% of FNs, and 57.4% of small-plus-low-contrast objects. This establishes geometric observability, not detector improvement.

The controlled Fold1 patch experiments confirmed the trade-off: P192 rescued five B0 FNs but lost 20 B0 TPs, while P256 rescued two and lost seven. Magnification therefore exposes complementary detections but removes full-image context and changes the false-positive/localization environment. The negative replacement result is evidence against adopting patches as the primary detector.

## 7. Why routing, fusion, and local redetection were rejected

### Routing

Prediction-time confidence, predicted size, predicted-box visibility, and instability were tested without using GT as a deployable signal. Useful rescue coverage required routing too many images/ROIs, so no practical simple rule was obtained. GT-derived size, contrast, CNR-like, background, and product-edge variables remain diagnostic only.

### Fusion

Complementary object outcomes did not translate to a useful deployable union-NMS result at the reference NMS IoU 0.7:

- B0 + M3: 282/20/8, F1 0.9527.
- B0 + P192: 284/85/6, F1 0.8619.
- B0 + M3 + P192: 284/91/6, F1 0.8541.

The oracle B0-plus-P192 capacity is 286 TP / 4 FN (recall 0.98621), but it assumes knowledge of which B0 FNs P192 rescues. It is not deployable performance. Simple fusion retained only three of the five oracle rescues and introduced many FPs.

### Candidate-guided local redetection

The best fixed-scale candidate-guided result tied B0 (281/15/9, F1 0.9590). It required 1,077 candidate crops over 124 images—8.69 local calls per image—versus P192's 7.21 fixed patches per image. Observable routing rescued at most one of nine B0 FNs and reduced F1. The policy audit found zero invariant or metric mismatches, so the rejection is not explained by the earlier analysis-code ambiguity.

## 8. Final safety-oriented inspection strategy

The frozen report concept is:

```text
X-ray input
  -> Conservative artifact control
  -> B2 primary detector
  -> confidence / localization / observable risk diagnostics
  -> DETECT | REINSPECT | PASS (conceptual)
  -> REINSPECT: secondary inspection or human review
```

- **DETECT:** return the detector-positive localization and confidence. Confidence 0.25 is the reporting threshold, not a final operational threshold. Confidence 0.40 is an EDA candidate only.
- **REINSPECT:** flag uncertainty or risk that can be observed at inference time, such as low/unstable confidence, localization instability, unusually small predicted boxes, or weak predicted-box local visibility. Inferred product-edge and abnormal background-gradient cues require further validation before deployment.
- **PASS:** remains a concept only. Because all 500 development images contain GT objects, no normal-negative specificity, PASS safety, or actual reinspection workload was measured.

The safety message is: **do not assume one AI model perfectly resolves every condition; transfer high-miss-risk or uncertain cases to reinspection.** This supports operator prioritization and potential workload reduction, but no numerical labor/time saving is claimed.

## 9. Limitations

1. All 500 development images are GT-positive; true-normal specificity, PASS rate, and real reinspection burden are unknown.
2. Stage 6 results use Fold1 only and are exploratory screening.
3. The same frozen four folds were used throughout development and selection; there is no untouched external held-out final evaluation.
4. Product-edge distance is derived from an inferred image boundary, not GT segmentation or a physically validated boundary.
5. Physical density, material, thickness, attenuation coefficient, and 3D geometry were not inferred.
6. Shape evidence is insufficient; position and background evidence remain exploratory.
7. Reporting confidence and EDA threshold candidates have not been operationally validated.
8. Real wall-clock deployment burden was not measured consistently for every method.
9. Machine, date, and resolution are confounded; subgroup differences do not establish equipment causality.

## 10. Competition rubric mapping

| Rubric | Evidence package | Report message |
|---|---|---|
| 데이터 이해/진단 (15) | Stage 0-1 role audit, structural EDA, artifact audit, grouped validation | Data provenance, duplication, temporal leakage, and colored shortcut risk were controlled before modeling. |
| AI 예측모델 개발 (40) | `07_final_model_comparison.csv`, B0-B6 OOF, B2 selection, Stage 6 negative ablations | Same-condition four-fold comparison selected B2 on precision, recall, F1, AP50, and mAP50-95 evidence. |
| 영향요인/오류분석 (15) | `07_failure_condition_evidence.csv`, Stage 4 effects, Stage 5 observability | Small area and weak image-derived visibility are the main supported miss associations, with uncertainty reported. |
| 현장 활용방안 (10) | `07_safety_policy_evidence.csv`, PASS/DETECT/REINSPECT workflow | Uncertain/high-risk cases are routed to reinspection instead of claiming full automation. |
| 창의성·차별성 (10) | Artifact control, leakage-aware folds, oracle/deployable distinction, negative-result preservation | The workflow connects shortcut control, explainable failure factors, and safety decisions while rejecting non-deployable gains. |
| 코드/재현성 (10) | Frozen folds, requirements, run manifests, deterministic scripts, machine-readable evidence | Every main decision is traceable to fixed data roles, code, metadata, and saved results. |

## Final freeze decisions

1. **Final selected detector:** B2.
2. **Stage 6 reference/global primary:** B0.
3. **Why the roles differ:** B2 wins the complete four-fold development comparison; B0 was intentionally frozen as the controlled Stage 6 ablation reference.
4. **P192 final model:** No.
5. **Routing/fusion/candidate-local operational adoption:** No; retained as exploratory negative evidence.
6. **Final safety strategy:** B2 detection plus inference-observable uncertainty diagnostics and a separately validated REINSPECT path; PASS remains unvalidated.
7. **Further model training:** Stop for the current competition evidence freeze. The next step is report assembly and, if later available, untouched external/field validation—not another development-model search.

## Evidence files created by this freeze

- `outputs/tables/07_final_model_comparison.csv`
- `outputs/tables/07_stage6_ablation_summary.csv`
- `outputs/tables/07_failure_condition_evidence.csv`
- `outputs/tables/07_safety_policy_evidence.csv`
- `outputs/tables/07_competition_rubric_mapping.csv`
- `outputs/eda/07_final_evidence_freeze.md`

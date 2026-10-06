# Stage 6 Fold1 deployable prediction-level fusion analysis

## Scope

This is a one-fold exploratory analysis. Fusion used prediction boxes and confidence only; GT was used after fusion solely for evaluation. No operational threshold was selected. P192 input is its existing source-coordinate output after the pre-existing 0.5 reference cross-patch NMS.

## Input integrity

B0, M3, and P192 each contained the same 124 Fold1 images and 290 identical TXT-GT boxes. Recomputed inputs reproduced B0 281/15/9, M3 281/15/9, and P192 266/29/24.

## Fusion method

Predictions at the existing 0.001 confidence floor were unioned in source-image coordinates. Deterministic class-agnostic greedy NMS retained the highest-confidence box and preserved its original confidence; no model bonus or GT information was used. The primary reference NMS IoU was 0.7, with 0.5/0.6/0.7 sensitivity. Weighted box fusion was not added because defining score aggregation would introduce another unvalidated fusion rule.

## Primary 0.7-NMS results

| Combination | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50-95 | Rescue | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F1_B0 | 281 | 15 | 9 | 0.9493 | 0.9690 | 0.9590 | 0.9625 | 0.3958 | 0 | 0 |
| M3_ONLY | 281 | 15 | 9 | 0.9493 | 0.9690 | 0.9590 | 0.9498 | 0.3650 | 2 | 2 |
| P192_ONLY | 266 | 29 | 24 | 0.9017 | 0.9172 | 0.9094 | 0.8913 | 0.3095 | 5 | 20 |
| F2_B0_M3 | 282 | 20 | 8 | 0.9338 | 0.9724 | 0.9527 | 0.9587 | 0.3821 | 2 | 1 |
| F3_B0_P192 | 284 | 85 | 6 | 0.7696 | 0.9793 | 0.8619 | 0.9678 | 0.4013 | 3 | 0 |
| F4_B0_M3_P192 | 284 | 91 | 6 | 0.7573 | 0.9793 | 0.8541 | 0.9643 | 0.3883 | 3 | 0 |

## Paired rescue and FP trade-off

- **F2_B0_M3:** rescued 2/9 B0 FN, lost 1 B0 TP; removed 1 geometrically matched B0 FP and introduced 6 unmatched fusion FP.
- **F3_B0_P192:** rescued 3/9 B0 FN, lost 0 B0 TP; removed 0 geometrically matched B0 FP and introduced 70 unmatched fusion FP.
- **F4_B0_M3_P192:** rescued 3/9 B0 FN, lost 0 B0 TP; removed 1 geometrically matched B0 FP and introduced 77 unmatched fusion FP.

## Subgroups and error types

- **F1_B0:** small 72/77 (0.9351); low contrast 77/81 (0.9506); small+low 25/29 (0.8621); localization/low-confidence/no-detection=4/5/0.
- **F2_B0_M3:** small 72/77 (0.9351); low contrast 76/81 (0.9383); small+low 25/29 (0.8621); localization/low-confidence/no-detection=5/3/0.
- **F3_B0_P192:** small 74/77 (0.9610); low contrast 78/81 (0.9630); small+low 26/29 (0.8966); localization/low-confidence/no-detection=1/5/0.
- **F4_B0_M3_P192:** small 73/77 (0.9481); low contrast 77/81 (0.9506); small+low 25/29 (0.8621); localization/low-confidence/no-detection=1/4/1.

## Oracle comparison

The prior B0+P192 oracle upper bound is TP 286, FN 4, Recall 0.9862. It assumes perfect GT-aware selection and is not deployable. At reference NMS 0.7, F3 preserves 3/5 oracle rescues (oracle gap 2) and F4 preserves 3/5 (oracle gap 2).
The missing oracle rescues show that simple prediction union/NMS cannot reproduce GT-aware object selection.

## NMS sensitivity

- **F2_B0_M3:** IoU 0.5: TP/FP/FN=281/9/9, F1=0.9690; IoU 0.6: TP/FP/FN=282/12/8, F1=0.9658; IoU 0.7: TP/FP/FN=282/20/8, F1=0.9527
- **F3_B0_P192:** IoU 0.5: TP/FP/FN=282/17/8, F1=0.9576; IoU 0.6: TP/FP/FN=282/32/8, F1=0.9338; IoU 0.7: TP/FP/FN=284/85/6, F1=0.8619
- **F4_B0_M3_P192:** IoU 0.5: TP/FP/FN=282/17/8, F1=0.9576; IoU 0.6: TP/FP/FN=283/42/7, F1=0.9203; IoU 0.7: TP/FP/FN=284/91/6, F1=0.8541
- Corresponding B0-only controls: IoU 0.5: TP/FP/FN=281/9/9, F1=0.9690; IoU 0.6: TP/FP/FN=281/9/9, F1=0.9690; IoU 0.7: TP/FP/FN=281/15/9, F1=0.9590. At IoU 0.5, F2 merely equals the same-threshold B0 control; the apparent FP reduction versus the original 0.7 result is an NMS-threshold effect, not a fusion gain. No grid point improves both FN and F1 over its corresponding B0-only control.

## Inference burden

Fold1 uses 894 retained P192 patches across 124 images: mean 7.21 patch model inputs/image. B0=1.0 full-image call; B0+M3=2 full-image calls; B0+P192≈8.21 total model inputs/image; B0+M3+P192≈9.21. No wall-clock time is inferred.

## Verdict

F2_B0_M3 reduces FN but does not improve F1 versus B0; it is a safety trade-off candidate, not a replacement.
All conclusions are Fold1 exploratory and threshold-sensitive; none defines an operational fusion threshold.

## Next experiment

Stop fusion expansion and retain B0 as primary. The one next experiment is the already pre-registered M6 CLAHE-to-bilateral Fold1 ablation, which tests preprocessing order without tuning this fusion or its thresholds.

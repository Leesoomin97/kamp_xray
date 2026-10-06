# Stage 6 Fold1 object-level strategy winner analysis — M6 update

## Scope

One-fold exploratory diagnostic over the same 290 Fold1 TXT-GT objects. GT-derived features are diagnostic only and are not deployable routing inputs. Product-edge distance remains an inferred-boundary proxy. No causal or physical material/density/thickness interpretation is made.

## 1. Join integrity

All eight strategies contain exactly 290 identical unique `(stem, object_id)` keys; duplicate and missing counts are zero. TP uses confidence 0.25 and IoU >= 0.50.

## 2. Global metrics

| Strategy | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 | 281 | 15 | 9 | 0.9493 | 0.9690 | 0.9590 | 0.9625 | 0.3958 |
| p192 | 266 | 29 | 24 | 0.9017 | 0.9172 | 0.9094 | 0.8913 | 0.3095 |
| p256 | 276 | 28 | 14 | 0.9079 | 0.9517 | 0.9293 | 0.9173 | 0.3829 |
| m2_p2 | 277 | 20 | 13 | 0.9327 | 0.9552 | 0.9438 | 0.9400 | 0.3820 |
| m3_bilateral | 281 | 15 | 9 | 0.9493 | 0.9690 | 0.9590 | 0.9498 | 0.3650 |
| m4_clahe | 276 | 29 | 14 | 0.9049 | 0.9517 | 0.9277 | 0.9280 | 0.3677 |
| m5_bilateral_clahe | 275 | 14 | 15 | 0.9516 | 0.9483 | 0.9499 | 0.9553 | 0.3675 |
| m6_clahe_bilateral | 277 | 21 | 13 | 0.9295 | 0.9552 | 0.9422 | 0.9485 | 0.3743 |

## 3. B0 FN rescue matrix

- `001_20200623_082414(1)#2`: p192=TP, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN, m6_clahe_bilateral=TP; rescue strategies=2
- `001_20200624_083202(6)#0`: p192=FN, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN, m6_clahe_bilateral=FN; rescue strategies=0
- `001_20200727_204958(2)#0`: p192=TP, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN, m6_clahe_bilateral=FN; rescue strategies=1
- `001_20200819_124414(0)#0`: p192=FN, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN, m6_clahe_bilateral=FN; rescue strategies=0
- `001_20200826_123714(5)#0`: p192=TP, p256=TP, m2_p2=TP, m3_bilateral=TP, m4_clahe=TP, m5_bilateral_clahe=FN, m6_clahe_bilateral=TP; rescue strategies=6
- `002_20200623_082311(8)#0`: p192=FN, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=TP, m6_clahe_bilateral=FN; rescue strategies=1
- `002_20200623_082318(5)#2`: p192=TP, p256=FN, m2_p2=TP, m3_bilateral=FN, m4_clahe=TP, m5_bilateral_clahe=FN, m6_clahe_bilateral=FN; rescue strategies=3
- `002_20200624_162540(3)#0`: p192=TP, p256=TP, m2_p2=TP, m3_bilateral=TP, m4_clahe=TP, m5_bilateral_clahe=TP, m6_clahe_bilateral=FN; rescue strategies=6
- `002_20200714_083057(5)#2`: p192=FN, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN, m6_clahe_bilateral=FN; rescue strategies=0

## 4. M6 rescue and role

M6 rescues 2/9 B0 FN: `001_20200623_082414(1)#2`; `001_20200826_123714(5)#0`.
Unique rescue=0; shared rescue=2. Both M6 rescues overlap P192. The seven-alternative rescue union remains 6/9 and all-fail remains 3.
One M6 rescue is small (8 px) and one is not (10 px), so rescue is not concentrated exclusively in small objects.
M6 changes B0's four localization failures to two, but only one original localization failure is rescued; two remain localization failures and one becomes no-detection. This is a mixed error-type redistribution, not repeated localization-specialist evidence.

## 5. Preprocessing order ablation

| Strategy | TP/FP/FN | F1 | Loc/LowConf/NoDet | Small | Low contrast | Small+low | Rescue/Unique/Loss |
|---|---|---:|---|---:|---:|---:|---|
| m3_bilateral | 281/15/9 | 0.9590 | 5/3/1 | 0.9351 | 0.9259 | 0.8621 | 2/0/2 |
| m4_clahe | 276/29/14 | 0.9277 | 6/6/2 | 0.8831 | 0.8889 | 0.7931 | 3/0/8 |
| m5_bilateral_clahe | 275/14/15 | 0.9499 | 3/10/2 | 0.9091 | 0.9136 | 0.7931 | 2/1/8 |
| m6_clahe_bilateral | 277/21/13 | 0.9422 | 2/8/3 | 0.9481 | 0.9383 | 0.8621 | 2/0/6 |

M5 and M6 differ on 18 objects: M6 gains 10 M5-FN objects and loses 8 M5-TP objects. Their B0-FN rescue sets are disjoint. Reversing the order raises TP by 2 and small recall by one object and reduces localization failures 3→2, but increases FP 14→21, FN confidence/no-detection burden, and lowers F1 .9499→.9422.

## 6. M6 feature-bin results

- `small` small_le_8 (n=77): best=m6_clahe_bilateral; M6-B0=+0.0130; B0-FN rescue=1; anecdotal_1_to_2_object_difference.
- `small_low_contrast` both (n=29): best=b0;p192;m3_bilateral;m6_clahe_bilateral; M6-B0=+0.0000; B0-FN rescue=0; no_alternative_gain_over_b0.
- `bbox_min_side_px` Q1 (n=77): best=m6_clahe_bilateral; M6-B0=+0.0130; B0-FN rescue=1; anecdotal_1_to_2_object_difference.
- `bbox_area_ratio` Q1 (n=78): best=m6_clahe_bilateral; M6-B0=+0.0256; B0-FN rescue=2; anecdotal_1_to_2_object_difference.
- `bbox_area_ratio` Q2 (n=71): best=b0;p256;m2_p2;m3_bilateral;m4_clahe;m6_clahe_bilateral; M6-B0=+0.0000; B0-FN rescue=0; no_alternative_gain_over_b0.
- `absolute_median_difference` Q3 (n=60): best=m6_clahe_bilateral; M6-B0=+0.0333; B0-FN rescue=2; anecdotal_1_to_2_object_difference.
- `cnr_like_robust` Q2 (n=73): best=b0;m3_bilateral;m6_clahe_bilateral; M6-B0=+0.0000; B0-FN rescue=0; no_alternative_gain_over_b0.
- `background_gradient_mean` Q3 (n=72): best=b0;m2_p2;m6_clahe_bilateral; M6-B0=+0.0000; B0-FN rescue=1; no_alternative_gain_over_b0.
- `image_edge_distance_norm` Q1 (n=73): best=b0;m6_clahe_bilateral; M6-B0=+0.0000; B0-FN rescue=0; no_alternative_gain_over_b0.
- `product_edge_distance_norm` Q1 (n=73): best=m6_clahe_bilateral; M6-B0=+0.0274; B0-FN rescue=2; anecdotal_1_to_2_object_difference.

M6 small recall is 73/77 versus B0 72/77 because it rescues one small B0 FN and loses no small B0 TP. It has no low-contrast or small+low rescue; low-contrast recall is one object below B0.

## 7. M6 two-dimensional condition results

- `size_x_contrast` [small, low_contrast] (n=29): best=b0;p192;m3_bilateral;m6_clahe_bilateral; M6-B0=+0.0000; rescue=0; no_alternative_gain_over_b0.
- `size_x_contrast` [small, not_low_contrast] (n=48): best=m6_clahe_bilateral; M6-B0=+0.0208; rescue=1; anecdotal_1_to_2_object_difference.
- `size_x_background` [small, low_bg_gradient] (n=71): best=m6_clahe_bilateral; M6-B0=+0.0141; rescue=1; anecdotal_1_to_2_object_difference.
- `size_x_product_edge` [small, far_proxy_edge] (n=40): best=b0;m3_bilateral;m6_clahe_bilateral; M6-B0=+0.0000; rescue=0; no_alternative_gain_over_b0.
- `contrast_x_background` [low_contrast, high_bg_gradient] (n=35): best=b0;p192;m2_p2;m6_clahe_bilateral; M6-B0=+0.0000; rescue=0; no_alternative_gain_over_b0.

## 8. All-fail objects

All eight strategies fail on 3 objects; M6 rescues none of the prior three all-fail cases.
- `001_20200624_083202(6)#0`: min-side=20px, abs contrast=21, background gradient=4.74, inferred product-edge norm=0.0829.
- `001_20200819_124414(0)#0`: min-side=8px, abs contrast=4, background gradient=6.41, inferred product-edge norm=0.0626.
- `002_20200714_083057(5)#2`: min-side=7px, abs contrast=1, background gradient=3.73, inferred product-edge norm=0.078.

## 9. Updated strategy classification

- **b0: A_GLOBAL_PRIMARY** — Best overall stability and the reference against which all alternatives are evaluated.
- **p192: B_EXPLORATORY_CONDITIONAL_SPECIALIST** — Largest B0-FN rescue set (5/9) and one remaining unique rescue after M6, but 20 B0-TP losses.
- **p256: C_NO_EVIDENCE_OF_BENEFIT** — Two shared rescues, no unique rescue, and seven B0-TP losses.
- **m2_p2: C_NO_EVIDENCE_OF_BENEFIT** — Three shared rescues, no unique rescue, and seven B0-TP losses.
- **m3_bilateral: C_NO_EVIDENCE_OF_BENEFIT** — Aggregate counts tie B0 but two rescues are offset by two losses; no unique rescue and lower AP.
- **m4_clahe: C_NO_EVIDENCE_OF_BENEFIT** — Three shared rescues, no unique rescue, and eight B0-TP losses.
- **m5_bilateral_clahe: C_NO_EVIDENCE_OF_BENEFIT** — One unique rescue remains anecdotal; global recall is lower and eight B0-TP objects are lost.
- **m6_clahe_bilateral: C_NO_EVIDENCE_OF_BENEFIT** — Two shared rescues, no unique rescue, six B0-TP losses, and lower global F1; localization/small gains are not repeated specialist evidence.

M6 shows an anecdotal localization/small-object signal but not a detection specialist pattern and no deployable expert evidence. No GT-derived condition here may be used directly as a router.

## 10. Conclusion and next step

B0 remains the global primary. P192 remains the only exploratory conditional specialist, with substantial replacement and compute costs. M6 does not expand the rescue union or reduce all-fail cases and is not promoted.
No further Stage6 training is justified by the completed M1–M6 Fold1 screen. The next single step is to freeze the Stage6 exploratory conclusion and update the competition evidence/report table with B0 as primary and the documented limitations.

# Stage 6 Fold1 object-level strategy winner analysis

## Scope and interpretation

This is a one-fold exploratory diagnostic over the same 290 Fold1 TXT-GT objects. GT-derived size, contrast, background, image-edge, and inferred product-edge features are diagnostic only and are not deployable routing inputs. Product-edge distance is an inferred-boundary proxy. No causal, physical-density, material, thickness, or final-generalization claim is made.

## 1. Join integrity

All seven object tables contained exactly 290 unique and identical `(stem, object_id)` keys, all assigned to Fold1. The Stage1B feature join retained all 290 objects. TP uses reporting confidence 0.25 and IoU >= 0.50.

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

B0 remains the global primary reference. M3 ties B0 at the fixed-threshold confusion counts but has lower AP50 and mAP50-95; equality of aggregate counts does not mean identical object outcomes.

## 3. B0 FN rescue matrix

- `001_20200623_082414(1)#2`: p192=TP, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN; rescue strategies=1
- `001_20200624_083202(6)#0`: p192=FN, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN; rescue strategies=0
- `001_20200727_204958(2)#0`: p192=TP, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN; rescue strategies=1
- `001_20200819_124414(0)#0`: p192=FN, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN; rescue strategies=0
- `001_20200826_123714(5)#0`: p192=TP, p256=TP, m2_p2=TP, m3_bilateral=TP, m4_clahe=TP, m5_bilateral_clahe=FN; rescue strategies=5
- `002_20200623_082311(8)#0`: p192=FN, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=TP; rescue strategies=1
- `002_20200623_082318(5)#2`: p192=TP, p256=FN, m2_p2=TP, m3_bilateral=FN, m4_clahe=TP, m5_bilateral_clahe=FN; rescue strategies=3
- `002_20200624_162540(3)#0`: p192=TP, p256=TP, m2_p2=TP, m3_bilateral=TP, m4_clahe=TP, m5_bilateral_clahe=TP; rescue strategies=6
- `002_20200714_083057(5)#2`: p192=FN, p256=FN, m2_p2=FN, m3_bilateral=FN, m4_clahe=FN, m5_bilateral_clahe=FN; rescue strategies=0

## 4. Rescue overlap and unique rescue

- **p192:** rescued 5/9 B0 FN; unique 2; shared 3.
- **p256:** rescued 2/9 B0 FN; unique 0; shared 2.
- **m2_p2:** rescued 3/9 B0 FN; unique 0; shared 3.
- **m3_bilateral:** rescued 2/9 B0 FN; unique 0; shared 2.
- **m4_clahe:** rescued 3/9 B0 FN; unique 0; shared 3.
- **m5_bilateral_clahe:** rescued 2/9 B0 FN; unique 1; shared 1.

## 5. Feature-bin winners

- `low_contrast` not_low_gt_4 (n=209): best=m3_bilateral; best alternative vs B0=+2 object(s); evidence=anecdotal_1_to_2_object_difference.
- `bbox_min_side_px` Q3 (n=84): best=p256;m2_p2;m3_bilateral; best alternative vs B0=+2 object(s); evidence=anecdotal_1_to_2_object_difference.
- `bbox_area_ratio` Q4 (n=64): best=m5_bilateral_clahe; best alternative vs B0=+2 object(s); evidence=anecdotal_1_to_2_object_difference.
- `absolute_median_difference` Q4 (n=71): best=m5_bilateral_clahe; best alternative vs B0=+2 object(s); evidence=anecdotal_1_to_2_object_difference.
- `image_edge_distance_norm` Q3 (n=74): best=m3_bilateral; best alternative vs B0=+2 object(s); evidence=anecdotal_1_to_2_object_difference.
- `absolute_median_difference` Q3 (n=60): best=m3_bilateral; best alternative vs B0=+1 object(s); evidence=anecdotal_1_to_2_object_difference.
- `cnr_like_robust` Q3 (n=71): best=m3_bilateral; best alternative vs B0=+1 object(s); evidence=anecdotal_1_to_2_object_difference.
- `cnr_like_robust` Q4 (n=73): best=m3_bilateral;m5_bilateral_clahe; best alternative vs B0=+1 object(s); evidence=anecdotal_1_to_2_object_difference.
- `background_gradient_mean` Q2 (n=72): best=p192;m3_bilateral;m4_clahe;m5_bilateral_clahe; best alternative vs B0=+1 object(s); evidence=anecdotal_1_to_2_object_difference.
- `product_edge_distance_norm` Q1 (n=73): best=m4_clahe; best alternative vs B0=+1 object(s); evidence=anecdotal_1_to_2_object_difference.
- `product_edge_distance_norm` Q3 (n=72): best=m2_p2;m3_bilateral;m5_bilateral_clahe; best alternative vs B0=+1 object(s); evidence=anecdotal_1_to_2_object_difference.

## 6. Two-dimensional condition winners

- `size_x_product_edge` [small, near_proxy_edge] (n=37): best=p192; best alternative vs B0=+3 object(s); evidence=limited_3_to_4_object_difference.
- `size_x_contrast` [not_small, not_low_contrast] (n=161): best=m3_bilateral;m4_clahe; best alternative vs B0=+2 object(s); evidence=anecdotal_1_to_2_object_difference.
- `size_x_background` [not_small, low_bg_gradient] (n=74): best=m5_bilateral_clahe; best alternative vs B0=+2 object(s); evidence=anecdotal_1_to_2_object_difference.
- `size_x_product_edge` [not_small, far_proxy_edge] (n=105): best=m5_bilateral_clahe; best alternative vs B0=+2 object(s); evidence=anecdotal_1_to_2_object_difference.
- `contrast_x_background` [not_low_contrast, low_bg_gradient] (n=99): best=m5_bilateral_clahe; best alternative vs B0=+2 object(s); evidence=anecdotal_1_to_2_object_difference.
- `size_x_background` [small, high_bg_gradient] (n=6): best=p192; best alternative vs B0=+1 object(s); evidence=anecdotal_1_to_2_object_difference.
- `contrast_x_background` [not_low_contrast, high_bg_gradient] (n=110): best=m3_bilateral;m4_clahe; best alternative vs B0=+1 object(s); evidence=anecdotal_1_to_2_object_difference.

## 7. All-fail objects

All seven strategies failed on 3 objects.
- `001_20200624_083202(6)#0`: min-side=20px, abs contrast=21, robust CNR-like=1.42, inferred product-edge norm=0.0829.
- `001_20200819_124414(0)#0`: min-side=8px, abs contrast=4, robust CNR-like=0.45, inferred product-edge norm=0.0626.
- `002_20200714_083057(5)#2`: min-side=7px, abs contrast=1, robust CNR-like=0.0519, inferred product-edge norm=0.078.

## 8. Conditional-specialist assessment

- **B0 — global primary:** best overall stability; M3 ties its fixed-threshold counts but not its AP metrics.
- **P192 — exploratory conditional specialist:** rescues 5/9 B0 FN and uniquely rescues two, but loses 20 B0 TP and is not a replacement detector.
- **P256, M2, M3, M4 — no evidence of net or unique conditional benefit:** their rescues are shared and offset by losses; M3's equal global counts arise from different objects.
- **M5 — insufficient specialist evidence:** one unique rescue is anecdotal. Its three localization failures are fewer than B0's four, but this is accompanied by ten low-confidence and two no-detection FN, so localization reduction is not a repeated condition-specific gain.
No strategy is promoted from one-fold subgroup wins alone; small cells remain anecdotal.

## 9. Size-only routing

Size alone is not a deployable strategy selector here: it is GT-derived in this audit, the alternatives both rescue and lose objects within size strata, and one-fold counts are sparse. Any later routing design would require prediction-time observable signals and independent validation.

## 10. Why this is not a deployable router

The comparison uses GT boxes and Stage1B GT-local features, examines only Fold1, and does not account for routing errors, fusion calibration, latency, or external/negative-image behavior. It identifies hypotheses, not an operational decision rule.

## 11. Next experiment

Run M6 (mild CLAHE followed by bilateral filtering) on the same B0-matched Fold1 protocol. This completes the pre-registered order ablation against M5 without introducing a new factor; only afterward should any specialist claim or 4-fold expansion be considered.

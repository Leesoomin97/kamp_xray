# B2 Final-performance Experiment Plan

## Frozen reference

- Selected detector: YOLOv8n with Conservative artifact-controlled input and training-only `visibility` augmentation (`hsv_v=0.05`).
- Frozen folds: `outputs/tables/01c_final_validation_folds.csv`, SHA-256 `963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8`.
- Common recipe: 30 epochs, AdamW, `lr0=0.01`, `lrf=0.1`, weight decay 0.0005, warmup 0.5 epochs, mosaic 0.5, full fine-tuning (`freeze=0`), seed 42, deterministic mode, reporting confidence 0.25, matching IoU 0.50.
- Existing B2 Fold1-4 run names:
  - `stage2b_b2_visibility_yolov8n_conservative_640_fold1_e30_v1`
  - `stage2b_b2_visibility_yolov8n_conservative_640_fold2_e30_v1`
  - `stage2b_b2_visibility_yolov8n_conservative_640_fold3_e30_v1`
  - `stage2b_b2_visibility_yolov8n_conservative_640_fold4_e30_v1`

## Planned sequence

1. `b2_1024_fold1_v1`: change only `imgsz` from 640 to 1024. Batch 8 is retained because the same KAMP V100 already completed the B0 1024 Fold1 run at batch 8. Treat Fold1 as screening evidence.
2. `b2_hardos_640_fold1_v1`: retain B2 640 and duplicate only eligible train images under `small_low_2x`.
3. `b2_hardos_1024_fold1_v1`: combine the two explicitly tested axes only after reviewing experiments 1 and 2.
4. Extend only a genuinely improved candidate to folds 2-4. Do not reinterpret Fold1 screening as OOF evidence.
5. Consider inference-only local/safety analysis only after a complete four-fold candidate exists.

No new backbone, general augmentation sweep, ensemble, operational threshold selection, or external held-out access is included.

## W&B role

W&B is opt-in telemetry. Local run manifests, `experiment_metadata.json`, frozen-fold files, checkpoints, and evaluation CSV/JSON remain the source of truth. Training proceeds if W&B import, authentication, network logging, or artifact upload fails.

Recommended groups are experiment-family specific: `B2-1024`, `B2-hardos-640`, and `B2-hardos-1024`. Run names remain globally unique and identical locally and in W&B.

## Decision criteria

Fold1 screening must compare the same fixed reporting/evaluation rules and examine overall Precision, Recall, F1, AP50, mAP50-95, TP/FP/FN, small recall, low-contrast recall, small-plus-low-contrast recall, localization failure, low-confidence FN, and no-detection FN. A candidate advances only if the intended subgroup improves without an unacceptable FP/F1 trade-off.

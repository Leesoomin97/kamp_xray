# Stage 6 M2 YOLOv8n-P2 Fold1 preflight

## Status

Preparation and static validation only. No training or dataset inference was run. M2 asks whether adding a P2/4 high-resolution detection path to the full-image detector improves small-object recall/localization without cropping.

## Architecture verification

- Configuration: `configs/stage6/yolov8n_p2.yaml`
- Parser/runtime target: `ultralytics==8.4.158`
- Class count: 1
- Detect inputs: layers `[18, 21, 24, 27]`
- Parsed strides: `[4, 8, 16, 32]` (P2/P3/P4/P5)
- Compatible transfer from pinned COCO `yolov8n.pt`: 219/437 state-dict items
- Pinned weight SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`

At 640 input, the one-class B0 Fold1 checkpoint has 3,011,043 parameters and 8.1917 GFLOPs; the one-class P2 model has 2,926,692 parameters and 12.3525 GFLOPs. M2 has six additional parsed layers, 84,351 fewer parameters (-2.80%), and 4.1608 more GFLOPs (+50.79%). FLOPs are a static complexity estimate, not KAMP runtime.

## B0 matching audit

The frozen fold file SHA-256 remains `963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8`. Fold1 contains 376 training and 124 validation source images. The Conservative integrity table contains 500 unique samples.

| Condition | B0 Fold1 | M2 Fold1 | Status |
|---|---|---|---|
| Train/validation source IDs | frozen Fold1, 376/124 | same frozen Fold1, 376/124 | matched |
| Input representation | Conservative | Conservative | matched |
| Input size | 640 | 640 | matched |
| Epochs / batch | 30 / 8 | 30 / 8 | matched |
| Fine-tuning | full, freeze=0 | full, freeze=0 | matched |
| Optimizer | AdamW | AdamW | matched |
| LR / final LR factor | 0.01 / 0.1 | 0.01 / 0.1 | matched |
| Warmup / weight decay | 0.5 epoch / 0.0005 | 0.5 epoch / 0.0005 | matched |
| Augmentation | HSV 0/0/0; translate 0.05; scale 0.2; flipLR 0.5; mosaic 0.5; others off | same | matched |
| Seed / deterministic / AMP | 42 / true / false | 42 / true / false | matched |
| Workers / patience | 2 / 0 | 2 / 0 | matched |
| Checkpoint rule | Ultralytics best.pt by validation fitness | same | matched |
| Evaluation | conf floor 0.001; reporting 0.25; NMS 0.70; match IoU 0.50 | same | matched |
| Detection architecture | YOLOv8n P3/P4/P5 | YOLOv8n-P2 P2/P3/P4/P5 | intentionally unmatched |
| Initialization | complete official COCO YOLOv8n checkpoint | compatible layers from the same checkpoint; new P2/four-scale head initialized by Ultralytics | unavoidable unmatched condition |

The initialization difference is inseparable from this architecture change and limits causal attribution. No additional seed run is authorized for this Fold1 preparation.

## Run separation

- Optional sanity: `stage6_m2_p2_yolov8n_640_fold1_sanity_e2_v1`; 2 epochs; pipeline/runtime only; `evidence_eligible=false`.
- Fold1 evidence: `stage6_m2_p2_yolov8n_640_fold1_b0matched_v1`; fixed 30 epochs; exploratory one-fold architecture evidence.
- Existing outputs are protected by run-directory and manifest collision checks and are not overwritten.

## Evaluation and interpretation

Evaluation uses the integrity-linked Conservative label path, not a raw Korean-path lookup. It produces standard prediction/metric tables and reports predefined subgroups: small (`bbox_min_side_px <= 8`), low contrast (`absolute_median_difference <= 4`), and their intersection.

M2 supports the P2 hypothesis only if small and small+low-contrast recall improve and localization failures decrease without an unacceptable FP/F1 trade-off. Overall F1 improvement with worse small-subgroup results does not support the hypothesis. An FP surge with worse F1 is reported as a trade-off. Any Fold1 result remains exploratory and does not establish 4-fold OOF superiority.

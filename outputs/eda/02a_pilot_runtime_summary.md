# Stage 2A Interrupted Pilot Runtime Summary

## Status

- **VERIFIED — purpose:** pipeline and CPU-runtime check only.
- **VERIFIED — representation:** `ARTIFACT_INPAINT_CONSERVATIVE`.
- **VERIFIED — frozen validation fold:** fold 1 (125 validation images).
- **VERIFIED — detector:** Ultralytics YOLOv8n initialized from official COCO `yolov8n.pt`. No supplied competition weight was used.
- **VERIFIED — input size / batch:** 416 px / 8.
- **VERIFIED — device:** CPU; CUDA unavailable.
- **VERIFIED — trainable scope:** backbone frozen with `freeze=22` (head-only pilot).
- **VERIFIED — requested duration:** 3 epochs.
- **VERIFIED — completed duration:** epochs 1 and 2 completed; epoch 3 was interrupted at approximately 85% and has no complete validation row.

## Recorded pilot metrics

| Epoch | Cumulative time | Precision | Recall | AP50 | mAP50–95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 225.129 s | 0.29279 | 0.19655 | 0.15334 | 0.02791 |
| 2 | 454.084 s | 0.50988 | 0.65862 | 0.48134 | 0.10660 |

The mean observed complete-epoch time was 227.042 seconds (about 3 minutes 47 seconds), including validation as recorded by Ultralytics.

## Checkpoint

- Current best checkpoint: `models/stage2a/pilots/conservative_416_fold1_e3_head/weights/best.pt`
- Best checkpoint corresponds to completed epoch 2.
- SHA-256: `177ac09a78923dfe21936b4316e4ae6af38e3a00e4e5001c2e4a6934f5a171bd`

## Interpretation limit

This interrupted, single-fold, short, frozen-backbone run is **not** a final baseline, an input-size comparison, a four-fold OOF result, or model-selection evidence. Its metrics must not be reported as competition performance. It establishes only that the data/training pipeline runs and provides a local CPU timing observation.

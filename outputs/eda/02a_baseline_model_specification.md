# Stage 2A Baseline Model Specification

## Fixed baseline recipe

- Architecture: Ultralytics YOLOv8n, single-class object detection.
- Implementation: `ultralytics==8.4.158`, `torch==2.14.1`.
- Initialization: official Ultralytics COCO `yolov8n.pt`; SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`. No competition weight is used.
- Input: approved artifact-controlled lossless PNG. Primary is Conservative; Local interpolation is a paired secondary ablation only.
- Channels: the grayscale-derived PNG is decoded by Ultralytics into its standard three-channel detector input.
- Validation: frozen 10-second temporal/similarity-component group-aware 4-fold assignment; SHA-256 `963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8`.
- Optimizer: AdamW; initial LR 0.01; final LR fraction 0.1; weight decay 0.0005; warmup 0.5 epoch.
- Batch: 8 for the first viability experiment.
- Epochs: 3 for the first viability experiment; the full baseline budget remains unresolved until runtime and learning behavior are reviewed.
- Trainable scope: `freeze=22` for the viability experiment.
- Seed: 42; deterministic mode enabled; AMP disabled.
- First KAMP-NOTE viability run workers/device: 2 / CUDA device 0. Both remain explicit CLI arguments; no automatic batch or worker scaling is performed.
- Augmentation: HSV disabled; degrees/shear/perspective/vertical flip disabled; translate 0.05; scale 0.2; horizontal flip 0.5; mosaic 0.5; mixup/copy-paste disabled.
- Early stopping: disabled (`patience=0`) for the fixed short run.
- Checkpoint selection: Ultralytics validation-fitness `best.pt` inside the single frozen fold run.
- Reporting match: IoU ≥ 0.50. Confusion counts use a documented 0.25 diagnostic threshold; raw confidences are retained and no safety threshold is selected.

Every run writes its recipe, environment, timestamps, fold hash, pretrained-weight hash, and checkpoint paths to `experiment_metadata.json`.

## Hardware and current constraint

The local host is an Intel Pentium 4405U CPU-only machine. Heavy execution has moved to the KAMP-NOTE PyTorch GPU-CUDA12 environment. The first remote 640 px run remains intentionally limited to one fold and three epochs; its actual GPU model, VRAM use, package versions, and runtime will be recorded by the run manifest.

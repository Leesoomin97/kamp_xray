# Final B2 Weight Backup Verification

Verification date: 2026-10-07

## Scope

- Read-only verification of `final_b2_weights_backup/manifest.json` against the locally archived files.
- Weight files remain excluded from Git and were not modified.
- The final selected detector is B2-640; its four frozen-fold `best.pt` checkpoints are separately preserved.

## B2-640 fold checkpoints

| Fold | Backup file | Exists | Size (bytes) | Actual SHA-256 | Manifest match |
|---:|---|---|---:|---|---|
| 1 | `b2_640_fold1_best.pt` | YES | 6,262,948 | `fa043d99a73ec27a1c634eadead4b1cf288f1a4173612b412d5e383704f0243b` | PASS |
| 2 | `b2_640_fold2_best.pt` | YES | 6,262,948 | `df208b5cd4551a2fe43d997cb9c6d046b12358e87bc303647fa053ae00482163` | PASS |
| 3 | `b2_640_fold3_best.pt` | YES | 6,262,948 | `aa86a3ce817f904ddd202dea9f2904d8a035726590baf4ed2d23054f825dbc3f` | PASS |
| 4 | `b2_640_fold4_best.pt` | YES | 6,262,948 | `09cc1aeb6745b9d983688be10ffcd895d2aae44cf7e7380b8d1011f9fffc56d8` | PASS |

All four files match both the byte size and SHA-256 recorded in `final_b2_weights_backup/manifest.json`.

## Pretrained initialization asset

| Asset | Status | Size (bytes) | SHA-256 |
|---|---|---:|---|
| `models/stage2a/pretrained/yolov8n.pt` | PRESENT | 6,549,796 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |

The hash matches the identifier enforced by `src/run_stage2a_training.py` and recorded in B2 metadata. This file is an official Ultralytics COCO pretrained initialization asset and contains no competition data. The runner never downloads it automatically; it must be placed at the documented path before training.

## Git status policy

- `*.pt` is ignored globally.
- `final_b2_weights_backup/` is explicitly ignored.
- Checkpoints must be transferred and archived outside Git with the hashes above.

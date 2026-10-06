# Stage 6 YOLOv8n-P2 architecture specification

## Provenance

- Runtime target: `ultralytics==8.4.158`.
- Project configuration: `configs/stage6/yolov8n_p2.yaml`.
- Structural source: the packaged Ultralytics 8.4.158 `cfg/models/v8/yolov8-p2.yaml`.
- Selected scale: nano (`scale: n`, depth 0.33, width 0.25, max channels 1024).
- Classes: one (`nc: 1`).

## Exact architectural difference from B0 YOLOv8n

B0 detects at P3/8, P4/16, and P5/32. M2 keeps the YOLOv8 backbone and adds one further upsampling/concatenation/C2f path to backbone P2, producing a P2/4 feature map. The bottom-up head is correspondingly extended and `Detect` consumes four maps: P2/4, P3/8, P4/16, and P5/32.

No attention block, custom operator, loss change, class change, input-resolution change, or additional augmentation is introduced.

## Static parser and complexity verification

The project configuration was parsed with the pinned local `ultralytics==8.4.158` implementation without training or dataset inference. The final `Detect` layer consumes layers `[18, 21, 24, 27]`, and the instantiated detection strides are `[4, 8, 16, 32]`; these are P2/P3/P4/P5. The one-class Fold1 B0 checkpoint and the one-class P2 YAML were compared at `imgsz=640` using the same Ultralytics FLOPs utility.

| Item | B0 YOLOv8n | M2 YOLOv8n-P2 | Difference |
|---|---:|---:|---:|
| Detect strides | 8/16/32 | 4/8/16/32 | adds P2/4 |
| Model layers | 23 | 29 | +6 |
| Parameters | 3,011,043 | 2,926,692 | -84,351 (-2.80%) |
| GFLOPs at 640 | 8.1917 | 12.3525 | +4.1608 (+50.79%) |

The lower parameter count does not imply lower compute: the new high-resolution path raises 640-input FLOPs substantially. These figures describe the parsed architectures, not measured KAMP runtime or detection performance.

## Initialization

The approved official COCO `yolov8n.pt` is SHA-256 pinned to `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`. `run_stage6_training.py` creates the P2 YAML model and calls Ultralytics compatible-layer transfer from that checkpoint. Backbone and matching head layers are transferred where shapes permit; newly introduced P2/four-scale-head parameters cannot come from the three-scale checkpoint and use Ultralytics initialization.

This partial-transfer exception is intrinsic to the architecture hypothesis and is recorded in every M2 run manifest. The parent backbone follows the same approved COCO-pretrained policy as B0, but the new P2/four-scale-head parameters use Ultralytics initialization. M2 therefore contains both the intended architecture change and this unavoidable initialization difference. Loss and all other frozen training settings remain the B0 settings; no additional seed training is authorized at this stage.

## Reproducibility and limitation

The configuration file, its SHA-256, pretrained checkpoint SHA-256, Ultralytics version, and run metadata are saved. The preparation-time structural check with Ultralytics 8.4.158 successfully built strides `[4, 8, 16, 32]` and transferred 219/437 state-dict items from the pinned YOLOv8n checkpoint. The runtime loader must still retain its transfer message and metadata for each M2 run. No Stage 6 training or inference was performed.

An optional 1-2 epoch Fold1 sanity run is permitted only to verify parsing, CUDA execution, runtime, checkpoint output, and evaluation compatibility. It is recorded as `evidence_eligible=false` and must not enter the M2/B0 performance comparison. The evidence run remains the fixed 30-epoch Fold1 experiment.

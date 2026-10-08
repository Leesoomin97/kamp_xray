# KAMP X-ray foreign-object detection — B2-640 reproduction package

This package reproduces the selected B2-640 YOLOv8n detector for foreign-object localization in finished-product X-ray images. Official TXT annotations are the ground truth. Colored rectangles embedded in the original BMP images are treated as annotation artifacts and are not used as predictive features.

## Folder structure

```text
submission_package_20261008/
├─ README.md, requirements.txt, LICENSE
├─ src/                         # preprocessing → training → evaluation → export
├─ data/source/                 # 500 canonical BMP + 500 official TXT
├─ outputs/tables/              # frozen folds, mappings, integrity metadata
├─ outputs/processed_data/      # 500 conservative PNG + YOLO labels
├─ models/b2_640/               # four frozen final fold weights + metadata
├─ models/stage2a/pretrained/   # fixed YOLOv8n initialization asset
├─ predictions/                 # evaluator-facing frozen OOF results
├─ reference/                   # immutable reference fold/OOF evidence
├─ metadata/                    # config, checksums, validation scope
└─ reproduction_runs/           # new verification outputs; initially empty
```

## What is included

- 500 canonical BMP images and their 500 official TXT annotations (`data/source/`)
- artifact-controlled conservative images and YOLO labels used by B2 (`outputs/processed_data/stage2a/conservative/`)
- frozen group-aware 4-fold assignment and integrity tables (`outputs/tables/`)
- complete preprocessing, training, evaluation, aggregation, and export source (`src/`)
- four final B2-640 fold weights and portable metadata (`models/b2_640/`)
- the official YOLOv8n initialization asset used by training (`models/stage2a/pretrained/yolov8n.pt`)
- frozen reference OOF evidence (`reference/oof/`) and evaluator-friendly prediction CSVs (`predictions/`)

## Environment

Validated training environment: Python 3.11.9, PyTorch 2.5.0+cu121, torchvision 0.20.0+cu121, Ultralytics 8.4.158, Tesla V100-SXM2-32GB. Install from the package root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows activation: `.venv\Scripts\activate`.

Before any run, verify package integrity:

```bash
python src/run_submission_pipeline.py verify
```

Stage I/O summary:

| Stage | Input | Main output |
|---|---|---|
| Preprocessing | `data/source/images`, official TXT and frozen mapping | `outputs/processed_data/stage1d`, preprocessing manifest |
| Fold materialization | conservative PNG/labels + frozen fold CSV | integrity table and host-local runtime split under `outputs/cache` |
| Training | one fold YAML + fixed B2 recipe + pretrained asset | new `models/stage2a/<tag>_foldN/weights/best.pt` |
| Inference/evaluation | fold `best.pt` + held-out fold | `reproduction_runs/<tag>/foldN_evaluation/` |
| OOF aggregation | four fold evaluation folders | `reproduction_runs/<tag>/oof/` |
| Prediction export | OOF image prediction table | bbox and image-summary CSVs under `reproduction_runs/<tag>/predictions/` |

## Fixed final recipe

YOLOv8n; conservative representation; visibility augmentation; `imgsz=640`; 30 epochs; batch 8; seed 42; no frozen layers; AdamW; `lr0=0.01`; `lrf=0.1`; weight decay 0.0005; warmup 0.5 epoch; `box=7.5`; `dfl=1.5`; `cls=0.5`; `hsv_v=0.05`; mosaic 0.5; no oversampling. See `metadata/final_b2_config.yaml`.

## FULL reproduction

One command performs preprocessing, fold materialization, four-fold training, held-out-fold inference/evaluation, OOF aggregation, and prediction export:

```bash
python src/run_submission_pipeline.py full --tag kamp_full_reproduction_v1 --device 0 --workers 2
```

Equivalent explicit preparation and training commands are:

```bash
python src/run_stage1d_artifact_control.py --workspace .
python src/prepare_stage2a_dataset.py --workspace .
python src/run_stage2a_training.py --project-root . --representation conservative --model yolov8n --augmentation-profile visibility --fold 1 --imgsz 640 --epochs 30 --batch 8 --device 0 --freeze 0 --seed 42 --workers 2 --box 7.5 --dfl 1.5 --cls 0.5 --oversampling-mode none --run-name kamp_full_reproduction_v1_fold1
python src/run_stage2a_training.py --project-root . --representation conservative --model yolov8n --augmentation-profile visibility --fold 2 --imgsz 640 --epochs 30 --batch 8 --device 0 --freeze 0 --seed 42 --workers 2 --box 7.5 --dfl 1.5 --cls 0.5 --oversampling-mode none --run-name kamp_full_reproduction_v1_fold2
python src/run_stage2a_training.py --project-root . --representation conservative --model yolov8n --augmentation-profile visibility --fold 3 --imgsz 640 --epochs 30 --batch 8 --device 0 --freeze 0 --seed 42 --workers 2 --box 7.5 --dfl 1.5 --cls 0.5 --oversampling-mode none --run-name kamp_full_reproduction_v1_fold3
python src/run_stage2a_training.py --project-root . --representation conservative --model yolov8n --augmentation-profile visibility --fold 4 --imgsz 640 --epochs 30 --batch 8 --device 0 --freeze 0 --seed 42 --workers 2 --box 7.5 --dfl 1.5 --cls 0.5 --oversampling-mode none --run-name kamp_full_reproduction_v1_fold4
```

The four explicit evaluation commands follow the same pattern; for example Fold 1:

```bash
python src/evaluate_stage2a_run.py --project-root . --run-dir models/stage2a/kamp_full_reproduction_v1_fold1 --output-dir reproduction_runs/kamp_full_reproduction_v1/fold1_evaluation --device 0 --conf-floor 0.001 --report-confidence 0.25 --nms-iou 0.7 --max-det 300
```

Repeat for folds 2–4, then aggregate and export:

```bash
python src/evaluate_stage2a_run.py --project-root . --run-dir models/stage2a/kamp_full_reproduction_v1_fold2 --output-dir reproduction_runs/kamp_full_reproduction_v1/fold2_evaluation --device 0 --conf-floor 0.001 --report-confidence 0.25 --nms-iou 0.7 --max-det 300
python src/evaluate_stage2a_run.py --project-root . --run-dir models/stage2a/kamp_full_reproduction_v1_fold3 --output-dir reproduction_runs/kamp_full_reproduction_v1/fold3_evaluation --device 0 --conf-floor 0.001 --report-confidence 0.25 --nms-iou 0.7 --max-det 300
python src/evaluate_stage2a_run.py --project-root . --run-dir models/stage2a/kamp_full_reproduction_v1_fold4 --output-dir reproduction_runs/kamp_full_reproduction_v1/fold4_evaluation --device 0 --conf-floor 0.001 --report-confidence 0.25 --nms-iou 0.7 --max-det 300
python src/aggregate_stage2a_oof.py --project-root . --fold-results reproduction_runs/kamp_full_reproduction_v1/fold1_evaluation reproduction_runs/kamp_full_reproduction_v1/fold2_evaluation reproduction_runs/kamp_full_reproduction_v1/fold3_evaluation reproduction_runs/kamp_full_reproduction_v1/fold4_evaluation --output-dir reproduction_runs/kamp_full_reproduction_v1/oof --run-name kamp_full_reproduction_v1_oof
python src/export_oof_predictions.py --oof-dir reproduction_runs/kamp_full_reproduction_v1/oof --bbox-output reproduction_runs/kamp_full_reproduction_v1/predictions/b2_640_frozen_4fold_oof_bbox_conf025.csv --image-output reproduction_runs/kamp_full_reproduction_v1/predictions/b2_640_frozen_4fold_oof_image_summary_conf025.csv --confidence 0.25
```

## QUICK verification

QUICK mode is intentionally separate from FULL reproduction. It does not train and does not overwrite the frozen reference. It verifies the four supplied weight hashes and repeats inference/evaluation/OOF aggregation:

```bash
python src/run_submission_pipeline.py quick --tag quick_verification_v1 --device 0
```

Use `--device cpu` only for a slower CPU verification. Outputs are created under `reproduction_runs/<tag>/`.

## Reference prediction artifact

`predictions/b2_640_frozen_4fold_oof_bbox_conf025.csv` contains one row per prediction at confidence 0.25 with columns `image_id, fold_id, prediction_id, x1, y1, x2, y2, confidence, class_id, class_name`. The companion image summary contains all 500 images. These are frozen 4-fold OOF development predictions, not an official or external test prediction.

Reference OOF metrics at confidence 0.25 and matching IoU 0.5 are: 500 images, 1,147 GT objects, TP 1,121, FP 51, FN 26, precision 0.9564846416, recall 0.9773321709, F1 0.9667960328, AP50 about 0.96039, and mAP50–95 about 0.38077.

## Reproducibility and limitations

The frozen fold file has SHA-256 `963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8`. Fold 1 was repeated with the same KAMP environment, seed, fold, and recipe and reproduced the same reported TP/FP/FN and derived metrics. This is execution reproducibility in that environment, not a claim of cross-GPU bitwise identity.

The Fold 1 repetition produced TP 281, FP 19, FN 9, precision 0.9366666667, recall 0.9689655172, F1 0.9525423729, AP50 0.9569807375, and mAP50–95 0.3736790631. Seven evaluation artifacts matched the original Fold 1 artifacts by SHA-256; their hashes and scope are recorded in `metadata/fold1_reproduction_evidence.json`.

No separate official held-out test split or submission schema was identified in the supplied files. Therefore the included prediction result is clearly labeled OOF. All 500 development images are GT-positive, so this package does not establish specificity, true-negative performance, automatic PASS safety, or real normal-product false-alarm workload. W&B is optional tracking only and is not required for preprocessing, inference, evaluation, or submission use; local files are the source of truth.

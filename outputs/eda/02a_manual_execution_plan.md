# Stage 2A Manual Execution Plan

## Planning-only runtime estimates

The 416 px estimate uses the two completed pilot epochs: 227.042 seconds per epoch including validation. The 640 px estimate uses the observed 377-second training plus 82-second validation time: 459 seconds per epoch. These projections exclude some startup, cache, checkpoint, and OS-load variation.

| Configuration | Estimated runtime |
|---|---:|
| 416 px × 3 epochs × 1 fold | 11 min 21 s |
| 416 px × 3 epochs × 4 folds | 45 min 25 s |
| 416 px × 10 epochs × 4 folds | 2 h 31 min 22 s |
| 640 px × 3 epochs × 1 fold | 22 min 57 s |
| 640 px × 3 epochs × 4 folds | 1 h 31 min 48 s |
| 640 px × 10 epochs × 4 folds | 5 h 06 min 00 s |

## One next KAMP-NOTE experiment

From the uploaded project root, run only `Conservative / fold 1 / 640 px / 3 epochs / frozen backbone` on GPU device 0. It is the smallest non-duplicative experiment that measures 640 px viability against the completed 416 px pipeline pilot. It does not start evaluation or another fold automatically.

```bash
python src/run_stage2a_training.py --project-root "$PWD" --representation conservative --fold 1 --imgsz 640 --epochs 3 --batch 8 --device 0 --freeze 22 --seed 42 --workers 2 --run-name kamp_conservative_640_fold1_e3_gpu_v1
```

The script regenerates host-local split files under `outputs/cache/stage2a_runtime_splits/`, writes model artifacts under `models/stage2a/kamp_conservative_640_fold1_e3_gpu_v1/`, and writes `outputs/run_manifests/kamp_conservative_640_fold1_e3_gpu_v1.json`. After completion, download the paths marked `[DOWNLOAD]` in the terminal manifest. Do not start evaluation or another fold yet.

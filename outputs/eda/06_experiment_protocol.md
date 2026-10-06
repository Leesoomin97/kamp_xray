# Stage 6 hypothesis-validation experiment protocol

## Status and scope

This document freezes the preparation rules for M1–M6. No Stage 6 detector has been trained or evaluated while preparing this protocol. The original dataset remains read-only, and only the approved 500-sample development corpus is eligible.

## Common frozen conditions

| Item | Fixed value |
|---|---|
| Development corpus | 500 approved logical samples; 1,147 TXT-GT objects |
| Validation | Frozen 10-second temporal/similarity-component group-aware 4-fold assignment |
| Fold SHA-256 | `963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8` |
| Fold sizes | 124 / 125 / 126 / 125 source images |
| Artifact control | Conservative artifact-controlled representation |
| Baseline architecture | YOLOv8n, except M2's explicitly isolated P2 head change |
| Input size | 640 |
| Epochs | 30 for evidence runs; M1 pipeline sanity is restricted to 1-3 epochs and M2 architecture sanity to 1-2 epochs; neither is evidence |
| Batch | 8 where technically feasible; any deviation must be declared before execution and recorded |
| Fine-tuning | Full fine-tuning, `freeze=0` |
| Seed | 42 |
| Initialization | Approved official Ultralytics COCO `yolov8n.pt` |
| Pretrained SHA-256 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| Optimizer | AdamW; `lr0=0.01`, `lrf=0.1`, weight decay 0.0005, warmup 0.5 epoch |
| AMP | Disabled, matching B0 |
| Augmentation | B0 profile: HSV off, translate 0.05, scale 0.2, horizontal flip 0.5, mosaic 0.5; all other listed geometric/mixing augmentations off |
| Reporting confidence | 0.25; diagnostic reporting threshold only |
| Matching | IoU ≥ 0.50 |
| Confidence floor / NMS | 0.001 / 0.70 |

No external held-out data, legacy split, RAW_RGB, grayscale-direct input, or competition-provenance weight is permitted.

## Independent hypotheses

- **M0:** Existing B0 YOLOv8n/Conservative baseline. Reused; no retraining.
- **B2:** Existing `hsv_v=0.05` visibility experiment. Reused; no retraining and not mixed with M1–M6.
- **M1:** 192 px fixed grid patches with 25% nominal overlap (144 px stride), resized by the detector to 640.
- **M1-256-CONTROLLED:** one additional scale ablation using 256 px fixed-grid patches with 25% nominal overlap (192 px stride). It uses a separate manifest/output tree, the identical GT clipping policy, and the same B0-progress replay. Existing M1-192 inputs and results remain immutable.
- **M2:** YOLOv8n with an added P2/4 detection branch; no attention or loss change.
- **M3:** Mild bilateral preprocessing only.
- **M4:** Mild CLAHE preprocessing only.
- **M5:** Mild bilateral followed by mild CLAHE.
- **M6:** Mild CLAHE followed by mild bilateral.

## Known common-condition exception

M2 cannot have bit-identical initialization to B0 because the added P2 path and four-scale detection head do not exist in `yolov8n.pt`. The parent backbone follows the same approved COCO-pretrained policy, while newly added P2/four-scale-head parameters use Ultralytics initialization. M2 therefore combines an architecture change with an unavoidable initialization difference. Optimizer, loss implementation, data, augmentation, folds, seed, resolution, and epoch budget remain unchanged; no extra seed run is authorized at this stage.

The M2 Fold1 evidence run otherwise matches B0: the same 376/124 frozen train/validation source IDs, Conservative images, 640 input, batch 8, 30 epochs, full fine-tuning, AdamW/LR/warmup/weight decay, B0 augmentation, seed 42, deterministic mode, workers 2, checkpoint-selection rule, and evaluation thresholds. A separately named 1-2 epoch M2 sanity run is pipeline-only (`evidence_eligible=false`) and must never enter the M2/B0 performance comparison.

## B0 and M1 training budgets

With batch 8, B0 has 47 training batches per epoch in every fold. Over 30 epochs this is 1,410 micro-batch iterations. Reproducing the Ultralytics 8.4.158 accumulation rule (`nbs=64`, warmup 0.5 epoch) yields approximately 180 optimizer updates per fold.

| Fold | B0 train images | B0 val images | Batches/epoch | Micro-batch iterations | Approx. optimizer updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 376 | 124 | 47 | 1,410 | 180 |
| 2 | 375 | 125 | 47 | 1,410 | 180 |
| 3 | 374 | 126 | 47 | 1,410 | 180 |
| 4 | 375 | 125 | 47 | 1,410 | 180 |

M1 has 2,682-2,692 training patches per fold and therefore 336-337 micro-batches per full patch epoch. Two distinct experiments are defined:

- **M1-CONTROLLED (primary ablation):** batch 8 and an exact B0 progress replay over 1,410 micro-batches. Warmup lasts 24 micro-batches, accumulation follows the same Ultralytics 8.4.158 rule, and the run is required to finish with approximately 180 optimizer updates. The ordinary 30-patch-epoch scheduler is disabled for this path. Instead, the B0 linear LR factor is replayed using a virtual B0 epoch `floor(global_micro_batch / 47)`: `lr_factor(e)=max(1-e/30,0)*(1-0.1)+0.1`, for virtual epochs 0 through 29. Non-bias LR starts at 0 and bias LR at 0.1. Because Ultralytics applies warmup only while `ni < 24`, the first post-warmup batch retains the `ni=23` values (non-bias 0.00958333; bias 0.01375) until virtual epoch 1 begins; thereafter the epochwise B0 factors are replayed, ending at 0.0013. For fold 1, 1,410 batches end in physical patch epoch 5 at batch 62 (four full 337-batch epochs plus 62 batches). Epoch count remains 30 only as a hard outer bound; it does not define the controlled LR trajectory.
- **M1-PRACTICAL (deferred):** uncapped 30 patch epochs, intended only for later performance optimization if M1-CONTROLLED is useful. It has substantially more updates and must never be pooled or labeled as the controlled scale ablation.

The controlled run manifest records target/actual micro-batch iterations, optimizer updates, warmup iterations, first/after-warmup/final parameter-group LRs, and the scheduler policy. OOF aggregation rejects sanity runs, and its recipe signature prevents controlled and practical runs from being combined. A later practical OOF, if authorized, must remain a separately named optimization result.

## M1 patch label and fold policy

- Patch grid: 192×192, stride 144, final tile edge-anchored.
- Every patch inherits its source image's frozen fold; source-image patches can never cross train/validation folds.
- Fully contained GT boxes are translated directly into patch coordinates.
- Every positive-area intersection between a source GT box and a patch is clipped to the patch and explicitly labeled. No visibility threshold is used and no visible fragment is silently treated as background.
- This policy is conservative against false-background supervision. Its trade-off is that very small truncated positive boxes can occur at patch boundaries; their mapping and retained-area fraction remain traceable.
- The planning manifest contains 3,584 kept patches: 1,604 positive and 1,980 true background-only. Unlabeled GT-fragment patches are zero, bbox-coordinate errors are zero, and all 1,147 source GT objects have at least one valid positive patch.
- Source coordinates and patch-to-source GT mapping are retained for evaluation and fusion.

Patch source-level evaluation uses one fixed provisional class-agnostic reference NMS at IoU 0.50 for the sanity and first controlled M1 runs. It is explicitly not final fusion. Patch-scale testing and fusion optimization are not combined; raw patch predictions and all suppression decisions are saved for later comparison without rerunning inference.

## Mild preprocessing parameters

All methods operate on the approved uint8 Conservative PNG and save lossless PNG outside the original dataset.

- Bilateral: OpenCV `d=5`, `sigmaColor=10`, `sigmaSpace=3`.
- CLAHE: OpenCV `clipLimit=1.5`, `tileGridSize=(8,8)`.
- M5: bilateral, then CLAHE with the same parameters.
- M6: CLAHE, then bilateral with the same parameters.

These are the exact Stage 5 mild settings. They are deterministic and introduce no color augmentation.

## Execution order and stopping rule

Run one fold at a time. The first run is an **M1 PIPELINE SANITY PILOT**, fold 1 for 2 epochs. Its only purposes are materialization, dataloader, coordinate transform, VRAM/runtime, checkpoint, evaluation, and fusion-pipeline verification. It is not model-comparison evidence, cannot enter the Stage 6 performance table, and cannot be used for model selection. Only after it succeeds may M1-CONTROLLED fold 1 be run. Do not start fold 2-4 automatically.

## First KAMP pipeline-sanity commands

From the uploaded project root:

```bash
python src/prepare_stage6_patch_dataset.py --project-root "$PWD" --materialize
python src/run_stage6_training.py --project-root "$PWD" --experiment m1_patch --run-purpose sanity --fold 1 --epochs 2 --imgsz 640 --batch 8 --device 0 --seed 42 --workers 2 --run-name stage6_m1_sanity_patch192_ov25_yolov8n_640_fold1_e2_v2
python src/evaluate_stage6_run.py --project-root "$PWD" --run-dir models/stage6/stage6_m1_sanity_patch192_ov25_yolov8n_640_fold1_e2_v2 --output-dir outputs/stage6_runs/stage6_m1_sanity_patch192_ov25_yolov8n_640_fold1_e2_v2 --device 0 --conf-floor 0.001 --report-confidence 0.25 --nms-iou 0.7 --max-det 300 --patch-fusion reference_nms --fusion-iou 0.5
```

The preparation command is a prerequisite, not a second experiment. Review its manifest before starting the sanity run.

## Controlled M1 fold-1 command after sanity success

```bash
python src/run_stage6_training.py --project-root "$PWD" --experiment m1_patch --run-purpose controlled --max-optimizer-steps 180 --fold 1 --epochs 30 --imgsz 640 --batch 8 --device 0 --seed 42 --workers 2 --run-name stage6_m1c_patch192_ov25_yolov8n_640_fold1_b0progress_v3
```

Evaluate it with the same fixed reference-NMS arguments and a matching output directory. M1-PRACTICAL remains deferred and has no authorized execution command in this protocol.

## M1-256 controlled scale ablation

M1-256-CONTROLLED is restricted to fold 1. It changes only patch size/stride relative to M1-192: 256 px, 25% overlap, stride 192. It uses `outputs/tables/06_patch256_manifest.csv` and `outputs/processed_data/stage6/patch256_overlap25/`; the existing 192 px manifest, derived data, and results are not modified. Every positive-area GT intersection is clipped and labeled, source-image folds are inherited, and all 1,147 GT objects must remain represented. Training reuses the controlled target of 1,410 micro-batches, 24 warmup iterations, approximately 180 optimizer updates, and the 47-micro-batch B0 virtual-epoch LR trajectory regardless of the 256-patch batches per physical epoch. Evaluation uses the same provisional fixed reference NMS (fusion IoU 0.50), which remains non-final.

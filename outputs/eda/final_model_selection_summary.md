# Final Model Selection and Stage 3 Operating-logic Evidence

## 1. Evidence scope and latest-file check

This document uses existing outputs only. No training, inference, fold generation, or threshold optimization was run.

The following requested evidence locations exist:

- `outputs/stage2a_runs/b2_640_repro_fold1_v1/`
- `outputs/stage2a_runs/b2_1024_fold1_v1/`
- `outputs/stage2a_runs/b2_hardos_640_fold1_v1/`
- `outputs/stage2a_runs/b2_box9_640_fold1_v1/`
- `outputs/tables/08_b2_hardos_fold1_paired/`
- `outputs/tables/09_b2_oof_threshold040_analysis/`
- `outputs/tables/10_stage3_positive_routing_simulation/`
- `outputs/tables/02b_b2_visibility_oof_v1/`
- `src/run_stage2a_training.py`

Each of the four new Fold1 evaluation directories contains metrics, image/object predictions, FP predictions, error types, confidence sweep, and evaluation metadata.

**Reproducibility limitation:** the corresponding localized `models/stage2a/<run>/experiment_metadata.json`, `args.yaml`, checkpoints, and `outputs/run_manifests/<run>.json` are absent for these four new runs. Their evaluation metadata directly verifies representation, fold, image size, epochs, batch, freeze, seed, model, and visibility augmentation. It does not record `box`, `dfl`, `cls`, or oversampling mode. Those fields therefore follow the frozen final specification/current runner for future reproduction but are not independently confirmed from localized training metadata for the completed runs.

## 2. Final model definition

The final detector is frozen as **B2-640**:

| Setting | Value | Verification status |
|---|---:|---|
| Architecture | YOLOv8n | VERIFIED in evaluation metadata |
| Representation | Conservative artifact control | VERIFIED |
| Input size | 640 | VERIFIED |
| Augmentation profile | `visibility` | VERIFIED |
| `hsv_v` | 0.05 | VERIFIED |
| Epochs / batch | 30 / 8 | VERIFIED |
| Freeze / seed | 0 / 42 | VERIFIED |
| Box / DFL / CLS | 7.5 / 1.5 / 0.5 | Frozen final specification and current runner defaults; historical localized args missing |
| Oversampling | none | Frozen final specification; historical localized training metadata missing |

W&B is an optional tracking layer. Local CSV/JSON, experiment metadata, run manifests, frozen folds, and checkpoints remain the source of truth.

## 3. Reproducibility confirmation

Existing B2 Fold1 and `b2_640_repro_fold1_v1` have identical TP, FP, FN, Precision, Recall, F1, AP50, and mAP50-95. More strongly, the following seven outputs have identical SHA-256 hashes across the two run directories:

- `metrics.json`
- `metrics.csv`
- `confidence_sweep.csv`
- `image_predictions.csv`
- `object_predictions.csv`
- `fp_predictions.csv`
- `error_types.csv`

Fold1 result: TP 281 / FP 19 / FN 9, Precision 0.9366667, Recall 0.9689655, F1 0.9525424, AP50 0.9569807, mAP50-95 0.3736791.

Therefore: **the same seed/fold/config was reproduced with identical saved evaluation results in the current KAMP code/environment.** This does not guarantee bitwise-identical results on another GPU, driver, CUDA, PyTorch, or Ultralytics environment.

## 4. Fold1 final-candidate comparison

| Experiment | TP / FP / FN | F1 | AP50 | mAP50-95 | Decision |
|---|---:|---:|---:|---:|---|
| B2-640 repro | 281 / 19 / 9 | 0.95254 | 0.95698 | 0.37368 | **SELECTED** |
| B2-1024 | 274 / 70 / 16 | 0.86435 | 0.91564 | 0.37511 | Reject |
| B2-HardOS-640 | 280 / 10 / 10 | 0.96552 | 0.94032 | 0.38808 | Reject |
| B2-Box9-640 | 279 / 15 / 11 | 0.95548 | 0.95309 | 0.38186 | Reject |

### Why B2-640 is retained

The decision criterion is not simply the largest average F1 or mAP. Foreign-object inspection prioritizes fewer missed objects, Recall, and stability of the vulnerable small/low-visibility subgroups. B2-640 has the lowest FN count among these Fold1 candidates and already has complete four-fold OOF evidence.

### Why B2-1024 is rejected

FN increased 9→16 and FP increased 19→70. Recall, Precision, F1, and AP50 all declined. mAP50-95 rose slightly, but that isolated change does not compensate for the detection-count degradation. Resizing to 1024 increases model-space sampling but cannot create additional sensor information absent from the source X-ray.

### Why B2-HardOS is rejected

Overall F1 and mAP50-95 improved, but the intended hard groups worsened:

- Small recall: 0.93506→0.89610.
- Low-contrast recall: 0.95062→0.92593.
- Small+low recall: 0.86207→0.82759.
- Three B0 FN objects were rescued, but none was small, low-contrast, or small+low.
- Four B0 TP objects were lost; three were small.

It is therefore not evidence that the intended hard-case weakness was improved.

### Why B2-Box9 is rejected

FP fell 19→15 and localization-failure FNs fell 4→2, but FN increased 9→11 and Recall decreased. This trade-off is inconsistent with the miss-minimization priority.

These are Fold1 screening comparisons, not independent final generalization estimates.

## 5. B2 four-fold OOF final performance

At the fixed reporting confidence 0.25 and matching IoU 0.50:

- Images / GT objects: 500 / 1,147
- TP / FP / FN: 1,121 / 51 / 26
- Precision: 0.9564846416
- Recall: 0.9773321709
- F1: 0.9667960328
- AP50: 0.9603903154
- mAP50-95: 0.3807657268
- Every frozen validation sample is covered exactly once.

These are development OOF results, not untouched external held-out performance. Confidence 0.25 is a reporting threshold, not a final operational threshold.

## 6. Development confidence-threshold analysis

| Confidence | TP / FP / FN | Precision | Recall | F1 | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0.25 | 1121 / 51 / 26 | 0.95648 | 0.97733 | 0.96680 | Reporting reference |
| 0.40 | 1121 / 30 / 26 | 0.97394 | 0.97733 | 0.97563 | Same TP/FN/Recall; 21 fewer FP (41.2%) |
| 0.45 | 1120 / 27 / 27 | 0.97646 | 0.97646 | 0.97646 | Highest listed F1 but one additional FN |
| 0.50 | 1115 / 27 / 32 | 0.97636 | 0.97210 | 0.97422 | More misses without an FP reduction versus 0.45 |

Confidence 0.40 is a strong single-threshold development candidate. For a three-way workflow, 0.45/0.25 provides a clearer candidate split between high-confidence DETECT and intermediate REINSPECT. Neither threshold is operationally frozen.

## 7. Object-level FN versus product-level signal

### Object level at confidence 0.40

- FN: 26
- LOW_CONFIDENCE: 15
- LOCALIZATION_FAILURE: 11
- Small FN: 12
- Low-contrast FN: 8
- Small+low FN: 7

`LOCALIZATION_FAILURE` does not mean the model produced no signal. It means a sufficiently confident prediction existed but its IoU with the GT object was below 0.50, so it is an FN under the object-detection matching rule.

### Image/product level from raw predictions

- Max raw confidence ≥0.25: 500/500 images.
- Max raw confidence ≥0.40: 500/500.
- Max raw confidence ≥0.45: 499/500.
- Max raw confidence ≥0.50: 496/500.

Thus every GT-positive development image has at least one raw prediction at confidence 0.40. This does not imply every GT object is correctly localized or detected. It also cannot estimate normal-product specificity because the approved 500-image corpus contains no GT-negative images.

## 8. Stage 3 operating-logic candidate

The development OOF candidate is:

- **DETECT:** image-level maximum raw confidence ≥0.45.
- **REINSPECT:** 0.25 ≤ image-level maximum raw confidence <0.45.
- **PASS candidate:** image-level maximum raw confidence <0.25.

On the 500 GT-positive OOF images:

- DETECT: 499
- REINSPECT: 1
- PASS candidate: 0
- DETECT + REINSPECT positive-set capture: 500/500

This is not evidence of a 0.2% production reinspection rate. Normal products are absent, so production workload, false alarms, specificity, and PASS safety remain unresolved.

Separately, 18/500 images (3.6%) contain at least one raw box in the 0.25-0.40 band, totaling 21 boxes. This is a low-confidence candidate-box prevalence, not an image-level reinspection rate and not a validated operational rule.

## 9. Error-analysis interpretation limits

- Contrast and CNR-like values are image-derived visibility proxies, not physical density, material, or thickness.
- Product-edge distance is based on an inferred image boundary, not GT physical product segmentation.
- Shape evidence remains limited.
- Machine, date, and resolution are confounded and do not support causal equipment claims.
- Stage6/Fold1 experiments are exploratory screens and must not be presented as four-fold final performance.
- No external untouched held-out final evaluation is available.

## 10. Model-development freeze

B2-640 is the single selected detector. The project has a reproducible Fold1 check, a four-fold OOF comparison, multiple controlled alternatives, error-linked selection logic, and a candidate Stage3 workflow. New model training, additional inference, or threshold re-optimization is not required for the current report package.

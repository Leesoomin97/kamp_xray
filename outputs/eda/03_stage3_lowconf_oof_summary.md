# Stage 3 Low-Confidence OOF Safety Analysis

## Integrity

- OOF images: 500
- GT objects: 1147
- Raw prediction candidates retained at floor: 4822
- IoU matching threshold: 0.50

## Stage2 reproduction check

- confidence=0.25: TP=1121, FP=51, FN=26, Precision=0.956485, Recall=0.977332, F1=0.966796
- This exactly reproduces the frozen B2 OOF reporting point.

## Threshold observations

- Highest F1 among evaluated thresholds: confidence=0.450, F1=0.976460.
- Lowest FN among evaluated thresholds: confidence=0.001, FN=13, Recall=0.988666, FP=3688.

## Safety interpretation

- confidence >= 0.40 is evaluated as a DETECT candidate region, not a finalized operational threshold.
- Predictions below 0.40 are evaluated as possible REINSPECT gray-zone candidates.
- Lowering the reinspection floor can recover additional GT objects but also introduces additional FP candidates.
- The final reinspection floor must therefore be selected from the empirical safety-versus-review-burden trade-off.

## Limitation

- All 500 development images contain GT foreign objects. There is no true-negative / normal-product corpus.
- Therefore specificity, false-positive rate on normal production, true PASS rate, and real production reinspection workload cannot be estimated from this dataset.
- Image-level DETECT / REINSPECT / PASS results are positive-corpus safety diagnostics only.

## Threshold status

- No final operational threshold is fixed by this analysis.
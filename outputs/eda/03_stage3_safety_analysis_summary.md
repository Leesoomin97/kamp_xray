# Stage 3 Safety Analysis - Initial Sweep

## Integrity

- GT-positive development images: 500
- GT objects: 1147
- IoU match criterion: 0.50
- Existing Stage 2 reporting confidence: 0.25

## Existing reporting point

- TP=1121
- FN=26
- Recall=0.977332
- FP=51
- Precision=0.956485
- F1=0.966796

## FP sweep support

- No trustworthy low-confidence raw FP table was automatically identified.
- Precision/F1 below confidence 0.25 are therefore not treated as valid.

## Routing interpretation

- DETECT uses confidence >= 0.25.
- REINSPECT is evaluated as a lower confidence gray zone.
- PASS means no retained candidate reached the reinspection threshold.
- Because every development image contains at least one GT object, PASS in this corpus is an unsafe pass, not evidence of true-negative specificity.
- Real production reinspection workload cannot be estimated without normal / true-negative images.

## Operational-variable constraint

- GT bbox size, GT contrast, GT CNR-like values, and GT localization IoU are useful failure-analysis variables.
- They must not be used as direct operating triggers for a completely missed object because they are unavailable at inference time.
- Acquisition metadata and model-side prediction signals may be considered operationally if they are available before the decision.

## Threshold status

- No final PASS / DETECT / REINSPECT threshold is selected by this script.
- Threshold sweep results are diagnostic evidence for Stage 3.
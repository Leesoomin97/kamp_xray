# Stage 6 selective routing feasibility — Fold 1

## Scope and integrity

- B0: TP=281, FP=15, FN=9; 192 controlled: TP=266, FP=29, FN=24; 256 controlled: TP=276, FP=28, FN=14.
- Paired result: 192 rescues 5/9 B0 FN and loses 20 B0 TP; 256 rescues 2/9 and loses 7 B0 TP; both 256 rescues are within the 192 rescue set.
- B0 remains the primary detector. GT/error labels are used only to score coverage, never as routing signals.
- Candidate-present B0 FN=9/9; no-candidate B0 FN=0. Candidate-based routing cannot recover no-candidate cases.

## Deployable signals

- Confidence, predicted-box size, and predicted-box 2x-ring visibility are computed only from B0 outputs and Conservative images.
- Local visibility uses predicted-box absolute median contrast, robust image-derived CNR-like proxy, and local background gradient. These are not physical density/material measurements.
- Stability available: True. When false, stability-based rules remain unresolved and are not imputed.

## Ranked burden sweep

- Candidate ROI count=1077, validation images=124. Burden is the fraction of B0 candidates ranked for routing; routed-image burden is separately reported.
- At candidate burdens 10/20/30/40/50%, confidence-only image routing covers rescued FN `4 / 5 / 5 / 5 / 5` of 5, while routing `62.1% / 84.7% / 100.0% / 100.0% / 100.0%` of images.
- Small predicted-box routing covers `2 / 2 / 2 / 3 / 3` of 5 at image burdens `14.5% / 25.8% / 41.1% / 64.5% / 71.8%`. The confidence-OR-size rule does not improve the observed Pareto frontier.
- Predicted-box local-visibility routing covers `2 / 4 / 4 / 5 / 5` of 5 at image burdens `48.4% / 70.2% / 85.5% / 93.5% / 98.4%`; it adds no lower-burden rescue advantage over confidence-only.
- Direct ROI-linked rescue coverage is lower than image routing because image routing executes the complete fixed 192-grid. This distinction is retained in the CSV and prevents incidental same-image coverage from being presented as ROI localization.
- Best observed available row by rescue coverage then image burden: `low_confidence_OR_high_instability_OR_low_visibility` at candidate burden 30%, rescue coverage 5/5, image burden 81.5%, expected fixed-grid 192-patch calls=724.
- The simplest high-coverage Pareto point is confidence-only at 10% candidate burden: 4/5 rescues but 77/124 images (62.1%) are routed. Full 5/5 coverage first occurs at 20% candidate burden and routes 105/124 images (84.7%), so the present full-grid second pass is not operationally selective.
- No operational threshold is selected. Reference 192 fusion remains provisional.

## Oracle upper bound

- Oracle only: retain all 281 B0 TP and add the five B0 FN rescued by 192, giving TP=286, FN=4, Recall=0.986207.
- FP is not fixed before an explicit deployable routing/fusion rule is evaluated. This is not deployable performance.

## Limitations

- All 124 Fold1 images are GT-positive; true-negative routing burden and production workload cannot be estimated.
- Existing 192/256 evaluation bundles do not include their training checkpoints or experiment_metadata.json.
- Stability conclusions require the separately prepared inference-only audit; no stability values are guessed.

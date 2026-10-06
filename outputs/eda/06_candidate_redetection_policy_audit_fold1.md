# Stage 6 candidate local re-detection policy audit

- candidate-scale feature rows checked: 5385
- raw-local selection mismatches: 0
- policy-dependent local-match rows: 0
- CONFIRM_CONFIDENCE transformation mismatches: 0
- REFINE_BOX transformation mismatches: 0
- CONFIRM_AND_REFINE transformation mismatches: 0
- reconstructed scale-summary value mismatches: 0
- legacy confidence trajectory column present: True

`confidence_trajectory_json` is the confidence of the final prediction associated with each GT object after policy application and evaluation. It is not the selected local-prediction confidence. The final associated candidate can differ by policy because bbox refinement changes which candidate overlaps a GT object.

The detailed target-object rows record both final prediction confidence and selected local confidence separately.
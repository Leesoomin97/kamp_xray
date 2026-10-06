# Stage 1A.5 — Sequence Group Sensitivity Summary

Scope: 500 approved Stage 1 samples only. No fold was created and no model was trained.

## 1. Results for candidate thresholds

- 3s: groups=395, mean=1.2658, median=1, max=6, singleton groups=340 (86.1%), samples in groups≥2=32.0%.
- 6s: groups=150, mean=3.3333, median=2, max=6, singleton groups=44 (29.3%), samples in groups≥2=91.2%.
- 10s: groups=130, mean=3.8462, median=5, max=6, singleton groups=25 (19.2%), samples in groups≥2=95.0%.
- 30s: groups=120, mean=4.1667, median=6, max=7, singleton groups=15 (12.5%), samples in groups≥2=97.0%.
- 60s: groups=119, mean=4.2017, median=6, max=7, singleton groups=13 (10.9%), samples in groups≥2=97.4%.

## 2. Group-size distribution

- **VERIFIED** — group IDs are deterministic and connect adjacent timestamps only within the same machine and filename date.
- **VERIFIED** — 6 folder/filename-date conflict samples are isolated rather than assigned by an undocumented assumption.
- **STRONGLY SUPPORTED** — the large change between 3s and 6s reflects the observed concentration of 4–6 second gaps.

## 3. Large-group concentration

- 3s: max=6, top-5 groups=6.0% of samples
- 6s: max=6, top-5 groups=6.0% of samples
- 10s: max=6, top-5 groups=6.0% of samples
- 30s: max=7, top-5 groups=6.2% of samples
- 60s: max=7, top-5 groups=6.2% of samples

- **EXPLORATORY** — concentration is a split-feasibility diagnostic, not evidence that every temporal group is one physical product.
- **VERIFIED** — no tested threshold creates a group of 10 or more samples; the largest group is 7 and the five largest groups contain at most 6.2% of samples. Group size alone does not make any candidate infeasible.

## 4. Near-duplicate coverage

- **VERIFIED** — temporal grouping captures 0/3 flagged dHash pairs at every tested threshold because all flagged pairs cross dates.
- **STRONGLY SUPPORTED** — temporal grouping alone is insufficient to constrain these similarity pairs; a separate similarity-component rule would be required if the candidates remain accepted.
- **EXPLORATORY** — dHash is a global structural screen and may be influenced by colored markings; no sample was collapsed.

## 5. Machine/date balance implications

- **VERIFIED** — groups never cross machines or dates, so the existing date-frequency imbalance remains visible rather than being merged away.
- **VERIFIED** — at 6s the machine-level group counts are 42/35/73, and at 10s they are 36/33/61; machine 3 contributes disproportionately many groups relative to its 33.4% sample share.
- **VERIFIED** — the two largest dates contain 42.4% of samples but 35.3% of 6s groups and 33.8% of 10s groups. Date concentration therefore remains relevant even though group counts are less concentrated than sample counts.
- **EXPLORATORY** — thresholds change group counts unevenly when machines/dates contain different gap patterns; group-aware fold construction will need composition checks for machine, date, resolution, object count, bbox minimum side, and bbox area ratio.

## 6. Validation feasibility

- **EXPLORATORY** — 3s preserves many groups but fragments the common 4–6 second acquisition pattern.
- **STRONGLY SUPPORTED** — 6s and 10s remain plausible candidates: both capture dominant short runs without crossing date boundaries.
- **EXPLORATORY** — 30s and 60s provide conservative sensitivity bounds. They do not create impractically large groups here, but can join independent samples separated by tens of seconds without process documentation.

## 7. Thresholds that appear too permissive

- **EXPLORATORY** — 3s is likely too permissive against leakage because it breaks most 4–6 second runs.

## 8. Thresholds that appear too aggressive

- **EXPLORATORY** — no threshold is rejected by group size alone. However, 30s and especially 60s are semantically aggressive without documented production-cycle boundaries and add little beyond 10s (130 groups versus 120/119); use them as stress rules rather than default assumptions.

## 9. Plausible thresholds for the next validation-design step

1. **6 seconds** — matches the dominant observed short-gap structure while retaining more independent groups.
2. **10 seconds** — a more conservative alternative covering additional short runs.

Neither is a final split decision. Similarity-pair components must be handled separately from time groups.

## 10. Remaining uncertainty

- **UNRESOLVED** — whether seconds-apart frames show repeated views of one item or successive products.
- **UNRESOLVED** — documented machine cycle/lot boundaries are unavailable.
- **UNRESOLVED** — folder date versus filename date for six conflicted samples.
- **UNRESOLVED** — whether all three dHash candidates should be enforced as similarity components after method-sensitivity review.

# Stage 1C — Validation Design Audit Summary

## 1. Input integrity

- **VERIFIED** — 500 approved samples, 1,147 GT objects, matching Stage 1A/1B object keys, complete 6s/10s assignments, and complete required pre-model features.

## 2. Near-duplicate manual review

- **STRONGLY SUPPORTED** — all 3 dHash pairs are class A scene-family similarities after original-BMP side-by-side review of silhouette, internal grayscale structure, TXT bbox layout, and artifact layout.
- All 3 links are forced into validation components. Different dates prevent claiming the same physical item.

## 3. 6s vs 10s group components

- **VERIFIED** — after transitive similarity merging: 6s components=147, median/max size=2.0/13; 10s components=127, median/max size=5.0/13.
- **STRONGLY SUPPORTED** — 10s materially reduces cross-fold adjacency at or below 10 seconds while retaining enough components for 3–5 folds. The remaining three ≤10s boundaries involve deliberately isolated folder/filename-date conflict samples.

## 4. Candidate CV schemes

- **VERIFIED** — compared group-aware 3/4/5-fold CV for both 6s and 10s components. Ordinary random KFold was not used.
- Assignment is deterministic, unseeded, largest-first, and balances pre-model sample categories plus object-feature quartile counts without model outputs.

## 5. Sample balance

- **VERIFIED** — selected 10s/4-fold image counts: F1=124, F2=125, F3=126, F4=125.
- Selected maximum TV distances: machine=0.022, date=0.181, resolution=0.019, object-count=0.023.

## 6. Object/X-ray difficulty balance

- **VERIFIED** — selected max |SMD|: bbox minimum side=0.041, absolute contrast=0.059, robust CNR-like=0.054, background gradient=0.106.
- KS values are descriptive distribution diagnostics only, not hypothesis-test decisions.

## 7. Joint-condition coverage

- **VERIFIED** — selected scheme has zero-coverage cells=0 and sparse (<5) cells=3 across six Stage 1B exploratory joint conditions.
- **UNRESOLVED** — small + complex background has only 12 objects overall, so some fold estimates remain intrinsically unstable.

## 8. Leakage audit

- **VERIFIED** — selected temporal-component violations=0; forced-similarity violations=0; cross-fold adjacent pairs ≤10s=3.
- Separate same-day sequence groups in different folds are residual dependence risk or ordinary same-day sampling, not automatically confirmed leakage.

## 9. Machine/date stress-test feasibility

- **STRONGLY SUPPORTED** — leave-one-machine-out is feasible only as a secondary equipment/domain-shift test because machine, SN, resolution, date, and object conditions are confounded.
- **EXPLORATORY** — latest-five-date holdout is retained as a temporal stress test, not primary model-selection validation.

## 10. Selected primary validation strategy

- **STRONGLY SUPPORTED** — primary design: 10-second temporal components + all 3 manually validated similarity links + deterministic group-aware 4-fold CV.
- Rationale: stronger observed short-gap protection than 6s, no confirmed group violation, acceptable sample/object balance, and the best transparent rare-condition/feature-balance trade-off among the required candidates.
- This design is for future model/preprocessing/hyperparameter comparison and later threshold development using development data only.

## 11. Frozen fold assignment status

- **VERIFIED** — `01c_final_validation_folds.csv` is frozen before model results. No randomness was used; generator version/hash/timestamp are recorded separately.
- It must not be modified because later model results are inconvenient. Revisions require a new version and pre-model justification.

## 12. Remaining uncertainty

- Ten seconds is an evidence-based analytical boundary, not a documented machine cycle.
- Reviewed pairs are strong scene-family matches but not proven identical physical items.
- Rare joint conditions remain low-n, and group-aware folds cannot remove all same-day dependence or acquisition confounding.
- Product-edge distance uses a strongly supported analysis boundary without physical segmentation GT.

## 13. Recommended next stage

- Freeze artifact-control experiment specifications against these exact folds, then run a separately authorized artifact-control ablation before any fair detector baseline. External held-out data remains untouched.

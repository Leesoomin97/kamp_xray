# Stage 1A — Validation Leakage Risk Assessment

Scope: approved allowlist only (500 logical labeled samples). No split was created.

## Evidence used

- **VERIFIED** — 3 machine folders, 3 SN identifiers, and 24 filename dates.
- **VERIFIED** — 463 within-machine/SN/date adjacent gaps; 374 are at most 10 seconds.
- **VERIFIED** — cross-stem exact duplicate pairs: 0.
- **EXPLORATORY** — 16×16 nearest-resize dHash, Hamming ≤ 8: 3 near pairs; same-machine 100.0%, same-date 0.0%, <=10 s 0.0%.
- The dHash screen uses global palette-decoded image structure. Colored markings may influence the metric; they were not interpreted as foreign-object features.
- **VERIFIED** — object and acquisition distributions vary across groups descriptively; this does not establish causation.

## A. Random image split

- Leakage risk: **HIGH/EXPLORATORY** when adjacent burst frames or flagged near duplicates cross folds.
- Distribution-shift risk: low within the same observed pool, but may overestimate deployment generalization.
- Sample-size feasibility: high.
- Advantage: simple and retains all group coverage.
- Limitation: ignores time, SN, machine, and similarity dependencies.

## B. Machine-group split

- Leakage risk: lower for machine-specific acquisition signatures.
- Distribution-shift risk: potentially high because only 3 machine groups are present and their sample/object distributions differ.
- Sample-size feasibility: limited; results can depend strongly on which machine is held out.
- Advantage: tests cross-machine transfer.
- Limitation: machine is entangled with SN, resolution, date, and object-condition distributions.

## C. Date-group split

- Leakage risk: lower for same-day temporal sequences.
- Distribution-shift risk: moderate to high when date frequencies and object conditions are uneven.
- Sample-size feasibility: feasible across 24 dates, but small dates require grouping.
- Advantage: approximates forward/temporal generalization when ordered chronologically.
- Limitation: dates may be confounded with machine/SN and do not prove independent production lots.

## D. Sequence/group-aware split

- Leakage risk: lowest among candidates for adjacent bursts and flagged near-duplicate components when groups are kept intact.
- Distribution-shift risk: controllable if group assignment is stratified descriptively by machine/date/object count.
- Sample-size feasibility: likely feasible, but the exploratory 10-second sequence rule and dHash threshold require sensitivity checks.
- Advantage: directly targets observed dependence structure without claiming a production-lot meaning.
- Limitation: grouping thresholds are analytical constructs, not documented process identifiers.

## Recommended validation strategies to test next

1. **Sequence/similarity-group-aware grouped validation** with all flagged pairs/components kept in one fold; test sensitivity to 6/10/60-second grouping windows.
2. **Date-grouped temporal validation** as a distribution-shift stress test.
3. **Leave-one-machine-out analysis** as a secondary cross-equipment stress test, not the sole score because only 3 machine groups exist.

No final validation strategy or split has been selected.

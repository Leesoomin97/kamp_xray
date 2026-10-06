# Git Snapshot Synchronization Audit

## Scope and method

- Compared `origin/backup/local-20261007` at `ebc8b4e3b0ef2b2543e25da7aa173c14177aa737` with `origin/backup/kamp-20261007` at `c32dd20abf6f99464483603ce79e9a4b7bc74c90`.
- Comparison used fetched remote refs and Git tree/blob inspection only. No checkout, merge, reset, clean, training, inference, or main-branch update was performed.
- “Identical” below means the Git blob is identical. Where explicitly stated as “text-equivalent,” the text is identical after line-ending normalization even though the blob differs.

## Branch comparison summary

| Classification | Count | Interpretation |
|---|---:|---|
| Local snapshot files | 1,084 | Tracked files in the local snapshot |
| KAMP snapshot files | 20,557 | Includes 19,368 derived files under `outputs/processed_data/` |
| Local-only | 261 | Mostly later local EDA/reporting products and 13 analysis scripts |
| KAMP-only | 19,734 | Mostly materialized derived data, cache/routing output, and KAMP run metadata |
| Present in both, identical blob | 346 | Safe to take from either branch |
| Present in both, different blob | 477 | Dominated by LF/CRLF differences and environment-specific paths; core semantic conflicts are called out below |

The two commits were created three minutes apart, but commit time alone was not used to decide which content is newer or preferable.

## Local-only files

### Reproducible source code

The KAMP snapshot has no source file that is absent from the local snapshot. The following 13 scripts exist only in the local snapshot and should be retained:

- `src/aggregate_stage3_lowconf_oof.py`
- `src/analyze_crop_enhancement_feasibility.py`
- `src/analyze_detection_failure_factors.py`
- `src/analyze_small_object_observability.py`
- `src/analyze_stage3_image_safety.py`
- `src/analyze_stage6_prediction_fusion.py`
- `src/analyze_stage6_strategy_winners.py`
- `src/analyze_stage6_strategy_winners_m6.py`
- `src/analyze_stage6_subgroups.py`
- `src/audit_stage3_eda.py`
- `src/audit_stage6_candidate_local_redetection.py`
- `src/plot_stage3_eda.py`
- `src/run_stage3_safety_analysis.py`

Two source files present in both branches have substantive local improvements and should use the local version:

- `src/analyze_stage2a_errors.py`: adds projected/native pixel-size diagnostics, confidence/IoU feature relationships, B0-vs-B2 paired comparisons, localization/FP diagnostics, summary paths, and expanded manifests.
- `src/analyze_stage6_candidate_local_redetection.py`: adds policy-independent local-prediction identifiers, Windows Unicode-safe image loading, project-root-based conservative-image reconstruction, and explicit trajectory semantics used by the invariant audit.

### Evidence and reports

Local-only evidence is concentrated in:

| Path group | Local-only count | Important contents |
|---|---:|---|
| `outputs/tables/` | 105 | Stage 2 error analysis, Stage 3 safety tables, Stage 4/5 factor and feasibility analysis, Stage 6 fusion/strategy audit, Stage 7/final comparison tables |
| `outputs/figures/` | 82 | Later diagnostic/report figures |
| `outputs/stage3_lowconf/` | 32 | Low-confidence Stage 3 evaluation products |
| `outputs/eda/` | 21 | Master summary, Stage 2–7 summaries, final model/evidence reports, W&B isolation plan |
| `outputs/run_manifests/` | 4 | Later local aggregation/error-analysis manifests |

Key local-only report/evidence paths include:

- `outputs/eda/00_03_eda_master_summary_verified.md`
- `outputs/eda/02b_b2_visibility_error_analysis_summary.md`
- `outputs/eda/03_stage3_safety_analysis_summary.md`
- `outputs/eda/04_detection_failure_factor_summary.md`
- `outputs/eda/05_small_object_enhancement_summary.md`
- `outputs/eda/06_candidate_redetection_policy_audit_fold1.md`
- `outputs/eda/06_prediction_fusion_analysis_fold1.md`
- `outputs/eda/06_strategy_winner_analysis_fold1_m6.md`
- `outputs/eda/07_final_evidence_freeze.md`
- `outputs/eda/final_model_selection_summary.md`
- `outputs/eda/final_report_evidence_map.md`
- `outputs/tables/final_model_experiment_comparison.csv`
- `outputs/tables/final_threshold_comparison.csv`
- `outputs/tables/final_stage3_routing_summary.csv`

The root-level W&B export `wandb_export_2026-10-06T05_32_09.866+09_00.csv` is also local-only. It is tracking evidence, not the local metadata source of truth.

## KAMP-only files

### KAMP training metadata that should be brought into final main

The following final-performance Fold 1 runs have evaluation outputs in both branches, but their training metadata and run manifests are KAMP-only:

- `models/stage2a/b2_640_repro_fold1_v1/{args.yaml,experiment_metadata.json,results.csv}`
- `models/stage2a/b2_1024_fold1_v1/{args.yaml,experiment_metadata.json,results.csv}`
- `models/stage2a/b2_hardos_640_fold1_v1/{args.yaml,experiment_metadata.json,results.csv}`
- `models/stage2a/b2_box9_640_fold1_v1/{args.yaml,experiment_metadata.json,results.csv}`
- `outputs/run_manifests/b2_640_repro_fold1_v1.json`
- `outputs/run_manifests/b2_1024_fold1_v1.json`
- `outputs/run_manifests/b2_hardos_640_fold1_v1.json`
- `outputs/run_manifests/b2_box9_640_fold1_v1.json`

Additional KAMP-only Stage 6 provenance includes 9 preparation/sanity/candidate-redetection manifests, Stage 6 run metadata/args/results files, and the sanity evaluation directory `outputs/stage6_runs/stage6_m1_sanity_patch192_ov25_yolov8n_640_fold1_e2_v3/`. These are useful provenance records if the final repository intends to preserve the complete experiment history.

### KAMP-only material that should not be copied into final main by default

| Path | Count | Recommendation |
|---|---:|---|
| `outputs/processed_data/` | 19,368 | Do not track in source main; derived training material should be regenerated or packaged separately |
| `outputs/cache/` | 50 | Exclude as runtime cache |
| `outputs/stage6_routing/fold1/` | 248 | Retain externally if raw routing predictions are required; summary evidence already exists in tracked tables/EDA |
| `outputs/failed_runs/` | 4 | Optional debugging provenance, not final evidence |
| `stage2a_evidence_bundle.tar.gz.part.{aa,ab,ac}` | 3 files, 56,513,752 bytes total | Exclude transfer chunks; local `.gitignore` is designed to exclude them |

The KAMP snapshot also contains filename-encoding variants of two root HWPX documents. These should not be selected automatically until their intended Korean filenames and content hashes are manually reconciled.

## Files present in both but different

### Core code requested for audit

All five requested code files are exact blob matches. Either branch is equivalent:

| File | Result | Recommendation |
|---|---|---|
| `src/run_stage2a_training.py` | Identical | Either branch |
| `src/evaluate_stage2a_run.py` | Identical | Either branch |
| `src/log_stage2a_eval_to_wandb.py` | Identical | Either branch |
| `src/b2_final_support.py` | Identical | Either branch |
| `src/wandb_logging.py` | Identical | Either branch |

`configs/stage6/yolov8n_p2.yaml` is also an exact match. No KAMP-only source or config was found.

### Project-control files

| File | Difference | Recommended source |
|---|---|---|
| `.gitignore` | Local adds `.env*`, agent/credential directories, all `.venv*`, runtime `runs/`, `outputs/cache/`, `outputs/processed_data/`, and named staging directories. KAMP tracked archive chunks because its rules do not cover `.part.aa` naming. | Local |
| `requirements.txt` | Local adds direct dependency `pandas==3.0.6`; KAMP lacks it although project analysis code imports pandas. | Local |
| `AGENTS.md` | Different project-instruction blobs; the local file is the current governing instruction set used for the project audit. | Local, subject to a final human encoding/readability check |
| `README.md` | Exact match, but both copies are empty. | Either, then author a submission README separately |

### Evidence-file differences

- All 8 evaluation files under each of `b2_640_repro_fold1_v1`, `b2_1024_fold1_v1`, `b2_hardos_640_fold1_v1`, and `b2_box9_640_fold1_v1` are text-equivalent after line-ending normalization. Their metrics/content do not conflict.
- All 9 files under `outputs/tables/02b_b2_visibility_oof_v1/` are text-equivalent after line-ending normalization.
- `outputs/tables/08_b2_hardos_fold1_paired/` (5 files), `09_b2_oof_threshold040_analysis/` (10 files), and `10_stage3_positive_routing_simulation/` (4 files) are exact blob matches.
- Core Stage 1 tables checked (`00_stage1_allowlist.csv`, `01b_object_xray_features.csv`, and `01c_final_validation_folds.csv`) are text-equivalent after line-ending normalization.
- Many other common CSV files differ only because the local snapshot uses LF and the KAMP snapshot uses CRLF. This is a repository-normalization issue rather than evidence disagreement.
- The B2 fold run manifests are a real path/provenance conflict: KAMP manifests point to their original KAMP Stage 2 evaluation outputs, while the local versions were later updated by the Stage 3 low-confidence evaluation and point to `outputs/stage3_lowconf/...`. Neither should silently overwrite the other. Preserve the original training/evaluation manifest and the later Stage 3 evaluation manifest under separately versioned names before finalizing main.

## Latest experiment result presence

| Result family | Local snapshot | KAMP snapshot | Difference |
|---|---|---|---|
| `b2_640_repro_fold1_v1` | 8 evaluation files | Same 8 evaluation files plus args, experiment metadata, results, run manifest | Evaluation text-equivalent; provenance metadata KAMP-only |
| `b2_1024_fold1_v1` | 8 evaluation files | Same 8 evaluation files plus args, experiment metadata, results, run manifest | Evaluation text-equivalent; provenance metadata KAMP-only |
| `b2_hardos_640_fold1_v1` | 8 evaluation files | Same 8 evaluation files plus args, experiment metadata, results, run manifest | Evaluation text-equivalent; provenance metadata KAMP-only |
| `b2_box9_640_fold1_v1` | 8 evaluation files | Same 8 evaluation files plus args, experiment metadata, results, run manifest | Evaluation text-equivalent; provenance metadata KAMP-only |
| `outputs/tables/08_b2_hardos_fold1_paired/` | Present, 5 files | Present, 5 files | Exact match |
| `outputs/tables/09_b2_oof_threshold040_analysis/` | Present, 10 files | Present, 10 files | Exact match |
| `outputs/tables/10_stage3_positive_routing_simulation/` | Present, 4 files | Present, 4 files | Exact match |
| `outputs/tables/02b_b2_visibility_oof_v1/` | Present, 9 files | Present, 9 files | Text-equivalent after line-ending normalization |

No listed latest result family is wholly missing from either branch. The missing elements are specifically the KAMP-only training provenance files for the four Fold 1 final-performance runs.

## Recommended source for final main

### A. Adopt the local version

- Entire `src/` tree: it is a strict functional superset; no source file exists only on KAMP, the five requested training/evaluation/W&B files are identical, and two shared analysis files have verified local fixes/extensions.
- `.gitignore`, `requirements.txt`, and the current `AGENTS.md`.
- Local-only Stage 2–7 EDA, tables, figures, audit scripts, and final report evidence.
- Common CSV/JSON evidence from local where the only difference is normalized line endings.

### B. Adopt the KAMP version

- KAMP-only `args.yaml`, `experiment_metadata.json`, `results.csv`, and run manifests for `b2_640_repro_fold1_v1`, `b2_1024_fold1_v1`, `b2_hardos_640_fold1_v1`, and `b2_box9_640_fold1_v1`.
- KAMP-only Stage 6 training/preparation metadata and manifests when retaining full experimental provenance.
- Do **not** adopt KAMP-only processed data, cache, archive chunks, or filename-corrupted root documents into source main automatically.

### C. Either branch is equivalent

- The five explicitly audited core code files.
- `configs/stage6/yolov8n_p2.yaml` and other common config content.
- The paired HardOS, threshold-0.40, and positive-routing tables, which are exact matches.
- The four latest Fold 1 evaluation directories and B2 OOF tables after line-ending normalization.
- `README.md` is identical but empty; equivalence does not mean it is submission-ready.

### Conflicts requiring an explicit decision

1. Version the B2 fold training/original-evaluation manifests separately from later Stage 3 low-confidence evaluation manifests; do not choose by overwrite.
2. Normalize repository line endings with a deliberate `.gitattributes` policy only when preparing main; this audit did not modify content.
3. Resolve the two root HWPX filename-encoding variants manually.
4. Write a real README before submission; both snapshots currently contain an empty file.

## Git-excluded artifacts requiring separate preservation

The absence of `*.pt` from both Git snapshots is expected and correct under `.gitignore`. Metadata confirms that final B2-640 training generated the following checkpoints; the audit records the metadata-declared paths but does not infer that the physical files still exist:

- `models/stage2a/stage2b_b2_visibility_yolov8n_conservative_640_fold1_e30_v1/weights/best.pt`
- `models/stage2a/stage2b_b2_visibility_yolov8n_conservative_640_fold2_e30_v1/weights/best.pt`
- `models/stage2a/stage2b_b2_visibility_yolov8n_conservative_640_fold3_e30_v1/weights/best.pt`
- `models/stage2a/stage2b_b2_visibility_yolov8n_conservative_640_fold4_e30_v1/weights/best.pt`

All four are separate-backup targets because the selected B2 result is a four-fold OOF development result and each fold requires its own checkpoint for exact reproduction. The generic initialization asset `models/stage2a/pretrained/yolov8n.pt` is also excluded from Git; its recorded SHA-256 is `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`, so it can be independently verified or reacquired from its documented Ultralytics provenance.

Rejected/diagnostic experiment checkpoints may be archived separately if full forensic reproducibility is desired, but they are not required to operate the final selected B2-640 detector. `last.pt` is not a required final artifact when `best.pt` and metadata are preserved.

## Unresolved issues

- Physical availability and checksum verification of the four B2 `best.pt` files remain outside this Git-tree audit.
- The same-path B2 manifests encode different evaluation phases and need versioned reconciliation before main is assembled.
- The root HWPX names differ by apparent encoding and were not semantically resolved.
- `README.md` is empty in both snapshots and is not submission-ready.
- KAMP-only raw routing predictions may be useful for forensic reruns, but whether to retain them in main versus external evidence storage is a packaging decision.
- This audit file itself was created after both snapshot commits and therefore is not yet present in either remote snapshot branch.

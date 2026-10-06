# Submission Workspace Readiness

## 1. Consolidated source basis

- Integration branch basis: `origin/backup/local-20261007`.
- KAMP content is incorporated selectively; backup branches remain unchanged.
- Local source is preferred because it is a functional superset of KAMP source and contains the final Stage 3–7 analysis/audit code.
- W&B remains an optional tracking layer. Local CSV/JSON/YAML, run manifests, frozen-fold metadata, and checkpoint hashes remain the source of truth.

## 2. Files incorporated from KAMP

For each of the following runs, `args.yaml`, `experiment_metadata.json`, `results.csv`, and the corresponding run manifest were incorporated:

- `b2_640_repro_fold1_v1`
- `b2_1024_fold1_v1`
- `b2_hardos_640_fold1_v1`
- `b2_box9_640_fold1_v1`

No weight, processed image, cache, W&B cache, dataset, or archive chunk was incorporated.

The four original B2 fold evaluation manifests and their later Stage 3 low-confidence reanalysis forms are preserved under separate `*_original_evaluation_manifest.json` and `*_stage3_reanalysis_manifest.json` names.

## 3. Final B2 model definition

Final selected detector: YOLOv8n B2-640 with Conservative artifact control, `visibility` augmentation (`hsv_v=0.05`), image size 640, 30 epochs, batch 8, seed 42, full fine-tuning, loss weights box/dfl/cls = 7.5/1.5/0.5, and no oversampling.

Frozen 4-fold OOF at reporting confidence 0.25: TP 1121, FP 51, FN 26, Precision 0.9564846416382252, Recall 0.977332170880558, F1 0.966796032772747, AP50 0.96039, mAP50-95 0.38077.

## 4. Weight backup verification

- All four B2-640 fold `best.pt` backups exist locally.
- Each is 6,262,948 bytes.
- All four actual SHA-256 values match `final_b2_weights_backup/manifest.json`.
- The directory and all `*.pt` files are excluded from Git.
- Detailed hashes: `outputs/eda/final_weight_backup_verification.md`.

## 5. Pretrained initialization asset

- `models/stage2a/pretrained/yolov8n.pt`: PRESENT, 6,549,796 bytes.
- SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- The runner verifies this hash and does not download the asset automatically.

## 6. README status

README is now populated with the project/GT/artifact definition, final model, validation and OOF metrics, Stage 3 candidate policy, exact supported training/evaluation/aggregation CLI examples, directory roles, environment guidance, manifest roles, future test-inference status, and limitations.

## 7. Requirements status

`requirements.txt` remains curated rather than a blind environment freeze. It covers direct imports used by training/evaluation/analysis: PyTorch, Ultralytics, NumPy, Pandas, OpenCV, Matplotlib, Polars compatibility runtime, and optional W&B 0.30.0. SciPy and scikit-learn are not imported by current `src/` code and are not added. W&B is lazily used only when requested; inference/evaluation remains independent of W&B authentication.

The local CPU recovery environment and KAMP GPU environment use different PyTorch/CUDA builds. Platform-appropriate PyTorch installation remains necessary; run metadata records the exact environment used for each completed experiment.

## 8. Final evidence status

All requested evidence files are PRESENT:

- `outputs/tables/final_model_experiment_comparison.csv`
- `outputs/tables/final_threshold_comparison.csv`
- `outputs/tables/final_stage3_routing_summary.csv`
- `outputs/eda/final_model_selection_summary.md`
- `outputs/eda/final_report_evidence_map.md`
- `outputs/eda/git_snapshot_sync_audit.md`

## 9. Intentionally excluded from Git

- Original dataset and any nested `dataset/`
- `outputs/processed_data/`
- Runtime cache and `labels.cache`
- `*.pt`, `*.pth`, `*.onnx`, `*.engine`
- `final_b2_weights_backup/`
- W&B local cache and credentials
- ZIP/7z/tar archives and split chunks including `.part.aa`-style names
- Local virtual environments and temporary staging directories

KAMP-only raw Stage 6 routing predictions are not required for the final report claims because aggregate routing tables, rule comparison, rescued-FN profiles, and analysis summaries are already retained. They remain external forensic artifacts and are not incorporated automatically.

The two HWPX files have identical content blobs across snapshots despite filename-encoding differences: `758f1491...` (87,294 bytes) and `39a08d19...` (77,132 bytes). The correctly named local copies are retained; the mojibake KAMP path variants are not added. These templates are not required in the source-code submission package.

## 10. Remaining submission work

- Finalize the official test inference policy and supported entry point.
- Generate the required test-data prediction result file without using test data for model selection.
- Assemble and verify the source-code submission ZIP, including source, curated requirements, README, permitted training data/package references, and prediction result.
- Prepare the final result-report PDF.
- Prepare the presentation PPT/PDF.
- Perform an offline clean-environment smoke test of the finalized inference entry point once it exists.

## Readiness conclusion

The consolidated development/evidence workspace is ready for submission packaging preparation. It is not yet a complete submission because the official test inference entry point/policy and test prediction result file are unresolved.

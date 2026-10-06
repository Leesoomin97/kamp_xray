from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage2a_common import (
    ap_metrics,
    fixed_threshold_metrics,
    print_output_manifest,
    project_root_from_script,
    read_csv,
    resolve_from_root,
    save_run_manifest,
    verify_frozen_folds,
    write_csv,
    write_json,
)


def load_images(path: Path) -> dict[str, dict[str, Any]]:
    images = {}
    for row in read_csv(path):
        if row["stem"] in images:
            raise RuntimeError(f"Duplicate source stem inside evaluation: {row['stem']}")
        images[row["stem"]] = {
            "stem": row["stem"], "fold_id": int(row["fold_id"]), "width": int(row["width"]), "height": int(row["height"]),
            "gt_boxes": json.loads(row["gt_boxes_json"]), "predictions": json.loads(row["prediction_boxes_json"]),
        }
    return images


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    fold_rows, fold_by_stem = verify_frozen_folds(project_root)
    if len(args.fold_results) != 4:
        raise ValueError("Exactly four evaluated fold result directories are required")
    all_images: dict[str, dict[str, Any]] = {}
    fold_metrics: list[dict[str, Any]] = []
    object_rows: list[dict[str, str]] = []
    fp_rows: list[dict[str, str]] = []
    image_rows: list[dict[str, str]] = []
    patch_inputs: list[dict[str, str]] = []
    fusion_audit: list[dict[str, str]] = []
    observed_folds: set[int] = set()
    experiment = None
    report_confidence = None
    training_signature = None
    for supplied in args.fold_results:
        directory = resolve_from_root(project_root, supplied)
        required = [directory / name for name in ("evaluation_metadata.json", "metrics.json", "image_predictions.csv", "object_predictions.csv", "fp_predictions.csv")]
        if any(not path.is_file() for path in required):
            raise FileNotFoundError(f"Incomplete Stage 6 fold result: {directory}")
        metadata = json.loads(required[0].read_text(encoding="utf-8"))
        metrics = json.loads(required[1].read_text(encoding="utf-8"))
        if not bool(metadata.get("training_config", {}).get("evidence_eligible", False)):
            raise RuntimeError("Pipeline-sanity runs cannot enter a Stage 6 OOF evidence aggregate")
        fold = int(metadata["fold"])
        if fold in observed_folds:
            raise RuntimeError(f"Fold supplied twice: {fold}")
        observed_folds.add(fold)
        experiment = experiment or metadata["experiment"]
        report_confidence = report_confidence if report_confidence is not None else float(metadata["report_confidence"])
        signature = {key: metadata["training_config"].get(key) for key in (
            "experiment", "run_purpose", "evidence_eligible", "comparison_role", "max_optimizer_steps",
            "imgsz", "epochs", "batch", "seed", "freeze", "architecture",
            "preprocessing", "patch_setting", "augmentation",
        )}
        training_signature = training_signature or signature
        if metadata["experiment"] != experiment or float(metadata["report_confidence"]) != report_confidence or signature != training_signature:
            raise RuntimeError("Fold evaluations do not share the same Stage 6 experiment recipe")
        images = load_images(required[2])
        expected = {row["stem"] for row in fold_rows if int(row["fold_id"]) == fold}
        if set(images) != expected or any(item["fold_id"] != fold for item in images.values()):
            raise RuntimeError(f"Fold {fold} differs from frozen source-image validation assignment")
        if set(images) & set(all_images):
            raise RuntimeError("An OOF source image appears in more than one fold")
        all_images.update(images)
        fold_metrics.append(metrics)
        image_rows.extend(read_csv(required[2]))
        object_rows.extend(read_csv(required[3]))
        fp_rows.extend(read_csv(required[4]))
        if (directory / "patch_prediction_fusion_inputs.csv").is_file():
            patch_inputs.extend(read_csv(directory / "patch_prediction_fusion_inputs.csv"))
            fusion_audit.extend(read_csv(directory / "patch_duplicate_fusion_audit.csv"))
    if observed_folds != {1, 2, 3, 4} or set(all_images) != set(fold_by_stem):
        raise RuntimeError("Four-fold Stage 6 OOF coverage is incomplete")

    aggregate = {
        **fixed_threshold_metrics(all_images, report_confidence, 0.5), **ap_metrics(all_images),
        "experiment": experiment, "n_folds": 4, "n_images": len(all_images),
        "n_gt_objects": sum(len(item["gt_boxes"]) for item in all_images.values()),
        "coverage_verified_exactly_once": True, "final_safety_threshold_selected": False,
        "patch_fusion_policy_final": False,
        "localization_failure_count": sum(row["error_type"] == "LOCALIZATION_FAILURE" for row in object_rows),
        "low_confidence_fn_count": sum(row["error_type"] == "LOW_CONFIDENCE" for row in object_rows),
        "no_detection_fn_count": sum(row["error_type"] == "NO_DETECTION" for row in object_rows),
    }
    for subgroup in ("small", "low_contrast", "small_low_contrast"):
        n_objects = sum(int(float(row[f"{subgroup}_n"])) for row in fold_metrics)
        tp = sum(int(float(row[f"{subgroup}_tp"])) for row in fold_metrics)
        fn = sum(int(float(row[f"{subgroup}_fn"])) for row in fold_metrics)
        aggregate.update({
            f"{subgroup}_n": n_objects,
            f"{subgroup}_tp": tp,
            f"{subgroup}_fn": fn,
            f"{subgroup}_recall": tp / (tp + fn) if tp + fn else None,
        })
    for key in ("precision", "recall", "f1", "ap50", "map50_95", "tp", "fp", "fn"):
        values = [float(row[key]) for row in fold_metrics]
        aggregate.update({f"fold_{key}_mean": statistics.fmean(values), f"fold_{key}_std": statistics.pstdev(values),
                          f"fold_{key}_min": min(values), f"fold_{key}_max": max(values)})
    sweep = [fixed_threshold_metrics(all_images, round(step * 0.05, 2), 0.5) for step in range(20)]
    output = resolve_from_root(project_root, args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"OOF output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fold_metrics.sort(key=lambda row: int(row["fold_id"]))
    files = {
        "06_fold_metrics.csv": fold_metrics,
        "06_image_predictions.csv": image_rows,
        "06_object_predictions.csv": object_rows,
        "06_error_types.csv": object_rows,
        "06_false_positive_predictions.csv": fp_rows,
        "06_confidence_sweep.csv": sweep,
        "06_oof_summary.csv": [aggregate],
    }
    if patch_inputs:
        files["06_patch_prediction_fusion_inputs.csv"] = patch_inputs
        files["06_patch_duplicate_fusion_audit.csv"] = fusion_audit
    generated: list[Path] = []
    for name, rows in files.items():
        path = output / name
        default_fields = ["stem"] if not rows else list(rows[0])
        write_csv(path, default_fields, rows)
        generated.append(path)
    write_json(output / "06_oof_summary.json", aggregate)
    write_json(output / "aggregation_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "experiment": experiment,
        "fold_result_directories": [str(resolve_from_root(project_root, path)) for path in args.fold_results],
        "training_signature": training_signature, "frozen_fold_sha256": "963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8",
        "coverage": "500 frozen source images exactly once", "model_performance_used_to_change_folds": False,
    })
    generated += [output / "06_oof_summary.json", output / "aggregation_metadata.json"]
    manifest = save_run_manifest(project_root, args.run_name, {
        "status": "COMPLETE", "stage": "Stage 6 OOF aggregation", "run_name": args.run_name,
        "project_root": str(project_root), "experiment": experiment, "folds": [1, 2, 3, 4],
        "ended_utc": datetime.now(timezone.utc).isoformat(), "generated_metric_paths": [str(path.resolve()) for path in generated],
    })
    print_output_manifest(stage="Stage 6 OOF aggregation", run_name=args.run_name, project_root=project_root,
        files=[manifest, *generated], directories=[output], downloads=[manifest, *generated],
        next_action="Run analyze_stage6_subgroups.py only after reviewing OOF coverage and metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate exactly four completed Stage 6 fold evaluations.")
    parser.add_argument("--fold-results", type=Path, nargs=4, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        root = args.project_root.resolve()
        manifest = save_run_manifest(root, args.run_name, {
            "status": "FAILED", "stage": "Stage 6 OOF aggregation", "run_name": args.run_name,
            "project_root": str(root), "ended_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        print_output_manifest(stage="Stage 6 OOF aggregation (failed)", run_name=args.run_name, project_root=root,
            files=[manifest], downloads=[manifest], next_action="Return the manifest and error; do not run subgroup analysis.")
        raise

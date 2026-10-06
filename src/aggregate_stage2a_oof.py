from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage2a_common import (
    ap_metrics, fixed_threshold_metrics, print_output_manifest, project_root_from_script,
    read_csv, resolve_from_root, save_run_manifest, verify_frozen_folds, write_csv, write_json,
)


def load_images(path: Path) -> dict[str, dict[str, Any]]:
    images = {}
    for row in read_csv(path):
        stem = row["stem"]
        if stem in images:
            raise RuntimeError(f"Duplicate stem inside evaluation: {stem}")
        images[stem] = {"stem": stem, "fold_id": int(row["fold_id"]), "width": int(row["width"]), "height": int(row["height"]),
                        "gt_boxes": json.loads(row["gt_boxes_json"]), "predictions": json.loads(row["prediction_boxes_json"])}
    return images


def main(args: argparse.Namespace) -> None:
    workspace = args.project_root.resolve(); fold_rows, fold_by_stem = verify_frozen_folds(workspace)
    if len(args.fold_results) != 4:
        raise ValueError("Exactly four evaluated fold result directories are required")
    all_images: dict[str, dict[str, Any]] = {}; fold_metrics = []; object_rows = []; fp_rows = []; image_rows = []
    observed_folds: set[int] = set(); representation = None; imgsz = None; report_confidence = None; training_config = None
    for directory in args.fold_results:
        result_dir = resolve_from_root(workspace, directory)
        required = [result_dir / name for name in ("evaluation_metadata.json", "metrics.json", "image_predictions.csv", "object_predictions.csv", "fp_predictions.csv")]
        if any(not path.is_file() for path in required):
            raise FileNotFoundError(f"Incomplete evaluated fold directory: {result_dir}")
        metadata = json.loads(required[0].read_text(encoding="utf-8")); metrics = json.loads(required[1].read_text(encoding="utf-8"))
        fold = int(metadata["fold"])
        if fold in observed_folds:
            raise RuntimeError(f"Fold supplied twice: {fold}")
        observed_folds.add(fold)
        representation = representation or metadata["representation"]; imgsz = imgsz or int(metadata["imgsz"])
        report_confidence = report_confidence if report_confidence is not None else float(metadata["report_confidence"])
        training_config = training_config or metadata.get("training_config")
        if metadata["representation"] != representation or int(metadata["imgsz"]) != imgsz or float(metadata["report_confidence"]) != report_confidence:
            raise RuntimeError("Fold evaluations do not share representation, image size, and reporting threshold")
        if metadata.get("training_config") != training_config:
            raise RuntimeError("Fold evaluations do not share the same training recipe")
        images = load_images(required[2])
        expected = {r["stem"] for r in fold_rows if int(r["fold_id"]) == fold}
        if set(images) != expected or any(item["fold_id"] != fold for item in images.values()):
            raise RuntimeError(f"Fold {fold} evaluation coverage differs from frozen validation assignment")
        overlap = set(all_images) & set(images)
        if overlap:
            raise RuntimeError(f"OOF sample appears twice: {sorted(overlap)[:3]}")
        all_images.update(images); fold_metrics.append(metrics)
        object_rows.extend(read_csv(required[3])); fp_rows.extend(read_csv(required[4])); image_rows.extend(read_csv(required[2]))
    if observed_folds != {1, 2, 3, 4} or set(all_images) != set(fold_by_stem):
        raise RuntimeError("Four-fold OOF coverage is incomplete")

    aggregate = {**fixed_threshold_metrics(all_images, report_confidence, 0.5), **ap_metrics(all_images),
                 "representation": representation, "imgsz": imgsz, "n_folds": 4, "n_images": len(all_images),
                 "n_gt_objects": sum(len(item["gt_boxes"]) for item in all_images.values()),
                 "coverage_verified_exactly_once": True, "final_safety_threshold_selected": False}
    metrics_to_summarize = ("precision", "recall", "f1", "ap50", "map50_95", "tp", "fp", "fn")
    variation = {}
    for key in metrics_to_summarize:
        values = [float(row[key]) for row in fold_metrics]
        variation[f"fold_{key}_mean"] = statistics.fmean(values)
        variation[f"fold_{key}_std"] = statistics.pstdev(values)
        variation[f"fold_{key}_min"] = min(values); variation[f"fold_{key}_max"] = max(values)
    summary = {**aggregate, **variation}
    sweep = [fixed_threshold_metrics(all_images, round(step * 0.05, 2), 0.5) for step in range(20)]
    output = resolve_from_root(workspace, args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"OOF output directory is not empty: {output}")
    fold_metrics = sorted(fold_metrics, key=lambda row: int(row["fold_id"]))
    write_csv(output / "02a_fold_metrics.csv", list(fold_metrics[0]), fold_metrics)
    write_csv(output / "02a_image_predictions.csv", list(image_rows[0]), image_rows)
    write_csv(output / "02a_object_predictions.csv", list(object_rows[0]), object_rows)
    write_csv(output / "02a_error_types.csv", list(object_rows[0]), object_rows)
    write_csv(output / "02a_false_positive_predictions.csv", list(fp_rows[0]) if fp_rows else ["stem", "fold_id", "prediction_index", "prediction_bbox_json", "confidence", "nearest_gt_iou", "fp_status"], fp_rows)
    write_csv(output / "02a_confidence_sweep.csv", list(sweep[0]), sweep)
    write_csv(output / "02a_oof_summary.csv", list(summary), [summary]); write_json(output / "02a_oof_summary.json", summary)
    write_json(output / "aggregation_metadata.json", {"representation": representation, "imgsz": imgsz,
                                                         "fold_result_directories": [str(resolve_from_root(workspace, path)) for path in args.fold_results],
                                                         "training_config": training_config,
                                                         "frozen_fold_sha256": "963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8",
                                                         "coverage": "500 frozen samples exactly once", "model_performance_used_to_change_folds": False})
    generated = [output / name for name in ("02a_fold_metrics.csv", "02a_image_predictions.csv", "02a_object_predictions.csv",
                 "02a_error_types.csv", "02a_false_positive_predictions.csv", "02a_confidence_sweep.csv",
                 "02a_oof_summary.csv", "02a_oof_summary.json", "aggregation_metadata.json")]
    manifest = {"status": "COMPLETE", "stage": "Stage 2A OOF aggregation", "run_name": args.run_name,
                "project_root": str(workspace), "representation": representation, "imgsz": imgsz,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "folds": [1, 2, 3, 4], "generated_checkpoint_paths": [],
                "generated_prediction_paths": [str(path.resolve()) for path in generated if "prediction" in path.name or "error_types" in path.name],
                "generated_metric_paths": [str(path.resolve()) for path in generated if path.is_file()],
                "success": True}
    manifest_path = save_run_manifest(workspace, args.run_name, manifest)
    print_output_manifest(stage="Stage 2A OOF aggregation", run_name=args.run_name, project_root=workspace,
                          files=[manifest_path, *generated], directories=[output],
                          downloads=[manifest_path, *generated],
                          next_action="Run Stage 2A error analysis locally only after verifying the OOF summary.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate four completed Stage 2A fold evaluations; never trains.")
    parser.add_argument("--fold-results", type=Path, nargs=4, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project-root", "--workspace", dest="project_root", type=Path, default=project_root_from_script())
    parsed = parser.parse_args()
    try:
        main(parsed)
    except BaseException as exc:
        root = parsed.project_root.resolve()
        failure = {"status": "FAILED", "stage": "Stage 2A OOF aggregation", "run_name": parsed.run_name,
                   "project_root": str(root), "ended_utc": datetime.now(timezone.utc).isoformat(),
                   "error_type": type(exc).__name__, "error": str(exc)}
        manifest_path = save_run_manifest(root, parsed.run_name, failure)
        print_output_manifest(stage="Stage 2A OOF aggregation (failed)", run_name=parsed.run_name, project_root=root,
                              files=[manifest_path], downloads=[manifest_path],
                              next_action="Return the manifest and error text; do not run error analysis.")
        raise

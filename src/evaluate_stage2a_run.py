from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage2a_common import (
    ap_metrics, box_iou, fixed_threshold_metrics, match_image, parse_yolo_labels, png_size,
    print_output_manifest, project_root_from_script, read_csv, resolve_from_root,
    save_run_manifest, validate_fold_dataset, write_csv, write_json,
)


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "experiment_metadata.json"
    if not path.is_file():
        raise RuntimeError(f"Completed experiment metadata missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE":
        raise RuntimeError("Only a completed run can be evaluated")
    return value


def main(args: argparse.Namespace) -> None:
    workspace = args.project_root.resolve(); run_dir = resolve_from_root(workspace, args.run_dir)
    metadata = load_run_metadata(run_dir)
    representation, fold, imgsz = str(metadata["representation"]), int(metadata["fold"]), int(metadata["imgsz"])
    _, expected_stems = validate_fold_dataset(workspace, representation, fold)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"best.pt missing: {best}")
    integrity = read_csv(workspace / "outputs" / "tables" / "02a_experiment_integrity.csv")
    role = "ARTIFACT_INPAINT_CONSERVATIVE" if representation == "conservative" else "ARTIFACT_LOCAL_INTERPOLATION"
    selected = {r["stem"]: r for r in integrity if r["representation"] == role and r["fold_id"] == str(fold)}
    if set(selected) != set(expected_stems):
        raise RuntimeError("Evaluation samples differ from frozen fold")

    run_name = str(metadata["run_name"])
    output = resolve_from_root(workspace, args.output_dir) if args.output_dir else workspace / "outputs" / "stage2a_runs" / run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("YOLO_CONFIG_DIR", str(workspace / "outputs" / "cache" / "ultralytics"))
    from ultralytics import YOLO

    image_paths = [workspace / Path(selected[stem]["image_path"]) for stem in expected_stems]
    model = YOLO(str(best))
    results = model.predict(source=[str(path) for path in image_paths], imgsz=imgsz, conf=args.conf_floor,
                            iou=args.nms_iou, max_det=args.max_det, device=args.device,
                            stream=True, verbose=False, save=False, save_txt=False,
                            save_conf=False, save_crop=False, project=str(output.parent),
                            name=output.name, exist_ok=True)
    images: dict[str, dict[str, Any]] = {}
    for result in results:
        stem = Path(result.path).stem
        if stem not in selected or stem in images:
            raise RuntimeError(f"Unexpected or duplicate inference result: {stem}")
        image_path = workspace / Path(selected[stem]["image_path"]); label_path = workspace / Path(selected[stem]["label_path"])
        width, height = png_size(image_path)
        gts = parse_yolo_labels(label_path, width, height)
        predictions = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().tolist(); confidence = result.boxes.conf.cpu().tolist()
            for box, score in zip(xyxy, confidence):
                predictions.append({"bbox": [float(v) for v in box], "confidence": float(score)})
        images[stem] = {"stem": stem, "fold_id": fold, "width": width, "height": height, "gt_boxes": gts,
                        "predictions": sorted(predictions, key=lambda p: -p["confidence"])}
    if set(images) != set(expected_stems):
        raise RuntimeError("Inference did not return every frozen validation sample exactly once")

    image_rows = [{"stem": stem, "fold_id": fold, "width": item["width"], "height": item["height"],
                   "gt_boxes_json": json.dumps(item["gt_boxes"], separators=(",", ":")),
                   "prediction_boxes_json": json.dumps(item["predictions"], separators=(",", ":")),
                   "prediction_count_at_floor": len(item["predictions"]), "confidence_floor": args.conf_floor}
                  for stem, item in sorted(images.items())]
    object_rows: list[dict[str, Any]] = []; fp_rows: list[dict[str, Any]] = []
    for stem, item in sorted(images.items()):
        gts, predictions = item["gt_boxes"], item["predictions"]
        matches, fps = match_image(gts, predictions, args.report_confidence, 0.5)
        for object_id, gt in enumerate(gts):
            if object_id in matches:
                prediction_index, overlap = matches[object_id]; prediction = predictions[prediction_index]
                status, error_type = "TP", "MATCHED"
            else:
                ranked = sorted(((box_iou(p["bbox"], gt), float(p["confidence"]), p) for p in predictions), reverse=True, key=lambda x: (x[0], x[1]))
                overlap, score, prediction = ranked[0] if ranked else (0.0, 0.0, None)
                status = "FN"
                if prediction is None:
                    error_type = "NO_DETECTION"
                elif overlap >= 0.5 and score < args.report_confidence:
                    error_type = "LOW_CONFIDENCE"
                elif score >= args.report_confidence and overlap >= 0.1:
                    error_type = "LOCALIZATION_FAILURE"
                else:
                    error_type = "NO_DETECTION"
            object_rows.append({"stem": stem, "object_id": object_id, "fold_id": fold,
                                "gt_bbox_json": json.dumps(gt, separators=(",", ":")),
                                "matched_prediction": prediction is not None and status == "TP",
                                "prediction_confidence": "" if prediction is None else prediction["confidence"],
                                "prediction_bbox_json": "" if prediction is None else json.dumps(prediction["bbox"], separators=(",", ":")),
                                "iou": overlap, "status": status, "localization_status": "IOU_GE_0.50" if status == "TP" else "NOT_MATCHED",
                                "error_type": error_type, "report_confidence_threshold": args.report_confidence})
        matched_prediction_indexes = {prediction_index for prediction_index, _ in matches.values()}
        for prediction_index in fps:
            prediction = predictions[prediction_index]
            nearest_iou = max((box_iou(prediction["bbox"], gt) for gt in gts), default=0.0)
            if nearest_iou >= 0.5:
                fp_type = "DUPLICATE_EXCESS"
            elif nearest_iou >= 0.1:
                fp_type = "LOCALIZATION_FALSE_POSITIVE"
            else:
                fp_type = "BACKGROUND_FALSE_POSITIVE"
            fp_rows.append({"stem": stem, "fold_id": fold, "prediction_index": prediction_index,
                            "prediction_bbox_json": json.dumps(prediction["bbox"], separators=(",", ":")),
                            "confidence": prediction["confidence"], "nearest_gt_iou": nearest_iou,
                            "fp_status": fp_type, "report_confidence_threshold": args.report_confidence,
                            "was_matched": prediction_index in matched_prediction_indexes})

    fixed = fixed_threshold_metrics(images, args.report_confidence, 0.5); ap = ap_metrics(images)
    metrics = {**fixed, **ap, "representation": representation, "fold_id": fold, "imgsz": imgsz,
               "n_images": len(images), "n_gt_objects": sum(len(i["gt_boxes"]) for i in images.values()),
               "confidence_floor": args.conf_floor, "nms_iou": args.nms_iou,
               "note": "Reporting threshold is fixed for confusion counts only and is not an operational threshold."}
    sweep = [fixed_threshold_metrics(images, round(step * 0.05, 2), 0.5) for step in range(20)]
    write_csv(output / "image_predictions.csv", list(image_rows[0]), image_rows)
    write_csv(output / "object_predictions.csv", list(object_rows[0]), object_rows)
    write_csv(output / "fp_predictions.csv", list(fp_rows[0]) if fp_rows else ["stem", "fold_id", "prediction_index", "prediction_bbox_json", "confidence", "nearest_gt_iou", "fp_status", "report_confidence_threshold", "was_matched"], fp_rows)
    write_csv(output / "error_types.csv", list(object_rows[0]), object_rows)
    write_csv(output / "confidence_sweep.csv", list(sweep[0]), sweep)
    write_csv(output / "metrics.csv", list(metrics), [metrics]); write_json(output / "metrics.json", metrics)
    training_config = {
        key: metadata.get(key)
        for key in (
            "epochs", "batch", "freeze", "seed", "workers", "model_sha256",
            "model_architecture", "pretrained_weight_identifier",
            "augmentation_profile", "augmentation",
        )
    }
    evaluation_metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
                                                       "model": str(best), "representation": representation, "fold": fold,
                                                       "imgsz": imgsz, "confidence_floor": args.conf_floor,
                                                       "report_confidence": args.report_confidence, "iou_match": 0.5,
                                                       "training_config": training_config,
                                                       "final_safety_threshold_selected": False}
    evaluation_metadata_path = output / "evaluation_metadata.json"
    write_json(evaluation_metadata_path, evaluation_metadata)
    metric_paths = [output / name for name in ("metrics.csv", "metrics.json", "confidence_sweep.csv")]
    prediction_paths = [output / name for name in ("image_predictions.csv", "object_predictions.csv", "fp_predictions.csv", "error_types.csv")]
    run_manifest = {**metadata, "status": "EVALUATION_COMPLETE", "stage": "Stage 2A evaluation",
                    "evaluation_ended_utc": datetime.now(timezone.utc).isoformat(),
                    "generated_checkpoint_paths": [str(best.resolve())],
                    "generated_prediction_paths": [str(path.resolve()) for path in prediction_paths],
                    "generated_metric_paths": [str(path.resolve()) for path in metric_paths],
                    "evaluation_metadata_path": str(evaluation_metadata_path.resolve())}
    manifest_path = save_run_manifest(workspace, run_name, run_manifest)
    print_output_manifest(stage="Stage 2A evaluation", run_name=run_name, project_root=workspace,
                          files=[manifest_path, evaluation_metadata_path, *metric_paths, *prediction_paths],
                          directories=[output], models=[best],
                          downloads=[manifest_path, evaluation_metadata_path, *metric_paths, *prediction_paths],
                          next_action="Review this one-fold evaluation; do not start another fold automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate exactly one completed Stage 2A run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--report-confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project-root", "--workspace", dest="project_root", type=Path, default=project_root_from_script())
    parsed = parser.parse_args()
    try:
        main(parsed)
    except BaseException as exc:
        root = parsed.project_root.resolve(); run_name = parsed.run_dir.name
        manifest_path = root / "outputs" / "run_manifests" / f"{run_name}.json"
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        failure = {**previous, "status": "EVALUATION_FAILED", "stage": "Stage 2A evaluation",
                   "run_name": run_name, "project_root": str(root),
                   "evaluation_ended_utc": datetime.now(timezone.utc).isoformat(),
                   "evaluation_error_type": type(exc).__name__, "evaluation_error": str(exc)}
        manifest_path = save_run_manifest(root, run_name, failure)
        print_output_manifest(stage="Stage 2A evaluation (failed)", run_name=run_name, project_root=root,
                              files=[manifest_path], downloads=[manifest_path],
                              next_action="Return the run manifest and error text; do not start another experiment.")
        raise

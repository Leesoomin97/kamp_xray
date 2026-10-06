from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage2a_common import (
    ap_metrics,
    box_iou,
    fixed_threshold_metrics,
    match_image,
    parse_yolo_labels,
    png_size,
    print_output_manifest,
    project_root_from_script,
    read_csv,
    resolve_from_root,
    save_run_manifest,
    write_csv,
    write_json,
)
from stage6_common import STAGE6_EXPERIMENTS, source_gt_by_stem, stage6_evaluation_sources


def load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "experiment_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETE" or metadata.get("stage") != "Stage 6 training":
        raise RuntimeError("Only a completed Stage 6 training run can be evaluated")
    return metadata


def source_inputs(project_root: Path, validation_stems: list[str]) -> dict[str, dict[str, Any]]:
    allow = source_gt_by_stem(project_root)
    integrity = {
        row["stem"]: row
        for row in read_csv(project_root / "outputs" / "tables" / "02a_experiment_integrity.csv")
        if row["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE"
    }
    images: dict[str, dict[str, Any]] = {}
    for stem in validation_stems:
        image_path = project_root / Path(integrity[stem]["image_path"])
        label_path = project_root / Path(integrity[stem]["label_path"])
        width, height = png_size(image_path)
        images[stem] = {
            "stem": stem,
            "width": width,
            "height": height,
            "gt_boxes": parse_yolo_labels(label_path, width, height),
            "predictions": [],
        }
    return images


def reference_nms(candidates: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    order = sorted(range(len(candidates)), key=lambda index: (-float(candidates[index]["confidence"]), index))
    kept: list[int] = []
    audit: list[dict[str, Any]] = []
    for index in order:
        suppressor = next((kept_index for kept_index in kept if box_iou(candidates[index]["bbox"], candidates[kept_index]["bbox"]) >= threshold), None)
        if suppressor is None:
            kept.append(index)
        audit.append({**candidates[index], "candidate_index": index, "kept_after_reference_nms": suppressor is None,
                      "suppressed_by_candidate_index": "" if suppressor is None else suppressor})
    return [candidates[index] for index in kept], audit


def object_and_fp_rows(images: dict[str, dict[str, Any]], fold: int, confidence: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    for stem, item in sorted(images.items()):
        gts, predictions = item["gt_boxes"], item["predictions"]
        matches, fps = match_image(gts, predictions, confidence, 0.5)
        for object_id, gt in enumerate(gts):
            if object_id in matches:
                prediction_index, overlap = matches[object_id]
                prediction = predictions[prediction_index]
                status, error_type = "TP", "MATCHED"
            else:
                ranked = sorted(((box_iou(prediction["bbox"], gt), float(prediction["confidence"]), prediction) for prediction in predictions), reverse=True, key=lambda row: (row[0], row[1]))
                overlap, score, prediction = ranked[0] if ranked else (0.0, 0.0, None)
                status = "FN"
                if prediction is None:
                    error_type = "NO_DETECTION"
                elif overlap >= 0.5 and score < confidence:
                    error_type = "LOW_CONFIDENCE"
                elif score >= confidence and overlap >= 0.1:
                    error_type = "LOCALIZATION_FAILURE"
                else:
                    error_type = "NO_DETECTION"
            object_rows.append({
                "stem": stem,
                "object_id": object_id,
                "fold_id": fold,
                "gt_bbox_json": json.dumps(gt, separators=(",", ":")),
                "matched_prediction": prediction is not None and status == "TP",
                "prediction_confidence": "" if prediction is None else prediction["confidence"],
                "prediction_bbox_json": "" if prediction is None else json.dumps(prediction["bbox"], separators=(",", ":")),
                "iou": overlap,
                "status": status,
                "localization_status": "IOU_GE_0.50" if status == "TP" else "NOT_MATCHED",
                "error_type": error_type,
                "report_confidence_threshold": confidence,
            })
        for prediction_index in fps:
            prediction = predictions[prediction_index]
            nearest = max((box_iou(prediction["bbox"], gt) for gt in gts), default=0.0)
            fp_type = "DUPLICATE_EXCESS" if nearest >= 0.5 else "LOCALIZATION_FALSE_POSITIVE" if nearest >= 0.1 else "BACKGROUND_FALSE_POSITIVE"
            fp_rows.append({
                "stem": stem,
                "fold_id": fold,
                "prediction_index": prediction_index,
                "prediction_bbox_json": json.dumps(prediction["bbox"], separators=(",", ":")),
                "confidence": prediction["confidence"],
                "nearest_gt_iou": nearest,
                "fp_status": fp_type,
                "report_confidence_threshold": confidence,
            })
    return object_rows, fp_rows


def fold_subgroup_metrics(project_root: Path, object_rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = {
        (row["stem"], int(row["object_id"])): row
        for row in read_csv(project_root / "outputs" / "tables" / "01b_object_xray_features.csv")
    }
    definitions = {
        "small": lambda row: float(row["bbox_min_side_px"]) <= 8,
        "low_contrast": lambda row: float(row["absolute_median_difference"]) <= 4,
        "small_low_contrast": lambda row: float(row["bbox_min_side_px"]) <= 8 and float(row["absolute_median_difference"]) <= 4,
    }
    result: dict[str, Any] = {}
    for name, predicate in definitions.items():
        selected = [row for row in object_rows if predicate(features[(row["stem"], int(row["object_id"]))])]
        tp = sum(row["status"] == "TP" for row in selected)
        fn = sum(row["status"] == "FN" for row in selected)
        result[f"{name}_n"] = len(selected)
        result[f"{name}_tp"] = tp
        result[f"{name}_fn"] = fn
        result[f"{name}_recall"] = tp / (tp + fn) if tp + fn else None
    return result


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    run_dir = resolve_from_root(project_root, args.run_dir)
    metadata = load_metadata(run_dir)
    experiment = str(metadata["experiment"])
    fold = int(metadata["fold"])
    imgsz = int(metadata["imgsz"])
    sources, validation_stems = stage6_evaluation_sources(project_root, experiment, fold)
    if {row.get("source_stem", row.get("stem")) for row in sources} != set(validation_stems):
        raise RuntimeError("Evaluation source images differ from the frozen validation fold")
    if any(row.get("materialization_status", "COMPLETE") != "COMPLETE" for row in sources):
        raise RuntimeError("Stage 6 evaluation source is not fully materialized")
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    run_name = str(metadata["run_name"])
    output = resolve_from_root(project_root, args.output_dir) if args.output_dir else project_root / "outputs" / "stage6_runs" / run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    images = source_inputs(project_root, validation_stems)
    is_patch = STAGE6_EXPERIMENTS[experiment]["dataset_kind"] == "patch"
    if is_patch:
        path_key = "patch_image_path"
        source_by_input_stem = {row["patch_id"]: row for row in sources}
    else:
        path_key = "image_path" if "image_path" in sources[0] else "output_path"
        source_by_input_stem = {row["stem"]: row for row in sources}
    input_paths = [project_root / Path(row[path_key]) for row in sources]

    os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / "outputs" / "cache" / "ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(best))
    results = model.predict(
        source=[str(path) for path in input_paths], imgsz=imgsz, conf=args.conf_floor,
        iou=args.nms_iou, max_det=args.max_det, device=args.device, stream=True,
        verbose=False, save=False, save_txt=False, save_conf=False, save_crop=False,
        project=str(output.parent), name=output.name, exist_ok=True,
    )
    raw_patch_rows: list[dict[str, Any]] = []
    observed_inputs: set[str] = set()
    direct_predictions: dict[str, list[dict[str, Any]]] = {stem: [] for stem in validation_stems}
    for result in results:
        input_stem = Path(result.path).stem
        if input_stem not in source_by_input_stem or input_stem in observed_inputs:
            raise RuntimeError(f"Unexpected or duplicate inference result: {input_stem}")
        observed_inputs.add(input_stem)
        source = source_by_input_stem[input_stem]
        source_stem = source.get("source_stem", source.get("stem"))
        boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
        scores = [] if result.boxes is None else result.boxes.conf.cpu().tolist()
        for prediction_index, (box, score) in enumerate(zip(boxes, scores)):
            translated = [float(value) for value in box]
            if is_patch:
                translated = [translated[0] + float(source["x0"]), translated[1] + float(source["y0"]),
                              translated[2] + float(source["x0"]), translated[3] + float(source["y0"])]
                translated[0] = max(0.0, min(translated[0], images[source_stem]["width"]))
                translated[2] = max(0.0, min(translated[2], images[source_stem]["width"]))
                translated[1] = max(0.0, min(translated[1], images[source_stem]["height"]))
                translated[3] = max(0.0, min(translated[3], images[source_stem]["height"]))
                raw_patch_rows.append({
                    "source_stem": source_stem, "patch_id": input_stem, "fold_id": fold,
                    "patch_x0": source["x0"], "patch_y0": source["y0"], "patch_prediction_index": prediction_index,
                    "patch_bbox_json": json.dumps([float(value) for value in box], separators=(",", ":")),
                    "source_bbox_json": json.dumps(translated, separators=(",", ":")), "confidence": float(score),
                })
            direct_predictions[source_stem].append({"bbox": translated, "confidence": float(score), "patch_id": input_stem if is_patch else ""})
    if observed_inputs != set(source_by_input_stem):
        raise RuntimeError("Inference did not return every validation input exactly once")

    fusion_audit_rows: list[dict[str, Any]] = []
    for stem in validation_stems:
        if is_patch and args.patch_fusion == "reference_nms":
            fused, audit = reference_nms(direct_predictions[stem], args.fusion_iou)
            images[stem]["predictions"] = fused
            for row in audit:
                fusion_audit_rows.append({
                    "source_stem": stem, "candidate_index": row["candidate_index"], "patch_id": row["patch_id"],
                    "source_bbox_json": json.dumps(row["bbox"], separators=(",", ":")), "confidence": row["confidence"],
                    "kept_after_reference_nms": row["kept_after_reference_nms"],
                    "suppressed_by_candidate_index": row["suppressed_by_candidate_index"], "fusion_iou": args.fusion_iou,
                })
        elif is_patch:
            raise RuntimeError("--patch-fusion none retains inputs but cannot produce comparable source-level metrics; use reference_nms for the pilot evaluation")
        else:
            images[stem]["predictions"] = sorted(direct_predictions[stem], key=lambda row: -row["confidence"])

    image_rows = [{
        "stem": stem, "fold_id": fold, "width": item["width"], "height": item["height"],
        "gt_boxes_json": json.dumps(item["gt_boxes"], separators=(",", ":")),
        "prediction_boxes_json": json.dumps(item["predictions"], separators=(",", ":")),
        "prediction_count_at_floor": len(item["predictions"]), "confidence_floor": args.conf_floor,
    } for stem, item in sorted(images.items())]
    object_rows, fp_rows = object_and_fp_rows(images, fold, args.report_confidence)
    fixed = fixed_threshold_metrics(images, args.report_confidence, 0.5)
    metrics = {
        **fixed, **ap_metrics(images), "experiment": experiment, "fold_id": fold, "imgsz": imgsz,
        "run_purpose": metadata.get("run_purpose"), "evidence_eligible": metadata.get("evidence_eligible", False),
        "comparison_role": metadata.get("comparison_role"),
        "n_images": len(images), "n_gt_objects": sum(len(item["gt_boxes"]) for item in images.values()),
        "confidence_floor": args.conf_floor, "nms_iou": args.nms_iou,
        "patch_fusion": args.patch_fusion if is_patch else "not_applicable",
        "fusion_iou": args.fusion_iou if is_patch else "",
        "fusion_policy_final": False,
        "localization_failure_count": sum(row["error_type"] == "LOCALIZATION_FAILURE" for row in object_rows),
        "low_confidence_fn_count": sum(row["error_type"] == "LOW_CONFIDENCE" for row in object_rows),
        "no_detection_fn_count": sum(row["error_type"] == "NO_DETECTION" for row in object_rows),
        "note": "0.25 is a fixed reporting threshold, not an operational threshold; patch reference NMS is provisional.",
    }
    metrics.update(fold_subgroup_metrics(project_root, object_rows))
    sweep = [fixed_threshold_metrics(images, round(step * 0.05, 2), 0.5) for step in range(20)]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "image_predictions.csv", list(image_rows[0]), image_rows)
    write_csv(output / "object_predictions.csv", list(object_rows[0]), object_rows)
    write_csv(output / "fp_predictions.csv", list(fp_rows[0]) if fp_rows else ["stem", "fold_id", "prediction_index", "prediction_bbox_json", "confidence", "nearest_gt_iou", "fp_status", "report_confidence_threshold"], fp_rows)
    write_csv(output / "error_types.csv", list(object_rows[0]), object_rows)
    write_csv(output / "confidence_sweep.csv", list(sweep[0]), sweep)
    write_csv(output / "metrics.csv", list(metrics), [metrics])
    write_json(output / "metrics.json", metrics)
    generated = [output / name for name in ("image_predictions.csv", "object_predictions.csv", "fp_predictions.csv", "error_types.csv", "confidence_sweep.csv", "metrics.csv", "metrics.json")]
    if is_patch:
        raw_fields = list(raw_patch_rows[0]) if raw_patch_rows else ["source_stem", "patch_id", "fold_id", "patch_x0", "patch_y0", "patch_prediction_index", "patch_bbox_json", "source_bbox_json", "confidence"]
        fusion_fields = list(fusion_audit_rows[0]) if fusion_audit_rows else ["source_stem", "candidate_index", "patch_id", "source_bbox_json", "confidence", "kept_after_reference_nms", "suppressed_by_candidate_index", "fusion_iou"]
        write_csv(output / "patch_prediction_fusion_inputs.csv", raw_fields, raw_patch_rows)
        write_csv(output / "patch_duplicate_fusion_audit.csv", fusion_fields, fusion_audit_rows)
        generated += [output / "patch_prediction_fusion_inputs.csv", output / "patch_duplicate_fusion_audit.csv"]
    evaluation_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir.resolve()),
        "experiment": experiment, "fold": fold, "imgsz": imgsz, "confidence_floor": args.conf_floor,
        "report_confidence": args.report_confidence, "iou_match": 0.5, "patch_fusion": metrics["patch_fusion"],
        "fusion_iou": metrics["fusion_iou"], "fusion_policy_final": False,
        "training_config": metadata, "final_safety_threshold_selected": False,
    }
    write_json(output / "evaluation_metadata.json", evaluation_metadata)
    generated.append(output / "evaluation_metadata.json")
    manifest = save_run_manifest(project_root, run_name, {**metadata, "status": "EVALUATION_COMPLETE", "stage": "Stage 6 evaluation",
        "evaluation_ended_utc": datetime.now(timezone.utc).isoformat(), "generated_checkpoint_paths": [str(best.resolve())],
        "generated_prediction_paths": [str(path.resolve()) for path in generated if "prediction" in path.name or "fusion" in path.name or "error_types" in path.name],
        "generated_metric_paths": [str(path.resolve()) for path in generated if path.is_file()]})
    print_output_manifest(stage="Stage 6 evaluation", run_name=run_name, project_root=project_root,
        files=[manifest, *generated], directories=[output], models=[best], downloads=[manifest, *generated],
        next_action="Review this one-fold evaluation; do not start another fold automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate exactly one completed Stage 6 fold run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--report-confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--patch-fusion", choices=("reference_nms", "none"), default="reference_nms")
    parser.add_argument("--fusion-iou", type=float, default=0.5)
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        root = args.project_root.resolve()
        run_name = args.run_dir.name
        manifest = save_run_manifest(root, run_name, {
            "status": "EVALUATION_FAILED", "stage": "Stage 6 evaluation", "run_name": run_name,
            "project_root": str(root), "evaluation_ended_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_error_type": type(exc).__name__, "evaluation_error": str(exc),
        })
        print_output_manifest(stage="Stage 6 evaluation (failed)", run_name=run_name, project_root=root,
            files=[manifest], downloads=[manifest], next_action="Return the manifest and error; do not start another fold.")
        raise

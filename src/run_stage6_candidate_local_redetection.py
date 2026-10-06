from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stage2a_common import box_iou, print_output_manifest, project_root_from_script, save_run_manifest, sha256, write_csv, write_json


ALLOWED_SCALES = (128, 160, 192, 224, 256)


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def crop_with_replicate(image: np.ndarray, bbox: list[float], size: int) -> tuple[np.ndarray, dict[str, int]]:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    x0 = math.floor(cx - size / 2)
    y0 = math.floor(cy - size / 2)
    x1, y1 = x0 + size, y0 + size
    height, width = image.shape[:2]
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - width), max(0, y1 - height)
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    crop = padded[y0 + top:y1 + top, x0 + left:x1 + left]
    if crop.shape[:2] != (size, size):
        raise RuntimeError(f"Unexpected crop shape {crop.shape[:2]} for size {size}")
    return crop, {"crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
                  "pad_left": left, "pad_top": top, "pad_right": right, "pad_bottom": bottom}


def matching_features(base: list[float], local: list[float]) -> dict[str, float | bool]:
    bcx, bcy = (base[0] + base[2]) / 2, (base[1] + base[3]) / 2
    lcx, lcy = (local[0] + local[2]) / 2, (local[1] + local[3]) / 2
    bw, bh = max(base[2] - base[0], 1e-12), max(base[3] - base[1], 1e-12)
    lw, lh = max(local[2] - local[0], 1e-12), max(local[3] - local[1], 1e-12)
    return {
        "base_local_iou": box_iou(base, local),
        "local_center_in_base": base[0] <= lcx <= base[2] and base[1] <= lcy <= base[3],
        "base_center_in_local": local[0] <= bcx <= local[2] and local[1] <= bcy <= local[3],
        "center_distance_over_base_diagonal": math.hypot(lcx - bcx, lcy - bcy) / max(math.hypot(bw, bh), 1e-12),
        "size_ratio_local_to_base": (lw * lh) / max(bw * bh, 1e-12),
    }


def main(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    if tuple(args.crop_scales) != ALLOWED_SCALES:
        raise ValueError(f"--crop-scales must be exactly {' '.join(map(str, ALLOWED_SCALES))}")
    if args.padding_mode != "replicate":
        raise ValueError("Only deterministic edge-replicate padding is approved")
    checkpoint = resolve(root, args.checkpoint)
    b0_path = resolve(root, args.b0_predictions)
    output = resolve(root, args.output_dir)
    manifest_path = root / "outputs/run_manifests" / f"{args.run_name}.json"
    if not checkpoint.is_file() or not b0_path.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else b0_path)
    if (output.exists() and any(output.iterdir())) or manifest_path.exists():
        raise FileExistsError(f"Output or run manifest already exists for {args.run_name}")

    integrity_rows = [row for row in load_csv(root / "outputs/tables/02a_experiment_integrity.csv")
                      if row["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE" and int(row["fold_id"]) == 1]
    integrity = {row["stem"]: row for row in integrity_rows}
    b0_rows = load_csv(b0_path)
    if len(b0_rows) != 124 or set(row["stem"] for row in b0_rows) != set(integrity):
        raise RuntimeError("B0 predictions do not match frozen Fold1 Conservative inputs")

    candidates: list[dict[str, Any]] = []
    image_cache: dict[str, np.ndarray] = {}
    for row in sorted(b0_rows, key=lambda item: item["stem"]):
        stem = row["stem"]
        image_path = root / Path(integrity[stem]["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read Conservative image: {image_path}")
        image_cache[stem] = image
        predictions = json.loads(row["prediction_boxes_json"])
        for candidate_index, prediction in enumerate(predictions):
            bbox = [float(value) for value in prediction["bbox"]]
            confidence = float(prediction["confidence"])
            if confidence < args.conf_floor:
                continue
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            candidates.append({
                "stem": stem, "candidate_id": f"{stem}__b{candidate_index:03d}", "candidate_index": candidate_index,
                "base_confidence": confidence, "base_bbox": bbox, "base_width": width, "base_height": height,
                "base_min_side": min(width, height), "base_area": width * height,
                "image_width": int(row["width"]), "image_height": int(row["height"]),
            })
    if len(candidates) != 1077:
        raise RuntimeError(f"Expected 1,077 B0 candidates at floor {args.conf_floor}, found {len(candidates)}")

    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "status": "RUNNING", "stage": "Stage 6 candidate-guided local re-detection inference",
        "run_name": args.run_name, "project_root": str(root), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "b0_predictions": str(b0_path),
        "b0_predictions_sha256": sha256(b0_path), "fold": 1, "candidate_definition": "all B0 predictions at confidence >=0.001",
        "candidate_count": len(candidates), "image_count": 124, "crop_scales": list(args.crop_scales),
        "padding_mode": args.padding_mode, "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
        "confidence_floor": args.conf_floor, "nms_iou": args.nms_iou, "max_det": args.max_det,
        "gt_used_for_inference_or_matching": False, "started_utc": started_utc, "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("ultralytics", "torch", "numpy", "opencv-python")},
    }
    save_run_manifest(root, args.run_name, metadata)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "outputs/cache/ultralytics"))
    import torch
    from ultralytics import YOLO

    device_index = int(str(args.device).split(",")[0]) if torch.cuda.is_available() and str(args.device).split(",")[0].isdigit() else None
    if device_index is not None:
        torch.cuda.set_device(device_index)
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(checkpoint))
    prediction_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for scale in args.crop_scales:
        for start in range(0, len(candidates), args.batch):
            batch_candidates = candidates[start:start + args.batch]
            crops: list[np.ndarray] = []
            crop_meta: list[dict[str, int]] = []
            for candidate in batch_candidates:
                crop, meta = crop_with_replicate(image_cache[candidate["stem"]], candidate["base_bbox"], scale)
                crops.append(crop); crop_meta.append(meta)
            results = model.predict(source=crops, imgsz=args.imgsz, batch=args.batch, conf=args.conf_floor,
                                    iou=args.nms_iou, max_det=args.max_det, device=args.device, verbose=False,
                                    save=False, save_txt=False, save_conf=False, save_crop=False,
                                    project=str(output), name="ultralytics_temp", exist_ok=True)
            if len(results) != len(batch_candidates):
                raise RuntimeError("Inference result count differs from candidate crop count")
            for candidate, meta, result in zip(batch_candidates, crop_meta, results):
                boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
                scores = [] if result.boxes is None else result.boxes.conf.cpu().tolist()
                local_rows: list[dict[str, Any]] = []
                for local_index, (crop_bbox, score) in enumerate(zip(boxes, scores)):
                    source_bbox = [float(crop_bbox[0]) + meta["crop_x0"], float(crop_bbox[1]) + meta["crop_y0"],
                                   float(crop_bbox[2]) + meta["crop_x0"], float(crop_bbox[3]) + meta["crop_y0"]]
                    source_bbox[0] = max(0.0, min(source_bbox[0], candidate["image_width"]))
                    source_bbox[2] = max(0.0, min(source_bbox[2], candidate["image_width"]))
                    source_bbox[1] = max(0.0, min(source_bbox[1], candidate["image_height"]))
                    source_bbox[3] = max(0.0, min(source_bbox[3], candidate["image_height"]))
                    match = matching_features(candidate["base_bbox"], source_bbox)
                    item = {
                        "stem": candidate["stem"], "candidate_id": candidate["candidate_id"], "candidate_index": candidate["candidate_index"],
                        "crop_scale": scale, "local_prediction_index": local_index, "base_confidence": candidate["base_confidence"],
                        "base_bbox_json": json.dumps(candidate["base_bbox"], separators=(",", ":")),
                        "local_confidence": float(score), "local_crop_bbox_json": json.dumps([float(v) for v in crop_bbox], separators=(",", ":")),
                        "source_bbox_json": json.dumps(source_bbox, separators=(",", ":")), **meta, **match,
                    }
                    prediction_rows.append(item); local_rows.append(item)
                selected = max(local_rows, key=lambda item: (float(item["base_local_iou"]), float(item["local_confidence"])), default=None)
                feature_rows.append({
                    "stem": candidate["stem"], "candidate_id": candidate["candidate_id"], "candidate_index": candidate["candidate_index"],
                    "crop_scale": scale, "base_confidence": candidate["base_confidence"],
                    "base_bbox_json": json.dumps(candidate["base_bbox"], separators=(",", ":")),
                    "base_width": candidate["base_width"], "base_height": candidate["base_height"],
                    "base_min_side": candidate["base_min_side"], "base_area": candidate["base_area"],
                    "image_width": candidate["image_width"], "image_height": candidate["image_height"],
                    "local_prediction_count": len(local_rows), "selected_local_prediction_index": "" if selected is None else selected["local_prediction_index"],
                    "selected_local_confidence": "" if selected is None else selected["local_confidence"],
                    "selected_source_bbox_json": "" if selected is None else selected["source_bbox_json"],
                    "selected_base_local_iou": "" if selected is None else selected["base_local_iou"],
                    "selected_local_center_in_base": "" if selected is None else selected["local_center_in_base"],
                    "selected_base_center_in_local": "" if selected is None else selected["base_center_in_local"],
                    "selected_center_distance_over_base_diagonal": "" if selected is None else selected["center_distance_over_base_diagonal"],
                    "selected_size_ratio_local_to_base": "" if selected is None else selected["size_ratio_local_to_base"],
                    **meta,
                })
    if len(feature_rows) != len(candidates) * len(args.crop_scales):
        raise RuntimeError("Candidate-scale output coverage is incomplete")
    if device_index is not None:
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_reserved() / 1_000_000_000
    else:
        peak_gb = None
    local_path = output / "local_predictions.csv"
    feature_path = output / "candidate_scale_features.csv"
    metadata_path = output / "inference_metadata.json"
    write_csv(local_path, list(prediction_rows[0]) if prediction_rows else ["stem", "candidate_id", "crop_scale"], prediction_rows)
    write_csv(feature_path, list(feature_rows[0]), feature_rows)
    counts = [sum(candidate["stem"] == stem for candidate in candidates) for stem in sorted(image_cache)]
    metadata.update({
        "status": "INFERENCE_COMPLETE", "ended_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started, "peak_cuda_reserved_gb": peak_gb,
        "candidate_count_distribution": {"mean": statistics.fmean(counts), "median": statistics.median(counts),
                                         "p90": float(np.percentile(counts, 90)), "max": max(counts)},
        "candidate_scale_rows": len(feature_rows), "local_prediction_rows": len(prediction_rows),
        "generated_prediction_paths": [str(local_path.resolve()), str(feature_path.resolve())],
        "generated_metric_paths": [str(metadata_path.resolve())],
    })
    write_json(metadata_path, metadata)
    manifest = save_run_manifest(root, args.run_name, metadata)
    print_output_manifest(stage="Stage 6 candidate local re-detection inference", run_name=args.run_name,
                          project_root=root, files=[local_path, feature_path, metadata_path, manifest], directories=[output],
                          models=[checkpoint], downloads=[local_path, feature_path, metadata_path, manifest],
                          next_action="Download these four files and run analyze_stage6_candidate_local_redetection.py locally.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference-only B0 candidate-guided local re-detection for frozen Fold1.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--b0-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--padding-mode", choices=("replicate",), default="replicate")
    parser.add_argument("--crop-scales", nargs="+", type=int, default=list(ALLOWED_SCALES))
    main(parser.parse_args())

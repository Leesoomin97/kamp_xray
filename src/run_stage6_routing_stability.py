from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stage2a_common import (
    print_output_manifest, project_root_from_script, read_csv, save_run_manifest, write_csv, write_json,
)


RUN_NAME = "stage6_routing_stability_fold1"
B0_RUN = "kamp_conservative_640_fold1_e30_fullft_v1"
SCALE = 0.90
CONF_FLOOR = 0.001


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def iou(a: list[float], b: list[float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, aa + ab - inter)


def restore_box(box: list[float], transform: str, width: int, height: int, dx: int = 0, dy: int = 0) -> list[float]:
    if transform == "horizontal_flip":
        return [width - box[2], box[1], width - box[0], box[3]]
    if transform == "scale_0.90":
        restored = [(box[0] - dx) / SCALE, (box[1] - dy) / SCALE,
                    (box[2] - dx) / SCALE, (box[3] - dy) / SCALE]
        return [max(0.0, min(width, restored[0])), max(0.0, min(height, restored[1])),
                max(0.0, min(width, restored[2])), max(0.0, min(height, restored[3]))]
    return box


def greedy_match(anchors: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[int, tuple[dict[str, Any], float]]:
    pairs = sorted(
        ((iou(anchor["bbox"], candidate["bbox"]), ai, ci) for ai, anchor in enumerate(anchors)
         for ci, candidate in enumerate(candidates)), reverse=True,
    )
    used_a, used_c, matches = set(), set(), {}
    for overlap, ai, ci in pairs:
        if ai in used_a or ci in used_c or overlap <= 0:
            continue
        used_a.add(ai); used_c.add(ci); matches[ai] = (candidates[ci], overlap)
    return matches


def main(root: Path, device: str) -> None:
    root = root.resolve()
    output = root / "outputs/stage6_routing/fold1"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Routing stability output is not empty: {output}")
    variants = output / "input_variants"
    b0_dir = root / "outputs/stage2a_runs" / B0_RUN
    image_rows = read_csv(b0_dir / "image_predictions.csv")
    integrity = {row["stem"]: row for row in read_csv(root / "outputs/tables/02a_experiment_integrity.csv")
                 if row["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE" and int(row["fold_id"]) == 1}
    checkpoint = root / "models/stage2a" / B0_RUN / "weights/best.pt"
    if len(image_rows) != 124 or not checkpoint.is_file():
        raise RuntimeError("B0 Fold1 images/checkpoint are incomplete")
    transform_metadata: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    sources: dict[str, list[Path]] = {"horizontal_flip": [], "scale_0.90": []}
    for row in image_rows:
        stem = row["stem"]
        image = read_gray(root / Path(integrity[stem]["image_path"]))
        height, width = image.shape
        flip_path = variants / "horizontal_flip" / f"{stem}.png"
        write_png(flip_path, cv2.flip(image, 1))
        sources["horizontal_flip"].append(flip_path)
        sw, sh = max(1, round(width * SCALE)), max(1, round(height * SCALE))
        dx, dy = (width - sw) // 2, (height - sh) // 2
        scaled = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.copyMakeBorder(scaled, dy, height - sh - dy, dx, width - sw - dx, cv2.BORDER_REFLECT_101)
        scale_path = variants / "scale_0.90" / f"{stem}.png"
        write_png(scale_path, canvas)
        sources["scale_0.90"].append(scale_path)
        transform_metadata[(stem, "horizontal_flip")] = (width, height, 0, 0)
        transform_metadata[(stem, "scale_0.90")] = (width, height, dx, dy)

    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "outputs/cache/ultralytics"))
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    transformed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    raw_rows = []
    for transform, paths in sources.items():
        results = model.predict(source=[str(path) for path in paths], imgsz=640, conf=CONF_FLOOR, iou=0.7,
                                max_det=300, device=device, stream=True, verbose=False, save=False,
                                project=str(output / "ultralytics_temp"), name=transform, exist_ok=True)
        for result in results:
            stem = Path(result.path).stem
            width, height, dx, dy = transform_metadata[(stem, transform)]
            boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
            scores = [] if result.boxes is None else result.boxes.conf.cpu().tolist()
            rows = []
            for index, (box, confidence) in enumerate(zip(boxes, scores)):
                restored = restore_box([float(value) for value in box], transform, width, height, dx, dy)
                item = {"bbox": restored, "confidence": float(confidence)}
                rows.append(item)
                raw_rows.append({"stem": stem, "transform": transform, "prediction_index": index,
                                 "confidence": confidence, "restored_bbox_json": json.dumps(restored, separators=(",", ":"))})
            transformed[(stem, transform)] = rows

    feature_rows = []
    for image_row in image_rows:
        stem = image_row["stem"]
        anchors = json.loads(image_row["prediction_boxes_json"])
        matches = {transform: greedy_match(anchors, transformed[(stem, transform)]) for transform in sources}
        for index, anchor in enumerate(anchors):
            confidences = [float(anchor["confidence"])]
            overlaps, displacements, sizes = [], [], []
            aw, ah = anchor["bbox"][2] - anchor["bbox"][0], anchor["bbox"][3] - anchor["bbox"][1]
            sizes.append(min(aw, ah))
            for transform in sources:
                if index not in matches[transform]:
                    confidences.append(0.0); overlaps.append(0.0); displacements.append(math.hypot(aw, ah)); sizes.append(0.0)
                    continue
                candidate, overlap = matches[transform][index]
                confidences.append(float(candidate["confidence"])); overlaps.append(overlap)
                acx, acy = (anchor["bbox"][0] + anchor["bbox"][2]) / 2, (anchor["bbox"][1] + anchor["bbox"][3]) / 2
                bcx, bcy = (candidate["bbox"][0] + candidate["bbox"][2]) / 2, (candidate["bbox"][1] + candidate["bbox"][3]) / 2
                displacements.append(math.hypot(acx - bcx, acy - bcy))
                sizes.append(min(candidate["bbox"][2] - candidate["bbox"][0], candidate["bbox"][3] - candidate["bbox"][1]))
            mean_size = float(np.mean(sizes))
            feature_rows.append({
                "stem": stem, "candidate_index": index, "original_confidence": anchor["confidence"],
                "confidence_std": float(np.std(confidences)), "confidence_range": max(confidences) - min(confidences),
                "bbox_iou_consistency": float(np.mean(overlaps)), "center_displacement_px": float(np.mean(displacements)),
                "predicted_size_cv": float(np.std(sizes) / max(mean_size, 1e-9)),
                "horizontal_flip_matched": index in matches["horizontal_flip"],
                "scale_0.90_matched": index in matches["scale_0.90"],
                "notes": "Inference-only deployable stability proxy; no GT-derived routing feature.",
            })

    output.mkdir(parents=True, exist_ok=True)
    raw_path, feature_path = output / "stability_predictions.csv", output / "stability_features.csv"
    write_csv(raw_path, list(raw_rows[0]), raw_rows)
    write_csv(feature_path, list(feature_rows[0]), feature_rows)
    metadata = output / "stability_metadata.json"
    write_json(metadata, {"status": "COMPLETE", "stage": "Stage 6 routing stability inference", "run_name": RUN_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint.resolve()), "fold": 1,
        "transforms": ["existing original prediction", "horizontal_flip", "scale_0.90_reflect_pad"],
        "scale": SCALE, "confidence_floor": CONF_FLOOR, "nms_iou": 0.7, "candidate_count": len(feature_rows),
        "gt_used_as_routing_signal": False})
    manifest = save_run_manifest(root, RUN_NAME, {"status": "COMPLETE", "stage": "Stage 6 routing stability inference",
        "run_name": RUN_NAME, "generated_prediction_paths": [str(raw_path.resolve()), str(feature_path.resolve())],
        "generated_metric_paths": [str(metadata.resolve())]})
    print_output_manifest(stage="Stage 6 routing stability inference", run_name=RUN_NAME, project_root=root,
        files=[raw_path, feature_path, metadata, manifest], directories=[output], models=[checkpoint],
        downloads=[raw_path, feature_path, metadata, manifest],
        next_action="Run analyze_stage6_selective_routing.py with --stability-features; do not retrain any model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference-only B0 perturbation stability audit for Fold1.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.project_root, args.device)

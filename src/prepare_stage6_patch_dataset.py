from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stage2a_common import (
    parse_yolo_labels, png_size, print_output_manifest, project_root_from_script,
    read_csv, save_run_manifest, sha256, verify_frozen_folds, write_csv,
)
from stage6_common import source_gt_by_stem


DEFAULT_PATCH_SIZE = 192
DEFAULT_OVERLAP = 0.25


def positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    values = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if values[-1] != last:
        values.append(last)
    return values


def intersection_area(box: list[float], tile: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = tile
    return max(0.0, min(box[2], x1) - max(box[0], x0)) * max(0.0, min(box[3], y1) - max(box[1], y0))


def fully_contained(box: list[float], tile: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = tile
    return box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1


def clipped_box(box: list[float], tile: tuple[int, int, int, int]) -> list[float]:
    x0, y0, x1, y1 = tile
    return [max(box[0], x0), max(box[1], y0), min(box[2], x1), min(box[3], y1)]


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def patch_label(box: list[float], x0: int, y0: int, patch_size: int) -> tuple[str, list[float]]:
    local = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
    cx = (local[0] + local[2]) / 2 / patch_size
    cy = (local[1] + local[3]) / 2 / patch_size
    bw = (local[2] - local[0]) / patch_size
    bh = (local[3] - local[1]) / patch_size
    return f"0 {cx:.10f} {cy:.10f} {bw:.10f} {bh:.10f}", local


def patch_configuration(patch_size: int, overlap: float) -> tuple[int, str, str, str]:
    if patch_size not in (192, 256) or abs(overlap - 0.25) > 1e-12:
        raise ValueError("Stage 6 supports only 192/256 px patches with overlap=0.25")
    stride = int(round(patch_size * (1.0 - overlap)))
    if patch_size == 192:
        return stride, "patch192_overlap25", "06_patch_manifest.csv", "stage6_m1_controlled_v2"
    return stride, "patch256_overlap25", "06_patch256_manifest.csv", "stage6_m1_patch256_controlled_v1"


def main(project_root: Path, materialize: bool, patch_size: int, overlap: float) -> None:
    project_root = project_root.resolve()
    stride, output_name, manifest_name, processing_version = patch_configuration(patch_size, overlap)
    _, fold_by_stem = verify_frozen_folds(project_root)
    allow = source_gt_by_stem(project_root)
    integrity = [
        row
        for row in read_csv(project_root / "outputs" / "tables" / "02a_experiment_integrity.csv")
        if row["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE"
    ]
    images = {row["stem"]: project_root / Path(row["image_path"]) for row in integrity}
    labels = {row["stem"]: project_root / Path(row["label_path"]) for row in integrity}
    if set(images) != set(allow) or set(labels) != set(allow):
        raise RuntimeError("Conservative Stage 6 source coverage differs from the approved 500 samples")

    output_root = project_root / "outputs" / "processed_data" / "stage6" / output_name
    rows: list[dict[str, Any]] = []
    covered_source_objects: set[tuple[str, int]] = set()
    for stem in sorted(allow):
        image_path, label_path = images[stem], labels[stem]
        width, height = png_size(image_path)
        boxes = parse_yolo_labels(label_path, width, height)
        source_image = read_grayscale(image_path) if materialize else None
        for y0 in positions(height, patch_size, stride):
            for x0 in positions(width, patch_size, stride):
                tile = (x0, y0, min(x0 + patch_size, width), min(y0 + patch_size, height))
                contained = [i for i, box in enumerate(boxes) if fully_contained(box, tile)]
                partial = [i for i, box in enumerate(boxes) if intersection_area(box, tile) > 0 and i not in contained]
                # Controlled M1 integrity policy: every positive-area GT intersection is
                # explicitly labeled after clipping. No visible GT fragment is silently
                # presented as background, even when only a small fraction remains.
                accepted_partial = list(partial)
                rejected_partial: list[int] = []
                disposition = "KEEP"
                patch_id = f"{stem}__x{x0:04d}_y{y0:04d}"
                patch_image = output_root / "images" / f"{patch_id}.png"
                patch_label_path = output_root / "labels" / f"{patch_id}.txt"
                mapping = []
                label_lines = []
                for object_id in contained:
                    line, local = patch_label(boxes[object_id], x0, y0, patch_size)
                    label_lines.append(line)
                    covered_source_objects.add((stem, object_id))
                    mapping.append({"source_object_id": object_id, "source_bbox": boxes[object_id], "patch_bbox": local,
                                    "clipped": False, "retained_area_fraction": 1.0})
                for object_id in accepted_partial:
                    clipped = clipped_box(boxes[object_id], tile)
                    line, local = patch_label(clipped, x0, y0, patch_size)
                    label_lines.append(line)
                    covered_source_objects.add((stem, object_id))
                    retained_fraction = intersection_area(boxes[object_id], tile) / max(1e-12, (boxes[object_id][2] - boxes[object_id][0]) * (boxes[object_id][3] - boxes[object_id][1]))
                    mapping.append({"source_object_id": object_id, "source_bbox": boxes[object_id], "patch_bbox": local,
                                    "clipped": True, "retained_area_fraction": retained_fraction})
                status = "PLANNED"
                output_hash = ""
                if materialize and disposition == "KEEP":
                    crop = source_image[y0:min(y0 + patch_size, height), x0:min(x0 + patch_size, width)]
                    if crop.shape != (patch_size, patch_size):
                        crop = cv2.copyMakeBorder(crop, 0, patch_size - crop.shape[0], 0, patch_size - crop.shape[1], cv2.BORDER_CONSTANT, value=0)
                    write_png(patch_image, crop)
                    patch_label_path.parent.mkdir(parents=True, exist_ok=True)
                    patch_label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
                    status = "COMPLETE"
                    output_hash = sha256(patch_image)
                rows.append({
                    "patch_id": patch_id,
                    "source_stem": stem,
                    "fold_id": fold_by_stem[stem],
                    "source_image_path": image_path.relative_to(project_root).as_posix(),
                    "source_label_path": label_path.relative_to(project_root).as_posix(),
                    "source_width": width,
                    "source_height": height,
                    "x0": x0,
                    "y0": y0,
                    "x1": tile[2],
                    "y1": tile[3],
                    "patch_size": patch_size,
                    "overlap_fraction": overlap,
                    "stride": stride,
                    "label_policy": "ALL_POSITIVE_AREA_INTERSECTIONS_CLIPPED",
                    "processing_version": processing_version,
                    "contained_object_count": len(contained),
                    "partial_object_count": len(partial),
                    "accepted_clipped_object_count": len(accepted_partial),
                    "rejected_partial_object_count": len(rejected_partial),
                    "labeled_object_count": len(contained) + len(accepted_partial),
                    "is_background_only": len(contained) == 0 and len(partial) == 0,
                    "disposition": disposition,
                    "partial_handling": "every positive-area intersection clipped and labeled" if accepted_partial else "not applicable",
                    "contains_unlabeled_small_partial_fragment": False,
                    "source_to_patch_gt_json": json.dumps(mapping, separators=(",", ":")),
                    "patch_image_path": patch_image.relative_to(project_root).as_posix(),
                    "patch_label_path": patch_label_path.relative_to(project_root).as_posix(),
                    "source_image_sha256": sha256(image_path),
                    "patch_image_sha256": output_hash,
                    "materialization_status": status if disposition == "KEEP" else "NOT_MATERIALIZED_BY_POLICY",
                })

    expected_source_objects = {
        (stem, object_id)
        for stem in allow
        for object_id in range(len(parse_yolo_labels(labels[stem], *png_size(images[stem]))))
    }
    if covered_source_objects != expected_source_objects or len(expected_source_objects) != 1147:
        raise RuntimeError("Patch policy does not retain all 1,147 source GT objects")
    if any(row["contains_unlabeled_small_partial_fragment"] for row in rows):
        raise RuntimeError("Unlabeled visible GT fragment remains in patch manifest")
    if any(int(row["fold_id"]) != int(fold_by_stem[row["source_stem"]]) for row in rows):
        raise RuntimeError("Patch source fold inheritance changed")
    coordinate_errors = 0
    for row in rows:
        for item in json.loads(row["source_to_patch_gt_json"]):
            x0, y0, x1, y1 = item["patch_bbox"]
            coordinate_errors += int(not (0 <= x0 < x1 <= patch_size and 0 <= y0 < y1 <= patch_size))
    if coordinate_errors:
        raise RuntimeError(f"Patch bbox coordinate errors: {coordinate_errors}")

    fieldnames = list(rows[0])
    manifest_table = project_root / "outputs" / "tables" / manifest_name
    write_csv(manifest_table, fieldnames, rows)
    if materialize:
        run_name = f"stage6_m1_patch{patch_size}_dataset_preparation"
        kept = [row for row in rows if row["disposition"] == "KEEP"]
        run_manifest = save_run_manifest(project_root, run_name, {
            "status": "COMPLETE", "stage": "Stage 6 patch dataset preparation", "run_name": run_name,
            "patch_size": patch_size, "overlap_fraction": overlap, "stride": stride,
            "label_policy": "ALL_POSITIVE_AREA_INTERSECTIONS_CLIPPED",
            "processing_version": processing_version,
            "candidate_patch_count": len(rows), "materialized_patch_count": len(kept),
            "positive_patch_count": sum(int(row["labeled_object_count"]) > 0 for row in kept),
            "true_background_patch_count": sum(str(row["is_background_only"]).lower() == "true" for row in kept),
            "source_sample_count": len({row["source_stem"] for row in rows}),
            "source_gt_count": len(expected_source_objects),
            "unlabeled_fragment_patch_count": 0,
            "coordinate_error_count": coordinate_errors,
            "source_fold_inheritance_error_count": 0,
            "generated_metric_paths": [str(manifest_table.resolve())],
        })
        print_output_manifest(stage="Stage 6 patch dataset preparation", run_name=run_name, project_root=project_root,
            files=[manifest_table, run_manifest], directories=[output_root], downloads=[manifest_table, run_manifest],
            next_action="Review patch counts and coordinate mappings, then run only the approved M1 fold-1 training command.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan or materialize a frozen-fold Stage 6 patch dataset.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--materialize", action="store_true", help="Write derived patch PNG/TXT files outside the original dataset.")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, choices=(192, 256))
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    main(args.project_root, args.materialize, args.patch_size, args.overlap)

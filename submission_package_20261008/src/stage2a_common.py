from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any


EXPECTED_FOLD_SHA256 = "963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8"
EXPECTED_PRETRAINED_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
REPRESENTATIONS = {
    "conservative": "ARTIFACT_INPAINT_CONSERVATIVE",
    "local": "ARTIFACT_LOCAL_INTERPOLATION",
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_root(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root.resolve() / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_folds(workspace: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    fold_path = workspace / "outputs" / "tables" / "01c_final_validation_folds.csv"
    observed = sha256(fold_path)
    if observed != EXPECTED_FOLD_SHA256:
        raise RuntimeError(f"Frozen fold SHA-256 changed: {observed}")
    rows = read_csv(fold_path)
    if len(rows) != 500 or len({r["stem"] for r in rows}) != 500:
        raise RuntimeError("Frozen fold table must contain 500 unique stems")
    counts = [sum(r["fold_id"] == str(fold) for r in rows) for fold in range(1, 5)]
    if counts != [124, 125, 126, 125]:
        raise RuntimeError(f"Frozen fold counts changed: {counts}")
    return rows, {r["stem"]: r["fold_id"] for r in rows}


def validate_fold_dataset(workspace: Path, representation: str, fold: int) -> tuple[Path, list[str]]:
    """Validate frozen membership and materialize a host-local runtime split.

    Previously prepared split files contain absolute paths from the machine on
    which they were generated.  They remain historical artifacts and are not
    used for execution.  This function writes current-host absolute paths only
    under outputs/cache, never inside the original dataset tree.
    """
    workspace = workspace.resolve()
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {representation}")
    folds, fold_by_stem = verify_frozen_folds(workspace)
    integrity_path = workspace / "outputs" / "tables" / "02a_experiment_integrity.csv"
    integrity = read_csv(integrity_path)
    role = REPRESENTATIONS[representation]
    selected = [r for r in integrity if r["representation"] == role]
    if len(selected) != 500 or any(r["status"] != "OK" for r in selected):
        raise RuntimeError(f"Stage 2A integrity is not clean for {role}")
    if any(not (workspace / Path(r["image_path"])).is_file() or not (workspace / Path(r["label_path"])).is_file() for r in selected):
        raise RuntimeError("Prepared image or label is missing")
    selected_by_stem = {r["stem"]: r for r in selected}
    val_stems = sorted(r["stem"] for r in folds if r["fold_id"] == str(fold))
    expected = sorted(r["stem"] for r in folds if r["fold_id"] == str(fold))
    train_stems = sorted(r["stem"] for r in folds if r["fold_id"] != str(fold))
    if set(train_stems) & set(val_stems) or set(train_stems) | set(val_stems) != set(fold_by_stem):
        raise RuntimeError("Frozen train/validation membership overlaps or omits samples")

    split_dir = workspace / "outputs" / "cache" / "stage2a_runtime_splits" / representation
    split_dir.mkdir(parents=True, exist_ok=True)
    train_file = split_dir / f"fold_{fold}_train.txt"
    val_file = split_dir / f"fold_{fold}_val.txt"
    yaml_path = split_dir / f"fold_{fold}.yaml"
    def image_path(stem: str) -> str:
        return (workspace / Path(selected_by_stem[stem]["image_path"])).resolve().as_posix()
    train_file.write_text("\n".join(image_path(stem) for stem in train_stems) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(image_path(stem) for stem in val_stems) + "\n", encoding="utf-8")
    yaml_path.write_text(
        f"train: {train_file.resolve().as_posix()}\n"
        f"val: {val_file.resolve().as_posix()}\n"
        "nc: 1\n"
        "names:\n  0: foreign_object\n",
        encoding="utf-8",
    )
    return yaml_path, val_stems


def save_run_manifest(workspace: Path, run_name: str, payload: dict[str, Any]) -> Path:
    path = workspace.resolve() / "outputs" / "run_manifests" / f"{run_name}.json"
    write_json(path, payload)
    return path


def print_output_manifest(*, stage: str, run_name: str, project_root: Path,
                          files: list[Path] | None = None, directories: list[Path] | None = None,
                          models: list[Path] | None = None, downloads: list[Path] | None = None,
                          next_action: str) -> None:
    existing_files = [p.resolve() for p in (files or []) if p.is_file()]
    existing_dirs = [p.resolve() for p in (directories or []) if p.is_dir()]
    existing_models = [p.resolve() for p in (models or []) if p.is_file()]
    existing_downloads = [p.resolve() for p in (downloads or []) if p.exists()]
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Stage:\n{stage}\n")
    print(f"Run name:\n{run_name}\n")
    print(f"Project root:\n{project_root.resolve()}\n")
    print("Generated files:")
    for path in existing_files:
        print(f"[FILE] {path}")
    print("\nGenerated directories:")
    for path in existing_dirs:
        print(f"[DIR] {path}")
    print("\nModel checkpoint(s):")
    for path in existing_models:
        print(f"[MODEL] {path}")
    print("\nFiles to download back to local project:")
    for path in existing_downloads:
        print(f"[DOWNLOAD] {path}")
    print(f"\nNext action:\n{next_action}")
    print("=" * 60)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if len(signature) < 24 or signature[:8] != b"\x89PNG\r\n\x1a\n" or signature[12:16] != b"IHDR":
        raise ValueError(f"Not a supported PNG: {path}")
    return struct.unpack(">II", signature[16:24])


def parse_yolo_labels(path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts:
            continue
        class_id, cx, cy, bw, bh = map(float, parts)
        if int(class_id) != 0:
            raise ValueError(f"Unexpected class ID in {path}: {class_id}")
        boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height])
    return boxes


def box_iou(a: list[float], b: list[float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


def match_image(gts: list[list[float]], predictions: list[dict[str, Any]], confidence: float, iou_threshold: float = 0.5) -> tuple[dict[int, tuple[int, float]], list[int]]:
    candidates = sorted((i for i, p in enumerate(predictions) if float(p["confidence"]) >= confidence), key=lambda i: (-float(predictions[i]["confidence"]), i))
    used_gt: set[int] = set()
    matches: dict[int, tuple[int, float]] = {}
    false_positive_indexes: list[int] = []
    for prediction_index in candidates:
        scores = [(box_iou(predictions[prediction_index]["bbox"], gt), gt_index) for gt_index, gt in enumerate(gts) if gt_index not in used_gt]
        best_iou, best_gt = max(scores, default=(0.0, -1))
        if best_iou >= iou_threshold:
            used_gt.add(best_gt)
            matches[best_gt] = (prediction_index, best_iou)
        else:
            false_positive_indexes.append(prediction_index)
    return matches, false_positive_indexes


def fixed_threshold_metrics(images: dict[str, dict[str, Any]], confidence: float, iou_threshold: float = 0.5) -> dict[str, float | int]:
    tp = fp = fn = 0
    for item in images.values():
        matches, fps = match_image(item["gt_boxes"], item["predictions"], confidence, iou_threshold)
        tp += len(matches); fp += len(fps); fn += len(item["gt_boxes"]) - len(matches)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"confidence_threshold": confidence, "iou_threshold": iou_threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def average_precision(images: dict[str, dict[str, Any]], iou_threshold: float) -> float:
    total_gt = sum(len(item["gt_boxes"]) for item in images.values())
    ranked = sorted(((float(pred["confidence"]), stem, index) for stem, item in images.items() for index, pred in enumerate(item["predictions"])), reverse=True)
    used: dict[str, set[int]] = {stem: set() for stem in images}
    cumulative_tp = cumulative_fp = 0
    recalls: list[float] = []; precisions: list[float] = []
    for _, stem, prediction_index in ranked:
        item = images[stem]; prediction = item["predictions"][prediction_index]
        candidates = [(box_iou(prediction["bbox"], gt), gt_index) for gt_index, gt in enumerate(item["gt_boxes"]) if gt_index not in used[stem]]
        best_iou, best_gt = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            used[stem].add(best_gt); cumulative_tp += 1
        else:
            cumulative_fp += 1
        recalls.append(cumulative_tp / total_gt if total_gt else 0.0)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    if not recalls:
        return 0.0
    return sum(max((p for r, p in zip(recalls, precisions) if r >= target), default=0.0) for target in (i / 100 for i in range(101))) / 101


def ap_metrics(images: dict[str, dict[str, Any]]) -> dict[str, Any]:
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    values = {f"ap{int(round(threshold * 100))}": average_precision(images, threshold) for threshold in thresholds}
    values["ap50"] = values["ap50"]
    values["map50_95"] = statistics.fmean(values[f"ap{int(round(threshold * 100))}"] for threshold in thresholds)
    values["ap_method"] = "single-class 101-point interpolated AP from predictions retained at confidence floor"
    return values


def safe_float(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

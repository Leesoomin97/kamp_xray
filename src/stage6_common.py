from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from stage2a_common import (
    EXPECTED_PRETRAINED_SHA256,
    read_csv,
    sha256,
    verify_frozen_folds,
)


STAGE6_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "m1_patch": {
        "experiment_id": "M1",
        "architecture": "YOLOv8n",
        "dataset_kind": "patch",
        "preprocessing": "Conservative artifact control only",
        "patch_setting": "192 px; 25% nominal overlap; 144 px stride; all positive-area GT intersections clipped/labeled",
        "patch_manifest": "06_patch_manifest.csv",
    },
    "m1_patch256": {
        "experiment_id": "M1-256-CONTROLLED",
        "architecture": "YOLOv8n",
        "dataset_kind": "patch",
        "preprocessing": "Conservative artifact control only",
        "patch_setting": "256 px; 25% nominal overlap; 192 px stride; all positive-area GT intersections clipped/labeled",
        "patch_manifest": "06_patch256_manifest.csv",
    },
    "m2_p2": {
        "experiment_id": "M2",
        "architecture": "YOLOv8n-P2",
        "dataset_kind": "conservative",
        "preprocessing": "Conservative artifact control only",
        "patch_setting": "none",
    },
    "m3_bilateral": {
        "experiment_id": "M3",
        "architecture": "YOLOv8n",
        "dataset_kind": "preprocessed",
        "preprocessing": "bilateral",
        "patch_setting": "none",
    },
    "m4_clahe": {
        "experiment_id": "M4",
        "architecture": "YOLOv8n",
        "dataset_kind": "preprocessed",
        "preprocessing": "clahe",
        "patch_setting": "none",
    },
    "m5_bilateral_clahe": {
        "experiment_id": "M5",
        "architecture": "YOLOv8n",
        "dataset_kind": "preprocessed",
        "preprocessing": "bilateral_clahe",
        "patch_setting": "none",
    },
    "m6_clahe_bilateral": {
        "experiment_id": "M6",
        "architecture": "YOLOv8n",
        "dataset_kind": "preprocessed",
        "preprocessing": "clahe_bilateral",
        "patch_setting": "none",
    },
}

BASELINE_AUGMENTATION = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.2,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.5,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

FIXED_TRAINING = {
    "imgsz": 640,
    "epochs": 30,
    "seed": 42,
    "freeze": 0,
    "optimizer": "AdamW",
    "lr0": 0.01,
    "lrf": 0.1,
    "warmup_epochs": 0.5,
    "weight_decay": 0.0005,
    "amp": False,
    "deterministic": True,
    "patience": 0,
}

BASELINE_BATCH_SIZE = 8
BASELINE_EPOCHS = 30
ULTRALYTICS_NOMINAL_BATCH_SIZE = 64
ULTRALYTICS_WARMUP_BIAS_LR = 0.1


def b0_fold_step_budget(fold: int, batch_size: int = BASELINE_BATCH_SIZE) -> dict[str, object]:
    """Reproduce the approximate B0 batch/optimizer-step budget for one fold.

    Optimizer updates follow the Ultralytics 8.4.158 accumulation rule used by
    the recorded B0 recipe. This is deterministic for the fixed dataset size,
    batch size, epoch count, warmup, and nominal batch size.
    """
    if fold not in (1, 2, 3, 4):
        raise ValueError("fold must be 1, 2, 3, or 4")
    validation_images = {1: 124, 2: 125, 3: 126, 4: 125}[fold]
    train_images = 500 - validation_images
    batches_per_epoch = math.ceil(train_images / batch_size)
    micro_batches = batches_per_epoch * BASELINE_EPOCHS
    warmup_iterations = round(0.5 * batches_per_epoch)
    last_optimizer_step = -1
    optimizer_updates = 0
    target_accumulation = ULTRALYTICS_NOMINAL_BATCH_SIZE / batch_size
    for ni in range(micro_batches):
        if ni < warmup_iterations and warmup_iterations > 0:
            accumulation = max(1, round(1 + (target_accumulation - 1) * ni / warmup_iterations))
        else:
            accumulation = max(1, round(target_accumulation))
        if ni - last_optimizer_step >= accumulation:
            optimizer_updates += 1
            last_optimizer_step = ni
    return {
        "fold": fold,
        "train_images": train_images,
        "validation_images": validation_images,
        "batch_size": batch_size,
        "epochs": BASELINE_EPOCHS,
        "batches_per_epoch": batches_per_epoch,
        "micro_batch_iterations": micro_batches,
        "warmup_iterations": warmup_iterations,
        "approx_optimizer_updates": optimizer_updates,
        "scheduler_policy": "ultralytics_8.4.158_linear_epoch_lr_replayed_on_b0_micro_batch_progress",
        "lr0": FIXED_TRAINING["lr0"],
        "lrf": FIXED_TRAINING["lrf"],
        "lr_first_non_bias": 0.0,
        "lr_first_bias": 0.1,
        "lr_after_warmup_non_bias": FIXED_TRAINING["lr0"] * (warmup_iterations - 1) / warmup_iterations,
        "lr_after_warmup_bias": ULTRALYTICS_WARMUP_BIAS_LR
        + (FIXED_TRAINING["lr0"] - ULTRALYTICS_WARMUP_BIAS_LR)
        * (warmup_iterations - 1)
        / warmup_iterations,
        "lr_final": FIXED_TRAINING["lr0"]
        * ((1 - (BASELINE_EPOCHS - 1) / BASELINE_EPOCHS) * (1 - FIXED_TRAINING["lrf"]) + FIXED_TRAINING["lrf"]),
    }


def ensure_safe_run_name(run_name: str) -> None:
    if any(token in run_name for token in ("/", "\\", "..")):
        raise ValueError("--run-name must be a single safe directory name")


def pretrained_path(project_root: Path) -> Path:
    path = project_root / "models" / "stage2a" / "pretrained" / "yolov8n.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Approved pretrained weight missing: {path}")
    observed = sha256(path)
    if observed != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError(f"Approved yolov8n.pt SHA-256 changed: {observed}")
    return path


def source_gt_by_stem(project_root: Path) -> dict[str, dict[str, str]]:
    eligible = [
        row
        for row in read_csv(project_root / "outputs" / "tables" / "00_stage1_allowlist.csv")
        if row["eligible_for_stage1_eda"].strip().lower() == "true"
    ]
    if len(eligible) != 500 or len({row["stem"] for row in eligible}) != 500:
        raise RuntimeError("Stage 1 allowlist is not 500 unique eligible samples")
    return {row["stem"]: row for row in eligible}


def _write_runtime_yaml(
    project_root: Path,
    experiment: str,
    fold: int,
    train_images: list[Path],
    val_images: list[Path],
) -> Path:
    cache = project_root / "outputs" / "cache" / "stage6_runtime_splits" / experiment
    cache.mkdir(parents=True, exist_ok=True)
    train_file = cache / f"fold_{fold}_train.txt"
    val_file = cache / f"fold_{fold}_val.txt"
    yaml_path = cache / f"fold_{fold}.yaml"
    train_file.write_text("\n".join(path.resolve().as_posix() for path in train_images) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(path.resolve().as_posix() for path in val_images) + "\n", encoding="utf-8")
    yaml_path.write_text(
        f"train: {train_file.resolve().as_posix()}\n"
        f"val: {val_file.resolve().as_posix()}\n"
        "nc: 1\n"
        "names:\n  0: foreign_object\n",
        encoding="utf-8",
    )
    return yaml_path


def prepare_stage6_runtime_dataset(project_root: Path, experiment: str, fold: int) -> tuple[Path, list[str]]:
    if experiment not in STAGE6_EXPERIMENTS:
        raise ValueError(f"Unsupported Stage 6 experiment: {experiment}")
    fold_rows, fold_by_stem = verify_frozen_folds(project_root)
    validation_stems = sorted(row["stem"] for row in fold_rows if int(row["fold_id"]) == fold)
    spec = STAGE6_EXPERIMENTS[experiment]
    kind = spec["dataset_kind"]

    if kind == "conservative":
        integrity = read_csv(project_root / "outputs" / "tables" / "02a_experiment_integrity.csv")
        rows = [row for row in integrity if row["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE"]
        by_stem = {row["stem"]: row for row in rows}
        if set(by_stem) != set(fold_by_stem):
            raise RuntimeError("Conservative Stage 6 inputs differ from frozen corpus")
        train = [project_root / Path(by_stem[stem]["image_path"]) for stem in sorted(by_stem) if int(fold_by_stem[stem]) != fold]
        val = [project_root / Path(by_stem[stem]["image_path"]) for stem in validation_stems]
    elif kind == "preprocessed":
        manifest = read_csv(project_root / "outputs" / "tables" / "06_preprocessing_manifest.csv")
        method = str(spec["preprocessing"])
        rows = [row for row in manifest if row["preprocessing_type"] == method]
        by_stem = {row["stem"]: row for row in rows}
        if set(by_stem) != set(fold_by_stem):
            raise RuntimeError(f"Preprocessing manifest coverage is incomplete for {method}")
        if any(row["materialization_status"] != "COMPLETE" for row in rows):
            raise RuntimeError(f"Materialize Stage 6 preprocessing before training: {method}")
        if any(row["fold_id"] != fold_by_stem[row["stem"]] for row in rows):
            raise RuntimeError("Preprocessing manifest fold metadata differs from frozen folds")
        if any(not (project_root / Path(row["output_label_path"])).is_file() for row in rows):
            raise RuntimeError(f"Materialized Stage 6 labels are incomplete for {method}")
        train = [project_root / Path(by_stem[stem]["output_path"]) for stem in sorted(by_stem) if int(fold_by_stem[stem]) != fold]
        val = [project_root / Path(by_stem[stem]["output_path"]) for stem in validation_stems]
    else:
        manifest = read_csv(project_root / "outputs" / "tables" / str(spec["patch_manifest"]))
        rows = [row for row in manifest if row["disposition"] == "KEEP"]
        if not rows or any(row["materialization_status"] != "COMPLETE" for row in rows):
            raise RuntimeError("Materialize the Stage 6 patch dataset before training")
        source_stems = {row["source_stem"] for row in rows}
        if source_stems != set(fold_by_stem):
            raise RuntimeError("Patch manifest does not cover every frozen source image")
        if any(row["fold_id"] != fold_by_stem[row["source_stem"]] for row in rows):
            raise RuntimeError("Patch manifest fold metadata differs from frozen folds")
        if any(not (project_root / Path(row["patch_label_path"])).is_file() for row in rows):
            raise RuntimeError("Materialized Stage 6 patch labels are incomplete")
        train = [project_root / Path(row["patch_image_path"]) for row in rows if int(row["fold_id"]) != fold]
        val = [project_root / Path(row["patch_image_path"]) for row in rows if int(row["fold_id"]) == fold]

    if not train or not val or any(not path.is_file() for path in train + val):
        raise RuntimeError("Stage 6 runtime dataset has missing or empty image lists")
    return _write_runtime_yaml(project_root, experiment, fold, train, val), validation_stems


def stage6_evaluation_sources(project_root: Path, experiment: str, fold: int) -> tuple[list[dict[str, str]], list[str]]:
    fold_rows, _ = verify_frozen_folds(project_root)
    validation_stems = sorted(row["stem"] for row in fold_rows if int(row["fold_id"]) == fold)
    spec = STAGE6_EXPERIMENTS[experiment]
    if spec["dataset_kind"] == "patch":
        rows = [
            row
            for row in read_csv(project_root / "outputs" / "tables" / str(spec["patch_manifest"]))
            if row["disposition"] == "KEEP" and int(row["fold_id"]) == fold
        ]
        return rows, validation_stems
    if spec["dataset_kind"] == "conservative":
        rows = [
            row
            for row in read_csv(project_root / "outputs" / "tables" / "02a_experiment_integrity.csv")
            if row["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE" and int(row["fold_id"]) == fold
        ]
        return rows, validation_stems
    method = str(spec["preprocessing"])
    rows = [
        row
        for row in read_csv(project_root / "outputs" / "tables" / "06_preprocessing_manifest.csv")
        if row["preprocessing_type"] == method and int(row["fold_id"]) == fold
    ]
    return rows, validation_stems


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

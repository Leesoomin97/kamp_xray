from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FOLD_HASH = "963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*parts: object) -> None:
    interpreter = os.environ.get("KAMP_PYTHON_EXECUTABLE", sys.executable)
    subprocess.run([interpreter, *map(str, parts)], cwd=ROOT, check=True)


def verify_inputs() -> None:
    required_counts = {
        ROOT / "data/source/images": ("*.bmp", 500),
        ROOT / "data/source/labels": ("*.txt", 500),
        ROOT / "outputs/processed_data/stage2a/conservative/images": ("*.png", 500),
        ROOT / "outputs/processed_data/stage2a/conservative/labels": ("*.txt", 500),
    }
    for directory, (pattern, expected) in required_counts.items():
        observed = len(list(directory.glob(pattern)))
        if observed != expected:
            raise RuntimeError(f"{directory}: expected {expected}, observed {observed}")
    gt_objects = sum(
        1
        for path in (ROOT / "data/source/labels").glob("*.txt")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if gt_objects != 1147:
        raise RuntimeError(f"Official GT object count mismatch: {gt_objects}")
    fold_path = ROOT / "outputs/tables/01c_final_validation_folds.csv"
    if sha256(fold_path) != EXPECTED_FOLD_HASH:
        raise RuntimeError("Frozen fold hash mismatch")
    weight_manifest = json.loads((ROOT / "models/b2_640/manifest.json").read_text(encoding="utf-8"))
    for fold in range(1, 5):
        path = ROOT / f"models/b2_640/fold{fold}/weights/best.pt"
        expected = weight_manifest["folds"][str(fold)]["sha256"]
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Frozen final weight mismatch for fold {fold}")
    pretrained = ROOT / "models/stage2a/pretrained/yolov8n.pt"
    if not pretrained.is_file() or sha256(pretrained) != "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36":
        raise RuntimeError("Pretrained initialization asset mismatch")
    with (ROOT / "outputs/tables/02a_experiment_integrity.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        integrity = list(csv.DictReader(handle))
    if len(integrity) != 1000 or any(row["status"] != "OK" for row in integrity):
        raise RuntimeError("Prepared dataset integrity table is incomplete")
    print("input_integrity=OK images=500 labels=500 gt_objects=1147 folds=4 frozen_weights=4")


def prepare() -> None:
    run("src/run_stage1d_artifact_control.py", "--workspace", ROOT)
    run("src/prepare_stage2a_dataset.py", "--workspace", ROOT)
    verify_inputs()


def evaluate_and_aggregate(run_dirs: list[Path], tag: str, device: str) -> None:
    output_root = ROOT / "reproduction_runs" / tag
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Choose a new --tag; output exists: {output_root}")
    fold_outputs: list[Path] = []
    for fold, run_dir in enumerate(run_dirs, start=1):
        output = output_root / f"fold{fold}_evaluation"
        run("src/evaluate_stage2a_run.py", "--project-root", ROOT,
            "--run-dir", run_dir, "--output-dir", output, "--device", device,
            "--conf-floor", 0.001, "--report-confidence", 0.25,
            "--nms-iou", 0.7, "--max-det", 300)
        fold_outputs.append(output)
    oof = output_root / "oof"
    run("src/aggregate_stage2a_oof.py", "--project-root", ROOT,
        "--fold-results", *fold_outputs, "--output-dir", oof,
        "--run-name", f"{tag}_oof")
    run("src/export_oof_predictions.py", "--oof-dir", oof,
        "--bbox-output", output_root / "predictions/b2_640_frozen_4fold_oof_bbox_conf025.csv",
        "--image-output", output_root / "predictions/b2_640_frozen_4fold_oof_image_summary_conf025.csv",
        "--confidence", 0.25)


def quick(tag: str, device: str) -> None:
    verify_inputs()
    evaluate_and_aggregate([ROOT / f"models/b2_640/fold{fold}" for fold in range(1, 5)], tag, device)


def full(tag: str, device: str, workers: int) -> None:
    prepare()
    run_dirs: list[Path] = []
    for fold in range(1, 5):
        run_name = f"{tag}_fold{fold}"
        run("src/run_stage2a_training.py", "--project-root", ROOT,
            "--representation", "conservative", "--model", "yolov8n",
            "--augmentation-profile", "visibility", "--fold", fold,
            "--imgsz", 640, "--epochs", 30, "--batch", 8,
            "--device", device, "--freeze", 0, "--seed", 42,
            "--workers", workers, "--box", 7.5, "--dfl", 1.5, "--cls", 0.5,
            "--oversampling-mode", "none", "--run-name", run_name)
        run_dirs.append(ROOT / "models/stage2a" / run_name)
    evaluate_and_aggregate(run_dirs, tag, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submission reproduction orchestrator; no model-selection logic.")
    parser.add_argument("mode", choices=("verify", "prepare", "quick", "full"))
    parser.add_argument("--tag", default="submission_verification_v1")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "verify":
        verify_inputs()
    elif args.mode == "prepare":
        prepare()
    elif args.mode == "quick":
        quick(args.tag, args.device)
    else:
        full(args.tag, args.device, args.workers)

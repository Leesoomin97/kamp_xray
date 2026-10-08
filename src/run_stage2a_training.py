from __future__ import annotations

import argparse
import csv
import importlib.metadata
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from stage2a_common import (
    EXPECTED_FOLD_SHA256, EXPECTED_PRETRAINED_SHA256, print_output_manifest,
    project_root_from_script, save_run_manifest, sha256, validate_fold_dataset, write_json,
)
from stage2b_augmentations import WEAK_GAUSSIAN_BLUR_CONFIG, make_weak_gaussian_blur_trainer
from b2_final_support import OVERSAMPLING_RULES, prepare_oversampled_training_split
from wandb_logging import OptionalWandbLogger, disable_ultralytics_builtin_wandb


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def default_wandb_group(args: argparse.Namespace) -> str:
    if args.oversampling_mode != "none":
        return f"B2-hardos-{args.imgsz}"
    if args.augmentation_profile == "visibility":
        return f"B2-{args.imgsz}"
    return args.experiment_family


METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}

MODEL_SPECS = {
    "yolov8n": {"filename": "yolov8n.pt", "expected_sha256": EXPECTED_PRETRAINED_SHA256},
    "yolov8s": {"filename": "yolov8s.pt", "expected_sha256": None},
}

BASELINE_AUGMENTATION = {
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0,
    "translate": 0.05, "scale": 0.2, "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5, "mosaic": 0.5, "mixup": 0.0, "copy_paste": 0.0,
}

AUGMENTATION_PROFILES = {
    "baseline": {**BASELINE_AUGMENTATION},
    # Preserve native object pixel scale by removing four-image mosaic only.
    "small_object": {**BASELINE_AUGMENTATION, "mosaic": 0.0},
    # Value-only jitter of +/-5%; hue/saturation remain disabled for grayscale-like X-ray input.
    "visibility": {**BASELINE_AUGMENTATION, "hsv_v": 0.05},
    "combined": {**BASELINE_AUGMENTATION, "mosaic": 0.0, "hsv_v": 0.05},
    "gaussian_blur": {
        **BASELINE_AUGMENTATION,
        "gaussian_blur": dict(WEAK_GAUSSIAN_BLUR_CONFIG),
    },
}


def read_training_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{str(key).strip(): str(value).strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def result_fitness(row: dict[str, str]) -> float:
    for key in ("fitness", "metrics/fitness"):
        if key in row and row[key] != "":
            return float(row[key])
    # Ultralytics detection checkpoint fitness: 0.1 * mAP50 + 0.9 * mAP50-95.
    return 0.1 * float(row[METRIC_COLUMNS["map50"]]) + 0.9 * float(row[METRIC_COLUMNS["map50_95"]])


def print_final_training_summary(*, args: argparse.Namespace, results_path: Path,
                                 elapsed_seconds: float, peak_gpu_memory_gb: float | None,
                                 best_path: Path, last_path: Path) -> None:
    rows = read_training_results(results_path)
    valid_rows = [row for row in rows if all(column in row and row[column] != "" for column in METRIC_COLUMNS.values())]
    best_row = max(valid_rows, key=result_fitness) if valid_rows else None
    final_row = valid_rows[-1] if valid_rows else None

    def epoch(row: dict[str, str] | None) -> str:
        if row is None:
            return "UNAVAILABLE"
        value = float(row.get("epoch", "nan"))
        return str(int(value)) if value.is_integer() else str(value)

    def metric(row: dict[str, str] | None, name: str) -> str:
        if row is None:
            return "UNAVAILABLE"
        return f"{float(row[METRIC_COLUMNS[name]]):.6f}"

    training_runtime_seconds = elapsed_seconds
    if final_row is not None and final_row.get("time", "") != "":
        training_runtime_seconds = float(final_row["time"])
    memory = f"{peak_gpu_memory_gb:.3f} GB (CUDA peak reserved)" if peak_gpu_memory_gb is not None else "UNAVAILABLE"
    print("\n" + "=" * 60)
    print("FINAL TRAINING SUMMARY")
    print("=" * 60)
    print(f"Run name: {args.run_name}")
    print(f"Representation: {args.representation}")
    print(f"Model: {args.model}")
    print(f"Augmentation profile: {args.augmentation_profile}")
    print(f"Fold: {args.fold}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs requested: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Freeze: {args.freeze}")
    print(f"Best epoch: {epoch(best_row)}")
    print("Best epoch metrics:")
    print(f"  Precision: {metric(best_row, 'precision')}")
    print(f"  Recall: {metric(best_row, 'recall')}")
    print(f"  mAP50: {metric(best_row, 'map50')}")
    print(f"  mAP50-95: {metric(best_row, 'map50_95')}")
    print(f"Final epoch: {epoch(final_row)}")
    print("Final epoch metrics:")
    print(f"  Precision: {metric(final_row, 'precision')}")
    print(f"  Recall: {metric(final_row, 'recall')}")
    print(f"  mAP50: {metric(final_row, 'map50')}")
    print(f"  mAP50-95: {metric(final_row, 'map50_95')}")
    print(f"Total runtime: {training_runtime_seconds:.1f} seconds ({training_runtime_seconds / 3600:.3f} hours)")
    print(f"Peak GPU memory: {memory}")
    print(f"best.pt: {best_path.resolve()}")
    print(f"last.pt: {last_path.resolve()}")
    print("Next action: Review this summary and run only the explicitly approved one-run evaluation command.")
    print("=" * 60)


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    model_spec = MODEL_SPECS[args.model]
    augmentation = AUGMENTATION_PROFILES[args.augmentation_profile]
    if not 1 <= args.fold <= 4:
        raise ValueError("--fold must be 1, 2, 3, or 4")
    if any(token in args.run_name for token in ("/", "\\", "..")):
        raise ValueError("--run-name must be a single safe directory name")
    run_root = project_root / "models" / "stage2a"
    run_dir = run_root / args.run_name
    manifest_path = project_root / "outputs" / "run_manifests" / f"{args.run_name}.json"
    if run_dir.exists() or manifest_path.exists():
        raise FileExistsError(f"Run name already exists; choose a unique --run-name: {args.run_name}")

    started = time.time(); started_utc = utc_now()
    wandb_logger: OptionalWandbLogger | None = None
    wandb_group = args.wandb_group or default_wandb_group(args)
    metadata: dict[str, object] = {
        "status": "STARTING", "stage": "Stage 2A training", "run_name": args.run_name,
        "project_root": str(project_root), "representation": args.representation, "fold": args.fold,
        "imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch, "device": args.device,
        "freeze": args.freeze, "seed": args.seed, "workers": args.workers,
        "fold_sha256": EXPECTED_FOLD_SHA256, "model_architecture": args.model,
        "pretrained_weight_identifier": f"Ultralytics COCO {model_spec['filename']}",
        "pretrained_weight_provenance": "Pre-positioned project asset; automatic weight download is disabled by the existence check.",
        "model_sha256": model_spec["expected_sha256"], "augmentation_profile": args.augmentation_profile,
        "experiment_family": args.experiment_family,
        "oversampling_mode": args.oversampling_mode,
        "oversampling_rule": OVERSAMPLING_RULES[args.oversampling_mode],
        "artifact_control": "Conservative chromatic-line repair" if args.representation == "conservative" else "Local interpolation chromatic-line repair",
        "wandb": {
            "requested": args.use_wandb, "entity": args.wandb_entity, "project": args.wandb_project,
            "group": wandb_group, "log_artifacts": args.wandb_log_artifacts,
            "role": "optional visualization/tracking layer; local metadata remains source of truth",
        },
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "wandb_group": wandb_group,
        "wandb_run_name": args.run_name,
        "wandb_run_id": None,
        "started_utc": started_utc,
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {name: installed_version(name) for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "Pillow", "PyYAML", "polars", "wandb")},
        "optimizer": "AdamW", "lr0": 0.01, "lrf": 0.1, "weight_decay": 0.0005,
        "box": args.box, "dfl": args.dfl, "cls": args.cls,
        "warmup_epochs": 0.5, "checkpoint_selection": "Ultralytics best.pt by validation fitness",
        "generated_checkpoint_paths": [], "generated_prediction_paths": [], "generated_metric_paths": [],
    }
    manifest_path = save_run_manifest(project_root, args.run_name, metadata)
    pending_path = run_root / "_metadata_pending" / f"{args.run_name}.json"
    try:
        yaml_path, val_stems = validate_fold_dataset(project_root, args.representation, args.fold)
        yaml_path, oversampling_manifest, oversampling_stats = prepare_oversampled_training_split(
            project_root=project_root, base_yaml=yaml_path, representation=args.representation,
            fold=args.fold, mode=args.oversampling_mode,
        )
        weight_path = project_root / "models" / "stage2a" / "pretrained" / str(model_spec["filename"])
        if not weight_path.is_file():
            raise RuntimeError(f"Pretrained weight is missing; place the approved file at: {weight_path}")
        observed_weight_sha256 = sha256(weight_path)
        expected_weight_sha256 = model_spec["expected_sha256"]
        if expected_weight_sha256 is not None and observed_weight_sha256 != expected_weight_sha256:
            raise RuntimeError(f"Pretrained weight SHA-256 changed: {weight_path}")
        metadata.update({
            "status": "RUNNING", "validation_samples": len(val_stems), "runtime_dataset_yaml": str(yaml_path.resolve()),
            "initialization": "official Ultralytics COCO pretrained weights; no competition weights",
            "model_sha256": observed_weight_sha256, "augmentation": augmentation,
            "oversampling": oversampling_stats,
            "oversampling_manifest": str(oversampling_manifest.resolve()) if oversampling_manifest else None,
        })
        write_json(pending_path, metadata); save_run_manifest(project_root, args.run_name, metadata)
        yolo_config_dir = project_root / "outputs" / "cache" / "ultralytics"
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
        import torch
        from ultralytics import YOLO

        disable_ultralytics_builtin_wandb()

        metadata["torch_cuda_available"] = torch.cuda.is_available()
        metadata["torch_cuda_version"] = torch.version.cuda
        metadata["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "CPU"
        wandb_config = {
            "experiment_family": args.experiment_family,
            "run_name": args.run_name,
            "model": args.model,
            "representation": args.representation,
            "augmentation_profile": args.augmentation_profile,
            "fold": args.fold,
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "seed": args.seed,
            "optimizer": "AdamW",
            "lr0": 0.01,
            "lrf": 0.1,
            "box": args.box,
            "dfl": args.dfl,
            "cls": args.cls,
            "hsv_v": augmentation["hsv_v"],
            "mosaic": augmentation["mosaic"],
            "freeze": args.freeze,
            "oversampling_mode": args.oversampling_mode,
            "oversampling_rule": OVERSAMPLING_RULES[args.oversampling_mode],
            "frozen_fold_hash": EXPECTED_FOLD_SHA256,
            "artifact_control": metadata["artifact_control"],
            "ultralytics_version": metadata["packages"]["ultralytics"],
            "torch_version": metadata["packages"]["torch"],
            "cuda_version": metadata["torch_cuda_version"],
            "device": args.device,
            "device_name": metadata["device_name"],
        }
        wandb_logger = OptionalWandbLogger(
            enabled=args.use_wandb, entity=args.wandb_entity, project=args.wandb_project,
            group=wandb_group, run_name=args.run_name, config=wandb_config,
        )
        metadata["wandb"].update({
            "status": wandb_logger.status, "run_id": wandb_logger.run_id, "error": wandb_logger.error,
        })
        metadata["wandb_run_id"] = wandb_logger.run_id
        write_json(pending_path, metadata); save_run_manifest(project_root, args.run_name, metadata)
        model = YOLO(str(weight_path))
        if wandb_logger.active:
            model.add_callback("on_fit_epoch_end", wandb_logger.on_fit_epoch_end)
        trainer_class = make_weak_gaussian_blur_trainer() if args.augmentation_profile == "gaussian_blur" else None
        cuda_device_index = int(str(args.device).split(",")[0]) if torch.cuda.is_available() and str(args.device).split(",")[0].isdigit() else None
        if cuda_device_index is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device_index)
        model.train(
            trainer=trainer_class,
            data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
            workers=args.workers, freeze=args.freeze, optimizer="AdamW", lr0=0.01, lrf=0.1,
            warmup_epochs=0.5, weight_decay=0.0005, box=args.box, dfl=args.dfl, cls=args.cls,
            seed=args.seed, deterministic=True, amp=False,
            hsv_h=augmentation["hsv_h"], hsv_s=augmentation["hsv_s"], hsv_v=augmentation["hsv_v"],
            degrees=augmentation["degrees"], translate=augmentation["translate"], scale=augmentation["scale"],
            shear=augmentation["shear"], perspective=augmentation["perspective"],
            flipud=augmentation["flipud"], fliplr=augmentation["fliplr"], mosaic=augmentation["mosaic"],
            mixup=augmentation["mixup"], copy_paste=augmentation["copy_paste"],
            close_mosaic=0, patience=0, save=True, plots=False,
            project=str(run_root), name=args.run_name, exist_ok=False, verbose=True,
        )
        peak_gpu_memory_gb = None
        if cuda_device_index is not None:
            torch.cuda.synchronize(cuda_device_index)
            peak_gpu_memory_gb = torch.cuda.max_memory_reserved(cuda_device_index) / 1_000_000_000
        best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
        result_files = [run_dir / "results.csv", run_dir / "args.yaml"]
        metadata.update({
            "status": "COMPLETE", "ended_utc": utc_now(), "elapsed_seconds": time.time() - started,
            "generated_checkpoint_paths": [str(path.resolve()) for path in (best, last) if path.is_file()],
            "generated_metric_paths": [str(path.resolve()) for path in result_files if path.is_file()],
        })
        experiment_metadata = run_dir / "experiment_metadata.json"
        write_json(experiment_metadata, metadata); pending_path.unlink(missing_ok=True)
        manifest_path = save_run_manifest(project_root, args.run_name, metadata)
        if args.wandb_log_artifacts and wandb_logger.active:
            wandb_logger.log_artifact(
                name=f"{args.run_name}-evidence", artifact_type="model-evidence",
                files=[best, experiment_metadata, manifest_path, run_dir / "args.yaml", run_dir / "results.csv"],
            )
        wandb_logger.finish(exit_code=0)
        metadata["wandb"].update({"status": wandb_logger.status, "error": wandb_logger.error})
        write_json(experiment_metadata, metadata)
        manifest_path = save_run_manifest(project_root, args.run_name, metadata)
        files = [experiment_metadata, manifest_path, *result_files]
        if oversampling_manifest:
            files.append(oversampling_manifest)
        downloads = [manifest_path, experiment_metadata, best, last, *result_files]
        if oversampling_manifest:
            downloads.append(oversampling_manifest)
        print_output_manifest(stage="Stage 2A training", run_name=args.run_name, project_root=project_root,
                              files=files, directories=[run_dir], models=[best, last], downloads=downloads,
                              next_action="Run the one-run evaluation command only after reviewing GPU runtime and VRAM use.")
        print_final_training_summary(args=args, results_path=run_dir / "results.csv",
                                     elapsed_seconds=float(metadata["elapsed_seconds"]),
                                     peak_gpu_memory_gb=peak_gpu_memory_gb,
                                     best_path=best, last_path=last)
    except BaseException as exc:
        if wandb_logger is not None:
            wandb_logger.finish(exit_code=1)
        metadata.update({"status": "FAILED_OR_INTERRUPTED", "ended_utc": utc_now(), "elapsed_seconds": time.time() - started,
                         "error_type": type(exc).__name__, "error": str(exc)})
        write_json(pending_path, metadata); manifest_path = save_run_manifest(project_root, args.run_name, metadata)
        print_output_manifest(stage="Stage 2A training (failed/interrupted)", run_name=args.run_name, project_root=project_root,
                              files=[manifest_path, pending_path], directories=[run_dir],
                              models=[run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"],
                              downloads=[manifest_path, pending_path, run_dir / "results.csv", run_dir / "weights" / "best.pt"],
                              next_action="Return the manifest and error text; do not start another experiment.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exactly one frozen-fold Stage 2A training experiment.")
    parser.add_argument("--representation", choices=("conservative", "local"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="yolov8n")
    parser.add_argument("--augmentation-profile", choices=tuple(AUGMENTATION_PROFILES), default="baseline")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freeze", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--box", type=float, default=7.5, help="Bounding-box loss weight (B2 default: 7.5).")
    parser.add_argument("--dfl", type=float, default=1.5, help="Distribution focal loss weight (B2 default: 1.5).")
    parser.add_argument("--cls", type=float, default=0.5, help="Classification loss weight (B2 default: 0.5).")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-family", default="B2-final-performance")
    parser.add_argument("--oversampling-mode", choices=tuple(OVERSAMPLING_RULES), default="none")
    parser.add_argument("--use-wandb", action="store_true", help="Enable optional fail-open training telemetry.")
    parser.add_argument("--wandb-entity", default=None,
                        help="Optional W&B entity. No personal entity is embedded in the submission package.")
    parser.add_argument("--wandb-project", default="KAMP-manufacturing-xray")
    parser.add_argument("--wandb-group", help="Defaults to B2-<imgsz> or B2-hardos-<imgsz>.")
    parser.add_argument("--wandb-log-artifacts", action="store_true", help="Optionally upload best.pt and lightweight evidence files.")
    parser.add_argument("--project-root", "--workspace", dest="project_root", type=Path, default=project_root_from_script())
    main(parser.parse_args())

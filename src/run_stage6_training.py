from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from stage2a_common import print_output_manifest, project_root_from_script, save_run_manifest, sha256, write_json
from stage6_common import (
    BASELINE_AUGMENTATION,
    BASELINE_BATCH_SIZE,
    FIXED_TRAINING,
    STAGE6_EXPERIMENTS,
    b0_fold_step_budget,
    ensure_safe_run_name,
    prepare_stage6_runtime_dataset,
    pretrained_path,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def make_controlled_m1_trainer(reference_budget: dict[str, object]):
    """Replay the B0 micro-batch warmup, accumulation, and LR trajectory for M1."""
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ControlledM1Trainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            self.stage6_optimizer_steps = 0
            self.stage6_micro_batch_iterations = 0
            self.stage6_target_micro_batch_iterations = int(reference_budget["micro_batch_iterations"])
            self.stage6_target_optimizer_steps = int(reference_budget["approx_optimizer_updates"])
            self.stage6_target_warmup_iterations = int(reference_budget["warmup_iterations"])
            self.stage6_actual_warmup_iterations = 0
            self.stage6_reference_batches_per_epoch = int(reference_budget["batches_per_epoch"])
            self.stage6_reference_epochs = int(reference_budget["epochs"])
            self.stage6_lr_snapshots: dict[str, dict[str, float]] = {}
            super().__init__(*args, **kwargs)

        def _setup_scheduler(self) -> None:
            # The ordinary epoch scheduler must not follow the much longer patch epoch.
            # preprocess_batch() applies the B0 virtual-epoch LR at every micro-batch.
            self.lf = lambda _epoch: 1.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

        def _get_warmup_iterations(self, _num_batches: int) -> int:
            return self.stage6_target_warmup_iterations

        def _b0_lr_factor(self, micro_batch_index: int) -> float:
            virtual_epoch = min(
                micro_batch_index // self.stage6_reference_batches_per_epoch,
                self.stage6_reference_epochs - 1,
            )
            return max(1 - virtual_epoch / self.stage6_reference_epochs, 0) * (1 - self.args.lrf) + self.args.lrf

        def _snapshot_lrs(self) -> dict[str, float]:
            return {
                f"pg{index}_{group.get('param_group', 'unknown')}": float(group["lr"])
                for index, group in enumerate(self.optimizer.param_groups)
            }

        def preprocess_batch(self, batch):
            ni = self.stage6_micro_batch_iterations
            if ni >= self.stage6_target_micro_batch_iterations:
                raise RuntimeError("Controlled M1 exceeded its B0 micro-batch budget")
            virtual_epoch = ni // self.stage6_reference_batches_per_epoch
            factor = self._b0_lr_factor(ni)
            if ni < self.stage6_target_warmup_iterations:
                self.stage6_actual_warmup_iterations += 1
            for group in self.optimizer.param_groups:
                target_lr = group["initial_lr"] * factor
                if ni < self.stage6_target_warmup_iterations:
                    start_lr = self.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0
                    group["lr"] = start_lr + (target_lr - start_lr) * ni / self.stage6_target_warmup_iterations
                elif virtual_epoch == 0:
                    # Ultralytics uses `ni < nw`, so B0 carries the ni=nw-1
                    # warmup LR until the scheduler advances at epoch 1.
                    start_lr = self.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0
                    group["lr"] = start_lr + (group["initial_lr"] - start_lr) * (
                        self.stage6_target_warmup_iterations - 1
                    ) / self.stage6_target_warmup_iterations
                else:
                    group["lr"] = target_lr
            if ni == 0:
                self.stage6_lr_snapshots["first"] = self._snapshot_lrs()
            if ni == self.stage6_target_warmup_iterations:
                self.stage6_lr_snapshots["after_warmup"] = self._snapshot_lrs()
            if ni == self.stage6_target_micro_batch_iterations - 1:
                self.stage6_lr_snapshots["final"] = self._snapshot_lrs()
            self.stage6_micro_batch_iterations += 1
            if self.stage6_micro_batch_iterations >= self.stage6_target_micro_batch_iterations:
                self.stop = True
            return super().preprocess_batch(batch)

        def optimizer_step(self) -> None:
            super().optimizer_step()
            self.stage6_optimizer_steps += 1
            if self.stage6_optimizer_steps > self.stage6_target_optimizer_steps:
                raise RuntimeError("Controlled M1 exceeded its B0 optimizer-update budget")

    return ControlledM1Trainer


def require_fixed_protocol(args: argparse.Namespace) -> None:
    if not 1 <= args.fold <= 4:
        raise ValueError("--fold must be 1, 2, 3, or 4")
    for name in ("imgsz", "seed"):
        expected = FIXED_TRAINING[name]
        if getattr(args, name) != expected:
            raise ValueError(f"Stage 6 protocol freezes --{name}={expected}")
    if args.experiment in {"m1_patch", "m1_patch256"}:
        expected_steps = b0_fold_step_budget(args.fold)["approx_optimizer_updates"]
        if args.experiment == "m1_patch256" and args.run_purpose != "controlled":
            raise ValueError("M1-256 is authorized only as a controlled B0-progress ablation")
        if args.run_purpose == "sanity":
            if not 1 <= args.epochs <= 3 or args.max_optimizer_steps is not None or args.batch != BASELINE_BATCH_SIZE:
                raise ValueError("M1 sanity runs require 1-3 epochs, batch=8, and no optimizer-step limit")
        elif args.run_purpose == "controlled":
            if args.epochs != FIXED_TRAINING["epochs"] or args.batch != BASELINE_BATCH_SIZE:
                raise ValueError("Controlled M1 requires epochs=30 and batch=8")
            if args.max_optimizer_steps != expected_steps:
                raise ValueError(f"Controlled M1 fold {args.fold} requires --max-optimizer-steps {expected_steps}")
        elif args.run_purpose == "practical":
            if args.epochs != FIXED_TRAINING["epochs"] or args.max_optimizer_steps is not None or args.batch != BASELINE_BATCH_SIZE:
                raise ValueError("Practical M1 requires epochs=30, batch=8, and no optimizer-step limit")
        else:
            raise ValueError("M1 requires --run-purpose sanity, controlled, or practical")
    elif args.experiment == "m2_p2":
        if args.run_purpose == "sanity":
            if not 1 <= args.epochs <= 2 or args.max_optimizer_steps is not None or args.batch != BASELINE_BATCH_SIZE:
                raise ValueError("M2 sanity runs require 1-2 epochs, batch=8, and no optimizer-step limit")
        elif (args.run_purpose != "evidence" or args.epochs != FIXED_TRAINING["epochs"]
              or args.max_optimizer_steps is not None or args.batch != BASELINE_BATCH_SIZE):
            raise ValueError("M2 evidence runs require --run-purpose evidence, epochs=30, batch=8, and no optimizer-step limit")
    else:
        if (args.run_purpose != "evidence" or args.epochs != FIXED_TRAINING["epochs"]
                or args.max_optimizer_steps is not None or args.batch != BASELINE_BATCH_SIZE):
            raise ValueError("M3-M6 require --run-purpose evidence, epochs=30, batch=8, and no optimizer-step limit")
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("--batch must be positive and --workers must be non-negative")
    ensure_safe_run_name(args.run_name)


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    require_fixed_protocol(args)
    spec = STAGE6_EXPERIMENTS[args.experiment]
    baseline_budget = b0_fold_step_budget(args.fold)
    run_root = project_root / "models" / "stage6"
    run_dir = run_root / args.run_name
    manifest_path = project_root / "outputs" / "run_manifests" / f"{args.run_name}.json"
    if run_dir.exists() or manifest_path.exists():
        raise FileExistsError(f"Run name already exists: {args.run_name}")

    started = time.time()
    metadata = {
        "status": "STARTING",
        "stage": "Stage 6 training",
        "run_name": args.run_name,
        "experiment": args.experiment,
        "experiment_id": spec["experiment_id"],
        "run_purpose": args.run_purpose,
        "evidence_eligible": args.run_purpose != "sanity",
        "comparison_role": {
            "sanity": "pipeline_only_not_for_model_comparison",
            "controlled": "primary_controlled_ablation",
            "practical": "separate_practical_optimization",
            "evidence": "standard_stage6_evidence",
        }[args.run_purpose],
        "project_root": str(project_root),
        "fold": args.fold,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "max_optimizer_steps": args.max_optimizer_steps,
        "b0_reference_step_budget": baseline_budget,
        "target_micro_batch_iterations": baseline_budget["micro_batch_iterations"] if args.run_purpose == "controlled" else None,
        "target_optimizer_updates": baseline_budget["approx_optimizer_updates"] if args.run_purpose == "controlled" else None,
        "target_warmup_iterations": baseline_budget["warmup_iterations"] if args.run_purpose == "controlled" else None,
        "scheduler_policy": baseline_budget["scheduler_policy"] if args.run_purpose == "controlled" else "ultralytics_default_epoch_scheduler",
        "freeze": FIXED_TRAINING["freeze"],
        "architecture": spec["architecture"],
        "preprocessing": spec["preprocessing"],
        "patch_setting": spec["patch_setting"],
        "augmentation": BASELINE_AUGMENTATION,
        "optimizer": FIXED_TRAINING["optimizer"],
        "lr0": FIXED_TRAINING["lr0"],
        "lrf": FIXED_TRAINING["lrf"],
        "warmup_epochs": FIXED_TRAINING["warmup_epochs"],
        "weight_decay": FIXED_TRAINING["weight_decay"],
        "amp": FIXED_TRAINING["amp"],
        "deterministic": FIXED_TRAINING["deterministic"],
        "checkpoint_selection": "Ultralytics best.pt by validation fitness",
        "started_utc": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "Pillow", "PyYAML", "polars")},
        "generated_checkpoint_paths": [],
        "generated_prediction_paths": [],
        "generated_metric_paths": [],
    }
    save_run_manifest(project_root, args.run_name, metadata)
    pending = run_root / "_metadata_pending" / f"{args.run_name}.json"
    try:
        dataset_yaml, validation_stems = prepare_stage6_runtime_dataset(project_root, args.experiment, args.fold)
        weights = pretrained_path(project_root)
        p2_config = project_root / "configs" / "stage6" / "yolov8n_p2.yaml"
        if args.experiment == "m2_p2" and not p2_config.is_file():
            raise FileNotFoundError(p2_config)
        metadata.update({
            "status": "RUNNING",
            "validation_source_samples": len(validation_stems),
            "runtime_dataset_yaml": str(dataset_yaml.resolve()),
            "pretrained_weight_path": str(weights.resolve()),
            "pretrained_weight_sha256": sha256(weights),
            "initialization": (
                "YOLOv8n-P2 YAML with compatible layers transferred from approved official COCO yolov8n.pt; new P2/head layers initialized by Ultralytics"
                if args.experiment == "m2_p2"
                else "approved official Ultralytics COCO yolov8n.pt"
            ),
            "p2_config_path": str(p2_config.resolve()) if args.experiment == "m2_p2" else None,
            "p2_config_sha256": sha256(p2_config) if args.experiment == "m2_p2" else None,
        })
        write_json(pending, metadata)
        save_run_manifest(project_root, args.run_name, metadata)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / "outputs" / "cache" / "ultralytics"))
        import torch
        from ultralytics import YOLO

        if args.experiment == "m2_p2":
            model = YOLO(str(p2_config)).load(str(weights))
        else:
            model = YOLO(str(weights))
        device_index = int(str(args.device).split(",")[0]) if torch.cuda.is_available() and str(args.device).split(",")[0].isdigit() else None
        if device_index is not None:
            torch.cuda.set_device(device_index)
            torch.cuda.reset_peak_memory_stats()
        trainer_class = make_controlled_m1_trainer(baseline_budget) if args.run_purpose == "controlled" else None
        train_kwargs = dict(
            data=str(dataset_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            freeze=FIXED_TRAINING["freeze"],
            optimizer=FIXED_TRAINING["optimizer"],
            lr0=FIXED_TRAINING["lr0"],
            lrf=FIXED_TRAINING["lrf"],
            warmup_epochs=FIXED_TRAINING["warmup_epochs"],
            weight_decay=FIXED_TRAINING["weight_decay"],
            seed=args.seed,
            deterministic=True,
            amp=False,
            **BASELINE_AUGMENTATION,
            close_mosaic=0,
            patience=0,
            save=True,
            plots=False,
            project=str(run_root),
            name=args.run_name,
            exist_ok=False,
            verbose=True,
        )
        if trainer_class is not None:
            train_kwargs["trainer"] = trainer_class
        model.train(**train_kwargs)
        actual_optimizer_steps = getattr(model.trainer, "stage6_optimizer_steps", None)
        actual_micro_batch_iterations = getattr(model.trainer, "stage6_micro_batch_iterations", None)
        actual_warmup_iterations = (
            getattr(model.trainer, "stage6_actual_warmup_iterations", None)
            if args.run_purpose == "controlled" else None
        )
        actual_lr_snapshots = getattr(model.trainer, "stage6_lr_snapshots", None)
        actual_batch_size = int(model.trainer.batch_size)
        actual_train_dataset_size = len(model.trainer.train_loader.dataset)
        actual_train_batches_per_epoch = len(model.trainer.train_loader)
        if args.run_purpose == "controlled" and actual_optimizer_steps != args.max_optimizer_steps:
            raise RuntimeError(f"Controlled optimizer-step budget mismatch: {actual_optimizer_steps} != {args.max_optimizer_steps}")
        if args.run_purpose == "controlled" and actual_micro_batch_iterations != baseline_budget["micro_batch_iterations"]:
            raise RuntimeError(
                f"Controlled micro-batch budget mismatch: {actual_micro_batch_iterations} "
                f"!= {baseline_budget['micro_batch_iterations']}"
            )
        if args.run_purpose == "controlled" and actual_batch_size != BASELINE_BATCH_SIZE:
            raise RuntimeError(f"Controlled batch size changed during training: {actual_batch_size} != {BASELINE_BATCH_SIZE}")
        peak_gb = None
        if device_index is not None:
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_reserved() / 1_000_000_000
        best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
        result_files = [run_dir / "results.csv", run_dir / "args.yaml"]
        metadata.update({
            "status": "COMPLETE",
            "ended_utc": utc_now(),
            "elapsed_seconds": time.time() - started,
            "peak_cuda_reserved_gb": peak_gb,
            "actual_optimizer_steps": actual_optimizer_steps,
            "target_micro_batch_iterations": baseline_budget["micro_batch_iterations"] if args.run_purpose == "controlled" else None,
            "actual_micro_batch_iterations": actual_micro_batch_iterations,
            "target_optimizer_updates": baseline_budget["approx_optimizer_updates"] if args.run_purpose == "controlled" else None,
            "actual_optimizer_updates": actual_optimizer_steps,
            "target_warmup_iterations": baseline_budget["warmup_iterations"] if args.run_purpose == "controlled" else None,
            "actual_warmup_iterations": actual_warmup_iterations,
            "lr_first": actual_lr_snapshots.get("first") if actual_lr_snapshots else None,
            "lr_after_warmup": actual_lr_snapshots.get("after_warmup") if actual_lr_snapshots else None,
            "lr_final": actual_lr_snapshots.get("final") if actual_lr_snapshots else None,
            "scheduler_policy": baseline_budget["scheduler_policy"] if args.run_purpose == "controlled" else "ultralytics_default_epoch_scheduler",
            "actual_batch_size": actual_batch_size,
            "actual_train_dataset_size": actual_train_dataset_size,
            "actual_train_batches_per_epoch": actual_train_batches_per_epoch,
            "generated_checkpoint_paths": [str(path.resolve()) for path in (best, last) if path.is_file()],
            "generated_metric_paths": [str(path.resolve()) for path in result_files if path.is_file()],
        })
        experiment_metadata = run_dir / "experiment_metadata.json"
        write_json(experiment_metadata, metadata)
        pending.unlink(missing_ok=True)
        manifest = save_run_manifest(project_root, args.run_name, metadata)
        print_output_manifest(
            stage="Stage 6 training",
            run_name=args.run_name,
            project_root=project_root,
            files=[experiment_metadata, manifest, *result_files],
            directories=[run_dir],
            models=[best, last],
            downloads=[experiment_metadata, manifest, best, last, *result_files],
            next_action="Evaluate only this completed fold with evaluate_stage6_run.py.",
        )
    except BaseException as exc:
        metadata.update({"status": "FAILED_OR_INTERRUPTED", "ended_utc": utc_now(), "elapsed_seconds": time.time() - started, "error_type": type(exc).__name__, "error": str(exc)})
        write_json(pending, metadata)
        manifest = save_run_manifest(project_root, args.run_name, metadata)
        print_output_manifest(
            stage="Stage 6 training (failed/interrupted)", run_name=args.run_name, project_root=project_root,
            files=[manifest, pending], directories=[run_dir], models=[run_dir / "weights" / "best.pt"],
            downloads=[manifest, pending, run_dir / "results.csv", run_dir / "weights" / "best.pt"],
            next_action="Return the manifest and error; do not start another experiment.",
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exactly one Stage 6 experiment and one frozen fold.")
    parser.add_argument("--experiment", choices=tuple(STAGE6_EXPERIMENTS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-purpose", choices=("sanity", "controlled", "practical", "evidence"), required=True)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    main(parser.parse_args())

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from stage2a_common import project_root_from_script, read_csv, resolve_from_root


def subgroup_metrics(objects: list[dict[str, str]], features: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    groups = {
        "small": lambda feature: float(feature["bbox_min_side_px"]) <= 8.0,
        "low_contrast": lambda feature: float(feature["absolute_median_difference"]) <= 4.0,
        "small_low_contrast": lambda feature: (
            float(feature["bbox_min_side_px"]) <= 8.0
            and float(feature["absolute_median_difference"]) <= 4.0
        ),
    }
    result: dict[str, Any] = {}
    for name, predicate in groups.items():
        selected = [row for row in objects if predicate(features[(row["stem"], row["object_id"])])]
        tp = sum(row["status"] == "TP" for row in selected)
        result[f"{name}_n"] = len(selected)
        result[f"{name}_recall"] = tp / len(selected) if selected else None
    return result


def main(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    run_dir = resolve_from_root(root, args.run_dir)
    evaluation_dir = resolve_from_root(root, args.evaluation_dir)
    metadata_path = run_dir / "experiment_metadata.json"
    metrics_path = evaluation_dir / "metrics.json"
    objects_path = evaluation_dir / "object_predictions.csv"
    required = [metadata_path, metrics_path, objects_path, evaluation_dir / "error_types.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Completed local evaluation evidence is missing: " + "; ".join(missing))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_id = metadata.get("wandb_run_id")
    if not run_id:
        raise RuntimeError(
            "Training metadata has no wandb_run_id; refusing to create or guess a W&B run. "
            "Evaluation outputs remain valid locally."
        )
    stored_entity = metadata.get("wandb_entity")
    stored_project = metadata.get("wandb_project")
    stored_group = metadata.get("wandb_group")
    stored_run_name = metadata.get("wandb_run_name")
    if not all((stored_entity, stored_project, stored_group, stored_run_name)):
        raise RuntimeError("Training metadata is missing one or more persisted W&B identity fields")
    if args.wandb_entity is not None and args.wandb_entity != stored_entity:
        raise RuntimeError("--wandb-entity differs from the entity persisted by training")
    if args.wandb_project is not None and args.wandb_project != stored_project:
        raise RuntimeError("--wandb-project differs from the project persisted by training")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    objects = read_csv(objects_path)
    feature_rows = read_csv(root / "outputs" / "tables" / "01b_object_xray_features.csv")
    features = {(row["stem"], row["object_id"]): row for row in feature_rows}
    object_keys = {(row["stem"], row["object_id"]) for row in objects}
    if not object_keys <= set(features):
        raise RuntimeError("Evaluation objects do not have complete Stage 1 feature coverage")

    payload = {
        "eval/tp": int(metrics["tp"]),
        "eval/fp": int(metrics["fp"]),
        "eval/fn": int(metrics["fn"]),
        "eval/precision": float(metrics["precision"]),
        "eval/recall": float(metrics["recall"]),
        "eval/f1": float(metrics["f1"]),
        "eval/ap50": float(metrics["ap50"]),
        "eval/map50_95": float(metrics["map50_95"]),
        "eval/localization_failure": sum(row["status"] == "FN" and row["error_type"] == "LOCALIZATION_FAILURE" for row in objects),
        "eval/low_confidence_fn": sum(row["status"] == "FN" and row["error_type"] == "LOW_CONFIDENCE" for row in objects),
        "eval/no_detection_fn": sum(row["status"] == "FN" and row["error_type"] == "NO_DETECTION" for row in objects),
        **{f"eval/{key}": value for key, value in subgroup_metrics(objects, features).items()},
    }

    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError("Optional package 'wandb' is required only for this explicit logging command.") from exc
    run = wandb.init(
        entity=str(stored_entity), project=str(stored_project), id=str(run_id), resume="must",
        reinit=True,
    )
    run.summary.update(payload)
    run.log(payload)
    if args.wandb_log_artifacts:
        artifact = wandb.Artifact(name=f"{metadata['run_name']}-evaluation", type="evaluation-evidence")
        for path in (metrics_path, evaluation_dir / "evaluation_metadata.json", root / "outputs" / "run_manifests" / f"{metadata['run_name']}.json"):
            if path.is_file():
                artifact.add_file(str(path.resolve()), name=path.name)
        run.log_artifact(artifact)
    run.finish()
    print(json.dumps({"run_name": metadata["run_name"], "wandb_run_id": run_id, **payload}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optionally attach completed local Stage 2A evaluation metrics to its W&B run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--wandb-entity", help="Optional assertion; the persisted training entity remains authoritative.")
    parser.add_argument("--wandb-project", help="Optional assertion; the persisted training project remains authoritative.")
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    parser.add_argument("--project-root", "--workspace", dest="project_root", type=Path, default=project_root_from_script())
    main(parser.parse_args())

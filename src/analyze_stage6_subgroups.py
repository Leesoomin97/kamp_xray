from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage2a_common import print_output_manifest, project_root_from_script, read_csv, resolve_from_root, save_run_manifest, write_csv, write_json


def number(row: dict[str, str], names: tuple[str, ...]) -> float:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return float(value)
    raise KeyError(f"None of these feature columns exist: {names}")


def main(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    oof_dir = resolve_from_root(project_root, args.oof_dir)
    predictions = read_csv(oof_dir / "06_object_predictions.csv")
    features = read_csv(project_root / "outputs" / "tables" / "01b_object_xray_features.csv")
    feature_by_key = {(row["stem"], int(row["object_id"])): row for row in features}
    if len(predictions) != 1147 or len(feature_by_key) != 1147:
        raise RuntimeError("Stage 6 subgroup analysis expects the unchanged 1,147-object corpus")
    joined = []
    for prediction in predictions:
        key = (prediction["stem"], int(prediction["object_id"]))
        if key not in feature_by_key:
            raise RuntimeError(f"Missing Stage 1 feature row: {key}")
        feature = feature_by_key[key]
        joined.append({
            **prediction,
            "bbox_min_side_px": number(feature, ("bbox_min_side_px",)),
            "absolute_median_contrast": number(feature, ("absolute_median_difference", "abs_median_contrast")),
        })

    masks = {
        "ALL": lambda row: True,
        "SMALL_MIN_SIDE_LE_8PX": lambda row: row["bbox_min_side_px"] <= 8,
        "LOW_CONTRAST_ABS_MEDIAN_LE_4": lambda row: row["absolute_median_contrast"] <= 4,
        "SMALL_AND_LOW_CONTRAST": lambda row: row["bbox_min_side_px"] <= 8 and row["absolute_median_contrast"] <= 4,
    }
    rows: list[dict[str, Any]] = []
    for subgroup, predicate in masks.items():
        part = [row for row in joined if predicate(row)]
        tp = sum(row["status"] == "TP" for row in part)
        fn = sum(row["status"] == "FN" for row in part)
        rows.append({
            "subgroup": subgroup, "n_objects": len(part), "tp": tp, "fn": fn,
            "recall": tp / (tp + fn) if tp + fn else "",
            "localization_failure": sum(row["error_type"] == "LOCALIZATION_FAILURE" for row in part),
            "low_confidence_fn": sum(row["error_type"] == "LOW_CONFIDENCE" for row in part),
            "no_detection_fn": sum(row["error_type"] == "NO_DETECTION" for row in part),
            "definition_status": "Stage 1B pre-specified exploratory threshold",
        })
    output = resolve_from_root(project_root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = output / "06_subgroup_metrics.csv"
    summary = output / "06_subgroup_metrics.json"
    write_csv(table, list(rows[0]), rows)
    write_json(summary, {"subgroups": rows, "physical_density_inferred": False, "final_safety_threshold_selected": False})
    manifest = save_run_manifest(project_root, args.run_name, {
        "status": "COMPLETE", "stage": "Stage 6 subgroup analysis", "run_name": args.run_name,
        "generated_metric_paths": [str(table.resolve()), str(summary.resolve())],
    })
    print_output_manifest(stage="Stage 6 subgroup analysis", run_name=args.run_name, project_root=project_root,
        files=[manifest, table, summary], directories=[output], downloads=[manifest, table, summary],
        next_action="Compare this OOF result with B0 and existing B2 using the frozen reporting rules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pre-specified Stage 6 OOF subgroups; never trains.")
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    main(parser.parse_args())

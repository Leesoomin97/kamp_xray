from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_stage6_candidate_local_redetection import (
    MATCH_THRESHOLDS,
    POLICIES,
    REPORT_CONFIDENCE,
    SCALES,
    apply_policy,
    evaluate,
    load_b0,
    load_csv,
    load_local_features,
)
from stage2a_common import box_iou, match_image, project_root_from_script


TARGETS = (
    ("002_20200624_162540(3)", 0),
    ("002_20200714_083057(5)", 2),
)


def close(a: Any, b: Any, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return a == b


def bbox_close(a: list[float], b: list[float], tolerance: float = 1e-7) -> bool:
    return len(a) == len(b) and all(close(x, y, tolerance) for x, y in zip(a, b))


def final_object_rows(images: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stem, item in sorted(images.items()):
        gts = item["gt_boxes"]
        predictions = item["predictions"]
        matches, _ = match_image(gts, predictions, REPORT_CONFIDENCE, 0.5)
        for object_id, gt in enumerate(gts):
            if object_id in matches:
                prediction_index, overlap = matches[object_id]
                status = "TP"
            else:
                ranked = sorted(
                    ((box_iou(prediction["bbox"], gt), float(prediction["confidence"]), index)
                     for index, prediction in enumerate(predictions)),
                    reverse=True,
                    key=lambda value: (value[0], value[1]),
                )
                overlap, _, prediction_index = ranked[0] if ranked else (0.0, 0.0, -1)
                status = "FN"
            prediction = predictions[prediction_index] if prediction_index >= 0 else None
            rows.append({
                "stem": stem,
                "object_id": object_id,
                "status": status,
                "final_candidate_id": "" if prediction is None else prediction["candidate_id"],
                "final_confidence": "" if prediction is None else float(prediction["confidence"]),
                "final_bbox": None if prediction is None else prediction["bbox"],
                "final_iou": overlap,
            })
    return rows


def raw_selection_audit(raw_path: Path, feature_path: Path) -> tuple[int, int]:
    raw = load_csv(raw_path)
    features = load_csv(feature_path)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in raw:
        grouped.setdefault((row["candidate_id"], int(row["crop_scale"])), []).append(row)
    mismatch = 0
    for row in features:
        key = (row["candidate_id"], int(row["crop_scale"]))
        selected = max(
            grouped.get(key, []),
            key=lambda item: (float(item["base_local_iou"]), float(item["local_confidence"])),
            default=None,
        )
        if selected is None:
            mismatch += int(row["selected_local_prediction_index"] != "")
            continue
        expected_bbox = json.loads(selected["source_bbox_json"])
        observed_bbox = json.loads(row["selected_source_bbox_json"])
        mismatch += int(
            int(row["selected_local_prediction_index"]) != int(selected["local_prediction_index"])
            or not close(row["selected_local_confidence"], selected["local_confidence"])
            or not close(row["selected_base_local_iou"], selected["base_local_iou"])
            or not bbox_close(observed_bbox, expected_bbox)
        )
    return len(features), mismatch


def scale_summary_mismatches(expected: pd.DataFrame, observed_path: Path) -> int:
    observed = pd.read_csv(observed_path, encoding="utf-8-sig")
    keys = ["crop_scale", "match_threshold", "policy"]
    merged = expected.merge(observed, on=keys, suffixes=("_expected", "_observed"), how="outer", indicator=True)
    mismatches = int((merged["_merge"] != "both").sum())
    for column in ("tp", "fp", "fn", "precision", "recall", "f1", "ap50", "map50_95",
                   "localization_failure", "low_confidence", "no_detection", "b0_fn_rescued", "b0_tp_lost"):
        left, right = f"{column}_expected", f"{column}_observed"
        if left not in merged or right not in merged:
            mismatches += len(merged)
            continue
        for a, b in zip(merged[left], merged[right]):
            mismatches += int(not close(a, b, 1e-7))
    return mismatches


def main(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    required = {
        "scale_summary": root / "outputs/tables/06_candidate_redetection_scale_summary_fold1.csv",
        "rescue_matrix": root / "outputs/tables/06_candidate_redetection_rescue_matrix_fold1.csv",
        "raw_local": root / "outputs/stage6_candidate_redetection/fold1/local_predictions.csv",
        "local_features": root / "outputs/stage6_candidate_redetection/fold1/candidate_scale_features.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required completed inference/analysis files are missing:\n" + "\n".join(missing))

    b0_path = root / "outputs/stage2a_runs/kamp_conservative_640_fold1_e30_fullft_v1/image_predictions.csv"
    base_images, candidates = load_b0(b0_path)
    local = load_local_features(required["local_features"], candidates)
    feature_rows, raw_selection_mismatches = raw_selection_audit(required["raw_local"], required["local_features"])
    _, b0_objects, _ = evaluate(root, base_images)
    b0_status = {(row["stem"], int(row["object_id"])): row["status"] for row in b0_objects}

    invariant_rows: list[dict[str, Any]] = []
    differing_policy_rows = 0
    transformation_mismatches = {policy: 0 for policy in POLICIES}
    expected_summary: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for scale in SCALES:
        for threshold in MATCH_THRESHOLDS:
            per_policy_predictions: dict[str, dict[str, dict[str, Any]]] = {}
            for policy in POLICIES:
                images = apply_policy(base_images, candidates, local, scale, threshold, policy)
                metrics, objects, _ = evaluate(root, images)
                result_status = {(row["stem"], int(row["object_id"])): row["status"] for row in objects}
                rescued = sum(b0_status[key] == "FN" and value == "TP" for key, value in result_status.items())
                lost = sum(b0_status[key] == "TP" and value == "FN" for key, value in result_status.items())
                expected_summary.append({"crop_scale": scale, "match_threshold": threshold, "policy": policy,
                                         **metrics, "b0_fn_rescued": rescued, "b0_tp_lost": lost})
                per_policy_predictions[policy] = {
                    prediction["candidate_id"]: prediction
                    for image in images.values()
                    for prediction in image["predictions"]
                }
                target_map = {(row["stem"], int(row["object_id"])): row for row in final_object_rows(images)}
                for stem, object_id in TARGETS:
                    outcome = target_map[(stem, object_id)]
                    candidate_id = outcome["final_candidate_id"]
                    selected = local[(candidate_id, scale)] if candidate_id else None
                    local_matched = bool(selected and selected["bbox"] is not None and selected["base_local_iou"] >= threshold)
                    target_rows.append({
                        "stem": stem,
                        "object_id": object_id,
                        "crop_scale": scale,
                        "match_threshold": threshold,
                        "policy": policy,
                        **outcome,
                        "selected_local_matched": local_matched,
                        "selected_local_confidence": "" if not local_matched else selected["confidence"],
                        "selected_base_local_iou": "" if not local_matched else selected["base_local_iou"],
                        "selected_local_bbox_json": "" if not local_matched else json.dumps(selected["bbox"], separators=(",", ":")),
                    })

            for candidate_id in candidates:
                selected = local[(candidate_id, scale)]
                matched = selected["bbox"] is not None and selected["base_local_iou"] >= threshold
                signatures = {
                    (selected["confidence"] if matched else None,
                     json.dumps(selected["bbox"], separators=(",", ":")) if matched else None,
                     selected["base_local_iou"] if matched else None)
                    for _policy in POLICIES
                }
                differs = len(signatures) != 1
                differing_policy_rows += int(differs)
                base = candidates[candidate_id]
                for policy in POLICIES:
                    actual = per_policy_predictions[policy][candidate_id]
                    expected_bbox = (selected["bbox"] if matched and policy in {"REFINE_BOX", "CONFIRM_AND_REFINE"}
                                     else base["base_bbox"])
                    expected_confidence = (max(float(base["base_confidence"]), float(selected["confidence"]))
                                           if matched and policy in {"CONFIRM_CONFIDENCE", "CONFIRM_AND_REFINE"}
                                           else float(base["base_confidence"]))
                    transformation_mismatches[policy] += int(
                        not bbox_close(actual["bbox"], expected_bbox)
                        or not close(actual["confidence"], expected_confidence)
                    )
                invariant_rows.append({
                    "stem": candidates[candidate_id]["stem"],
                    "candidate_id": candidate_id,
                    "crop_scale": scale,
                    "match_threshold": threshold,
                    "matched_local_prediction_id": "" if not matched else selected.get("prediction_id", ""),
                    "matched_local_confidence": "" if not matched else selected["confidence"],
                    "matched_local_bbox_json": "" if not matched else json.dumps(selected["bbox"], separators=(",", ":")),
                    "policy_variant_count": len(signatures),
                    "policy_invariant": not differs,
                })

    summary_mismatches = scale_summary_mismatches(pd.DataFrame(expected_summary), required["scale_summary"])
    rescue = pd.read_csv(required["rescue_matrix"], encoding="utf-8-sig")
    trajectory_semantics_ok = "confidence_trajectory_json" in rescue.columns

    output_table = root / "outputs/tables/06_candidate_redetection_policy_invariant_fold1.csv"
    output_targets = root / "outputs/tables/06_candidate_redetection_policy_target_audit_fold1.csv"
    output_report = root / "outputs/eda/06_candidate_redetection_policy_audit_fold1.md"
    for path in (output_table, output_targets, output_report):
        if path.exists():
            raise FileExistsError(f"Audit output already exists: {path}")
    pd.DataFrame(invariant_rows).to_csv(output_table, index=False, encoding="utf-8-sig")
    pd.DataFrame(target_rows).to_csv(output_targets, index=False, encoding="utf-8-sig")
    lines = [
        "# Stage 6 candidate local re-detection policy audit",
        "",
        f"- candidate-scale feature rows checked: {feature_rows}",
        f"- raw-local selection mismatches: {raw_selection_mismatches}",
        f"- policy-dependent local-match rows: {differing_policy_rows}",
        f"- CONFIRM_CONFIDENCE transformation mismatches: {transformation_mismatches['CONFIRM_CONFIDENCE']}",
        f"- REFINE_BOX transformation mismatches: {transformation_mismatches['REFINE_BOX']}",
        f"- CONFIRM_AND_REFINE transformation mismatches: {transformation_mismatches['CONFIRM_AND_REFINE']}",
        f"- reconstructed scale-summary value mismatches: {summary_mismatches}",
        f"- legacy confidence trajectory column present: {trajectory_semantics_ok}",
        "",
        "`confidence_trajectory_json` is the confidence of the final prediction associated with each GT object after policy application and evaluation. It is not the selected local-prediction confidence. The final associated candidate can differ by policy because bbox refinement changes which candidate overlaps a GT object.",
        "",
        "The detailed target-object rows record both final prediction confidence and selected local confidence separately.",
    ]
    output_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "raw_selection_mismatches": raw_selection_mismatches,
        "policy_dependent_local_match_rows": differing_policy_rows,
        "policy_transformation_mismatches": transformation_mismatches,
        "scale_summary_mismatches": summary_mismatches,
        "outputs": [str(output_table.resolve()), str(output_targets.resolve()), str(output_report.resolve())],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Stage 6 candidate-local matching and policy invariants without inference.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    main(parser.parse_args())

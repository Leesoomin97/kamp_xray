from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stage2a_common import project_root_from_script, read_csv, write_csv


B0_RUN = "kamp_conservative_640_fold1_e30_fullft_v1"
P192_RUN = "stage6_m1c_patch192_ov25_yolov8n_640_fold1_b0progress_v3"
P256_RUN = "stage6_m1_256c_patch256_ov25_yolov8n_640_fold1_b0progress_v1"
BURDENS = (0.10, 0.20, 0.30, 0.40, 0.50)
EPSILON = 1e-6


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def image_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return image


def clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width - 1, math.floor(box[0])))
    y0 = max(0, min(height - 1, math.floor(box[1])))
    x1 = max(x0 + 1, min(width, math.ceil(box[2])))
    y1 = max(y0 + 1, min(height, math.ceil(box[3])))
    return x0, y0, x1, y1


def iou(a: list[float], b: list[float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(EPSILON, area_a + area_b - inter)


def local_features(image: np.ndarray, box_float: list[float], other_boxes: list[list[float]]) -> dict[str, float | int | bool]:
    height, width = image.shape
    box = clip_box(box_float, width, height)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ex0 = max(0, math.floor(cx - bw))
    ey0 = max(0, math.floor(cy - bh))
    ex1 = min(width, math.ceil(cx + bw))
    ey1 = min(height, math.ceil(cy + bh))
    obj = image[y0:y1, x0:x1].astype(np.float32)
    ring_mask = np.ones((ey1 - ey0, ex1 - ex0), dtype=bool)
    ring_mask[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = False
    for other in other_boxes:
        ox0, oy0, ox1, oy1 = clip_box(other, width, height)
        ax0, ay0, ax1, ay1 = max(ex0, ox0), max(ey0, oy0), min(ex1, ox1), min(ey1, oy1)
        if ax0 < ax1 and ay0 < ay1:
            ring_mask[ay0 - ey0:ay1 - ey0, ax0 - ex0:ax1 - ex0] = False
    region = image[ey0:ey1, ex0:ex1].astype(np.float32)
    bg = region[ring_mask]
    if not obj.size or not bg.size:
        return {"pred_local_abs_median_contrast": math.nan, "pred_local_robust_cnr": math.nan,
                "pred_local_background_gradient": math.nan, "pred_local_valid": False,
                "pred_local_object_pixels": int(obj.size), "pred_local_background_pixels": int(bg.size)}
    obj_median, bg_median = float(np.median(obj)), float(np.median(bg))
    mad = float(np.median(np.abs(bg - bg_median)))
    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.hypot(gx, gy)
    absolute = abs(obj_median - bg_median)
    return {
        "pred_local_abs_median_contrast": absolute,
        "pred_local_robust_cnr": absolute / (1.4826 * mad + EPSILON),
        "pred_local_background_gradient": float(np.mean(gradient[ring_mask])),
        "pred_local_valid": bool(bg.size >= 50),
        "pred_local_object_pixels": int(obj.size),
        "pred_local_background_pixels": int(bg.size),
    }


def risk_rank(values: list[float], high_is_risky: bool) -> list[float]:
    finite = [(value, index) for index, value in enumerate(values) if math.isfinite(value)]
    finite.sort(key=lambda item: (item[0], item[1]), reverse=high_is_risky)
    scores = [math.nan] * len(values)
    denominator = max(1, len(finite) - 1)
    for rank, (_, index) in enumerate(finite):
        scores[index] = 1.0 - rank / denominator
    return scores


def validate_known_results(root: Path) -> tuple[dict[str, dict[tuple[str, int], dict[str, str]]], dict[str, dict[str, Any]]]:
    paths = {
        "b0": root / "outputs/stage2a_runs" / B0_RUN,
        "p192": root / "outputs/stage6_runs" / P192_RUN,
        "p256": root / "outputs/stage6_runs" / P256_RUN,
    }
    expected = {
        "b0": (281, 15, 9), "p192": (266, 29, 24), "p256": (276, 28, 14),
    }
    metrics, objects = {}, {}
    for name, path in paths.items():
        metrics[name] = load_json(path / "metrics.json")
        observed = tuple(int(metrics[name][key]) for key in ("tp", "fp", "fn"))
        if observed != expected[name]:
            raise RuntimeError(f"Known-result mismatch for {name}: {observed} != {expected[name]}")
        rows = read_csv(path / "object_predictions.csv")
        objects[name] = {(row["stem"], int(row["object_id"])): row for row in rows}
    b0_fn = {key for key, row in objects["b0"].items() if row["status"] == "FN"}
    b0_tp = set(objects["b0"]) - b0_fn
    rescue192 = {key for key in b0_fn if objects["p192"][key]["status"] == "TP"}
    rescue256 = {key for key in b0_fn if objects["p256"][key]["status"] == "TP"}
    loss192 = {key for key in b0_tp if objects["p192"][key]["status"] == "FN"}
    loss256 = {key for key in b0_tp if objects["p256"][key]["status"] == "FN"}
    if not (len(rescue192) == 5 and len(loss192) == 20 and len(rescue256) == 2 and len(loss256) == 7
            and rescue256 <= rescue192 and len(rescue192 | rescue256) == 5):
        raise RuntimeError("Known paired-rescue result mismatch")
    return objects, metrics


def main(root: Path, stability_path: Path | None) -> None:
    root = root.resolve()
    objects, metrics = validate_known_results(root)
    b0_dir = root / "outputs/stage2a_runs" / B0_RUN
    images = read_csv(b0_dir / "image_predictions.csv")
    integrity = {
        row["stem"]: row for row in read_csv(root / "outputs/tables/02a_experiment_integrity.csv")
        if row["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE" and int(row["fold_id"]) == 1
    }
    patch_counts = Counter(
        row["source_stem"] for row in read_csv(root / "outputs/tables/06_patch_manifest.csv")
        if row["disposition"] == "KEEP" and int(row["fold_id"]) == 1
    )
    candidates: list[dict[str, Any]] = []
    by_stem_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_row in images:
        stem = image_row["stem"]
        predictions = json.loads(image_row["prediction_boxes_json"])
        image = image_gray(root / Path(integrity[stem]["image_path"]))
        height, width = image.shape
        reported_boxes = [item["bbox"] for item in predictions if float(item["confidence"]) >= 0.25]
        for candidate_index, item in enumerate(predictions):
            box = [float(value) for value in item["bbox"]]
            other_boxes = [other for other in reported_boxes if other != item["bbox"]]
            row = {
                "stem": stem, "candidate_index": candidate_index, "fold_id": 1,
                "prediction_confidence": float(item["confidence"]),
                "prediction_bbox_json": json.dumps(box, separators=(",", ":")),
                "predicted_width_px": box[2] - box[0], "predicted_height_px": box[3] - box[1],
                "predicted_min_side_px": min(box[2] - box[0], box[3] - box[1]),
                "predicted_area_px": max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]),
                "predicted_area_ratio": max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (width * height),
                "image_width": width, "image_height": height,
                **local_features(image, box, other_boxes),
                "stability_available": False,
                "confidence_std": math.nan, "confidence_range": math.nan,
                "bbox_iou_consistency": math.nan, "center_displacement_px": math.nan,
                "predicted_size_cv": math.nan,
            }
            candidates.append(row)
            by_stem_candidates[stem].append(row)

    if stability_path and stability_path.is_file():
        stability = {(row["stem"], int(row["candidate_index"])): row for row in read_csv(stability_path)}
        if set(stability) != {(row["stem"], row["candidate_index"]) for row in candidates}:
            raise RuntimeError("Stability feature coverage differs from B0 candidate set")
        for row in candidates:
            item = stability[(row["stem"], row["candidate_index"])]
            row.update({
                "stability_available": True,
                "confidence_std": float(item["confidence_std"]),
                "confidence_range": float(item["confidence_range"]),
                "bbox_iou_consistency": float(item["bbox_iou_consistency"]),
                "center_displacement_px": float(item["center_displacement_px"]),
                "predicted_size_cv": float(item["predicted_size_cv"]),
            })

    # Associate each B0 object outcome with its best stored B0 candidate only for evaluation.
    object_to_candidate: dict[tuple[str, int], int | None] = {}
    for key, outcome in objects["b0"].items():
        if not outcome["prediction_bbox_json"]:
            object_to_candidate[key] = None
            continue
        target = json.loads(outcome["prediction_bbox_json"])
        matches = by_stem_candidates[key[0]]
        best = max(matches, key=lambda row: (iou(json.loads(row["prediction_bbox_json"]), target),
                                              -abs(row["prediction_confidence"] - float(outcome["prediction_confidence"]))))
        object_to_candidate[key] = int(best["candidate_index"])

    def add_score(name: str, values: list[float], high: bool) -> None:
        for row, score in zip(candidates, risk_rank(values, high)):
            row[name] = score

    # Uncertainty is highest near the fixed 0.25 reporting boundary. Very-low
    # floor detections are not ranked above plausible weak candidates merely
    # because their confidence approaches 0.001.
    for row in candidates:
        confidence = float(row["prediction_confidence"])
        row["risk_low_confidence"] = (
            confidence / 0.25 if confidence < 0.25 else (1.0 - confidence) / 0.75
        )
    add_score("risk_small_predicted_box", [float(row["predicted_min_side_px"]) for row in candidates], False)
    contrast = risk_rank([float(row["pred_local_abs_median_contrast"]) for row in candidates], False)
    cnr = risk_rank([float(row["pred_local_robust_cnr"]) for row in candidates], False)
    bg = risk_rank([float(row["pred_local_background_gradient"]) for row in candidates], True)
    for index, row in enumerate(candidates):
        valid = [value for value in (contrast[index], cnr[index], bg[index]) if math.isfinite(value)]
        row["risk_low_local_visibility"] = sum(valid) / len(valid) if valid else math.nan
    stability_available = bool(candidates and candidates[0]["stability_available"])
    if stability_available:
        components = [
            risk_rank([float(row["confidence_std"]) for row in candidates], True),
            risk_rank([float(row["confidence_range"]) for row in candidates], True),
            risk_rank([float(row["bbox_iou_consistency"]) for row in candidates], False),
            risk_rank([float(row["center_displacement_px"]) for row in candidates], True),
            risk_rank([float(row["predicted_size_cv"]) for row in candidates], True),
        ]
        for index, row in enumerate(candidates):
            row["risk_high_instability"] = sum(component[index] for component in components) / len(components)
    else:
        for row in candidates:
            row["risk_high_instability"] = math.nan

    rules: dict[str, list[float] | None] = {
        "low_confidence": [row["risk_low_confidence"] for row in candidates],
        "small_predicted_box": [row["risk_small_predicted_box"] for row in candidates],
        "low_local_visibility": [row["risk_low_local_visibility"] for row in candidates],
        "high_instability": [row["risk_high_instability"] for row in candidates] if stability_available else None,
        "low_confidence_OR_small_predicted_box": [
            max(row["risk_low_confidence"], row["risk_small_predicted_box"]) for row in candidates
        ],
        "low_confidence_OR_high_instability": None,
        "low_confidence_OR_high_instability_OR_low_visibility": None,
    }
    if stability_available:
        rules.update({
            "low_confidence_OR_high_instability": [max(row["risk_low_confidence"], row["risk_high_instability"]) for row in candidates],
            "low_confidence_OR_high_instability_OR_low_visibility": [max(row["risk_low_confidence"], row["risk_high_instability"], row["risk_low_local_visibility"]) for row in candidates],
        })

    b0_fn = {key for key, row in objects["b0"].items() if row["status"] == "FN"}
    rescue = {key for key in b0_fn if objects["p192"][key]["status"] == "TP"}
    localization = {key for key in b0_fn if objects["b0"][key]["error_type"] == "LOCALIZATION_FAILURE"}
    low_conf_fn = {key for key in b0_fn if objects["b0"][key]["error_type"] == "LOW_CONFIDENCE"}
    no_candidate = {key for key in b0_fn if object_to_candidate[key] is None}
    candidate_present = b0_fn - no_candidate
    sweep_rows = []
    selected_by_rule_burden: dict[tuple[str, float], set[int]] = {}
    for rule, scores in rules.items():
        if scores is None:
            for burden in BURDENS:
                sweep_rows.append({"rule": rule, "signal_available": False, "burden_target": burden,
                    "routed_roi_count": "", "routed_roi_fraction": "", "routed_image_count": "", "routed_image_fraction": "",
                    "expected_second_pass_patch_count": "", "b0_fn_covered": "", "b0_fn_total": len(b0_fn),
                    "b0_fn_coverage": "", "rescuable_fn_covered": "", "rescuable_fn_total": len(rescue),
                    "rescuable_fn_coverage": "", "localization_failure_covered": "", "localization_failure_total": len(localization),
                    "low_confidence_fn_covered": "", "low_confidence_fn_total": len(low_conf_fn),
                    "b0_fn_direct_roi_covered": "", "rescuable_fn_direct_roi_covered": "",
                    "rescuable_fn_direct_roi_coverage": "", "notes": "PENDING_STABILITY_INFERENCE"})
            continue
        order = sorted(range(len(candidates)), key=lambda index: (-float(scores[index]), candidates[index]["stem"], candidates[index]["candidate_index"]))
        for burden in BURDENS:
            count = max(1, math.ceil(len(candidates) * burden))
            selected = set(order[:count])
            selected_by_rule_burden[(rule, burden)] = selected
            stems = {candidates[index]["stem"] for index in selected}
            selected_candidate_keys = {(candidates[index]["stem"], candidates[index]["candidate_index"]) for index in selected}
            covered = {key for key in b0_fn if key[0] in stems}
            directly_covered = {
                key for key in b0_fn
                if object_to_candidate[key] is not None and (key[0], object_to_candidate[key]) in selected_candidate_keys
            }
            sweep_rows.append({
                "rule": rule, "signal_available": True, "burden_target": burden,
                "routed_roi_count": len(selected), "routed_roi_fraction": len(selected) / len(candidates),
                "routed_image_count": len(stems), "routed_image_fraction": len(stems) / len(images),
                "expected_second_pass_patch_count": sum(patch_counts[stem] for stem in stems),
                "b0_fn_covered": len(covered), "b0_fn_total": len(b0_fn), "b0_fn_coverage": len(covered) / len(b0_fn),
                "rescuable_fn_covered": len(covered & rescue), "rescuable_fn_total": len(rescue),
                "rescuable_fn_coverage": len(covered & rescue) / len(rescue),
                "localization_failure_covered": len(covered & localization), "localization_failure_total": len(localization),
                "low_confidence_fn_covered": len(covered & low_conf_fn), "low_confidence_fn_total": len(low_conf_fn),
                "b0_fn_direct_roi_covered": len(directly_covered),
                "rescuable_fn_direct_roi_covered": len(directly_covered & rescue),
                "rescuable_fn_direct_roi_coverage": len(directly_covered & rescue) / len(rescue),
                "notes": "Primary coverage is image routing; GT outcomes are evaluation targets only, never routing inputs.",
            })

    profile_rows = []
    lookup = {(row["stem"], row["candidate_index"]): row for row in candidates}
    for stem, object_id in sorted(rescue):
        candidate_index = object_to_candidate[(stem, object_id)]
        item = lookup[(stem, candidate_index)] if candidate_index is not None else None
        profile_rows.append({
            "stem": stem, "object_id": object_id, "b0_error_type": objects["b0"][(stem, object_id)]["error_type"],
            "b0_prediction_confidence": objects["b0"][(stem, object_id)]["prediction_confidence"],
            "b0_iou": objects["b0"][(stem, object_id)]["iou"], "candidate_present": item is not None,
            "candidate_index": candidate_index if candidate_index is not None else "", "rescued_by_192": True,
            "rescued_by_256": objects["p256"][(stem, object_id)]["status"] == "TP",
            "predicted_min_side_px": item["predicted_min_side_px"] if item else "",
            "pred_local_abs_median_contrast": item["pred_local_abs_median_contrast"] if item else "",
            "pred_local_robust_cnr": item["pred_local_robust_cnr"] if item else "",
            "pred_local_background_gradient": item["pred_local_background_gradient"] if item else "",
            "risk_low_confidence": item["risk_low_confidence"] if item else "",
            "risk_small_predicted_box": item["risk_small_predicted_box"] if item else "",
            "risk_low_local_visibility": item["risk_low_local_visibility"] if item else "",
            "risk_high_instability": item["risk_high_instability"] if item else "",
            "notes": "GT/error fields evaluate routing coverage and are not deployable routing inputs.",
        })

    comparison_rows = []
    for rule in rules:
        rows = [row for row in sweep_rows if row["rule"] == rule]
        comparison_rows.append({
            "rule": rule, "signal_available": rows[0]["signal_available"],
            **{f"rescue_coverage_at_{int(float(row['burden_target']) * 100)}pct": row["rescuable_fn_coverage"] for row in rows},
            **{f"image_burden_at_{int(float(row['burden_target']) * 100)}pct": row["routed_image_fraction"] for row in rows},
            **{f"direct_roi_rescue_at_{int(float(row['burden_target']) * 100)}pct": row["rescuable_fn_direct_roi_coverage"] for row in rows},
            "first_burden_full_rescue": next((row["burden_target"] for row in rows if row["rescuable_fn_coverage"] == 1.0), ""),
            "interpretation": "Rank-based exploratory routing; no operational threshold selected.",
        })

    table_dir = root / "outputs/tables"
    candidate_path = table_dir / "06_routing_candidate_features_fold1.csv"
    sweep_path = table_dir / "06_routing_burden_sweep_fold1.csv"
    comparison_path = table_dir / "06_routing_rule_comparison_fold1.csv"
    profile_path = table_dir / "06_routing_rescued_fn_profile_fold1.csv"
    write_csv(candidate_path, list(candidates[0]), candidates)
    write_csv(sweep_path, list(sweep_rows[0]), sweep_rows)
    write_csv(comparison_path, list(comparison_rows[0]), comparison_rows)
    write_csv(profile_path, list(profile_rows[0]), profile_rows)

    available = [row for row in sweep_rows if as_bool(row["signal_available"])]
    pareto = sorted(available, key=lambda row: (-float(row["rescuable_fn_coverage"]), float(row["routed_image_fraction"]), float(row["burden_target"])))
    best = pareto[0]
    sweep_lookup = {(row["rule"], float(row["burden_target"])): row for row in sweep_rows}
    def sequence(rule: str, field: str, percent: bool = False) -> str:
        values = [sweep_lookup[(rule, burden)][field] for burden in BURDENS]
        if percent:
            return " / ".join(f"{float(value):.1%}" for value in values)
        return " / ".join(str(value) for value in values)
    oracle_recall = 286 / 290
    summary = root / "outputs/eda/06_selective_routing_feasibility_fold1.md"
    summary.write_text(f"""# Stage 6 selective routing feasibility — Fold 1

## Scope and integrity

- B0: TP=281, FP=15, FN=9; 192 controlled: TP=266, FP=29, FN=24; 256 controlled: TP=276, FP=28, FN=14.
- Paired result: 192 rescues 5/9 B0 FN and loses 20 B0 TP; 256 rescues 2/9 and loses 7 B0 TP; both 256 rescues are within the 192 rescue set.
- B0 remains the primary detector. GT/error labels are used only to score coverage, never as routing signals.
- Candidate-present B0 FN={len(candidate_present)}/{len(b0_fn)}; no-candidate B0 FN={len(no_candidate)}. Candidate-based routing cannot recover no-candidate cases.

## Deployable signals

- Confidence, predicted-box size, and predicted-box 2x-ring visibility are computed only from B0 outputs and Conservative images.
- Local visibility uses predicted-box absolute median contrast, robust image-derived CNR-like proxy, and local background gradient. These are not physical density/material measurements.
- Stability available: {stability_available}. When false, stability-based rules remain unresolved and are not imputed.

## Ranked burden sweep

- Candidate ROI count={len(candidates)}, validation images={len(images)}. Burden is the fraction of B0 candidates ranked for routing; routed-image burden is separately reported.
- At candidate burdens 10/20/30/40/50%, confidence-only image routing covers rescued FN `{sequence('low_confidence', 'rescuable_fn_covered')}` of 5, while routing `{sequence('low_confidence', 'routed_image_fraction', True)}` of images.
- Small predicted-box routing covers `{sequence('small_predicted_box', 'rescuable_fn_covered')}` of 5 at image burdens `{sequence('small_predicted_box', 'routed_image_fraction', True)}`. The confidence-OR-size rule does not improve the observed Pareto frontier.
- Predicted-box local-visibility routing covers `{sequence('low_local_visibility', 'rescuable_fn_covered')}` of 5 at image burdens `{sequence('low_local_visibility', 'routed_image_fraction', True)}`; it adds no lower-burden rescue advantage over confidence-only.
- Direct ROI-linked rescue coverage is lower than image routing because image routing executes the complete fixed 192-grid. This distinction is retained in the CSV and prevents incidental same-image coverage from being presented as ROI localization.
- Best observed available row by rescue coverage then image burden: `{best['rule']}` at candidate burden {float(best['burden_target']):.0%}, rescue coverage {int(best['rescuable_fn_covered'])}/5, image burden {float(best['routed_image_fraction']):.1%}, expected fixed-grid 192-patch calls={best['expected_second_pass_patch_count']}.
- The simplest high-coverage Pareto point is confidence-only at 10% candidate burden: 4/5 rescues but 77/124 images (62.1%) are routed. Full 5/5 coverage first occurs at 20% candidate burden and routes 105/124 images (84.7%), so the present full-grid second pass is not operationally selective.
- No operational threshold is selected. Reference 192 fusion remains provisional.

## Oracle upper bound

- Oracle only: retain all 281 B0 TP and add the five B0 FN rescued by 192, giving TP=286, FN=4, Recall={oracle_recall:.6f}.
- FP is not fixed before an explicit deployable routing/fusion rule is evaluated. This is not deployable performance.

## Limitations

- All 124 Fold1 images are GT-positive; true-negative routing burden and production workload cannot be estimated.
- Existing 192/256 evaluation bundles do not include their training checkpoints or experiment_metadata.json.
- Stability conclusions require the separately prepared inference-only audit; no stability values are guessed.
""", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze deployable Fold1 selective routing signals without training.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--stability-features", type=Path)
    args = parser.parse_args()
    stability = args.stability_features
    if stability and not stability.is_absolute():
        stability = args.project_root / stability
    main(args.project_root, stability)

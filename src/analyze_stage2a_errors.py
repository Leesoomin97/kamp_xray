from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from run_stage1a_structural_eda import parse_bmp, quantile, spearman, write_png
from run_stage1b_xray_difficulty_eda import boundary_component, image_arrays
from run_stage1d_artifact_control import mask_points, repair_conservative, repair_local
from stage2a_common import (
    print_output_manifest, project_root_from_script, read_csv, resolve_from_root,
    save_run_manifest, verify_frozen_folds, write_csv,
)


FEATURES = {
    "bbox_min_side_px": "size",
    "bbox_area_ratio": "size",
    "bbox_center_x_norm": "position",
    "bbox_center_y_norm": "position",
    "image_edge_distance_px": "position",
    "product_edge_distance_px": "position",
    "absolute_median_difference": "visibility",
    "signed_median_difference": "visibility",
    "cnr_like_robust": "visibility",
    "bbox_aspect_ratio": "background_shape",
    "object_iqr": "background_shape",
    "background_gradient_mean": "background_shape",
    "background_iqr": "background_shape",
}


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def key(row: dict[str, str]) -> tuple[str, int]:
    return row["stem"], int(row["object_id"])


def summary_row(feature: str, group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(r["status"] == "TP" for r in rows); total = len(rows)
    folds = sorted({int(r["fold_id"]) for r in rows})
    return {"feature": feature, "group_or_range": group, "object_count": total, "tp": tp,
            "fn": total - tp, "recall": tp / total if total else "", "fold_coverage": len(folds),
            "fold_ids": "|".join(map(str, folds)), "notes": "Exploratory quantile group; not an operational threshold."}


def quantile_table(feature: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [r for r in rows if not math.isnan(f(r.get(feature)))]
    values = [f(r[feature]) for r in valid]
    if not values:
        return []
    cuts = [quantile(values, q) for q in (.25, .5, .75)]
    groups = [[] for _ in range(4)]
    for row in valid:
        value = f(row[feature]); index = sum(value > cut for cut in cuts)
        groups[index].append(row)
    bounds = [f"(-inf,{cuts[0]:.8g}]", f"({cuts[0]:.8g},{cuts[1]:.8g}]",
              f"({cuts[1]:.8g},{cuts[2]:.8g}]", f"({cuts[2]:.8g},inf)"]
    return [summary_row(feature, f"Q{i + 1} {bounds[i]}", group) for i, group in enumerate(groups)]


def finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [f(row.get(field)) for row in rows if not math.isnan(f(row.get(field)))]


def distribution_fields(values: list[float], prefix: str) -> dict[str, Any]:
    if not values:
        return {f"{prefix}_n": 0, f"{prefix}_median": "", f"{prefix}_q25": "", f"{prefix}_q75": ""}
    return {f"{prefix}_n": len(values), f"{prefix}_median": quantile(values, .5),
            f"{prefix}_q25": quantile(values, .25), f"{prefix}_q75": quantile(values, .75)}


def confidence_iou_relationships(feature: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if not math.isnan(f(row.get(feature)))]
    values = [f(row[feature]) for row in valid]
    if not values:
        return []
    cuts = [quantile(values, q) for q in (.25, .5, .75)]
    groups = [[] for _ in range(4)]
    for row in valid:
        groups[sum(f(row[feature]) > cut for cut in cuts)].append(row)
    result = []
    for index, group in enumerate(groups):
        confidence = finite_values(group, "prediction_confidence"); overlaps = finite_values(group, "iou")
        result.append({"feature": feature, "exploratory_quantile": f"Q{index + 1}",
                       "lower_bound": "" if index == 0 else cuts[index - 1],
                       "upper_bound": "" if index == 3 else cuts[index],
                       "object_count": len(group), "tp": sum(row["status"] == "TP" for row in group),
                       "fn": sum(row["status"] == "FN" for row in group),
                       "localization_failure_count": sum(row["error_type"] == "LOCALIZATION_FAILURE" for row in group),
                       **distribution_fields(confidence, "confidence"), **distribution_fields(overlaps, "iou"),
                       "notes": "Exploratory feature quartile; confidence excludes objects with no retained prediction; IoU is diagnostic localization overlap."})
    return result


def continuous_correlations(feature: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for outcome in ("prediction_confidence", "iou"):
        pairs = [(f(row.get(feature)), f(row.get(outcome))) for row in rows]
        pairs = [(x, y) for x, y in pairs if not math.isnan(x) and not math.isnan(y)]
        result.append({"feature": feature, "outcome": outcome, "n": len(pairs),
                       "spearman": spearman([x for x, _ in pairs], [y for _, y in pairs]) if len(pairs) >= 2 else "",
                       "interpretation": "Descriptive monotonic association only; correlation does not imply causation.",
                       "limitation": "Prediction confidence is absent for objects with no retained prediction." if outcome == "prediction_confidence" else "IoU includes unmatched/best-candidate localization diagnostics."})
    return result


def detectability_bins(feature: str, rows: list[dict[str, Any]], requested_bins: int = 10) -> list[dict[str, Any]]:
    """Create distribution-derived bins without imposing a fixed pixel threshold."""
    valid = [row for row in rows if not math.isnan(f(row.get(feature)))]
    values = sorted(f(row[feature]) for row in valid)
    if not values:
        return []
    cuts = sorted({quantile(values, index / requested_bins) for index in range(1, requested_bins)})
    groups: list[list[dict[str, Any]]] = [[] for _ in range(len(cuts) + 1)]
    for row in valid:
        groups[sum(f(row[feature]) > cut for cut in cuts)].append(row)
    result = []
    for index, group in enumerate(groups):
        if not group:
            continue
        group_values = [f(row[feature]) for row in group]
        confidence = finite_values(group, "prediction_confidence")
        overlaps = finite_values(group, "iou")
        tp = sum(row["status"] == "TP" for row in group)
        result.append({
            "feature": feature, "bin_id": len(result) + 1,
            "binning_method": f"empirical {requested_bins}-quantile cuts; duplicate cuts collapsed",
            "lower_observed": min(group_values), "upper_observed": max(group_values),
            "object_count": len(group), "tp": tp, "fn": len(group) - tp,
            "recall": tp / len(group), "fold_coverage": len({int(row["fold_id"]) for row in group}),
            **distribution_fields(confidence, "confidence"), **distribution_fields(overlaps, "iou"),
            "notes": "Distribution-derived diagnostic bin; not an empirical detection limit or safety threshold.",
        })
    return result


def pixel_transition_rows(bins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for lower, upper in zip(bins, bins[1:]):
        result.append({
            "lower_bin_id": lower["bin_id"], "upper_bin_id": upper["bin_id"],
            "candidate_boundary_projected_px": (f(lower["upper_observed"]) + f(upper["lower_observed"])) / 2,
            "lower_bin_range": f"{lower['lower_observed']}..{lower['upper_observed']}",
            "upper_bin_range": f"{upper['lower_observed']}..{upper['upper_observed']}",
            "lower_object_count": lower["object_count"], "upper_object_count": upper["object_count"],
            "lower_recall": lower["recall"], "upper_recall": upper["recall"],
            "recall_change_with_increasing_size": f(upper["recall"]) - f(lower["recall"]),
            "notes": "Adjacent empirical-bin contrast only; the largest change is a Stage 3 candidate, not a fixed threshold.",
        })
    return result


def pixel_recall_plot(path: Path, bins: list[dict[str, Any]]) -> None:
    """Write a dependency-free PNG; exact bin bounds/counts remain in the companion CSV."""
    width, height, left, top, bottom = 1000, 600, 70, 40, 70
    canvas = bytearray([255] * (width * height * 3))
    plot_height = height - top - bottom
    def paint(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            for x in range(max(0, x0), min(width, x1)):
                offset = (y * width + x) * 3
                canvas[offset:offset + 3] = bytes(color)
    paint(left - 2, top, left + 2, height - bottom + 2, (40, 40, 40))
    paint(left - 2, height - bottom - 2, width - 20, height - bottom + 2, (40, 40, 40))
    if bins:
        slot = (width - left - 40) / len(bins)
        for index, row in enumerate(bins):
            x0 = int(left + index * slot + slot * .15); x1 = int(left + (index + 1) * slot - slot * .15)
            y1 = height - bottom
            y0 = int(y1 - f(row["recall"]) * plot_height)
            paint(x0, y0, x1, y1, (68, 114, 196))
            fn_fraction = int(row["fn"]) / int(row["object_count"])
            paint(x0, y0, x1, min(y1, y0 + max(2, int(fn_fraction * plot_height))), (192, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, width, height, bytes(canvas))


def group_metrics(rows: list[dict[str, Any]], feature: str, getter: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[getter(row)].append(row)
    result = []
    for label, group in sorted(groups.items()):
        stems = {r["stem"] for r in group}; tp = sum(r["status"] == "TP" for r in group)
        result.append({"feature": feature, "group": label, "image_count": len(stems), "object_count": len(group),
                       "tp": tp, "fn": len(group) - tp, "recall": tp / len(group) if group else "",
                       "notes": "Descriptive association only; acquisition factors are confounded."})
    return result


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    return quantile(sorted_values, p)


def bootstrap(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], unit: str,
              repetitions: int, seed: int) -> tuple[float, float, float, int]:
    selected = [r for r in rows if predicate(r)]
    if not selected:
        return math.nan, math.nan, math.nan, 0
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        label = row["stem"] if unit == "image" else row["final_group_component_id"]
        by_unit[label].append(row)
    labels = sorted(by_unit); rng = random.Random(seed); estimates = []
    for _ in range(repetitions):
        sampled = [item for _ in labels for item in by_unit[rng.choice(labels)]]
        estimates.append(sum(r["status"] == "TP" for r in sampled) / len(sampled))
    estimates.sort(); observed = sum(r["status"] == "TP" for r in selected) / len(selected)
    return observed, percentile(estimates, .025), percentile(estimates, .975), len(selected)


def min_point_distance(box: list[float], points: list[tuple[int, int]]) -> float:
    if not points:
        return math.nan
    x0, y0, x1, y1 = box
    def d(point: tuple[int, int]) -> float:
        x, y = point; dx = max(x0 - x, 0, x - x1); dy = max(y0 - y, 0, y - y1)
        return math.hypot(dx, dy)
    return min(d(point) for point in points)


def local_complexity(gray: list[bytes], box: list[float]) -> tuple[float, float]:
    height, width = len(gray), len(gray[0]); x0, y0, x1, y1 = box
    margin = max(4, int(round(max(x1 - x0, y1 - y0))))
    ax0, ay0 = max(1, int(x0) - margin), max(1, int(y0) - margin)
    ax1, ay1 = min(width - 1, int(x1) + margin), min(height - 1, int(y1) + margin)
    values, gradients = [], []
    for y in range(ay0, ay1):
        for x in range(ax0, ax1):
            if x0 <= x <= x1 and y0 <= y <= y1:
                continue
            values.append(float(gray[y][x]))
            gradients.append(math.hypot((gray[y][x + 1] - gray[y][x - 1]) / 2,
                                        (gray[y + 1][x] - gray[y - 1][x]) / 2))
    return ((quantile(values, .75) - quantile(values, .25)) if values else math.nan,
            statistics.fmean(gradients) if gradients else math.nan)


def annotate_case(path: Path, source: Path, gt_boxes: list[list[float]], prediction_boxes: list[list[float]], representation: str) -> None:
    image = parse_bmp(source); width, height = int(image["width"]), int(image["height"])
    gray, mask_rows, _, _ = image_arrays(image); points = mask_points(mask_rows)
    displayed = repair_conservative(gray, mask_rows, points) if representation == "conservative" else repair_local(gray, mask_rows, points)
    rgb = bytearray(value for row in displayed for value in row for _ in range(3))
    def draw(box: list[float], color: tuple[int, int, int]) -> None:
        x0, y0, x1, y1 = [max(0, int(round(v))) for v in box]
        x0, x1 = min(width - 1, x0), min(width - 1, x1); y0, y1 = min(height - 1, y0), min(height - 1, y1)
        for thickness in range(2):
            for x in range(x0, x1 + 1):
                for y in (max(0, y0 - thickness), min(height - 1, y1 + thickness)):
                    rgb[(y * width + x) * 3:(y * width + x) * 3 + 3] = bytes(color)
            for y in range(y0, y1 + 1):
                for x in (max(0, x0 - thickness), min(width - 1, x1 + thickness)):
                    rgb[(y * width + x) * 3:(y * width + x) * 3 + 3] = bytes(color)
    for box in gt_boxes: draw(box, (0, 255, 255))
    for box in prediction_boxes: draw(box, (255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True); write_png(path, width, height, bytes(rgb))


def main(args: argparse.Namespace) -> None:
    workspace = args.project_root.resolve(); oof = resolve_from_root(workspace, args.oof_dir); output = resolve_from_root(workspace, args.output_dir)
    figures = resolve_from_root(workspace, args.figures_dir) if args.figures_dir else output / "figures"
    summary_path = resolve_from_root(workspace, args.summary_path) if args.summary_path else output / "02a_error_analysis_summary.md"
    required = [oof / name for name in ("02a_object_predictions.csv", "02a_image_predictions.csv",
                                         "02a_false_positive_predictions.csv", "02a_oof_summary.csv")]
    if any(not path.is_file() for path in required):
        raise RuntimeError("Complete four-fold OOF outputs are required; analysis was not run.")
    fold_rows, fold_id_by_stem = verify_frozen_folds(workspace)
    fold_by_stem = {row["stem"]: row for row in fold_rows}
    objects = read_csv(required[0]); images = read_csv(required[1]); fps = read_csv(required[2])
    oof_summary = read_csv(required[3])[0]; representation = oof_summary["representation"]
    if representation not in {"conservative", "local"}:
        raise RuntimeError("OOF representation is not an approved Stage 1D detector input")
    if len({r["stem"] for r in images}) != 500 or {r["stem"] for r in images} != set(fold_id_by_stem):
        raise RuntimeError("OOF coverage is not exactly the frozen 500-sample corpus")
    features = {key(r): r for r in read_csv(workspace / "outputs/tables/01b_object_xray_features.csv")}
    if len(objects) != 1147 or set(map(key, objects)) != set(features):
        raise RuntimeError("OOF object identifiers do not match the 1,147 Stage 1B objects")
    metadata = {r["stem"]: r for r in read_csv(workspace / "outputs/tables/01a_sample_metadata.csv")}
    geometry = {(r["image_id"], int(r["object_id"])): r for r in read_csv(workspace / "outputs/tables/01a_object_geometry.csv")}
    allowlist = {r["stem"]: r for r in read_csv(workspace / "outputs/tables/00_stage1_allowlist.csv")}
    joined: list[dict[str, Any]] = []
    for prediction in objects:
        row = {**features[key(prediction)], **prediction, **{f"metadata_{k}": v for k, v in metadata[prediction["stem"]].items()}}
        row["final_group_component_id"] = fold_by_stem[row["stem"]]["final_group_component_id"]
        object_geometry = geometry[key(prediction)]
        image_width, image_height = f(object_geometry["image_width"]), f(object_geometry["image_height"])
        detector_input_size = int(float(oof_summary["imgsz"]))
        row["native_bbox_min_side_px"] = f(object_geometry["bbox_min_side_px"])
        row["letterbox_scale"] = min(detector_input_size / image_width, detector_input_size / image_height)
        row["projected_bbox_min_side_px"] = row["native_bbox_min_side_px"] * row["letterbox_scale"]
        joined.append(row)

    pixel_object_rows = [{
        "stem": row["stem"], "object_id": row["object_id"], "fold_id": row["fold_id"],
        "status": row["status"], "error_type": row["error_type"],
        "native_bbox_min_side_px": row["native_bbox_min_side_px"],
        "detector_input_size": int(float(oof_summary["imgsz"])), "letterbox_scale": row["letterbox_scale"],
        "projected_bbox_min_side_px": row["projected_bbox_min_side_px"],
        "prediction_confidence": row["prediction_confidence"], "iou": row["iou"],
        "notes": "Projected size assumes aspect-ratio-preserving letterbox; padding does not change bbox size.",
    } for row in joined]
    native_bins = detectability_bins("native_bbox_min_side_px", joined)
    projected_bins = detectability_bins("projected_bbox_min_side_px", joined)
    pixel_bins = [{"size_basis": "native", **row} for row in native_bins] + [{"size_basis": "projected_640", **row} for row in projected_bins]
    transitions = pixel_transition_rows(projected_bins)
    sharpest_transition = max(transitions, key=lambda row: f(row["recall_change_with_increasing_size"])) if transitions else None
    pixel_summary = {
        "detector_input_size": int(float(oof_summary["imgsz"])),
        "projection_assumption": "aspect-ratio-preserving letterbox",
        "binning": "empirical deciles with duplicate cut values collapsed",
        "object_count": len(joined),
        "sharpest_adjacent_recall_increase": sharpest_transition,
        "stage3_use": "Candidate empirical detection-limit evidence only; no threshold selected.",
    }
    write_csv(output / "02b_object_pixel_size_diagnostics.csv", list(pixel_object_rows[0]), pixel_object_rows)
    write_csv(output / "02b_pixel_size_detectability.csv", list(pixel_bins[0]), pixel_bins)
    write_csv(output / "02b_pixel_size_transition_scan.csv", list(transitions[0]) if transitions else ["notes"], transitions)
    (output / "02b_pixel_size_detectability_summary.json").write_text(json.dumps(pixel_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pixel_recall_plot(figures / "pixel_size" / "02b_projected_pixel_recall.png", projected_bins)

    status_feature_rows = []
    diagnostic_features = [*FEATURES, "native_bbox_min_side_px", "projected_bbox_min_side_px"]
    for feature in diagnostic_features:
        for status in ("TP", "FN"):
            group = [row for row in joined if row["status"] == status]
            values = finite_values(group, feature)
            status_feature_rows.append({
                "feature": feature, "status": status, **distribution_fields(values, "value"),
                "notes": "Observed OOF status distribution; image-derived associations do not establish causality.",
            })
    write_csv(output / "02b_feature_status_summary.csv", list(status_feature_rows[0]), status_feature_rows)

    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature, family in FEATURES.items():
        tables[family].extend(quantile_table(feature, joined))
    write_csv(output / "02a_error_by_size.csv", list(tables["size"][0]), tables["size"])
    write_csv(output / "02a_error_by_position.csv", list(tables["position"][0]), tables["position"])
    write_csv(output / "02a_error_by_visibility.csv", list(tables["visibility"][0]), tables["visibility"])
    write_csv(output / "02a_error_by_background_shape.csv", list(tables["background_shape"][0]), tables["background_shape"])
    relationship_rows = [item for feature in FEATURES for item in confidence_iou_relationships(feature, joined)]
    correlation_rows = [item for feature in FEATURES for item in continuous_correlations(feature, joined)]
    write_csv(output / "02a_confidence_iou_feature_relationships.csv", list(relationship_rows[0]), relationship_rows)
    write_csv(output / "02a_confidence_iou_correlations.csv", list(correlation_rows[0]), correlation_rows)

    thresholds = {"small": 8.0, "low_contrast": 4.0, "near_edge": 127.0, "complex_background": 5.3050633,
                  "low_cnr": 0.51797428, "high_heterogeneity": 14.25, "near_product_edge": 43.613332}
    flags = {
        "small + low contrast": lambda r: f(r["bbox_min_side_px"]) <= thresholds["small"] and f(r["absolute_median_difference"]) <= thresholds["low_contrast"],
        "small + near image edge": lambda r: f(r["bbox_min_side_px"]) <= thresholds["small"] and f(r["image_edge_distance_px"]) <= thresholds["near_edge"],
        "small + complex background": lambda r: f(r["bbox_min_side_px"]) <= thresholds["small"] and f(r["background_gradient_mean"]) >= thresholds["complex_background"],
        "low contrast + complex background": lambda r: f(r["absolute_median_difference"]) <= thresholds["low_contrast"] and f(r["background_gradient_mean"]) >= thresholds["complex_background"],
        "low CNR + high heterogeneity": lambda r: f(r["cnr_like_robust"]) <= thresholds["low_cnr"] and f(r["object_iqr"]) >= thresholds["high_heterogeneity"],
        "near product edge + low contrast": lambda r: f(r["product_edge_distance_px"]) <= thresholds["near_product_edge"] and f(r["absolute_median_difference"]) <= thresholds["low_contrast"],
    }
    interactions = []
    for name, predicate in flags.items():
        group = [r for r in joined if predicate(r)]; item = summary_row("interaction", name, group)
        item["definition"] = "Stage 1B pre-specified exploratory thresholds"; interactions.append(item)
    write_csv(output / "02a_interaction_errors.csv", list(interactions[0]), interactions)

    if args.baseline_error_dir:
        baseline_dir = resolve_from_root(workspace, args.baseline_error_dir)
        comparison_rows = []
        comparisons = [
            ("interaction", baseline_dir / "02a_interaction_errors.csv", interactions),
            ("size", baseline_dir / "02a_error_by_size.csv", tables["size"]),
            ("position", baseline_dir / "02a_error_by_position.csv", tables["position"]),
            ("visibility", baseline_dir / "02a_error_by_visibility.csv", tables["visibility"]),
            ("background_shape", baseline_dir / "02a_error_by_background_shape.csv", tables["background_shape"]),
        ]
        for family, baseline_path, candidate_rows in comparisons:
            if not baseline_path.is_file():
                raise FileNotFoundError(f"Baseline comparison table missing: {baseline_path}")
            baseline_rows = read_csv(baseline_path)
            baseline_by_key = {(row["feature"], row["group_or_range"]): row for row in baseline_rows}
            for candidate in candidate_rows:
                comparison_key = (candidate["feature"], candidate["group_or_range"])
                baseline = baseline_by_key.get(comparison_key)
                if baseline is None:
                    raise RuntimeError(f"Baseline comparison group missing: {comparison_key}")
                comparison_rows.append({
                    "family": family, "feature": candidate["feature"], "group_or_range": candidate["group_or_range"],
                    "object_count": candidate["object_count"], "baseline_tp": baseline["tp"], "b2_tp": candidate["tp"],
                    "baseline_fn": baseline["fn"], "b2_fn": candidate["fn"],
                    "fn_reduction": int(baseline["fn"]) - int(candidate["fn"]),
                    "baseline_recall": baseline["recall"], "b2_recall": candidate["recall"],
                    "recall_change": f(candidate["recall"]) - f(baseline["recall"]),
                    "notes": "Paired frozen-corpus OOF comparison at confidence 0.25 and IoU 0.50; no operational threshold selected.",
                })
        write_csv(output / "02b_b0_vs_b2_failure_comparison.csv", list(comparison_rows[0]), comparison_rows)

    acquisition = []
    acquisition.extend(group_metrics(joined, "machine", lambda r: r["machine"]))
    acquisition.extend(group_metrics(joined, "date", lambda r: r["date"]))
    acquisition.extend(group_metrics(joined, "resolution", lambda r: r["resolution"]))
    write_csv(output / "02a_error_by_acquisition_group.csv", list(acquisition[0]), acquisition)

    error_distribution = []
    for scope, fold_filter in [("OOF_OVERALL", None), *[(f"FOLD_{fold}", fold) for fold in range(1, 5)]]:
        scoped_objects = [row for row in joined if fold_filter is None or int(row["fold_id"]) == fold_filter]
        scoped_fps = [row for row in fps if fold_filter is None or int(row["fold_id"]) == fold_filter]
        counts = Counter("TP" if row["status"] == "TP" else row["error_type"] for row in scoped_objects)
        counts.update(f"FP_{row['fp_status']}" for row in scoped_fps)
        for error_type, count in sorted(counts.items()):
            error_distribution.append({"scope": scope, "error_type": error_type, "count": count,
                                       "object_count": len(scoped_objects), "fp_count": len(scoped_fps),
                                       "notes": "TP/FN are GT-object outcomes; FP counts are unmatched predictions. No true-negative images exist."})
    write_csv(output / "02a_error_type_distribution.csv", list(error_distribution[0]), error_distribution)

    fp_analysis = []
    by_stem_fps: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fps: by_stem_fps[row["stem"]].append(row)
    for stem, stem_fps in by_stem_fps.items():
        source = workspace / Path(allowlist[stem]["canonical_raw_path"]); image = parse_bmp(source)
        gray, mask_rows, _, _ = image_arrays(image); points = mask_points(mask_rows)
        component, _ = boundary_component(gray, mask_rows, "OTSU_DARK_LCC")
        width, height = int(image["width"]), int(image["height"])
        boundary = []
        for index in component:
            x, y = index % width, index // width
            if x == 0 or y == 0 or x == width - 1 or y == height - 1 or any((ny * width + nx) not in component for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1))):
                boundary.append((x, y))
        for fp in stem_fps:
            box = json.loads(fp["prediction_bbox_json"]); cx = int((box[0] + box[2]) / 2); cy = int((box[1] + box[3]) / 2)
            edge = min(box[0], box[1], width - box[2], height - box[3])
            repair_distance = min_point_distance(box, points); boundary_distance = min_point_distance(box, boundary)
            background_iqr, background_gradient = local_complexity(gray, box)
            fp_analysis.append({**fp, "prediction_center_inside_inferred_product": (cy * width + cx) in component,
                                "machine": fold_by_stem[stem]["machine"], "date": fold_by_stem[stem]["date"],
                                "resolution": fold_by_stem[stem]["resolution"],
                                "image_edge_distance_px": edge, "nearest_repair_mask_distance_px": repair_distance,
                                "inferred_product_boundary_distance_px": boundary_distance,
                                "local_background_iqr": background_iqr, "local_background_gradient_mean": background_gradient,
                                "notes": "Product region and repair relation are heuristic diagnostics, not product GT."})
    fp_fields = list(fp_analysis[0]) if fp_analysis else ["stem", "fold_id", "prediction_bbox_json", "confidence", "nearest_gt_iou", "fp_status", "notes"]
    write_csv(output / "02a_false_positive_analysis.csv", fp_fields, fp_analysis)
    fp_group_rows = []
    for feature in ("machine", "date", "resolution"):
        values = sorted({row[feature] for row in fold_rows})
        for value in values:
            group = [row for row in fp_analysis if row[feature] == value]
            total_images = sum(row[feature] == value for row in fold_rows)
            fp_group_rows.append({"group_feature": feature, "group_value": value, "fp_count": len(group),
                                  "images_with_fp": len({row["stem"] for row in group}), "total_images": total_images,
                                  "fp_per_image": len(group) / total_images if total_images else "",
                                  "notes": "Descriptive only; machine/date/resolution are confounded and all 500 images contain GT objects."})
    for feature in ("local_background_gradient_mean", "local_background_iqr"):
        valid = [row for row in fp_analysis if not math.isnan(f(row.get(feature)))]
        values = [f(row[feature]) for row in valid]
        if values:
            cuts = [quantile(values, q) for q in (.25, .5, .75)]
            for index in range(4):
                group = [row for row in valid if sum(f(row[feature]) > cut for cut in cuts) == index]
                fp_group_rows.append({"group_feature": feature, "group_value": f"FP_Q{index + 1}",
                                      "fp_count": len(group), "images_with_fp": len({row["stem"] for row in group}),
                                      "total_images": "", "fp_per_image": "",
                                      "notes": "Exploratory distribution among FP predictions only; not an FP rate and no TN-image denominator exists."})
    write_csv(output / "02a_false_positive_group_concentration.csv", list(fp_group_rows[0]), fp_group_rows)

    fn_rows = [r for r in joined if r["status"] == "FN"]
    med_contrast = quantile([f(r["absolute_median_difference"]) for r in joined], .5)
    med_cnr = quantile([f(r["cnr_like_robust"]) for r in joined], .5)
    fn_cases = []
    for row in fn_rows:
        labels = [name for name, pred in flags.items() if pred(row)]
        if (f(row["bbox_min_side_px"]) <= thresholds["small"]): labels.append("small")
        if f(row["absolute_median_difference"]) <= thresholds["low_contrast"]: labels.append("low_contrast")
        if f(row["cnr_like_robust"]) <= thresholds["low_cnr"]: labels.append("low_cnr")
        if f(row["product_edge_distance_px"]) <= thresholds["near_product_edge"]: labels.append("near_product_edge")
        if f(row["background_gradient_mean"]) >= thresholds["complex_background"]: labels.append("complex_background")
        if not labels and f(row["absolute_median_difference"]) > med_contrast and f(row["cnr_like_robust"]) > med_cnr: labels.append("apparently_easy_miss")
        fn_cases.append({"stem": row["stem"], "object_id": row["object_id"], "fold_id": row["fold_id"],
                         "error_type": row["error_type"], "conditions": "|".join(labels) or "other",
                         "condition_count": len(labels), "bbox_min_side_px": row["bbox_min_side_px"],
                         "absolute_median_difference": row["absolute_median_difference"], "cnr_like_robust": row["cnr_like_robust"],
                         "product_edge_distance_px": row["product_edge_distance_px"], "background_gradient_mean": row["background_gradient_mean"],
                         "gt_bbox_json": row["gt_bbox_json"], "prediction_bbox_json": row["prediction_bbox_json"],
                         "notes": "Representative categories are descriptive; no causal claim."})
    write_csv(output / "02a_false_negative_cases.csv", list(fn_cases[0]) if fn_cases else ["stem", "object_id", "conditions"], fn_cases)
    localization_cases = []
    for row in joined:
        if row["error_type"] != "LOCALIZATION_FAILURE":
            continue
        localization_cases.append({"stem": row["stem"], "object_id": row["object_id"], "fold_id": row["fold_id"],
                                   "prediction_confidence": row["prediction_confidence"], "iou": row["iou"],
                                   "bbox_min_side_px": row["bbox_min_side_px"], "bbox_area_ratio": row["bbox_area_ratio"],
                                   "image_edge_distance_px": row["image_edge_distance_px"],
                                   "product_edge_distance_px": row["product_edge_distance_px"],
                                   "signed_median_difference": row["signed_median_difference"],
                                   "absolute_median_difference": row["absolute_median_difference"],
                                   "cnr_like_robust": row["cnr_like_robust"], "object_iqr": row["object_iqr"],
                                   "background_gradient_mean": row["background_gradient_mean"], "background_iqr": row["background_iqr"],
                                   "bbox_aspect_ratio": row["bbox_aspect_ratio"], "machine": row["machine"], "date": row["date"],
                                   "resolution": row["resolution"], "gt_bbox_json": row["gt_bbox_json"],
                                   "prediction_bbox_json": row["prediction_bbox_json"],
                                   "notes": "Prediction retained at reporting threshold but IoU < 0.50; diagnostic, not an operational rule."})
    localization_fields = list(localization_cases[0]) if localization_cases else ["stem", "object_id", "fold_id", "prediction_confidence", "iou", "notes"]
    write_csv(output / "02a_localization_failure_cases.csv", localization_fields, localization_cases)

    uncertainty = []
    bootstrap_groups = {"overall": lambda r: True, "small": lambda r: f(r["bbox_min_side_px"]) <= thresholds["small"],
                        "low_contrast": lambda r: f(r["absolute_median_difference"]) <= thresholds["low_contrast"],
                        "low_cnr": lambda r: f(r["cnr_like_robust"]) <= thresholds["low_cnr"], **flags}
    for index, (name, predicate) in enumerate(bootstrap_groups.items()):
        estimate, low, high, count = bootstrap(joined, predicate, args.bootstrap_unit, args.bootstrap_replicates, args.seed + index)
        uncertainty.append({"subgroup": name, "object_count": count, "recall": estimate, "ci_low": low, "ci_high": high,
                            "confidence_level": .95, "bootstrap_unit": args.bootstrap_unit, "replicates": args.bootstrap_replicates,
                            "notes": "Cluster bootstrap resamples frozen images or 10-second group components."})
    write_csv(output / "02a_recall_uncertainty.csv", list(uncertainty[0]), uncertainty)

    hypotheses = []
    ordered_patterns = sorted(interactions, key=lambda r: (f(r["recall"]) if r["recall"] != "" else 2, -int(r["object_count"])))
    for row in ordered_patterns[:4]:
        hypotheses.append({"observed_failure_pattern": row["group_or_range"], "evidence": f"n={row['object_count']}, FN={row['fn']}, recall={row['recall']}",
                           "candidate_explanation": "Candidate detectability interaction; association is not causation.",
                           "possible_intervention": "Test one controlled scale/representation/training intervention in Stage 2B.",
                           "priority": "HIGH" if int(row["object_count"]) >= 20 else "EXPLORATORY", "risk_or_tradeoff": "Sparse cells or confounding may make the pattern unstable.",
                           "needs_experiment": True})
    write_csv(output / "02a_improvement_hypotheses.csv", list(hypotheses[0]), hypotheses)
    evidence = [{"competition_area": "AI 예측모델 개발", "finding": "Frozen four-fold OOF baseline", "evidence": "02a_oof_summary.csv", "limitation": "Development validation only"},
                {"competition_area": "영향요인 및 오류분석", "finding": "GT/image-derived subgroup associations", "evidence": "02a_error_by_*.csv and 02a_interaction_errors.csv", "limitation": "No causality"},
                {"competition_area": "창의성·차별성", "finding": "Artifact-controlled paired representation design", "evidence": "Stage 1D plus paired Stage 2A runs", "limitation": "Requires completed ablation"},
                {"competition_area": "현장 활용", "finding": "Confidence retained for later safety analysis", "evidence": "02a_confidence_sweep.csv", "limitation": "No operational threshold selected"}]
    write_csv(output / "02a_competition_evidence.csv", list(evidence[0]), evidence)

    image_by_stem = {r["stem"]: r for r in images}
    for index, case in enumerate(sorted(fn_cases, key=lambda r: (-int(r["condition_count"]), f(r["bbox_min_side_px"])))[:12]):
        image_row = image_by_stem[case["stem"]]
        annotate_case(figures / "fn_cases" / f"fn_{index:02d}_{case['stem']}.png",
                      workspace / Path(allowlist[case["stem"]]["canonical_raw_path"]), json.loads(image_row["gt_boxes_json"]),
                      [p["bbox"] for p in json.loads(image_row["prediction_boxes_json"]) if f(p["confidence"]) >= .25], representation)
    for index, row in enumerate(sorted(fp_analysis, key=lambda r: -f(r["confidence"]))[:12]):
        image_row = image_by_stem[row["stem"]]
        annotate_case(figures / "fp_cases" / f"fp_{index:02d}_{row['stem']}.png",
                      workspace / Path(allowlist[row["stem"]]["canonical_raw_path"]), json.loads(image_row["gt_boxes_json"]),
                      [json.loads(row["prediction_bbox_json"])], representation)
    for index, case in enumerate(sorted(localization_cases, key=lambda row: f(row["iou"]), reverse=True)[:12]):
        annotate_case(figures / "localization_failure_cases" / f"localization_{index:02d}_{case['stem']}.png",
                      workspace / Path(allowlist[case["stem"]]["canonical_raw_path"]), [json.loads(case["gt_bbox_json"])],
                      [json.loads(case["prediction_bbox_json"])], representation)

    tp_count = sum(row["status"] == "TP" for row in joined); fn_count = sum(row["status"] == "FN" for row in joined)
    summary_text = f"""# Stage 2A OOF Error Analysis Summary

## Scope and observed facts

- Frozen OOF scope: 500 labeled development images and {len(joined)} GT objects.
- Reporting threshold: confidence 0.25; this is not an operational threshold.
- Object outcomes: TP={tp_count}, FN={fn_count}; unmatched predictions FP={len(fp_analysis)}.
- Localization failures (confidence retained but IoU < 0.50): {len(localization_cases)}.
- All 500 images contain labels; there is no true-negative image corpus for image-level specificity/FPR estimation.

## Derived diagnostics

- Size, area ratio, image-edge distance, inferred product-edge distance, signed/absolute contrast, robust CNR-like proxy, object IQR, background gradient/IQR, and bbox aspect ratio are reported with counts by exploratory quartile.
- Prediction-confidence and localization-IoU relationships are reported both by exploratory quartile and Spearman association.
- Machine, date, and resolution summaries are descriptive and may be mutually confounded.
- Pre-specified interaction tables always retain object counts; sparse cells must not be overinterpreted.
- Product-edge distance is inferred from an unsupervised boundary proxy rather than GT segmentation.

## Pixel-size detectability

- Native and 640-letterbox projected bbox minimum-side sizes are reported per object and in distribution-derived bins.
- The sharpest adjacent empirical-bin recall change is recorded in `02b_pixel_size_detectability_summary.json`.
- This is candidate evidence for Stage 3 empirical-limit analysis only; no detection or safety threshold is selected.

## Hypotheses and limitations

- Associations identify candidate failure conditions for controlled follow-up; they do not establish causality.
- Intensity/contrast/CNR-like variables are image-derived proxies. They do not identify physical density, material, or thickness.
- FP background quartiles describe the FP set only and are not false-positive rates because no true-negative image denominator exists.
- No PASS/DETECT/REINSPECT or Stage 3 operational threshold is selected here.
"""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")
    generated = sorted({path for root in (output, figures) for path in root.rglob("*") if path.is_file()} | {summary_path})
    manifest = {"status": "COMPLETE", "stage": "Stage 2A error analysis", "run_name": args.run_name,
                "project_root": str(workspace), "representation": representation,
                "bootstrap_unit": args.bootstrap_unit, "bootstrap_replicates": args.bootstrap_replicates,
                "seed": args.seed, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "generated_checkpoint_paths": [], "generated_prediction_paths": [],
                "generated_metric_paths": [str(path.resolve()) for path in generated]}
    manifest_path = save_run_manifest(workspace, args.run_name, manifest)
    print_output_manifest(stage="Stage 2A error analysis", run_name=args.run_name, project_root=workspace,
                          files=[manifest_path, *generated], directories=[output, figures, summary_path.parent],
                          downloads=[manifest_path, output, figures, summary_path],
                          next_action="Review the generated error tables before proposing any Stage 2B intervention.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze completed Stage 2A OOF predictions; never trains or infers.")
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--baseline-error-dir", type=Path)
    parser.add_argument("--bootstrap-unit", choices=("image", "group"), default="group")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project-root", "--workspace", dest="project_root", type=Path, default=project_root_from_script())
    parsed = parser.parse_args()
    try:
        main(parsed)
    except BaseException as exc:
        root = parsed.project_root.resolve()
        failure = {"status": "FAILED", "stage": "Stage 2A error analysis", "run_name": parsed.run_name,
                   "project_root": str(root), "ended_utc": datetime.now(timezone.utc).isoformat(),
                   "error_type": type(exc).__name__, "error": str(exc)}
        manifest_path = save_run_manifest(root, parsed.run_name, failure)
        print_output_manifest(stage="Stage 2A error analysis (failed)", run_name=parsed.run_name, project_root=root,
                              files=[manifest_path], downloads=[manifest_path],
                              next_action="Return the manifest and error text; do not proceed to Stage 2B.")
        raise

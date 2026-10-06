from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_stage6_run import fold_subgroup_metrics, object_and_fp_rows
from stage2a_common import (ap_metrics, box_iou, fixed_threshold_metrics, match_image,
                            print_output_manifest, project_root_from_script,
                            save_run_manifest, sha256)


SCALES = (128, 160, 192, 224, 256)
MATCH_THRESHOLDS = (0.1, 0.3, 0.5)
POLICIES = ("CONFIRM_CONFIDENCE", "REFINE_BOX", "CONFIRM_AND_REFINE")
REPORT_CONFIDENCE = 0.25
MATCH_IOU = 0.5


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_b0(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    images: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    for row in load_csv(path):
        predictions = json.loads(row["prediction_boxes_json"])
        clean = []
        for index, pred in enumerate(predictions):
            bbox = [float(v) for v in pred["bbox"]]
            confidence = float(pred["confidence"])
            candidate_id = f"{row['stem']}__b{index:03d}"
            clean.append({"bbox": bbox, "confidence": confidence, "candidate_id": candidate_id})
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            candidates[candidate_id] = {"stem": row["stem"], "candidate_id": candidate_id, "candidate_index": index,
                                        "base_bbox": bbox, "base_confidence": confidence, "base_min_side": min(width, height),
                                        "base_area": width * height, "image_width": int(row["width"]), "image_height": int(row["height"])}
        images[row["stem"]] = {"stem": row["stem"], "width": int(row["width"]), "height": int(row["height"]),
                               "gt_boxes": json.loads(row["gt_boxes_json"]), "predictions": clean}
    if len(images) != 124 or len(candidates) != 1077 or sum(len(x["gt_boxes"]) for x in images.values()) != 290:
        raise RuntimeError("B0 Fold1 integrity mismatch")
    return images, candidates


def load_local_features(path: Path, candidates: dict[str, dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = load_csv(path)
    if len(rows) != len(candidates) * len(SCALES):
        raise RuntimeError("Candidate-scale coverage mismatch")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["candidate_id"], int(row["crop_scale"]))
        if row["candidate_id"] not in candidates or key in result:
            raise RuntimeError(f"Unexpected/duplicate candidate-scale row: {key}")
        result[key] = {
            "local_prediction_count": int(row["local_prediction_count"]),
            "confidence": None if row["selected_local_confidence"] == "" else float(row["selected_local_confidence"]),
            "bbox": None if row["selected_source_bbox_json"] == "" else json.loads(row["selected_source_bbox_json"]),
            "base_local_iou": None if row["selected_base_local_iou"] == "" else float(row["selected_base_local_iou"]),
            "center_distance": None if row["selected_center_distance_over_base_diagonal"] == "" else float(row["selected_center_distance_over_base_diagonal"]),
            "size_ratio": None if row["selected_size_ratio_local_to_base"] == "" else float(row["selected_size_ratio_local_to_base"]),
        }
    return result


def clone_images(base: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {stem: {"stem": stem, "width": item["width"], "height": item["height"], "gt_boxes": item["gt_boxes"], "predictions": []}
            for stem, item in base.items()}


def apply_policy(base_images: dict[str, dict[str, Any]], candidates: dict[str, dict[str, Any]],
                 local: dict[tuple[str, int], dict[str, Any]], scale: int, threshold: float, policy: str,
                 routed: set[str] | None = None) -> dict[str, dict[str, Any]]:
    images = clone_images(base_images)
    for candidate_id, base in candidates.items():
        selected = local[(candidate_id, scale)]
        matched = (routed is None or candidate_id in routed) and selected["bbox"] is not None and selected["base_local_iou"] >= threshold
        bbox = list(base["base_bbox"])
        confidence = float(base["base_confidence"])
        if matched and policy in {"REFINE_BOX", "CONFIRM_AND_REFINE"}:
            bbox = list(selected["bbox"])
        if matched and policy in {"CONFIRM_CONFIDENCE", "CONFIRM_AND_REFINE"}:
            confidence = max(confidence, float(selected["confidence"]))
        images[base["stem"]]["predictions"].append({"bbox": bbox, "confidence": confidence, "candidate_id": candidate_id})
    for item in images.values():
        item["predictions"].sort(key=lambda x: (-x["confidence"], x["candidate_id"]))
    return images


def evaluate(root: Path, images: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    objects, fps = object_and_fp_rows(images, 1, REPORT_CONFIDENCE)
    metrics = {**fixed_threshold_metrics(images, REPORT_CONFIDENCE, MATCH_IOU), **ap_metrics(images),
               **fold_subgroup_metrics(root, objects)}
    metrics.update({name.lower(): sum(row["error_type"] == name for row in objects)
                    for name in ("LOCALIZATION_FAILURE", "LOW_CONFIDENCE", "NO_DETECTION")})
    return metrics, objects, fps


def fp_pair_counts(b0_fp: list[dict[str, Any]], test_fp: list[dict[str, Any]]) -> tuple[int, int]:
    matched = 0
    for stem in {row["stem"] for row in b0_fp + test_fp}:
        a = [row for row in b0_fp if row["stem"] == stem]
        b = [row for row in test_fp if row["stem"] == stem]
        edges = []
        for ai, ar in enumerate(a):
            for bi, br in enumerate(b):
                overlap = box_iou(json.loads(ar["prediction_bbox_json"]), json.loads(br["prediction_bbox_json"]))
                if overlap >= 0.5:
                    edges.append((overlap, ai, bi))
        ua: set[int] = set(); ub: set[int] = set()
        for _, ai, bi in sorted(edges, reverse=True):
            if ai not in ua and bi not in ub:
                ua.add(ai); ub.add(bi); matched += 1
    return len(b0_fp) - matched, len(test_fp) - matched


def object_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["stem"], int(row["object_id"])): row for row in rows}


def paired_counts(b0: dict[tuple[str, int], dict[str, Any]], test: dict[tuple[str, int], dict[str, Any]]) -> tuple[int, int]:
    rescue = sum(b0[key]["status"] == "FN" and row["status"] == "TP" for key, row in test.items())
    loss = sum(b0[key]["status"] == "TP" and row["status"] == "FN" for key, row in test.items())
    return rescue, loss


def consensus_images(base_images: dict[str, dict[str, Any]], candidates: dict[str, dict[str, Any]],
                     local: dict[tuple[str, int], dict[str, Any]], scales: tuple[int, ...], threshold: float) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    images = clone_images(base_images)
    audit: dict[str, dict[str, float]] = {}
    for candidate_id, base in candidates.items():
        valid = [(scale, local[(candidate_id, scale)]) for scale in scales
                 if local[(candidate_id, scale)]["bbox"] is not None and local[(candidate_id, scale)]["base_local_iou"] >= threshold]
        agreeing: set[int] = set()
        pair_ious = []
        for (ia, (_, a)), (ib, (_, b)) in combinations(enumerate(valid), 2):
            overlap = box_iou(a["bbox"], b["bbox"])
            pair_ious.append(overlap)
            if overlap >= 0.5:
                agreeing.update((ia, ib))
        if len(agreeing) >= 2:
            chosen = [valid[i][1] for i in sorted(agreeing)]
            bbox = np.median(np.asarray([item["bbox"] for item in chosen], dtype=float), axis=0).tolist()
            median_local_confidence = float(np.median([item["confidence"] for item in chosen]))
            confidence = max(float(base["base_confidence"]), median_local_confidence)
            bw = max(float(base["base_bbox"][2]) - float(base["base_bbox"][0]), 1e-12)
            bh = max(float(base["base_bbox"][3]) - float(base["base_bbox"][1]), 1e-12)
            centers = np.asarray([[(item["bbox"][0] + item["bbox"][2]) / 2,
                                   (item["bbox"][1] + item["bbox"][3]) / 2] for item in chosen], dtype=float)
            center_consistency = float(np.max(np.linalg.norm(centers - np.median(centers, axis=0), axis=1))
                                       / max(math.hypot(bw, bh), 1e-12))
            confirmed = True
        else:
            bbox, confidence, confirmed = list(base["base_bbox"]), float(base["base_confidence"]), False
            median_local_confidence, center_consistency = 0.0, float("nan")
        images[base["stem"]]["predictions"].append({"bbox": bbox, "confidence": confidence, "candidate_id": candidate_id})
        audit[candidate_id] = {"valid_scale_count": len(valid), "agreement_count": len(agreeing),
                               "median_pairwise_iou": float(np.median(pair_ious)) if pair_ious else 0.0,
                               "median_local_confidence": median_local_confidence,
                               "center_consistency_over_base_diagonal": center_consistency,
                               "confirmed": int(confirmed)}
    for item in images.values():
        item["predictions"].sort(key=lambda x: (-x["confidence"], x["candidate_id"]))
    return images, audit


def predicted_visibility(root: Path, candidates: dict[str, dict[str, Any]]) -> dict[str, float]:
    import cv2

    integrity = {row["stem"]: row for row in load_csv(root / "outputs/tables/02a_experiment_integrity.csv")
                 if row["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE" and int(row["fold_id"]) == 1}
    cache: dict[str, np.ndarray] = {}
    result: dict[str, float] = {}
    for candidate_id, item in candidates.items():
        stem = item["stem"]
        if stem not in cache:
            image = cv2.imread(str(root / Path(integrity[stem]["image_path"])), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Cannot read image for visibility: {stem}")
            cache[stem] = image
        image = cache[stem]; h, w = image.shape
        x0, y0, x1, y1 = item["base_bbox"]
        ix0, iy0, ix1, iy1 = max(0, math.floor(x0)), max(0, math.floor(y0)), min(w, math.ceil(x1)), min(h, math.ceil(y1))
        bw, bh = max(x1 - x0, 1), max(y1 - y0, 1); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ox0, oy0 = max(0, math.floor(cx - bw)), max(0, math.floor(cy - bh))
        ox1, oy1 = min(w, math.ceil(cx + bw)), min(h, math.ceil(cy + bh))
        obj = image[iy0:iy1, ix0:ix1].reshape(-1)
        outer = image[oy0:oy1, ox0:ox1]
        mask = np.ones(outer.shape, dtype=bool)
        mask[max(0, iy0-oy0):max(0, iy1-oy0), max(0, ix0-ox0):max(0, ix1-ox0)] = False
        ring = outer[mask]
        result[candidate_id] = abs(float(np.median(obj)) - float(np.median(ring))) if len(obj) and len(ring) else float("inf")
    return result


def write_report(path: Path, scale: pd.DataFrame, rescue: pd.DataFrame, multi: pd.DataFrame, burden: pd.DataFrame,
                 candidate_stats: dict[str, float], p192_rescues: set[tuple[str, int]]) -> None:
    best_f1 = scale.sort_values(["f1", "recall"], ascending=False).iloc[0]
    best_recall = scale.sort_values(["recall", "f1"], ascending=False).iloc[0]
    lines = ["# Stage 6 candidate-guided local re-detection — Fold1 analysis", "",
             "## Scope", "", "Inference-only exploratory analysis. Candidate generation, crop construction, local matching, and correction used no GT. TXT GT was introduced only for final evaluation. No threshold or crop scale is adopted operationally.", "",
             "## Candidate integrity and burden", "",
             f"B0 contributed 1,077 candidates over 124 images: mean {candidate_stats['mean']:.2f}, median {candidate_stats['median']:.0f}, P90 {candidate_stats['p90']:.0f}, max {candidate_stats['max']:.0f}. One-scale all-candidate re-detection therefore costs 8.69 local crops/image in addition to B0, versus 7.21 P192 fixed-grid patches/image.", "",
             "## Fixed-scale policy sweep", "",
             f"Best observed F1 configuration (diagnostic only): scale={int(best_f1.crop_scale)}, policy={best_f1.policy}, base-local IoU={best_f1.match_threshold:.1f}, TP/FP/FN={int(best_f1.tp)}/{int(best_f1.fp)}/{int(best_f1.fn)}, F1={best_f1.f1:.4f}.",
             f"Best observed Recall configuration (diagnostic only): scale={int(best_recall.crop_scale)}, policy={best_recall.policy}, IoU={best_recall.match_threshold:.1f}, Recall={best_recall.recall:.4f}, F1={best_recall.f1:.4f}.",
             "These observations are a predeclared scale-response sweep, not operational threshold selection.", "",
             "## B0 FN rescue trajectories", ""]
    for row in rescue.itertuples():
        lines.append(f"- `{row.stem}#{int(row.object_id)}` [{row.b0_error_type}], policy={row.policy}, match={row.match_threshold}: first rescue={row.first_rescue_scale}, rescue scales={row.rescue_scales}, stable adjacent={row.stable_adjacent_rescue_scales}, P192 rescue={row.p192_rescue}.")
    lines += ["", "## Multi-scale consensus", ""]
    for row in multi[multi.row_type == "CONSENSUS"].itertuples():
        lines.append(f"- {row.multiscale_rule}, match={row.match_threshold}: TP/FP/FN={int(row.tp)}/{int(row.fp)}/{int(row.fn)}, F1={row.f1:.4f}, rescue={int(row.b0_fn_rescued)}, loss={int(row.b0_tp_lost)}.")
    lines += ["", "Per-object best-scale rows are oracle upper bounds only and are not deployable.", "",
              "## Candidate subset burden sweep", ""]
    for row in burden[burden.row_type == "ROUTING"].itertuples():
        lines.append(f"- {row.signal} at {int(row.burden_percent)}%: candidates={int(row.routed_candidates)}, images={int(row.routed_images)}, calls/image={row.local_calls_per_image:.2f}, rescue={int(row.b0_fn_rescued)}/9, localization rescue={int(row.localization_fn_rescued)}, low-confidence rescue={int(row.low_confidence_fn_rescued)}, F1={row.f1:.4f}.")
    lines += ["", "All routing signals are prediction-time observable, but their Fold1 ranking performance is diagnostic and not a deployable router. GT is used only to score coverage after routing.", "",
              "## P192 comparison", "",
              f"The existing P192 fixed-grid model rescues {len(p192_rescues)}/9 B0 FN. Candidate-local common/unique rescue relationships are reported in the rescue matrix; fixed-grid and candidate-local costs are not directly interchangeable because one covers the image and the other repeats inference around every candidate.", "",
              "## Decision framework", "",
              "A candidate-guided policy is promising only if it reduces FN, preserves or improves F1, avoids material FP growth and B0-TP loss, improves localization/low-confidence errors, and offers a burden advantage over 7.21 fixed-grid patches/image. Otherwise B0 remains primary and the method is rejected.", "",
              "## Required decision questions", ""]
    for crop_scale in SCALES:
        row = scale[scale.crop_scale == crop_scale].sort_values(["f1", "recall"], ascending=False).iloc[0]
        lines.append(f"- Scale {crop_scale}: best diagnostic fixed-scale row has TP/FP/FN={int(row.tp)}/{int(row.fp)}/{int(row.fn)}, F1={row.f1:.4f}, rescue/loss={int(row.b0_fn_rescued)}/{int(row.b0_tp_lost)}.")
    low_conf_best = scale.sort_values(["low_confidence", "f1"], ascending=[True, False]).iloc[0]
    loc_best = scale.sort_values(["localization_failure", "f1"], ascending=[True, False]).iloc[0]
    best_multi = multi[multi.row_type == "CONSENSUS"].sort_values(["f1", "recall"], ascending=False).iloc[0]
    any_rescue = rescue.groupby(["stem", "object_id"])["candidate_local_any_rescue"].max()
    p192_flags = rescue.groupby(["stem", "object_id"])["p192_rescue"].max()
    common = int((any_rescue & p192_flags).sum())
    local_only = int((any_rescue & ~p192_flags).sum())
    p192_only = int((~any_rescue & p192_flags).sum())
    lines += [
        f"- FN reduction: best observed fixed row has FN={int(best_recall.fn)} versus B0 FN=9; this is exploratory, not a selected setting.",
        "- Scale response and possible context loss must be judged from the five predeclared rows above; monotonic improvement is not assumed.",
        f"- Confidence confirmation: the row with the fewest low-confidence FN leaves {int(low_conf_best.low_confidence)} such errors.",
        f"- Box refinement: the row with the fewest localization failures leaves {int(loc_best.localization_failure)} such errors versus B0=4; M6=2 is context only, not part of this inference.",
        f"- Confirmation-only preservation: the best-F1 row records {int(best_f1.b0_tp_lost)} B0 TP losses and {int(best_f1.new_fp)} new FP; candidates were never deleted, but bbox refinement can still change GT matching.",
        f"- P192 rescue overlap: common={common}, candidate-local-only={local_only}, P192-only={p192_only} among the nine B0 FN.",
        f"- Multi-scale: best fixed consensus is {best_multi.multiscale_rule} at match={best_multi.match_threshold}, F1={best_multi.f1:.4f}; it is not automatically preferred to a single scale.",
        "- Burden: all-candidate one/two/three-scale calls are 8.69/17.37/26.06 per image, so even one scale exceeds the 7.21 fixed-grid P192 reference before routing.",
        "- Practicality: use the burden table to determine whether an observable candidate subset reaches the same rescue coverage with fewer calls. If not, candidate-guided local reinspection is not operationally justified by this Fold1 experiment.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    inference_dir = (root / args.inference_dir).resolve() if not args.inference_dir.is_absolute() else args.inference_dir.resolve()
    out_paths = [root / "outputs/tables/06_candidate_redetection_scale_summary_fold1.csv",
                 root / "outputs/tables/06_candidate_redetection_rescue_matrix_fold1.csv",
                 root / "outputs/tables/06_candidate_redetection_paired_errors_fold1.csv",
                 root / "outputs/tables/06_candidate_redetection_multiscale_fold1.csv",
                 root / "outputs/tables/06_candidate_redetection_burden_fold1.csv",
                 root / "outputs/eda/06_candidate_local_redetection_analysis_fold1.md"]
    manifest_path = root / "outputs/run_manifests" / f"{args.run_name}.json"
    if any(path.exists() for path in out_paths) or manifest_path.exists():
        raise FileExistsError("Candidate re-detection analysis output already exists; do not overwrite it")
    metadata = json.loads((inference_dir / "inference_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "INFERENCE_COMPLETE" or metadata.get("gt_used_for_inference_or_matching") is not False:
        raise RuntimeError("Inference metadata is incomplete or does not attest GT-free inference")
    b0_path = root / "outputs/stage2a_runs/kamp_conservative_640_fold1_e30_fullft_v1/image_predictions.csv"
    base_images, candidates = load_b0(b0_path)
    local = load_local_features(inference_dir / "candidate_scale_features.csv", candidates)
    b0_metrics, b0_objects, b0_fp = evaluate(root, base_images)
    if (b0_metrics["tp"], b0_metrics["fp"], b0_metrics["fn"]) != (281, 15, 9):
        raise RuntimeError("B0 reference mismatch")
    b0_map = object_map(b0_objects)
    p192_rows = load_csv(root / "outputs/stage6_runs/stage6_m1c_patch192_ov25_yolov8n_640_fold1_b0progress_v3/object_predictions.csv")
    p192_map = {(r["stem"], int(r["object_id"])): r for r in p192_rows}
    p192_rescues = {key for key, row in p192_map.items() if b0_map[key]["status"] == "FN" and row["status"] == "TP"}

    scale_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    evaluated: dict[tuple[int, float, str], tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]] = {}
    for scale in SCALES:
        for threshold in MATCH_THRESHOLDS:
            for policy in POLICIES:
                images = apply_policy(base_images, candidates, local, scale, threshold, policy)
                metrics, objects, fps = evaluate(root, images); test_map = object_map(objects)
                rescue_count, loss_count = paired_counts(b0_map, test_map)
                removed_fp, new_fp = fp_pair_counts(b0_fp, fps)
                evaluated[(scale, threshold, policy)] = (metrics, test_map)
                scale_rows.append({"crop_scale": scale, "match_threshold": threshold, "policy": policy, **metrics,
                                   "b0_fn_rescued": rescue_count, "b0_tp_lost": loss_count,
                                   "b0_fp_removed": removed_fp, "new_fp": new_fp})
                for key, row in test_map.items():
                    before = b0_map[key]
                    if before["status"] != row["status"] or before["error_type"] != row["error_type"]:
                        paired_rows.append({"crop_scale": scale, "match_threshold": threshold, "policy": policy,
                                            "stem": key[0], "object_id": key[1], "b0_status": before["status"],
                                            "result_status": row["status"], "b0_error_type": before["error_type"],
                                            "result_error_type": row["error_type"], "result_confidence": row["prediction_confidence"],
                                            "result_iou": row["iou"]})
    scale_df = pd.DataFrame(scale_rows)
    paired_df = pd.DataFrame(paired_rows)

    rescue_rows: list[dict[str, Any]] = []
    b0_fn_keys = [key for key, row in b0_map.items() if row["status"] == "FN"]
    for key in b0_fn_keys:
        for threshold in MATCH_THRESHOLDS:
            for policy in POLICIES:
                statuses = {scale: evaluated[(scale, threshold, policy)][1][key]["status"] for scale in SCALES}
                rescue_scales = [scale for scale in SCALES if statuses[scale] == "TP"]
                stable = sorted({scale for a, b in zip(SCALES[:-1], SCALES[1:]) if statuses[a] == statuses[b] == "TP" for scale in (a, b)})
                confidences = {scale: evaluated[(scale, threshold, policy)][1][key]["prediction_confidence"] for scale in SCALES}
                ious = {scale: evaluated[(scale, threshold, policy)][1][key]["iou"] for scale in SCALES}
                rescue_rows.append({"stem": key[0], "object_id": key[1], "b0_error_type": b0_map[key]["error_type"],
                                    "b0_confidence": b0_map[key]["prediction_confidence"], "b0_iou": b0_map[key]["iou"],
                                    "policy": policy, "match_threshold": threshold,
                                    **{f"scale_{scale}_status": statuses[scale] for scale in SCALES},
                                    "first_rescue_scale": min(rescue_scales) if rescue_scales else "",
                                    "rescue_scales": ";".join(map(str, rescue_scales)),
                                    "stable_adjacent_rescue_scales": ";".join(map(str, stable)),
                                    "rescue_scale_count": len(rescue_scales),
                                    "confidence_trajectory_json": json.dumps(confidences, ensure_ascii=False),
                                    "iou_trajectory_json": json.dumps(ious, ensure_ascii=False),
                                    "p192_rescue": key in p192_rescues})
    rescue_df = pd.DataFrame(rescue_rows)
    any_rescue_by_object = rescue_df.groupby(["stem", "object_id"])["rescue_scale_count"].max().gt(0)
    rescue_df["candidate_local_any_rescue"] = [bool(any_rescue_by_object.loc[(row.stem, row.object_id)])
                                                 for row in rescue_df.itertuples()]
    rescue_df["rescue_relation_to_p192"] = np.select(
        [rescue_df["candidate_local_any_rescue"] & rescue_df["p192_rescue"],
         rescue_df["candidate_local_any_rescue"] & ~rescue_df["p192_rescue"],
         ~rescue_df["candidate_local_any_rescue"] & rescue_df["p192_rescue"]],
        ["COMMON", "CANDIDATE_LOCAL_ONLY", "P192_ONLY"], default="NEITHER")

    multi_rows: list[dict[str, Any]] = []
    for threshold in MATCH_THRESHOLDS:
        for name, scales in {"MS1_160_192": (160, 192), "MS2_160_192_224": (160, 192, 224)}.items():
            images, audit = consensus_images(base_images, candidates, local, scales, threshold)
            metrics, objects, _ = evaluate(root, images); test_map = object_map(objects)
            rescue_count, loss_count = paired_counts(b0_map, test_map)
            multi_rows.append({"row_type": "CONSENSUS", "multiscale_rule": name, "scales": ";".join(map(str, scales)),
                               "match_threshold": threshold, **metrics, "b0_fn_rescued": rescue_count,
                               "b0_tp_lost": loss_count, "confirmed_candidates": sum(v["confirmed"] for v in audit.values()),
                               "mean_agreement_count": np.mean([v["agreement_count"] for v in audit.values()]),
                               "mean_local_confidence": np.mean([v["median_local_confidence"] for v in audit.values() if v["confirmed"]]),
                               "mean_local_bbox_iou_consistency": np.mean([v["median_pairwise_iou"] for v in audit.values() if v["confirmed"]]),
                               "mean_center_dispersion_over_base_diagonal": np.mean([v["center_consistency_over_base_diagonal"] for v in audit.values() if v["confirmed"]])})
        for policy in POLICIES:
            oracle_tp = sum(any(evaluated[(scale, threshold, policy)][1][key]["status"] == "TP" for scale in SCALES) for key in b0_map)
            multi_rows.append({"row_type": "PER_OBJECT_SCALE_ORACLE", "multiscale_rule": "BEST_OF_128_160_192_224_256",
                               "scales": ";".join(map(str, SCALES)), "match_threshold": threshold, "policy": policy,
                               "tp": oracle_tp, "fn": 290 - oracle_tp, "recall": oracle_tp / 290,
                               "note": "GT-aware diagnostic upper bound; not deployable"})
    multi_df = pd.DataFrame(multi_rows)

    visibility = predicted_visibility(root, candidates)
    rank_features: dict[str, dict[str, float]] = {
        "low_base_confidence": {cid: float(item["base_confidence"]) for cid, item in candidates.items()},
        "small_predicted_box": {cid: float(item["base_min_side"]) for cid, item in candidates.items()},
        "low_predicted_local_visibility": visibility,
    }
    instability: dict[str, float] = {}
    for cid in candidates:
        selections = [local[(cid, scale)] for scale in (160, 192, 224)]
        if any(item["bbox"] is None for item in selections):
            instability[cid] = 1.0
        else:
            pair_ious = [box_iou(a["bbox"], b["bbox"]) for a, b in combinations(selections, 2)]
            instability[cid] = 1 - float(np.median(pair_ious))
    rank_features["high_instability"] = instability
    burden_rows: list[dict[str, Any]] = []
    for scale_count in (1, 2, 3):
        burden_rows.append({"row_type": "REFERENCE", "signal": f"all_candidates_{scale_count}_scale",
                            "burden_percent": 100, "routed_candidates": len(candidates), "routed_images": 124,
                            "local_calls_per_image": len(candidates) * scale_count / 124,
                            "note": "Inference burden reference; no GT-based routing"})
    for signal, values in rank_features.items():
        reverse = signal == "high_instability"
        order = sorted(values, key=lambda cid: (values[cid], cid), reverse=reverse)
        for percent in (10, 20, 30, 40, 50):
            n = math.ceil(len(order) * percent / 100)
            routed = set(order[:n])
            images = apply_policy(base_images, candidates, local, 192, 0.3, "CONFIRM_AND_REFINE", routed)
            metrics, objects, _ = evaluate(root, images); test_map = object_map(objects)
            rescued_keys = {key for key, row in test_map.items() if b0_map[key]["status"] == "FN" and row["status"] == "TP"}
            burden_rows.append({"row_type": "ROUTING", "signal": signal, "burden_percent": percent,
                                "reference_scale": 192, "reference_policy": "CONFIRM_AND_REFINE", "reference_match_threshold": 0.3,
                                "routed_candidates": len(routed), "routed_images": len({candidates[cid]["stem"] for cid in routed}),
                                "local_calls_per_image": len(routed) / 124, **metrics, "b0_fn_rescued": len(rescued_keys),
                                "localization_fn_rescued": sum(b0_map[key]["error_type"] == "LOCALIZATION_FAILURE" for key in rescued_keys),
                                "low_confidence_fn_rescued": sum(b0_map[key]["error_type"] == "LOW_CONFIDENCE" for key in rescued_keys)})
    burden_rows.append({"row_type": "REFERENCE", "signal": "fixed_grid_p192", "burden_percent": 100,
                        "routed_candidates": "", "routed_images": 124, "local_calls_per_image": 894 / 124,
                        "b0_fn_rescued": 5, "note": "Existing separate P192 patch detector; reference comparison only"})
    burden_df = pd.DataFrame(burden_rows)

    outputs = {
        root / "outputs/tables/06_candidate_redetection_scale_summary_fold1.csv": scale_df,
        root / "outputs/tables/06_candidate_redetection_rescue_matrix_fold1.csv": rescue_df,
        root / "outputs/tables/06_candidate_redetection_paired_errors_fold1.csv": paired_df,
        root / "outputs/tables/06_candidate_redetection_multiscale_fold1.csv": multi_df,
        root / "outputs/tables/06_candidate_redetection_burden_fold1.csv": burden_df,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    counts = pd.Series([sum(item["stem"] == stem for item in candidates.values()) for stem in base_images])
    stats = {"mean": float(counts.mean()), "median": float(counts.median()), "p90": float(counts.quantile(.9)), "max": float(counts.max())}
    report_path = root / "outputs/eda/06_candidate_local_redetection_analysis_fold1.md"
    write_report(report_path, scale_df, rescue_df, multi_df, burden_df, stats, p192_rescues)
    generated = [*outputs.keys(), report_path]
    manifest = save_run_manifest(root, args.run_name, {
        "run_name": args.run_name,
        "stage": "Stage 6 candidate local re-detection analysis",
        "status": "ANALYSIS_COMPLETE",
        "fold": 1,
        "inference_dir": str(inference_dir),
        "inference_metadata_sha256": sha256(inference_dir / "inference_metadata.json"),
        "gt_use": "final evaluation and retrospective diagnostics only",
        "generated_paths": [str(path.resolve()) for path in generated],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    print_output_manifest(stage="Stage 6 candidate local re-detection analysis", run_name=args.run_name,
                          project_root=root, files=[*generated, manifest], directories=[], models=[],
                          downloads=[*generated, manifest], next_action="Review the Fold1 feasibility report; do not promote a threshold or policy without further validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze completed Stage 6 candidate-guided local re-detection inference.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--inference-dir", type=Path, default=Path("outputs/stage6_candidate_redetection/fold1"))
    parser.add_argument("--run-name", default="stage6_candidate_local_redetection_analysis_fold1_v1")
    main(parser.parse_args())

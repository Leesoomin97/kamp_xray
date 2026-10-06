from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evaluate_stage6_run import fold_subgroup_metrics, object_and_fp_rows
from stage2a_common import ap_metrics, box_iou, fixed_threshold_metrics, match_image, write_csv


ROOT = Path(__file__).resolve().parents[1]
REPORT_CONFIDENCE = 0.25
MATCH_IOU = 0.5
MAX_DET = 300
NMS_THRESHOLDS = (0.5, 0.6, 0.7)

RUNS = {
    "b0": ROOT / "outputs/stage2a_runs/kamp_conservative_640_fold1_e30_fullft_v1",
    "m3": ROOT / "outputs/stage6_runs/stage6_m3_bilateral_yolov8n_640_fold1_b0matched_v2",
    "p192": ROOT / "outputs/stage6_runs/stage6_m1c_patch192_ov25_yolov8n_640_fold1_b0progress_v3",
}

COMBINATIONS = {
    "F1_B0": ("b0",),
    "M3_ONLY": ("m3",),
    "P192_ONLY": ("p192",),
    "F2_B0_M3": ("b0", "m3"),
    "F3_B0_P192": ("b0", "p192"),
    "F4_B0_M3_P192": ("b0", "m3", "p192"),
}


def load_images(folder: Path, model_name: str) -> dict[str, dict[str, Any]]:
    path = folder / "image_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stem = row["stem"]
            predictions = json.loads(row["prediction_boxes_json"])
            for index, prediction in enumerate(predictions):
                prediction["bbox"] = [float(value) for value in prediction["bbox"]]
                prediction["confidence"] = float(prediction["confidence"])
                prediction["model"] = model_name
                prediction["source_prediction_index"] = index
                prediction["contributors"] = [model_name]
            result[stem] = {
                "stem": stem, "fold_id": int(row["fold_id"]), "width": int(row["width"]),
                "height": int(row["height"]), "gt_boxes": json.loads(row["gt_boxes_json"]),
                "predictions": predictions,
            }
    return result


def validate_inputs(inputs: dict[str, dict[str, dict[str, Any]]]) -> None:
    expected = {"b0": (281, 15, 9), "m3": (281, 15, 9), "p192": (266, 29, 24)}
    stems = set(inputs["b0"])
    if len(stems) != 124:
        raise RuntimeError(f"Expected 124 Fold1 images, found {len(stems)}")
    for name, images in inputs.items():
        if set(images) != stems:
            raise RuntimeError(f"{name} image membership differs from B0")
        if sum(len(item["gt_boxes"]) for item in images.values()) != 290:
            raise RuntimeError(f"{name} does not contain 290 GT objects")
        for stem in stems:
            a, b = inputs["b0"][stem], images[stem]
            if (a["width"], a["height"], a["gt_boxes"], a["fold_id"]) != (b["width"], b["height"], b["gt_boxes"], b["fold_id"]):
                raise RuntimeError(f"{name} geometry/GT/fold mismatch: {stem}")
        observed = fixed_threshold_metrics(images, REPORT_CONFIDENCE, MATCH_IOU)
        if (observed["tp"], observed["fp"], observed["fn"]) != expected[name]:
            raise RuntimeError(f"{name} prediction integrity mismatch: {observed}")
    metadata = json.loads((RUNS["p192"] / "evaluation_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("patch_fusion") != "reference_nms" or float(metadata.get("fusion_iou")) != 0.5:
        raise RuntimeError("P192 source-image predictions are not the expected reference-NMS output")


def union_nms(candidates: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    order = sorted(range(len(candidates)), key=lambda i: (-float(candidates[i]["confidence"]), str(candidates[i]["model"]), i))
    kept: list[dict[str, Any]] = []
    for index in order:
        candidate = candidates[index]
        suppressor = next((item for item in kept if box_iou(candidate["bbox"], item["bbox"]) >= threshold), None)
        if suppressor is None:
            kept.append({**candidate, "contributors": list(candidate["contributors"])})
        else:
            suppressor["contributors"] = sorted(set(suppressor["contributors"]) | set(candidate["contributors"]))
    return kept[:MAX_DET]


def fuse(inputs: dict[str, dict[str, dict[str, Any]]], members: tuple[str, ...], threshold: float) -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    for stem in sorted(inputs["b0"]):
        base = inputs["b0"][stem]
        candidates = [prediction for member in members for prediction in inputs[member][stem]["predictions"]]
        images[stem] = {"stem": stem, "fold_id": 1, "width": base["width"], "height": base["height"],
                        "gt_boxes": base["gt_boxes"], "predictions": union_nms(candidates, threshold)}
    return images


def error_counts(object_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {name: sum(row["error_type"] == name for row in object_rows)
            for name in ("LOCALIZATION_FAILURE", "LOW_CONFIDENCE", "NO_DETECTION")}


def paired_fp_counts(b0_fp: list[dict[str, Any]], fused_fp: list[dict[str, Any]]) -> tuple[int, int, int]:
    by_stem_a: dict[str, list[dict[str, Any]]] = {}
    by_stem_b: dict[str, list[dict[str, Any]]] = {}
    for row in b0_fp:
        by_stem_a.setdefault(row["stem"], []).append(row)
    for row in fused_fp:
        by_stem_b.setdefault(row["stem"], []).append(row)
    matched = 0
    for stem in set(by_stem_a) | set(by_stem_b):
        a, b = by_stem_a.get(stem, []), by_stem_b.get(stem, [])
        edges = []
        for ai, ar in enumerate(a):
            abox = json.loads(ar["prediction_bbox_json"])
            for bi, br in enumerate(b):
                overlap = box_iou(abox, json.loads(br["prediction_bbox_json"]))
                if overlap >= 0.5:
                    edges.append((overlap, ai, bi))
        used_a: set[int] = set(); used_b: set[int] = set()
        for _, ai, bi in sorted(edges, reverse=True):
            if ai not in used_a and bi not in used_b:
                used_a.add(ai); used_b.add(bi); matched += 1
    return len(b0_fp) - matched, len(fused_fp) - matched, matched


def prediction_contributors(images: dict[str, dict[str, Any]], stem: str, object_id: int) -> tuple[str, float, float, str]:
    item = images[stem]
    matches, _ = match_image(item["gt_boxes"], item["predictions"], REPORT_CONFIDENCE, MATCH_IOU)
    if object_id not in matches:
        return "", 0.0, 0.0, ""
    index, overlap = matches[object_id]
    pred = item["predictions"][index]
    return ";".join(pred["contributors"]), float(pred["confidence"]), float(overlap), json.dumps(pred["bbox"], separators=(",", ":"))


def evaluate(images: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    objects, fps = object_and_fp_rows(images, 1, REPORT_CONFIDENCE)
    metrics = {**fixed_threshold_metrics(images, REPORT_CONFIDENCE, MATCH_IOU), **ap_metrics(images),
               **fold_subgroup_metrics(ROOT, objects)}
    metrics.update(error_counts(objects))
    return metrics, objects, fps


def main() -> None:
    inputs = {name: load_images(folder, name) for name, folder in RUNS.items()}
    validate_inputs(inputs)
    output_dir = ROOT / "outputs/stage6_fusion/fold1"
    output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    evaluated: dict[tuple[str, float], tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    for threshold in NMS_THRESHOLDS:
        for combination, members in COMBINATIONS.items():
            images = fuse(inputs, members, threshold)
            metrics, objects, fps = evaluate(images)
            evaluated[(combination, threshold)] = (metrics, objects, fps, images)
            sensitivity.append({"fusion_method": "union_class_agnostic_nms", "combination": combination,
                                "members": ";".join(members), "nms_iou": threshold, **metrics})
            for stem, item in images.items():
                for index, prediction in enumerate(item["predictions"]):
                    all_predictions.append({"fusion_method": "union_class_agnostic_nms", "combination": combination,
                                            "nms_iou": threshold, "stem": stem, "prediction_index": index,
                                            "bbox_json": json.dumps(prediction["bbox"], separators=(",", ":")),
                                            "confidence": prediction["confidence"],
                                            "kept_model": prediction["model"], "contributors": ";".join(prediction["contributors"])})

    b0_metrics, b0_objects, b0_fp, _ = evaluated[("F1_B0", 0.7)]
    b0_by_key = {(r["stem"], int(r["object_id"])): r for r in b0_objects}
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    rescued: list[dict[str, Any]] = []
    for combination in COMBINATIONS:
        metrics, objects, fps, images = evaluated[(combination, 0.7)]
        by_key = {(r["stem"], int(r["object_id"])): r for r in objects}
        rescues = sum(b0_by_key[k]["status"] == "FN" and row["status"] == "TP" for k, row in by_key.items())
        losses = sum(b0_by_key[k]["status"] == "TP" and row["status"] == "FN" for k, row in by_key.items())
        removed_fp, new_fp, retained_fp = paired_fp_counts(b0_fp, fps)
        patch_manifest = ROOT / "outputs/tables/06_patch_manifest.csv"
        with patch_manifest.open(encoding="utf-8-sig", newline="") as handle:
            patch_rows = [r for r in csv.DictReader(handle) if int(r["fold_id"]) == 1 and r["disposition"] == "KEEP"]
        patch_mean = len(patch_rows) / 124
        members = COMBINATIONS[combination]
        full_calls = int("b0" in members) + int("m3" in members)
        patch_calls = patch_mean if "p192" in members else 0.0
        summary.append({"fusion_method": "union_class_agnostic_nms", "combination": combination, "members": ";".join(members),
                        "nms_iou": 0.7, **metrics, "b0_fn_rescued": rescues, "b0_tp_lost": losses,
                        "b0_fp_removed": removed_fp, "new_fp": new_fp, "geometrically_retained_fp": retained_fp,
                        "full_image_inference_calls_per_image": full_calls, "mean_patch_inference_calls_per_image": patch_calls,
                        "approx_total_model_inputs_per_image": full_calls + patch_calls})
        if combination.startswith("F") and combination != "F1_B0":
            for key, row in by_key.items():
                before = b0_by_key[key]
                transition = f"{before['status']}_TO_{row['status']}"
                contributors, confidence, overlap, bbox = prediction_contributors(images, key[0], key[1])
                paired.append({"combination": combination, "stem": key[0], "object_id": key[1],
                               "b0_status": before["status"], "fusion_status": row["status"], "transition": transition,
                               "b0_error_type": before["error_type"], "fusion_error_type": row["error_type"],
                               "final_contributors": contributors, "final_confidence": confidence if row["status"] == "TP" else "",
                               "final_iou": overlap if row["status"] == "TP" else "", "final_bbox_json": bbox if row["status"] == "TP" else ""})
                if transition == "FN_TO_TP":
                    rescued.append({"combination": combination, "stem": key[0], "object_id": key[1],
                                    "b0_error_type": before["error_type"], "contributing_model_predictions": contributors,
                                    "final_confidence": confidence, "final_iou": overlap, "final_bbox_json": bbox})

    table_dir = ROOT / "outputs/tables"
    write_csv(table_dir / "06_fusion_summary_fold1.csv", list(summary[0]), summary)
    write_csv(table_dir / "06_fusion_paired_errors_fold1.csv", list(paired[0]), paired)
    write_csv(table_dir / "06_fusion_rescued_objects_fold1.csv", list(rescued[0]) if rescued else ["combination", "stem", "object_id"], rescued)
    write_csv(table_dir / "06_fusion_nms_sensitivity_fold1.csv", list(sensitivity[0]), sensitivity)
    write_csv(output_dir / "fused_predictions.csv", list(all_predictions[0]), all_predictions)
    write_report(summary, sensitivity, paired, rescued, b0_metrics, patch_mean)


def write_report(summary: list[dict[str, Any]], sensitivity: list[dict[str, Any]], paired: list[dict[str, Any]],
                 rescued: list[dict[str, Any]], b0_metrics: dict[str, Any], patch_mean: float) -> None:
    by_name = {row["combination"]: row for row in summary}
    lines = ["# Stage 6 Fold1 deployable prediction-level fusion analysis", "",
             "## Scope", "",
             "This is a one-fold exploratory analysis. Fusion used prediction boxes and confidence only; GT was used after fusion solely for evaluation. No operational threshold was selected. P192 input is its existing source-coordinate output after the pre-existing 0.5 reference cross-patch NMS.", "",
             "## Input integrity", "",
             "B0, M3, and P192 each contained the same 124 Fold1 images and 290 identical TXT-GT boxes. Recomputed inputs reproduced B0 281/15/9, M3 281/15/9, and P192 266/29/24.", "",
             "## Fusion method", "",
             "Predictions at the existing 0.001 confidence floor were unioned in source-image coordinates. Deterministic class-agnostic greedy NMS retained the highest-confidence box and preserved its original confidence; no model bonus or GT information was used. The primary reference NMS IoU was 0.7, with 0.5/0.6/0.7 sensitivity. Weighted box fusion was not added because defining score aggregation would introduce another unvalidated fusion rule.", "",
             "## Primary 0.7-NMS results", "",
             "| Combination | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50-95 | Rescue | Loss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in COMBINATIONS:
        r = by_name[name]
        lines.append(f"| {name} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['ap50']:.4f} | {r['map50_95']:.4f} | {r['b0_fn_rescued']} | {r['b0_tp_lost']} |")
    lines += ["", "## Paired rescue and FP trade-off", ""]
    for name in ("F2_B0_M3", "F3_B0_P192", "F4_B0_M3_P192"):
        r = by_name[name]
        lines.append(f"- **{name}:** rescued {r['b0_fn_rescued']}/9 B0 FN, lost {r['b0_tp_lost']} B0 TP; removed {r['b0_fp_removed']} geometrically matched B0 FP and introduced {r['new_fp']} unmatched fusion FP.")
    lines += ["", "## Subgroups and error types", ""]
    for name in ("F1_B0", "F2_B0_M3", "F3_B0_P192", "F4_B0_M3_P192"):
        r = by_name[name]
        lines.append(f"- **{name}:** small {r['small_tp']}/{r['small_n']} ({r['small_recall']:.4f}); low contrast {r['low_contrast_tp']}/{r['low_contrast_n']} ({r['low_contrast_recall']:.4f}); small+low {r['small_low_contrast_tp']}/{r['small_low_contrast_n']} ({r['small_low_contrast_recall']:.4f}); localization/low-confidence/no-detection={r['LOCALIZATION_FAILURE']}/{r['LOW_CONFIDENCE']}/{r['NO_DETECTION']}.")
    lines += ["", "## Oracle comparison", "",
              f"The prior B0+P192 oracle upper bound is TP 286, FN 4, Recall 0.9862. It assumes perfect GT-aware selection and is not deployable. At reference NMS 0.7, F3 preserves {by_name['F3_B0_P192']['b0_fn_rescued']}/5 oracle rescues (oracle gap {5-by_name['F3_B0_P192']['b0_fn_rescued']}) and F4 preserves {by_name['F4_B0_M3_P192']['b0_fn_rescued']}/5 (oracle gap {5-by_name['F4_B0_M3_P192']['b0_fn_rescued']}).",
              "The missing oracle rescues show that simple prediction union/NMS cannot reproduce GT-aware object selection.", "",
              "## NMS sensitivity", ""]
    for name in ("F2_B0_M3", "F3_B0_P192", "F4_B0_M3_P192"):
        rows = [r for r in sensitivity if r["combination"] == name]
        lines.append(f"- **{name}:** " + "; ".join(f"IoU {r['nms_iou']:.1f}: TP/FP/FN={r['tp']}/{r['fp']}/{r['fn']}, F1={r['f1']:.4f}" for r in rows))
    b0_sensitivity = {r["nms_iou"]: r for r in sensitivity if r["combination"] == "F1_B0"}
    lines.append("- Corresponding B0-only controls: " + "; ".join(
        f"IoU {threshold:.1f}: TP/FP/FN={b0_sensitivity[threshold]['tp']}/{b0_sensitivity[threshold]['fp']}/{b0_sensitivity[threshold]['fn']}, F1={b0_sensitivity[threshold]['f1']:.4f}"
        for threshold in NMS_THRESHOLDS
    ) + ". At IoU 0.5, F2 merely equals the same-threshold B0 control; the apparent FP reduction versus the original 0.7 result is an NMS-threshold effect, not a fusion gain. No grid point improves both FN and F1 over its corresponding B0-only control.")
    lines += ["", "## Inference burden", "",
              f"Fold1 uses 894 retained P192 patches across 124 images: mean {patch_mean:.2f} patch model inputs/image. B0=1.0 full-image call; B0+M3=2 full-image calls; B0+P192≈{1+patch_mean:.2f} total model inputs/image; B0+M3+P192≈{2+patch_mean:.2f}. No wall-clock time is inferred.", "",
              "## Verdict", ""]
    candidates = [by_name[n] for n in ("F2_B0_M3", "F3_B0_P192", "F4_B0_M3_P192")]
    best = max(candidates, key=lambda r: (r["f1"], r["recall"], -r["fp"]))
    if best["f1"] > b0_metrics["f1"] and best["fn"] < b0_metrics["fn"]:
        verdict = f"{best['combination']} is the best exploratory deployable fusion candidate at reference NMS 0.7 because both F1 and FN improve versus B0."
    elif best["fn"] < b0_metrics["fn"]:
        verdict = f"{best['combination']} reduces FN but does not improve F1 versus B0; it is a safety trade-off candidate, not a replacement."
    else:
        verdict = "No tested fusion improves both FN and F1 versus B0; retain B0 as primary."
    lines += [verdict, "All conclusions are Fold1 exploratory and threshold-sensitive; none defines an operational fusion threshold.", "",
              "## Next experiment", "",
              "Stop fusion expansion and retain B0 as primary. The one next experiment is the already pre-registered M6 CLAHE-to-bilateral Fold1 ablation, which tests preprocessing order without tuning this fusion or its thresholds.", ""]
    (ROOT / "outputs/eda/06_prediction_fusion_analysis_fold1.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

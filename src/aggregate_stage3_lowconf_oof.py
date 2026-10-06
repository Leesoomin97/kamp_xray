from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path.cwd()

INPUT_ROOT = ROOT / "outputs" / "stage3_lowconf"

OUT = (
    ROOT
    / "outputs"
    / "tables"
    / "03_stage3_safety_analysis_v1"
)

EDA_OUT = ROOT / "outputs" / "eda"

OUT.mkdir(parents=True, exist_ok=True)
EDA_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IOU_THRESHOLD = 0.50

THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

# Stage2 reporting threshold
BASE_REPORT_THRESHOLD = 0.25

# Stage3 DETECT 후보
DETECT_THRESHOLD = 0.40

REINSPECT_THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
]


# ============================================================
# HELPERS
# ============================================================

def parse_boxes(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    parsed = json.loads(value)

    if not isinstance(parsed, list):
        return []

    return parsed


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def evaluate_image(gt_boxes, predictions, conf_threshold):
    """
    Confidence threshold 적용 후 confidence 내림차순 greedy matching.
    IoU >= 0.50이면 아직 매칭되지 않은 GT 중
    IoU가 가장 높은 GT와 매칭.
    """

    kept = [
        p for p in predictions
        if float(p["confidence"]) >= conf_threshold
    ]

    kept = sorted(
        kept,
        key=lambda x: float(x["confidence"]),
        reverse=True,
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for pred in kept:
        pred_box = pred["bbox"]

        best_iou = -1.0
        best_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = bbox_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if (
            best_gt_idx is not None
            and best_iou >= IOU_THRESHOLD
        ):
            matched_gt.add(best_gt_idx)
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - tp

    return tp, fp, fn


# ============================================================
# 1. LOAD 4 FOLDS
# ============================================================

frames = []

for fold in range(1, 5):
    path = (
        INPUT_ROOT
        / f"b2_fold{fold}"
        / "image_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing fold file: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "stem",
        "fold_id",
        "gt_boxes_json",
        "prediction_boxes_json",
        "confidence_floor",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path.name} missing columns: {sorted(missing)}"
        )

    frames.append(df)

images = pd.concat(
    frames,
    ignore_index=True,
)

if len(images) != 500:
    raise RuntimeError(
        f"Expected 500 OOF images, found {len(images)}"
    )

if images["stem"].nunique() != 500:
    raise RuntimeError(
        "OOF stem duplication detected."
    )

if sorted(images["fold_id"].unique().tolist()) != [1, 2, 3, 4]:
    raise RuntimeError(
        f"Unexpected folds: {sorted(images['fold_id'].unique())}"
    )


# ============================================================
# 2. PARSE RAW GT / PREDICTIONS
# ============================================================

records = []

total_gt = 0
total_raw_predictions = 0

for _, row in images.iterrows():

    gt_boxes = parse_boxes(
        row["gt_boxes_json"]
    )

    predictions = parse_boxes(
        row["prediction_boxes_json"]
    )

    total_gt += len(gt_boxes)
    total_raw_predictions += len(predictions)

    max_conf = (
        max(
            [float(p["confidence"]) for p in predictions],
            default=np.nan,
        )
    )

    records.append({
        "stem": row["stem"],
        "fold_id": int(row["fold_id"]),
        "gt_boxes": gt_boxes,
        "predictions": predictions,
        "max_prediction_confidence": max_conf,
        "prediction_count_at_floor": len(predictions),
    })

if total_gt != 1147:
    raise RuntimeError(
        f"Expected 1147 GT objects, found {total_gt}"
    )


# ============================================================
# 3. EXACT OOF THRESHOLD SWEEP
# ============================================================

sweep_rows = []

for threshold in THRESHOLDS:

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for rec in records:

        tp, fp, fn = evaluate_image(
            rec["gt_boxes"],
            rec["predictions"],
            threshold,
        )

        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = (
        tp_total / (tp_total + fp_total)
        if (tp_total + fp_total) > 0
        else np.nan
    )

    recall = (
        tp_total / (tp_total + fn_total)
        if (tp_total + fn_total) > 0
        else np.nan
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (
            not np.isnan(precision)
            and not np.isnan(recall)
            and (precision + recall) > 0
        )
        else np.nan
    )

    sweep_rows.append({
        "confidence_threshold": threshold,
        "iou_threshold": IOU_THRESHOLD,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

sweep = pd.DataFrame(sweep_rows)


# ============================================================
# 4. VALIDATE AGAINST KNOWN B2 @ 0.25
# ============================================================

r025 = sweep[
    np.isclose(
        sweep["confidence_threshold"],
        BASE_REPORT_THRESHOLD,
    )
].iloc[0]

expected = {
    "tp": 1121,
    "fp": 51,
    "fn": 26,
}

actual = {
    "tp": int(r025["tp"]),
    "fp": int(r025["fp"]),
    "fn": int(r025["fn"]),
}

if actual != expected:
    raise RuntimeError(
        "\n".join([
            "0.25 validation failed.",
            f"Expected: {expected}",
            f"Actual:   {actual}",
            "Greedy matching implementation does not reproduce Stage2 B2.",
        ])
    )


sweep.to_csv(
    OUT / "03e_lowconf_oof_threshold_sweep.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 5. COMPARE AGAINST DETECT THRESHOLD = 0.40
# ============================================================

detect_row = sweep[
    np.isclose(
        sweep["confidence_threshold"],
        DETECT_THRESHOLD,
    )
].iloc[0]

detect_tp = int(detect_row["tp"])
detect_fp = int(detect_row["fp"])
detect_fn = int(detect_row["fn"])

tradeoff_rows = []

for r in REINSPECT_THRESHOLDS:

    if r >= DETECT_THRESHOLD:
        continue

    low_row = sweep[
        np.isclose(
            sweep["confidence_threshold"],
            r,
        )
    ].iloc[0]

    low_tp = int(low_row["tp"])
    low_fp = int(low_row["fp"])
    low_fn = int(low_row["fn"])

    recovered_gt = low_tp - detect_tp

    additional_candidates = (
        (low_tp + low_fp)
        - (detect_tp + detect_fp)
    )

    additional_fp = (
        low_fp - detect_fp
    )

    residual_fn = low_fn

    tradeoff_rows.append({
        "detect_threshold": DETECT_THRESHOLD,
        "reinspect_lower_threshold": r,

        "detect_tp": detect_tp,
        "detect_fp": detect_fp,
        "detect_fn": detect_fn,

        "tp_if_reinspect_candidates_reviewed": low_tp,
        "fp_candidates_at_reinspect_floor": low_fp,
        "residual_fn_after_reinspect_floor": residual_fn,

        "gt_objects_recovered_vs_detect_only": recovered_gt,
        "additional_fp_candidates_vs_detect_only": additional_fp,
        "additional_total_candidates_vs_detect_only": additional_candidates,

        "object_recall_at_reinspect_floor":
            float(low_row["recall"]),

        "object_precision_at_reinspect_floor":
            float(low_row["precision"]),

        "f1_at_reinspect_floor":
            float(low_row["f1"]),
    })

tradeoff = pd.DataFrame(tradeoff_rows)

tradeoff.to_csv(
    OUT / "03f_reinspect_object_tradeoff.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. IMAGE-LEVEL DETECT / REINSPECT / PASS
# ============================================================

routing_rows = []

for r in REINSPECT_THRESHOLDS:

    if r >= DETECT_THRESHOLD:
        continue

    detect_images = 0
    reinspect_images = 0
    pass_images = 0

    detect_candidate_count = 0
    reinspect_candidate_count = 0

    for rec in records:

        preds = rec["predictions"]

        confs = [
            float(p["confidence"])
            for p in preds
        ]

        high = [
            c for c in confs
            if c >= DETECT_THRESHOLD
        ]

        gray = [
            c for c in confs
            if r <= c < DETECT_THRESHOLD
        ]

        detect_candidate_count += len(high)
        reinspect_candidate_count += len(gray)

        if high:
            detect_images += 1
        elif gray:
            reinspect_images += 1
        else:
            pass_images += 1

    routing_rows.append({
        "detect_threshold": DETECT_THRESHOLD,
        "reinspect_lower_threshold": r,
        "total_positive_images": len(records),

        "detect_images": detect_images,
        "reinspect_images": reinspect_images,
        "pass_images": pass_images,

        "detect_rate_positive_corpus":
            detect_images / len(records),

        "reinspect_rate_positive_corpus":
            reinspect_images / len(records),

        "unsafe_pass_rate_positive_corpus":
            pass_images / len(records),

        "detect_candidate_count":
            detect_candidate_count,

        "gray_zone_candidate_count":
            reinspect_candidate_count,

        "note":
            (
                "All 500 OOF images contain GT foreign objects. "
                "PASS therefore means unsafe pass on this positive corpus. "
                "This dataset cannot estimate specificity or production "
                "reinspection workload on normal products."
            ),
    })

routing = pd.DataFrame(routing_rows)

routing.to_csv(
    OUT / "03g_detect_reinspect_pass_routing.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 7. CANDIDATE CONFIDENCE DISTRIBUTION
# ============================================================

candidate_rows = []

for rec in records:

    for idx, pred in enumerate(rec["predictions"]):

        candidate_rows.append({
            "stem": rec["stem"],
            "fold_id": rec["fold_id"],
            "prediction_index": idx,
            "confidence": float(pred["confidence"]),
            "bbox_json": json.dumps(
                pred["bbox"],
                ensure_ascii=False,
            ),
        })

candidates = pd.DataFrame(candidate_rows)

candidates.to_csv(
    OUT / "03h_all_lowconf_oof_candidates.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. SUMMARY
# ============================================================

best_f1_row = sweep.loc[
    sweep["f1"].idxmax()
]

min_fn_row = sweep.loc[
    sweep["fn"].idxmin()
]

summary_lines = [
    "# Stage 3 Low-Confidence OOF Safety Analysis",
    "",
    "## Integrity",
    "",
    f"- OOF images: {len(records)}",
    f"- GT objects: {total_gt}",
    f"- Raw prediction candidates retained at floor: {total_raw_predictions}",
    f"- IoU matching threshold: {IOU_THRESHOLD:.2f}",
    "",
    "## Stage2 reproduction check",
    "",
    (
        f"- confidence=0.25: "
        f"TP={int(r025['tp'])}, "
        f"FP={int(r025['fp'])}, "
        f"FN={int(r025['fn'])}, "
        f"Precision={r025['precision']:.6f}, "
        f"Recall={r025['recall']:.6f}, "
        f"F1={r025['f1']:.6f}"
    ),
    "- This exactly reproduces the frozen B2 OOF reporting point.",
    "",
    "## Threshold observations",
    "",
    (
        f"- Highest F1 among evaluated thresholds: "
        f"confidence={best_f1_row['confidence_threshold']:.3f}, "
        f"F1={best_f1_row['f1']:.6f}."
    ),
    (
        f"- Lowest FN among evaluated thresholds: "
        f"confidence={min_fn_row['confidence_threshold']:.3f}, "
        f"FN={int(min_fn_row['fn'])}, "
        f"Recall={min_fn_row['recall']:.6f}, "
        f"FP={int(min_fn_row['fp'])}."
    ),
    "",
    "## Safety interpretation",
    "",
    (
        "- confidence >= 0.40 is evaluated as a DETECT candidate region, "
        "not a finalized operational threshold."
    ),
    (
        "- Predictions below 0.40 are evaluated as possible REINSPECT "
        "gray-zone candidates."
    ),
    (
        "- Lowering the reinspection floor can recover additional GT objects "
        "but also introduces additional FP candidates."
    ),
    (
        "- The final reinspection floor must therefore be selected from the "
        "empirical safety-versus-review-burden trade-off."
    ),
    "",
    "## Limitation",
    "",
    (
        "- All 500 development images contain GT foreign objects. "
        "There is no true-negative / normal-product corpus."
    ),
    (
        "- Therefore specificity, false-positive rate on normal production, "
        "true PASS rate, and real production reinspection workload cannot be "
        "estimated from this dataset."
    ),
    (
        "- Image-level DETECT / REINSPECT / PASS results are positive-corpus "
        "safety diagnostics only."
    ),
    "",
    "## Threshold status",
    "",
    "- No final operational threshold is fixed by this analysis.",
]

summary_path = (
    EDA_OUT
    / "03_stage3_lowconf_oof_summary.md"
)

summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)


# ============================================================
# 9. PRINT
# ============================================================

print()
print("=== OOF THRESHOLD SWEEP ===")
print(
    sweep.to_string(index=False)
)

print()
print("=== REINSPECT TRADE-OFF ===")
print(
    tradeoff.to_string(index=False)
)

print()
print("=== IMAGE ROUTING ===")
print(
    routing[
        [
            "detect_threshold",
            "reinspect_lower_threshold",
            "detect_images",
            "reinspect_images",
            "pass_images",
            "unsafe_pass_rate_positive_corpus",
            "gray_zone_candidate_count",
        ]
    ].to_string(index=False)
)

print()
print("=== VALIDATION ===")
print(
    "0.25 reproduction:",
    actual,
)

print()
print("Saved:")
print(
    OUT / "03e_lowconf_oof_threshold_sweep.csv"
)
print(
    OUT / "03f_reinspect_object_tradeoff.csv"
)
print(
    OUT / "03g_detect_reinspect_pass_routing.csv"
)
print(summary_path)

print()
print("Stage 3 low-confidence OOF aggregation complete.")

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


# ============================================================
# HELPERS
# ============================================================

def parse_json_list(value):
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


def match_image(gt_boxes, predictions, conf_threshold):
    """
    prediction confidence >= threshold만 사용.
    confidence 내림차순 greedy matching.
    IoU >= 0.50이면 아직 매칭되지 않은 GT 중
    가장 높은 IoU의 GT와 매칭.
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
    matched_predictions = 0
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
            matched_predictions += 1
        else:
            fp += 1

    gt_count = len(gt_boxes)
    matched_gt_count = len(matched_gt)
    missed_gt_count = gt_count - matched_gt_count

    any_gt_detected = matched_gt_count >= 1
    all_gt_detected = (
        gt_count > 0
        and matched_gt_count == gt_count
    )

    return {
        "gt_count": gt_count,
        "matched_gt_count": matched_gt_count,
        "missed_gt_count": missed_gt_count,
        "prediction_count": len(kept),
        "fp_count": fp,
        "any_gt_detected": any_gt_detected,
        "all_gt_detected": all_gt_detected,
    }


# ============================================================
# 1. LOAD OOF IMAGE PREDICTIONS
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
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "stem",
        "fold_id",
        "gt_boxes_json",
        "prediction_boxes_json",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path} missing columns: {sorted(missing)}"
        )

    frames.append(df)


images = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# 2. INTEGRITY CHECK
# ============================================================

if len(images) != 500:
    raise RuntimeError(
        f"Expected 500 OOF images, found {len(images)}"
    )

if images["stem"].nunique() != 500:
    raise RuntimeError(
        "Duplicate OOF stems detected."
    )

if sorted(images["fold_id"].unique().tolist()) != [1, 2, 3, 4]:
    raise RuntimeError(
        f"Unexpected folds: {sorted(images['fold_id'].unique())}"
    )


records = []

total_gt = 0

for _, row in images.iterrows():

    gt_boxes = parse_json_list(
        row["gt_boxes_json"]
    )

    predictions = parse_json_list(
        row["prediction_boxes_json"]
    )

    total_gt += len(gt_boxes)

    records.append({
        "stem": row["stem"],
        "fold_id": int(row["fold_id"]),
        "gt_boxes": gt_boxes,
        "predictions": predictions,
        "gt_count": len(gt_boxes),
    })


if total_gt != 1147:
    raise RuntimeError(
        f"Expected 1147 GT objects, found {total_gt}"
    )


# ============================================================
# 3. GT COUNT DISTRIBUTION
# ============================================================

gt_count_df = pd.DataFrame([
    {
        "stem": r["stem"],
        "fold_id": r["fold_id"],
        "gt_count": r["gt_count"],
    }
    for r in records
])

gt_distribution = (
    gt_count_df
    .groupby("gt_count")
    .size()
    .reset_index(name="image_count")
    .sort_values("gt_count")
)

gt_distribution.to_csv(
    OUT / "03i_gt_count_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 4. IMAGE-LEVEL SAFETY SWEEP
# ============================================================

summary_rows = []
detail_rows = []

for threshold in THRESHOLDS:

    threshold_results = []

    for rec in records:

        result = match_image(
            rec["gt_boxes"],
            rec["predictions"],
            threshold,
        )

        row = {
            "stem": rec["stem"],
            "fold_id": rec["fold_id"],
            "confidence_threshold": threshold,
            **result,
        }

        threshold_results.append(row)
        detail_rows.append(row)

    df = pd.DataFrame(threshold_results)

    total_images = len(df)

    any_detect_images = int(
        df["any_gt_detected"].sum()
    )

    all_detect_images = int(
        df["all_gt_detected"].sum()
    )

    zero_detect_images = int(
        (~df["any_gt_detected"]).sum()
    )

    partial_detect_images = int(
        (
            df["any_gt_detected"]
            & (~df["all_gt_detected"])
        ).sum()
    )

    total_missed_gt = int(
        df["missed_gt_count"].sum()
    )

    summary_rows.append({
        "confidence_threshold": threshold,
        "total_images": total_images,

        "any_gt_detected_images":
            any_detect_images,

        "any_gt_detected_rate":
            any_detect_images / total_images,

        "all_gt_detected_images":
            all_detect_images,

        "all_gt_detected_rate":
            all_detect_images / total_images,

        "partial_detection_images":
            partial_detect_images,

        "zero_gt_detected_images":
            zero_detect_images,

        "zero_gt_detected_rate":
            zero_detect_images / total_images,

        "total_missed_gt_objects":
            total_missed_gt,
    })


image_summary = pd.DataFrame(
    summary_rows
)

image_detail = pd.DataFrame(
    detail_rows
)


image_summary.to_csv(
    OUT / "03j_image_level_safety_sweep.csv",
    index=False,
    encoding="utf-8-sig",
)

image_detail.to_csv(
    OUT / "03k_image_level_safety_detail.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 5. STRATIFY BY GT COUNT
# ============================================================

stratified_rows = []

for threshold in THRESHOLDS:

    sub = image_detail[
        np.isclose(
            image_detail["confidence_threshold"],
            threshold,
        )
    ].copy()

    for gt_count, g in sub.groupby("gt_count"):

        n = len(g)

        stratified_rows.append({
            "confidence_threshold": threshold,
            "gt_count": int(gt_count),
            "image_count": n,

            "any_gt_detected_images":
                int(g["any_gt_detected"].sum()),

            "any_gt_detected_rate":
                float(g["any_gt_detected"].mean()),

            "all_gt_detected_images":
                int(g["all_gt_detected"].sum()),

            "all_gt_detected_rate":
                float(g["all_gt_detected"].mean()),

            "partial_detection_images":
                int(
                    (
                        g["any_gt_detected"]
                        & (~g["all_gt_detected"])
                    ).sum()
                ),

            "zero_gt_detected_images":
                int((~g["any_gt_detected"]).sum()),

            "total_missed_gt_objects":
                int(g["missed_gt_count"].sum()),
        })


stratified = pd.DataFrame(
    stratified_rows
)

stratified.to_csv(
    OUT / "03l_image_safety_by_gt_count.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. SINGLE-OBJECT IMAGE ANALYSIS
# ============================================================

single_rows = []

single_records = [
    r for r in records
    if r["gt_count"] == 1
]

for threshold in THRESHOLDS:

    detected = 0
    missed = 0

    for rec in single_records:

        result = match_image(
            rec["gt_boxes"],
            rec["predictions"],
            threshold,
        )

        if result["any_gt_detected"]:
            detected += 1
        else:
            missed += 1

    total = len(single_records)

    single_rows.append({
        "confidence_threshold": threshold,
        "single_object_images": total,
        "detected_images": detected,
        "missed_images": missed,
        "image_detection_rate":
            detected / total if total else np.nan,
        "image_miss_rate":
            missed / total if total else np.nan,
    })


single_summary = pd.DataFrame(
    single_rows
)

single_summary.to_csv(
    OUT / "03m_single_object_image_safety.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 7. DETECT THRESHOLD 0.40 DETAIL
# ============================================================

T_MAIN = 0.40

main_detail = image_detail[
    np.isclose(
        image_detail["confidence_threshold"],
        T_MAIN,
    )
].copy()

main_failures = main_detail[
    ~main_detail["all_gt_detected"]
].copy()

main_failures = main_failures.sort_values(
    [
        "gt_count",
        "missed_gt_count",
        "stem",
    ],
    ascending=[
        True,
        False,
        True,
    ],
)

main_failures.to_csv(
    OUT / "03n_threshold_040_incomplete_images.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. SUMMARY
# ============================================================

r040 = image_summary[
    np.isclose(
        image_summary["confidence_threshold"],
        0.40,
    )
].iloc[0]

r025 = image_summary[
    np.isclose(
        image_summary["confidence_threshold"],
        0.25,
    )
].iloc[0]

single040 = single_summary[
    np.isclose(
        single_summary["confidence_threshold"],
        0.40,
    )
].iloc[0]

single025 = single_summary[
    np.isclose(
        single_summary["confidence_threshold"],
        0.25,
    )
].iloc[0]


summary_lines = [
    "# Stage 3 Image-Level Safety Analysis",
    "",
    "## Dataset scope",
    "",
    f"- OOF images: {len(records)}",
    f"- GT objects: {total_gt}",
    f"- Single-object images: {len(single_records)}",
    "",
    "## Confidence 0.25",
    "",
    (
        f"- At least one GT detected: "
        f"{int(r025['any_gt_detected_images'])}/{len(records)} "
        f"({r025['any_gt_detected_rate']:.6f})"
    ),
    (
        f"- All GT objects detected: "
        f"{int(r025['all_gt_detected_images'])}/{len(records)} "
        f"({r025['all_gt_detected_rate']:.6f})"
    ),
    (
        f"- Images with zero GT detected: "
        f"{int(r025['zero_gt_detected_images'])}"
    ),
    "",
    "## Confidence 0.40",
    "",
    (
        f"- At least one GT detected: "
        f"{int(r040['any_gt_detected_images'])}/{len(records)} "
        f"({r040['any_gt_detected_rate']:.6f})"
    ),
    (
        f"- All GT objects detected: "
        f"{int(r040['all_gt_detected_images'])}/{len(records)} "
        f"({r040['all_gt_detected_rate']:.6f})"
    ),
    (
        f"- Partial-detection images: "
        f"{int(r040['partial_detection_images'])}"
    ),
    (
        f"- Images with zero GT detected: "
        f"{int(r040['zero_gt_detected_images'])}"
    ),
    (
        f"- Total missed GT objects: "
        f"{int(r040['total_missed_gt_objects'])}"
    ),
    "",
    "## Single-object images",
    "",
    (
        f"- confidence 0.25: "
        f"{int(single025['detected_images'])}/"
        f"{int(single025['single_object_images'])} detected "
        f"({single025['image_detection_rate']:.6f})"
    ),
    (
        f"- confidence 0.40: "
        f"{int(single040['detected_images'])}/"
        f"{int(single040['single_object_images'])} detected "
        f"({single040['image_detection_rate']:.6f})"
    ),
    "",
    "## Interpretation",
    "",
    (
        "- 'At least one GT detected' approximates product-level rejection "
        "when any foreign object is sufficient to reject the product."
    ),
    (
        "- 'All GT detected' is stricter and reveals residual object-level misses "
        "hidden by multi-object images."
    ),
    (
        "- Single-object images remove the masking effect where one detected object "
        "can hide another missed object within the same product image."
    ),
    "",
    "## Limitation",
    "",
    (
        "- All 500 images are GT-positive. "
        "There is no normal / true-negative corpus."
    ),
    (
        "- Therefore production PASS specificity and real reinspection workload "
        "cannot be estimated."
    ),
]

summary_path = (
    EDA_OUT
    / "03_stage3_image_safety_summary.md"
)

summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)


# ============================================================
# 9. PRINT
# ============================================================

print()
print("=== GT COUNT DISTRIBUTION ===")
print(
    gt_distribution.to_string(index=False)
)

print()
print("=== IMAGE LEVEL SAFETY ===")
print(
    image_summary[
        [
            "confidence_threshold",
            "any_gt_detected_images",
            "any_gt_detected_rate",
            "all_gt_detected_images",
            "all_gt_detected_rate",
            "partial_detection_images",
            "zero_gt_detected_images",
            "total_missed_gt_objects",
        ]
    ].to_string(index=False)
)

print()
print("=== SINGLE OBJECT IMAGE SAFETY ===")
print(
    single_summary.to_string(index=False)
)

print()
print("=== CONF 0.40 INCOMPLETE IMAGE COUNT ===")
print(len(main_failures))

print()
print("Saved:")
print(
    OUT / "03j_image_level_safety_sweep.csv"
)
print(
    OUT / "03l_image_safety_by_gt_count.csv"
)
print(
    OUT / "03m_single_object_image_safety.csv"
)
print(
    OUT / "03n_threshold_040_incomplete_images.csv"
)
print(summary_path)

print()
print("Stage 3 image-level safety analysis complete.")
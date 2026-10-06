from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()

SRC = (
    ROOT
    / "outputs"
    / "tables"
    / "02b_b2_visibility_error_analysis_v1"
)

OBJ_PATH = SRC / "02b_object_pixel_size_diagnostics.csv"
FP_REPORT_PATH = SRC / "02a_false_positive_analysis.csv"

OUT = ROOT / "outputs" / "tables" / "03_stage3_safety_analysis_v1"
EDA = ROOT / "outputs" / "eda"

OUT.mkdir(parents=True, exist_ok=True)
EDA.mkdir(parents=True, exist_ok=True)

# ==================================================
# 0. LOAD + INTEGRITY
# ==================================================

obj = pd.read_csv(OBJ_PATH)

required = {
    "stem",
    "object_id",
    "fold_id",
    "status",
    "error_type",
    "prediction_confidence",
    "iou",
    "projected_bbox_min_side_px",
}

missing = required - set(obj.columns)

if missing:
    raise RuntimeError(
        f"Object diagnostics missing required columns: {sorted(missing)}"
    )

obj["prediction_confidence"] = pd.to_numeric(
    obj["prediction_confidence"],
    errors="coerce",
)

obj["iou"] = pd.to_numeric(
    obj["iou"],
    errors="coerce",
)

if len(obj) != 1147:
    raise RuntimeError(
        f"Expected 1147 GT objects, found {len(obj)}"
    )

n_images = obj["stem"].nunique()

if n_images != 500:
    raise RuntimeError(
        f"Expected 500 images, found {n_images}"
    )

IOU_MATCH = 0.50
REPORT_THRESHOLD = 0.25


# ==================================================
# 1. SEARCH FOR RAW B2 FP CANDIDATE TABLE
# ==================================================

def confidence_col(df):
    candidates = [
        "prediction_confidence",
        "confidence",
        "conf",
        "score",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


raw_fp_candidates = []

for p in (ROOT / "outputs").rglob("*.csv"):

    lowpath = str(p).lower()

    # B2 / visibility 결과 우선
    if (
        "b2" not in lowpath
        and "visibility" not in lowpath
        and p.parent != SRC
    ):
        continue

    if p == OBJ_PATH:
        continue

    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    cc = confidence_col(df)

    if cc is None:
        continue

    # FP 가능성이 있는 파일명/컬럼만
    name_signal = (
        "fp" in p.name.lower()
        or "false_positive" in p.name.lower()
    )

    col_signal = any(
        c in df.columns
        for c in [
            "fp_status",
            "fp_type",
            "nearest_gt_iou",
        ]
    )

    if not (name_signal or col_signal):
        continue

    vals = pd.to_numeric(
        df[cc],
        errors="coerce",
    ).dropna()

    if len(vals) == 0:
        continue

    raw_fp_candidates.append({
        "path": p,
        "rows": len(df),
        "conf_col": cc,
        "min_conf": float(vals.min()),
        "max_conf": float(vals.max()),
        "df": df,
    })


# 가장 raw에 가까워 보이는 것 선택
def fp_score(x):
    s = 0
    p = str(x["path"]).lower()

    if "oof" in p:
        s += 5
    if "b2" in p:
        s += 4
    if "visibility" in p:
        s += 3
    if "fp" in x["path"].name.lower():
        s += 3
    if x["min_conf"] < REPORT_THRESHOLD:
        s += 10

    s += min(x["rows"], 10000) / 10000

    return s


raw_fp = None
raw_fp_info = None

if raw_fp_candidates:
    raw_fp_info = max(
        raw_fp_candidates,
        key=fp_score,
    )

    # 0.25 미만 FP가 실제 보존된 경우에만
    # low-threshold precision 계산에 사용
    if raw_fp_info["min_conf"] < REPORT_THRESHOLD:
        raw_fp = raw_fp_info["df"].copy()


# ==================================================
# 2. FALLBACK: REPORTING-THRESHOLD FP TABLE
# ==================================================

report_fp = None
report_fp_conf_col = None

if FP_REPORT_PATH.exists():

    report_fp = pd.read_csv(FP_REPORT_PATH)
    report_fp_conf_col = confidence_col(report_fp)

    if report_fp_conf_col is not None:
        report_fp[report_fp_conf_col] = pd.to_numeric(
            report_fp[report_fp_conf_col],
            errors="coerce",
        )


# ==================================================
# 3. OBJECT-LEVEL CONFIDENCE SWEEP
# ==================================================

thresholds = [
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
    0.40,
    0.50,
    0.60,
    0.70,
]

rows = []

for t in thresholds:

    conf_ok = (
        obj["prediction_confidence"].fillna(-1)
        >= t
    )

    iou_ok = (
        obj["iou"].fillna(-1)
        >= IOU_MATCH
    )

    tp_mask = conf_ok & iou_ok
    localization_mask = conf_ok & (~iou_ok)

    tp = int(tp_mask.sum())
    fn = int(len(obj) - tp)

    localization_failure = int(
        localization_mask.sum()
    )

    low_conf_or_absent = int(
        (~conf_ok).sum()
    )

    recall = tp / len(obj)

    fp = np.nan
    precision = np.nan
    f1 = np.nan
    fp_valid = False
    fp_source = ""

    # ----------------------------------------------
    # raw low-confidence FP가 있으면 전체 sweep 가능
    # ----------------------------------------------
    if raw_fp is not None:

        cc = raw_fp_info["conf_col"]

        vals = pd.to_numeric(
            raw_fp[cc],
            errors="coerce",
        )

        min_available = raw_fp_info["min_conf"]

        if t >= min_available:
            fp = int((vals >= t).sum())
            fp_valid = True
            fp_source = str(
                raw_fp_info["path"].relative_to(ROOT)
            )

    # ----------------------------------------------
    # raw FP가 없으면 >=0.25 구간만
    # reporting FP table 사용
    # ----------------------------------------------
    elif (
        report_fp is not None
        and report_fp_conf_col is not None
        and t >= REPORT_THRESHOLD
    ):

        vals = pd.to_numeric(
            report_fp[report_fp_conf_col],
            errors="coerce",
        )

        fp = int((vals >= t).sum())
        fp_valid = True
        fp_source = str(
            FP_REPORT_PATH.relative_to(ROOT)
        )

    if fp_valid:

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else np.nan
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else np.nan
        )

    rows.append({
        "confidence_threshold": t,
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "localization_failure_at_threshold": localization_failure,
        "below_threshold_or_absent": low_conf_or_absent,
        "fp": fp,
        "precision": precision,
        "f1": f1,
        "fp_valid": fp_valid,
        "fp_source": fp_source,
    })


sweep = pd.DataFrame(rows)

sweep.to_csv(
    OUT / "03a_confidence_threshold_sweep.csv",
    index=False,
    encoding="utf-8-sig",
)


# ==================================================
# 4. FN MECHANISM AS THRESHOLD CHANGES
# ==================================================

mechanism_rows = []

for t in thresholds:

    conf = obj["prediction_confidence"].fillna(-1)
    iou = obj["iou"].fillna(-1)

    tp = (
        (conf >= t)
        & (iou >= IOU_MATCH)
    )

    low_conf = (
        (conf < t)
        & (iou >= IOU_MATCH)
    )

    localization = (
        (conf >= t)
        & (iou < IOU_MATCH)
    )

    both_or_none = (
        (conf < t)
        & (iou < IOU_MATCH)
    )

    mechanism_rows.append({
        "confidence_threshold": t,
        "tp": int(tp.sum()),
        "low_confidence_only": int(low_conf.sum()),
        "localization_only": int(localization.sum()),
        "low_confidence_and_low_iou": int(both_or_none.sum()),
        "total_fn": int((~tp).sum()),
    })


mechanism = pd.DataFrame(mechanism_rows)

mechanism.to_csv(
    OUT / "03b_fn_mechanism_by_threshold.csv",
    index=False,
    encoding="utf-8-sig",
)


# ==================================================
# 5. IMAGE-LEVEL POSITIVE-CORPUS ROUTING
#
# DETECT:
#   candidate confidence >= 0.25
#
# REINSPECT:
#   lower safety threshold <= candidate confidence < 0.25
#
# PASS:
#   no candidate reaches lower threshold
#
# IMPORTANT:
# all 500 images are GT-positive.
# PASS here = unsafe pass on this development corpus.
# ==================================================

image_max = (
    obj.groupby("stem")["prediction_confidence"]
    .max()
    .rename("max_object_candidate_confidence")
    .reset_index()
)

# raw FP 후보가 있으면 image max confidence에 포함
if raw_fp is not None and "stem" in raw_fp.columns:

    cc = raw_fp_info["conf_col"]

    fp_tmp = raw_fp[["stem", cc]].copy()

    fp_tmp[cc] = pd.to_numeric(
        fp_tmp[cc],
        errors="coerce",
    )

    fp_max = (
        fp_tmp.groupby("stem")[cc]
        .max()
        .rename("max_fp_candidate_confidence")
        .reset_index()
    )

    image_max = image_max.merge(
        fp_max,
        on="stem",
        how="left",
    )

    image_max["max_candidate_confidence"] = (
        image_max[
            [
                "max_object_candidate_confidence",
                "max_fp_candidate_confidence",
            ]
        ]
        .max(axis=1)
    )

else:

    image_max["max_candidate_confidence"] = (
        image_max["max_object_candidate_confidence"]
    )


reinspect_thresholds = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
]

routing_rows = []

for r in reinspect_thresholds:

    conf = (
        image_max["max_candidate_confidence"]
        .fillna(-1)
    )

    detect = conf >= REPORT_THRESHOLD

    reinspect = (
        (conf >= r)
        & (conf < REPORT_THRESHOLD)
    )

    passed = conf < r

    detect_n = int(detect.sum())
    reinspect_n = int(reinspect.sum())
    pass_n = int(passed.sum())

    routing_rows.append({
        "detect_threshold": REPORT_THRESHOLD,
        "reinspect_threshold": r,
        "total_positive_images": len(image_max),
        "detect_images": detect_n,
        "reinspect_images": reinspect_n,
        "pass_images": pass_n,

        # 모든 이미지가 이물 GT-positive이므로
        # PASS는 이 corpus에서는 unsafe pass
        "unsafe_pass_rate_positive_corpus":
            pass_n / len(image_max),

        "reinspect_rate_positive_corpus":
            reinspect_n / len(image_max),

        "detect_rate_positive_corpus":
            detect_n / len(image_max),

        "limitation":
            (
                "All 500 development images contain GT objects. "
                "This does not estimate production specificity, "
                "normal-product PASS rate, or real reinspection workload."
            ),
    })


routing = pd.DataFrame(routing_rows)

routing.to_csv(
    OUT / "03c_positive_corpus_routing_sweep.csv",
    index=False,
    encoding="utf-8-sig",
)


# ==================================================
# 6. OBJECT SIZE x THRESHOLD DIAGNOSTIC
# analytical only — NOT an operational trigger
# ==================================================

SIZE_CANDIDATE = 11.338688085676036

obj["size_diag"] = np.where(
    obj["projected_bbox_min_side_px"]
    < SIZE_CANDIDATE,
    "<11.34px",
    ">=11.34px",
)

size_rows = []

for t in thresholds:

    for size_group, g in obj.groupby("size_diag"):

        conf = g["prediction_confidence"].fillna(-1)
        iou = g["iou"].fillna(-1)

        hit = (
            (conf >= t)
            & (iou >= IOU_MATCH)
        )

        size_rows.append({
            "confidence_threshold": t,
            "size_group": size_group,
            "n": len(g),
            "tp": int(hit.sum()),
            "fn": int((~hit).sum()),
            "recall": float(hit.mean()),
            "note":
                (
                    "GT-derived size diagnostic only; "
                    "not directly observable for a completely missed object."
                ),
        })


size_sweep = pd.DataFrame(size_rows)

size_sweep.to_csv(
    OUT / "03d_size_threshold_diagnostic.csv",
    index=False,
    encoding="utf-8-sig",
)


# ==================================================
# 7. WRITE SUMMARY
# ==================================================

r025 = sweep[
    np.isclose(
        sweep["confidence_threshold"],
        REPORT_THRESHOLD,
    )
].iloc[0]

lowest_pass = routing.sort_values(
    "unsafe_pass_rate_positive_corpus"
).iloc[0]

lines = []

lines.append("# Stage 3 Safety Analysis - Initial Sweep")
lines.append("")
lines.append("## Integrity")
lines.append("")
lines.append(
    f"- GT-positive development images: {n_images}"
)
lines.append(
    f"- GT objects: {len(obj)}"
)
lines.append(
    f"- IoU match criterion: {IOU_MATCH:.2f}"
)
lines.append(
    f"- Existing Stage 2 reporting confidence: {REPORT_THRESHOLD:.2f}"
)
lines.append("")
lines.append("## Existing reporting point")
lines.append("")
lines.append(
    f"- TP={int(r025['tp'])}"
)
lines.append(
    f"- FN={int(r025['fn'])}"
)
lines.append(
    f"- Recall={r025['recall']:.6f}"
)

if bool(r025["fp_valid"]):
    lines.append(
        f"- FP={int(r025['fp'])}"
    )
    lines.append(
        f"- Precision={r025['precision']:.6f}"
    )
    lines.append(
        f"- F1={r025['f1']:.6f}"
    )

lines.append("")
lines.append("## FP sweep support")
lines.append("")

if raw_fp is not None:
    lines.append(
        "- Low-confidence raw B2 FP candidates were found."
    )
    lines.append(
        f"- Source: {raw_fp_info['path'].relative_to(ROOT)}"
    )
    lines.append(
        f"- Minimum stored FP confidence: {raw_fp_info['min_conf']:.6f}"
    )
else:
    lines.append(
        "- No trustworthy low-confidence raw FP table was automatically identified."
    )
    lines.append(
        "- Precision/F1 below confidence 0.25 are therefore not treated as valid."
    )

lines.append("")
lines.append("## Routing interpretation")
lines.append("")
lines.append(
    "- DETECT uses confidence >= 0.25."
)
lines.append(
    "- REINSPECT is evaluated as a lower confidence gray zone."
)
lines.append(
    "- PASS means no retained candidate reached the reinspection threshold."
)
lines.append(
    "- Because every development image contains at least one GT object, "
    "PASS in this corpus is an unsafe pass, not evidence of true-negative specificity."
)
lines.append(
    "- Real production reinspection workload cannot be estimated without "
    "normal / true-negative images."
)
lines.append("")
lines.append("## Operational-variable constraint")
lines.append("")
lines.append(
    "- GT bbox size, GT contrast, GT CNR-like values, and GT localization IoU "
    "are useful failure-analysis variables."
)
lines.append(
    "- They must not be used as direct operating triggers for a completely missed "
    "object because they are unavailable at inference time."
)
lines.append(
    "- Acquisition metadata and model-side prediction signals may be considered "
    "operationally if they are available before the decision."
)
lines.append("")
lines.append("## Threshold status")
lines.append("")
lines.append(
    "- No final PASS / DETECT / REINSPECT threshold is selected by this script."
)
lines.append(
    "- Threshold sweep results are diagnostic evidence for Stage 3."
)

summary_path = (
    EDA
    / "03_stage3_safety_analysis_summary.md"
)

summary_path.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ==================================================
# 8. PRINT
# ==================================================

print()
print("=== CONFIDENCE SWEEP ===")
print(
    sweep[
        [
            "confidence_threshold",
            "tp",
            "fn",
            "recall",
            "fp",
            "precision",
            "f1",
            "fp_valid",
        ]
    ].to_string(index=False)
)

print()
print("=== POSITIVE-CORPUS ROUTING ===")
print(
    routing[
        [
            "detect_threshold",
            "reinspect_threshold",
            "detect_images",
            "reinspect_images",
            "pass_images",
            "unsafe_pass_rate_positive_corpus",
        ]
    ].to_string(index=False)
)

print()
print("=== RAW FP STATUS ===")

if raw_fp is not None:
    print("LOW-CONFIDENCE FP SOURCE FOUND")
    print(raw_fp_info["path"])
    print(
        "min confidence:",
        raw_fp_info["min_conf"],
    )
    print(
        "rows:",
        raw_fp_info["rows"],
    )
else:
    print(
        "No low-confidence raw FP source found automatically."
    )
    print(
        "Below 0.25: Recall is valid, "
        "but Precision/F1 must not be interpreted."
    )

print()
print("=== OUTPUT ===")
print(OUT)
print(summary_path)
print()
print("Stage 3 initial sweep complete.")
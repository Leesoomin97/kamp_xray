from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path.cwd()

DIR = (
    ROOT
    / "outputs"
    / "tables"
    / "03_stage3_safety_analysis_v1"
)

OUT = (
    ROOT
    / "outputs"
    / "eda"
    / "03_stage3_eda_audit.md"
)


# ============================================================
# LOAD
# ============================================================

sweep = pd.read_csv(
    DIR / "03e_lowconf_oof_threshold_sweep.csv"
)

tradeoff = pd.read_csv(
    DIR / "03f_reinspect_object_tradeoff.csv"
)

image = pd.read_csv(
    DIR / "03j_image_level_safety_sweep.csv"
)

by_gt = pd.read_csv(
    DIR / "03l_image_safety_by_gt_count.csv"
)

single = pd.read_csv(
    DIR / "03m_single_object_image_safety.csv"
)

incomplete = pd.read_csv(
    DIR / "03n_threshold_040_incomplete_images.csv"
)

zero_gt = pd.read_csv(
    DIR / "03o_zero_correct_detection_gt_cases.csv"
)

zero_pred = pd.read_csv(
    DIR / "03p_zero_correct_detection_highconf_predictions.csv"
)


# ============================================================
# HELPERS
# ============================================================

checks = []


def add_check(name, condition, detail):
    checks.append({
        "check": name,
        "status": "PASS" if condition else "FAIL",
        "detail": detail,
    })


def row_at(df, threshold):
    return df[
        np.isclose(
            df["confidence_threshold"].astype(float),
            threshold,
        )
    ].iloc[0]


# ============================================================
# 1. OBJECT-LEVEL CONSISTENCY
# ============================================================

r025 = row_at(sweep, 0.25)
r040 = row_at(sweep, 0.40)
r001 = row_at(sweep, 0.001)

add_check(
    "B2 OOF @0.25 reproduction",
    (
        int(r025["tp"]) == 1121
        and int(r025["fp"]) == 51
        and int(r025["fn"]) == 26
    ),
    (
        f"TP={int(r025['tp'])}, "
        f"FP={int(r025['fp'])}, "
        f"FN={int(r025['fn'])}"
    ),
)

add_check(
    "0.25 -> 0.40 recall preserved",
    (
        int(r025["tp"]) == int(r040["tp"])
        and int(r025["fn"]) == int(r040["fn"])
    ),
    (
        f"0.25: TP={int(r025['tp'])}, FN={int(r025['fn'])}; "
        f"0.40: TP={int(r040['tp'])}, FN={int(r040['fn'])}"
    ),
)

add_check(
    "0.40 reduces FP vs 0.25",
    int(r040["fp"]) < int(r025["fp"]),
    (
        f"FP {int(r025['fp'])} -> {int(r040['fp'])}"
    ),
)

add_check(
    "0.001 minimum-FN point",
    int(r001["fn"]) == sweep["fn"].min(),
    (
        f"FN={int(r001['fn'])}, "
        f"Recall={float(r001['recall']):.6f}, "
        f"FP={int(r001['fp'])}"
    ),
)


# ============================================================
# 2. IMAGE-LEVEL CONSISTENCY
# ============================================================

img040 = row_at(image, 0.40)
img025 = row_at(image, 0.25)

add_check(
    "0.40 object FN equals image missed-object total",
    int(r040["fn"]) == int(img040["total_missed_gt_objects"]),
    (
        f"object FN={int(r040['fn'])}, "
        f"image missed GT={int(img040['total_missed_gt_objects'])}"
    ),
)

add_check(
    "0.40 image partition sums to 500",
    (
        int(img040["all_gt_detected_images"])
        + int(img040["partial_detection_images"])
        + int(img040["zero_gt_detected_images"])
        == 500
    ),
    (
        f"all={int(img040['all_gt_detected_images'])}, "
        f"partial={int(img040['partial_detection_images'])}, "
        f"zero={int(img040['zero_gt_detected_images'])}"
    ),
)

add_check(
    "0.25 and 0.40 image safety identical",
    (
        int(img025["any_gt_detected_images"])
        == int(img040["any_gt_detected_images"])
        and int(img025["all_gt_detected_images"])
        == int(img040["all_gt_detected_images"])
        and int(img025["total_missed_gt_objects"])
        == int(img040["total_missed_gt_objects"])
    ),
    (
        f"0.25 any/all/missed="
        f"{int(img025['any_gt_detected_images'])}/"
        f"{int(img025['all_gt_detected_images'])}/"
        f"{int(img025['total_missed_gt_objects'])}; "
        f"0.40="
        f"{int(img040['any_gt_detected_images'])}/"
        f"{int(img040['all_gt_detected_images'])}/"
        f"{int(img040['total_missed_gt_objects'])}"
    ),
)


# ============================================================
# 3. GT-COUNT STRATIFICATION
# ============================================================

gt040 = by_gt[
    np.isclose(
        by_gt["confidence_threshold"].astype(float),
        0.40,
    )
].copy()

add_check(
    "GT-count image total = 500",
    int(gt040["image_count"].sum()) == 500,
    f"sum={int(gt040['image_count'].sum())}",
)

add_check(
    "GT-count object total = 1147",
    int(
        (
            gt040["gt_count"]
            * gt040["image_count"]
        ).sum()
    ) == 1147,
    (
        "objects="
        f"{int((gt040['gt_count'] * gt040['image_count']).sum())}"
    ),
)


# ============================================================
# 4. SINGLE-OBJECT CONSISTENCY
# ============================================================

s040 = row_at(single, 0.40)

single_gt040 = gt040[
    gt040["gt_count"] == 1
].iloc[0]

add_check(
    "Single-object image count",
    int(s040["single_object_images"]) == 176,
    f"n={int(s040['single_object_images'])}",
)

add_check(
    "Single-object results match GT-count table",
    (
        int(s040["detected_images"])
        == int(single_gt040["any_gt_detected_images"])
        and int(s040["missed_images"])
        == int(single_gt040["zero_gt_detected_images"])
    ),
    (
        f"detected={int(s040['detected_images'])}, "
        f"missed={int(s040['missed_images'])}"
    ),
)


# ============================================================
# 5. FAILURE-CASE CONSISTENCY
# ============================================================

add_check(
    "Incomplete-image table count",
    len(incomplete) == 25,
    f"rows={len(incomplete)}",
)

add_check(
    "Zero-correct-detection case count",
    len(zero_gt) == 5,
    f"rows={len(zero_gt)}",
)

add_check(
    "High-confidence wrong prediction case count",
    len(zero_pred) == 5,
    f"rows={len(zero_pred)}",
)

add_check(
    "Zero-case stems match",
    set(zero_gt["stem"]) == set(zero_pred["stem"]),
    (
        f"GT stems={len(set(zero_gt['stem']))}, "
        f"prediction stems={len(set(zero_pred['stem']))}"
    ),
)

add_check(
    "All high-confidence zero-case predictions are IoU < 0.5",
    (
        pd.to_numeric(
            zero_pred["nearest_gt_iou"],
            errors="coerce",
        ) < 0.5
    ).all(),
    (
        f"max nearest IoU="
        f"{pd.to_numeric(zero_pred['nearest_gt_iou'], errors='coerce').max():.6f}"
    ),
)

add_check(
    "All zero-case predictions have confidence >= 0.40",
    (
        pd.to_numeric(
            zero_pred["confidence"],
            errors="coerce",
        ) >= 0.40
    ).all(),
    (
        f"min confidence="
        f"{pd.to_numeric(zero_pred['confidence'], errors='coerce').min():.6f}"
    ),
)


# ============================================================
# 6. TRADE-OFF CONSISTENCY
# ============================================================

t001 = tradeoff[
    np.isclose(
        tradeoff["reinspect_lower_threshold"].astype(float),
        0.001,
    )
].iloc[0]

add_check(
    "0.001 trade-off matches threshold sweep",
    (
        int(t001["tp_if_reinspect_candidates_reviewed"])
        == int(r001["tp"])
        and int(t001["fp_candidates_at_reinspect_floor"])
        == int(r001["fp"])
        and int(t001["residual_fn_after_reinspect_floor"])
        == int(r001["fn"])
    ),
    (
        f"TP={int(t001['tp_if_reinspect_candidates_reviewed'])}, "
        f"FP={int(t001['fp_candidates_at_reinspect_floor'])}, "
        f"FN={int(t001['residual_fn_after_reinspect_floor'])}"
    ),
)


# ============================================================
# 7. WRITE AUDIT REPORT
# ============================================================

audit = pd.DataFrame(checks)

fail_count = int((audit["status"] == "FAIL").sum())
pass_count = int((audit["status"] == "PASS").sum())

lines = [
    "# Stage 3 EDA Audit",
    "",
    f"- PASS: {pass_count}",
    f"- FAIL: {fail_count}",
    "",
    "| Check | Status | Detail |",
    "|---|---|---|",
]

for _, r in audit.iterrows():
    lines.append(
        f"| {r['check']} | {r['status']} | {r['detail']} |"
    )

lines += [
    "",
    "## Key verified values",
    "",
    (
        f"- Object-level @0.25: "
        f"TP={int(r025['tp'])}, FP={int(r025['fp'])}, "
        f"FN={int(r025['fn'])}, Recall={float(r025['recall']):.6f}"
    ),
    (
        f"- Object-level @0.40: "
        f"TP={int(r040['tp'])}, FP={int(r040['fp'])}, "
        f"FN={int(r040['fn'])}, Recall={float(r040['recall']):.6f}"
    ),
    (
        f"- Image-level @0.40: "
        f"any detected={int(img040['any_gt_detected_images'])}/500, "
        f"all detected={int(img040['all_gt_detected_images'])}/500, "
        f"zero correct={int(img040['zero_gt_detected_images'])}/500"
    ),
    (
        f"- Single-object @0.40: "
        f"{int(s040['detected_images'])}/"
        f"{int(s040['single_object_images'])}"
    ),
    (
        f"- Low-confidence extreme @0.001: "
        f"TP={int(r001['tp'])}, FP={int(r001['fp'])}, "
        f"FN={int(r001['fn'])}"
    ),
    "",
]

OUT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print()
print(audit.to_string(index=False))
print()
print(f"PASS={pass_count}, FAIL={fail_count}")
print()
print("Saved:")
print(OUT)

if fail_count > 0:
    raise RuntimeError(
        "Stage 3 audit found inconsistencies. "
        "Resolve FAIL items before making final EDA figures."
    )

print()
print("Stage 3 EDA audit complete.")
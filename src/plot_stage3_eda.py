from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()

TABLE_DIR = (
    ROOT
    / "outputs"
    / "tables"
    / "03_stage3_safety_analysis_v1"
)

FIG_DIR = (
    ROOT
    / "outputs"
    / "figures"
    / "stage3"
)

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

sweep = pd.read_csv(
    TABLE_DIR / "03e_lowconf_oof_threshold_sweep.csv"
)

image = pd.read_csv(
    TABLE_DIR / "03j_image_level_safety_sweep.csv"
)

single = pd.read_csv(
    TABLE_DIR / "03m_single_object_image_safety.csv"
)

zero_gt = pd.read_csv(
    TABLE_DIR / "03o_zero_correct_detection_gt_cases.csv"
)

zero_pred = pd.read_csv(
    TABLE_DIR / "03p_zero_correct_detection_highconf_predictions.csv"
)


# ============================================================
# FIGURE 1
# Precision / Recall / F1 vs confidence threshold
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    sweep["confidence_threshold"],
    sweep["precision"],
    marker="o",
    label="Precision",
)

ax.plot(
    sweep["confidence_threshold"],
    sweep["recall"],
    marker="o",
    label="Recall",
)

ax.plot(
    sweep["confidence_threshold"],
    sweep["f1"],
    marker="o",
    label="F1",
)

ax.axvline(
    0.25,
    linestyle="--",
    linewidth=1,
    label="Stage2 reporting threshold = 0.25",
)

ax.axvline(
    0.40,
    linestyle=":",
    linewidth=1,
    label="Stage3 DETECT candidate = 0.40",
)

ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Score")
ax.set_title(
    "Stage3 OOF Precision / Recall / F1 by Confidence Threshold"
)
ax.set_ylim(0, 1.03)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_DIR / "03a_precision_recall_f1_vs_threshold.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# FP / FN trade-off vs confidence threshold
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    sweep["confidence_threshold"],
    sweep["fp"],
    marker="o",
    label="FP",
)

ax.plot(
    sweep["confidence_threshold"],
    sweep["fn"],
    marker="o",
    label="FN",
)

ax.axvline(
    0.25,
    linestyle="--",
    linewidth=1,
    label="0.25",
)

ax.axvline(
    0.40,
    linestyle=":",
    linewidth=1,
    label="0.40",
)

ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Object count")
ax.set_title(
    "Stage3 OOF FP / FN Trade-off by Confidence Threshold"
)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_DIR / "03b_fp_fn_tradeoff_vs_threshold.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3
# Image-level any / all detection rates
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    image["confidence_threshold"],
    image["any_gt_detected_rate"],
    marker="o",
    label="At least one GT detected",
)

ax.plot(
    image["confidence_threshold"],
    image["all_gt_detected_rate"],
    marker="o",
    label="All GT detected",
)

ax.axvline(
    0.25,
    linestyle="--",
    linewidth=1,
    label="0.25",
)

ax.axvline(
    0.40,
    linestyle=":",
    linewidth=1,
    label="0.40",
)

ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Image-level detection rate")
ax.set_title(
    "Stage3 Image-level Safety by Confidence Threshold"
)
ax.set_ylim(0.90, 1.005)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_DIR / "03c_image_level_detection_vs_threshold.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 4
# Single-object image detection rate
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    single["confidence_threshold"],
    single["image_detection_rate"],
    marker="o",
)

ax.axvline(
    0.25,
    linestyle="--",
    linewidth=1,
    label="0.25",
)

ax.axvline(
    0.40,
    linestyle=":",
    linewidth=1,
    label="0.40",
)

ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Detection rate")
ax.set_title(
    "Stage3 Single-object Image Detection Rate"
)
ax.set_ylim(0.90, 1.005)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_DIR / "03d_single_object_detection_vs_threshold.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 5
# Zero-correct-detection 5 cases
# GT candidate confidence vs high-confidence wrong prediction
# ============================================================

merged = zero_gt.merge(
    zero_pred[
        [
            "stem",
            "confidence",
            "nearest_gt_iou",
        ]
    ],
    on="stem",
    how="inner",
    suffixes=(
        "_gt_candidate",
        "_wrong_prediction",
    ),
)

if len(merged) != 5:
    raise RuntimeError(
        f"Expected 5 zero-correct-detection cases, found {len(merged)}"
    )

merged = merged.sort_values(
    "projected_bbox_min_side_px"
).reset_index(drop=True)

labels = [
    f"Case {i + 1}"
    for i in range(len(merged))
]

x = np.arange(len(merged))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width / 2,
    merged["prediction_confidence"],
    width,
    label="GT-matched candidate confidence",
)

ax.bar(
    x + width / 2,
    merged["confidence"],
    width,
    label="High-confidence wrong prediction",
)

ax.axhline(
    0.40,
    linestyle=":",
    linewidth=1,
    label="DETECT candidate threshold = 0.40",
)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Zero-correct-detection case")
ax.set_ylabel("Confidence")
ax.set_title(
    "Stage3 Five Zero-correct-detection Cases"
)
ax.set_ylim(0, 0.7)
ax.grid(axis="y", alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_DIR / "03e_zero_correct_detection_confidence_cases.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 6
# Zero-correct-detection 5 cases:
# projected object size vs nearest IoU
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    merged["projected_bbox_min_side_px"],
    merged["nearest_gt_iou"],
    s=70,
)

for i, row in merged.iterrows():
    ax.annotate(
        f"Case {i + 1}",
        (
            row["projected_bbox_min_side_px"],
            row["nearest_gt_iou"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )

ax.axhline(
    0.50,
    linestyle="--",
    linewidth=1,
    label="IoU match threshold = 0.50",
)

ax.set_xlabel("Projected GT minimum side at 640 input (px)")
ax.set_ylabel("Nearest prediction IoU")
ax.set_title(
    "Stage3 Residual Failure Cases: Size vs Localization"
)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_DIR / "03f_zero_correct_detection_size_vs_iou.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SAVE CASE TABLE USED IN FIGURES
# ============================================================

merged[
    [
        "stem",
        "fold_id",
        "error_type",
        "native_bbox_min_side_px",
        "projected_bbox_min_side_px",
        "prediction_confidence",
        "iou",
        "confidence",
        "nearest_gt_iou",
    ]
].to_csv(
    TABLE_DIR / "03q_zero_correct_detection_plot_cases.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# PRINT
# ============================================================

print()
print("Saved Stage3 figures to:")
print(FIG_DIR)

for p in sorted(FIG_DIR.glob("03*.png")):
    print("-", p.name)

print()
print(
    "Saved case table:",
    TABLE_DIR / "03q_zero_correct_detection_plot_cases.csv",
)

print()
print("Stage3 EDA plotting complete.")
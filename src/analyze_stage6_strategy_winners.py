from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
EDA = ROOT / "outputs" / "eda"

RUNS = {
    "b0": ROOT / "outputs/stage2a_runs/kamp_conservative_640_fold1_e30_fullft_v1",
    "p192": ROOT / "outputs/stage6_runs/stage6_m1c_patch192_ov25_yolov8n_640_fold1_b0progress_v3",
    "p256": ROOT / "outputs/stage6_runs/stage6_m1_256c_patch256_ov25_yolov8n_640_fold1_b0progress_v1",
    "m2_p2": ROOT / "outputs/stage6_runs/stage6_m2_p2_yolov8n_640_fold1_b0matched_v1",
    "m3_bilateral": ROOT / "outputs/stage6_runs/stage6_m3_bilateral_yolov8n_640_fold1_b0matched_v2",
    "m4_clahe": ROOT / "outputs/stage6_runs/stage6_m4_clahe_yolov8n_640_fold1_b0matched_v1",
    "m5_bilateral_clahe": ROOT / "outputs/stage6_runs/stage6_m5_bilateral_clahe_yolov8n_640_fold1_b0matched_v1",
}

EXPECTED = {
    "b0": (281, 15, 9), "p192": (266, 29, 24), "p256": (276, 28, 14),
    "m2_p2": (277, 20, 13), "m3_bilateral": (281, 15, 9),
    "m4_clahe": (276, 29, 14), "m5_bilateral_clahe": (275, 14, 15),
}

EXPECTED_STAGE6_DETAIL = {
    "p192": {"localization_failure_count": 12, "low_confidence_fn_count": 11, "no_detection_fn_count": 1,
              "small_tp": 72, "low_contrast_tp": 76, "small_low_contrast_tp": 25},
    "p256": {"localization_failure_count": 11, "low_confidence_fn_count": 2, "no_detection_fn_count": 1,
              "small_tp": 67, "low_contrast_tp": 73, "small_low_contrast_tp": 22},
    "m2_p2": {"localization_failure_count": 4, "low_confidence_fn_count": 7, "no_detection_fn_count": 2,
               "small_tp": 68, "low_contrast_tp": 74, "small_low_contrast_tp": 23},
    "m3_bilateral": {"localization_failure_count": 5, "low_confidence_fn_count": 3, "no_detection_fn_count": 1,
                       "small_tp": 72, "low_contrast_tp": 75, "small_low_contrast_tp": 25},
    "m4_clahe": {"localization_failure_count": 6, "low_confidence_fn_count": 6, "no_detection_fn_count": 2,
                  "small_tp": 68, "low_contrast_tp": 72, "small_low_contrast_tp": 23},
    "m5_bilateral_clahe": {"localization_failure_count": 3, "low_confidence_fn_count": 10, "no_detection_fn_count": 2,
                            "small_tp": 70, "low_contrast_tp": 74, "small_low_contrast_tp": 23},
}

FEATURES = [
    "bbox_min_side_px", "bbox_area_px", "bbox_area_ratio",
    "absolute_median_difference", "signed_median_difference", "cnr_like_robust",
    "background_gradient_mean", "background_iqr", "object_iqr",
    "image_edge_distance_px", "image_edge_distance_norm",
    "product_edge_distance_px", "product_edge_distance_norm",
]


def require_files() -> None:
    required = ["metrics.json", "metrics.csv", "object_predictions.csv", "image_predictions.csv",
                "fp_predictions.csv", "error_types.csv", "confidence_sweep.csv", "evaluation_metadata.json"]
    for name, folder in RUNS.items():
        missing = [item for item in required if not (folder / item).is_file()]
        if missing:
            raise FileNotFoundError(f"{name} missing: {missing}")
    if not (TABLES / "01b_object_xray_features.csv").is_file():
        raise FileNotFoundError(TABLES / "01b_object_xray_features.csv")


def load_and_verify() -> tuple[pd.DataFrame, dict[str, dict]]:
    require_files()
    metrics: dict[str, dict] = {}
    frames: list[pd.DataFrame] = []
    reference_keys = None
    for strategy, folder in RUNS.items():
        metric = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        observed = (int(metric["tp"]), int(metric["fp"]), int(metric["fn"]))
        if observed != EXPECTED[strategy]:
            raise RuntimeError(f"{strategy} metric mismatch: {observed} != {EXPECTED[strategy]}")
        metrics[strategy] = metric
        obj = pd.read_csv(folder / "object_predictions.csv")
        if len(obj) != 290 or obj[["stem", "object_id"]].duplicated().any():
            raise RuntimeError(f"{strategy} object table is not 290 unique GT objects")
        direct = (int((obj.status == "TP").sum()), len(pd.read_csv(folder / "fp_predictions.csv")), int((obj.status == "FN").sum()))
        if direct != EXPECTED[strategy]:
            raise RuntimeError(f"{strategy} direct object/FP recomputation mismatch: {direct} != {EXPECTED[strategy]}")
        if strategy in EXPECTED_STAGE6_DETAIL:
            for field, expected in EXPECTED_STAGE6_DETAIL[strategy].items():
                if int(metric[field]) != expected:
                    raise RuntimeError(f"{strategy} {field} mismatch: {metric[field]} != {expected}")
        keys = set(zip(obj.stem, obj.object_id))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise RuntimeError(f"{strategy} object keys differ from B0")
        obj = obj[["stem", "object_id", "fold_id", "status", "prediction_confidence", "iou", "error_type"]].copy()
        obj.rename(columns={
            "fold_id": f"{strategy}_fold_id", "status": f"{strategy}_status",
            "prediction_confidence": f"{strategy}_confidence", "iou": f"{strategy}_iou",
            "error_type": f"{strategy}_error_type",
        }, inplace=True)
        obj[f"{strategy}_tp"] = (obj[f"{strategy}_status"] == "TP").astype(int)
        frames.append(obj)

    matrix = frames[0]
    for frame in frames[1:]:
        matrix = matrix.merge(frame, on=["stem", "object_id"], how="inner", validate="one_to_one")
    fold_cols = [f"{name}_fold_id" for name in RUNS]
    if not (matrix[fold_cols].nunique(axis=1) == 1).all() or set(matrix[fold_cols[0]]) != {1}:
        raise RuntimeError("Fold IDs are inconsistent")
    matrix.rename(columns={fold_cols[0]: "fold_id"}, inplace=True)
    matrix.drop(columns=fold_cols[1:], inplace=True)

    feats = pd.read_csv(TABLES / "01b_object_xray_features.csv")
    feats = feats[["stem", "object_id", *FEATURES]].copy()
    matrix = matrix.merge(feats, on=["stem", "object_id"], how="left", validate="one_to_one")
    if matrix[FEATURES].isna().all(axis=1).any():
        raise RuntimeError("At least one Fold1 object has no Stage1B features")

    tp_cols = [f"{name}_tp" for name in RUNS]
    alternatives = [name for name in RUNS if name != "b0"]
    matrix["all_success"] = matrix[tp_cols].eq(1).all(axis=1)
    matrix["all_fail"] = matrix[tp_cols].eq(0).all(axis=1)
    matrix["b0_advantage"] = (matrix.b0_tp == 1) & matrix[[f"{x}_tp" for x in alternatives]].eq(0).any(axis=1)
    matrix["alternative_rescue"] = (matrix.b0_tp == 0) & matrix[[f"{x}_tp" for x in alternatives]].eq(1).any(axis=1)
    for strategy in alternatives:
        matrix[f"{strategy}_rescue"] = (matrix.b0_tp == 0) & (matrix[f"{strategy}_tp"] == 1)
    matrix["rescue_strategy_count"] = matrix[[f"{x}_rescue" for x in alternatives]].sum(axis=1)
    matrix["small"] = matrix.bbox_min_side_px <= 8
    matrix["low_contrast"] = matrix.absolute_median_difference <= 4
    matrix["small_low_contrast"] = matrix.small & matrix.low_contrast
    rescue_counts = {name: int(matrix[f"{name}_rescue"].sum()) for name in alternatives}
    if rescue_counts["p192"] != 5 or rescue_counts["p256"] != 2:
        raise RuntimeError(f"Patch rescue mismatch: {rescue_counts}")
    p192_keys = set(zip(matrix.loc[matrix.p192_rescue, "stem"], matrix.loc[matrix.p192_rescue, "object_id"]))
    p256_keys = set(zip(matrix.loc[matrix.p256_rescue, "stem"], matrix.loc[matrix.p256_rescue, "object_id"]))
    if not p256_keys <= p192_keys or len(p192_keys | p256_keys) != 5:
        raise RuntimeError("P256 rescue set is not the expected subset of the five-object P192 rescue union")
    return matrix, metrics


def global_summary(matrix: pd.DataFrame, metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for strategy, m in metrics.items():
        rows.append({
            "row_type": "GLOBAL_METRIC", "group": strategy, "n": 290,
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "ap50": m["ap50"], "map50_95": m["map50_95"],
            "feature_medians_json": "", "object_ids": "",
        })
    groups = ["all_success", "all_fail", "b0_advantage", "alternative_rescue"]
    groups += [f"{x}_rescue" for x in RUNS if x != "b0"]
    for group in groups:
        sub = matrix[matrix[group]]
        med = {feature: (None if sub[feature].dropna().empty else float(sub[feature].median())) for feature in FEATURES}
        ids = ";".join(f"{r.stem}#{int(r.object_id)}" for r in sub.itertuples())
        rows.append({
            "row_type": "OBJECT_GROUP", "group": group, "n": len(sub), "tp": "", "fp": "", "fn": "",
            "precision": "", "recall": "", "f1": "", "ap50": "", "map50_95": "",
            "feature_medians_json": json.dumps(med, ensure_ascii=False), "object_ids": ids,
        })
    classifications = {
        "b0": ("global_primary", "Best overall stability; tied fixed-threshold counts only by M3 and retains higher AP metrics."),
        "p192": ("conditional_specialist", "Largest B0-FN rescue set (5/9) and two unique rescues, but 20 B0-TP losses and worse global metrics."),
        "p256": ("no_evidence_of_benefit", "Two rescues are a subset of P192, no unique rescue, and seven B0-TP losses."),
        "m2_p2": ("no_evidence_of_benefit", "Three shared rescues, no unique rescue, seven B0-TP losses, and worse small-subgroup/global metrics."),
        "m3_bilateral": ("no_evidence_of_benefit", "Ties B0 confusion counts but swaps two rescued for two lost objects, has no unique rescue, and lower AP metrics."),
        "m4_clahe": ("no_evidence_of_benefit", "Three shared rescues, no unique rescue, eight B0-TP losses, and worse global/low-contrast metrics."),
        "m5_bilateral_clahe": ("insufficient_specialist_evidence", "One unique rescue is anecdotal; localization failures decrease but low-confidence/no-detection errors increase and global recall falls."),
    }
    for strategy, (classification, basis) in classifications.items():
        rows.append({"row_type": "STRATEGY_CLASSIFICATION", "group": strategy, "n": 290,
                     "classification": classification, "basis": basis})
    return pd.DataFrame(rows)


def rescue_tables(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    alternatives = [name for name in RUNS if name != "b0"]
    sets = {name: set(zip(matrix.loc[matrix[f"{name}_rescue"], "stem"], matrix.loc[matrix[f"{name}_rescue"], "object_id"])) for name in alternatives}
    overlap = []
    for a, b in combinations(alternatives, 2):
        inter, union = sets[a] & sets[b], sets[a] | sets[b]
        overlap.append({"strategy_a": a, "strategy_b": b, "rescue_a": len(sets[a]), "rescue_b": len(sets[b]),
                        "overlap": len(inter), "union": len(union), "jaccard": len(inter) / len(union) if union else np.nan})
    unique_rows = []
    for strategy in alternatives:
        rescued = matrix[matrix[f"{strategy}_rescue"]]
        unique = rescued[rescued.rescue_strategy_count == 1]
        shared = rescued[rescued.rescue_strategy_count > 1]
        summary = {}
        for feature in FEATURES:
            vals = rescued[feature].dropna()
            summary[feature] = {"median": None if vals.empty else float(vals.median()),
                                "q25": None if vals.empty else float(vals.quantile(.25)),
                                "q75": None if vals.empty else float(vals.quantile(.75))}
        unique_rows.append({
            "strategy": strategy, "total_b0_fn_rescue": len(rescued), "unique_rescue_count": len(unique),
            "shared_rescue_count": len(shared),
            "rescued_objects": ";".join(f"{r.stem}#{int(r.object_id)}" for r in rescued.itertuples()),
            "unique_objects": ";".join(f"{r.stem}#{int(r.object_id)}" for r in unique.itertuples()),
            "rescued_feature_summary_json": json.dumps(summary, ensure_ascii=False),
        })
    return pd.DataFrame(overlap), pd.DataFrame(unique_rows)


def summarize_cell(sub: pd.DataFrame, prefix: dict) -> dict:
    result = dict(prefix)
    result["n"] = len(sub)
    for strategy in RUNS:
        tp = int(sub[f"{strategy}_tp"].sum())
        result[f"{strategy}_tp"] = tp
        result[f"{strategy}_rate"] = tp / len(sub) if len(sub) else np.nan
        if strategy != "b0":
            result[f"{strategy}_minus_b0"] = result[f"{strategy}_rate"] - result["b0_rate"]
            result[f"{strategy}_b0_fn_rescue"] = int(sub[f"{strategy}_rescue"].sum())
    rates = {s: result[f"{s}_rate"] for s in RUNS}
    best_value = max(rates.values()) if rates else np.nan
    result["best_strategy"] = ";".join(s for s, value in rates.items() if np.isclose(value, best_value))
    best_alternative_tp = max(result[f"{s}_tp"] for s in RUNS if s != "b0")
    alternative_gain_objects = best_alternative_tp - result["b0_tp"]
    result["best_alternative_minus_b0_objects"] = alternative_gain_objects
    if len(sub) < 5:
        result["evidence_note"] = "insufficient_evidence"
    elif alternative_gain_objects <= 0:
        result["evidence_note"] = "no_alternative_gain_over_b0"
    elif alternative_gain_objects <= 2:
        result["evidence_note"] = "anecdotal_1_to_2_object_difference"
    elif alternative_gain_objects < 5:
        result["evidence_note"] = "limited_3_to_4_object_difference"
    else:
        result["evidence_note"] = "exploratory_pattern"
    return result


def feature_bins(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predefined = {
        "small": pd.Series(np.where(matrix.small, "small_le_8", "not_small_gt_8"), index=matrix.index),
        "low_contrast": pd.Series(np.where(matrix.low_contrast, "low_le_4", "not_low_gt_4"), index=matrix.index),
        "small_low_contrast": pd.Series(np.where(matrix.small_low_contrast, "both", "not_both"), index=matrix.index),
    }
    for name, labels in predefined.items():
        for label in sorted(labels.unique()):
            rows.append(summarize_cell(matrix[labels == label], {"feature": name, "bin": label, "bin_rule": "predefined"}))
    continuous = ["bbox_min_side_px", "bbox_area_ratio", "absolute_median_difference", "cnr_like_robust",
                  "background_gradient_mean", "image_edge_distance_norm", "product_edge_distance_norm"]
    for feature in continuous:
        valid = matrix[feature].notna()
        labels = pd.qcut(matrix.loc[valid, feature], q=4, labels=False, duplicates="drop")
        for code in sorted(labels.unique()):
            idx = labels[labels == code].index
            sub = matrix.loc[idx]
            rule = f"[{sub[feature].min():.8g}, {sub[feature].max():.8g}]"
            rows.append(summarize_cell(sub, {"feature": feature, "bin": f"Q{int(code)+1}", "bin_rule": rule}))
    return pd.DataFrame(rows)


def two_dimensional(matrix: pd.DataFrame) -> pd.DataFrame:
    bg_median = float(matrix.background_gradient_mean.median())
    edge_median = float(matrix.product_edge_distance_norm.median())
    axes = {
        "size_x_contrast": (np.where(matrix.small, "small", "not_small"), np.where(matrix.low_contrast, "low_contrast", "not_low_contrast"), "predefined <=8 and <=4"),
        "size_x_background": (np.where(matrix.small, "small", "not_small"), np.where(matrix.background_gradient_mean <= bg_median, "low_bg_gradient", "high_bg_gradient"), f"background median={bg_median:.8g}"),
        "size_x_product_edge": (np.where(matrix.small, "small", "not_small"), np.where(matrix.product_edge_distance_norm <= edge_median, "near_proxy_edge", "far_proxy_edge"), f"inferred product-edge median={edge_median:.8g}"),
        "contrast_x_background": (np.where(matrix.low_contrast, "low_contrast", "not_low_contrast"), np.where(matrix.background_gradient_mean <= bg_median, "low_bg_gradient", "high_bg_gradient"), f"contrast <=4; background median={bg_median:.8g}"),
    }
    rows = []
    for analysis, (a, b, rule) in axes.items():
        labels = pd.Series(a, index=matrix.index).astype(str) + " | " + pd.Series(b, index=matrix.index).astype(str)
        for label in sorted(labels.unique()):
            left, right = label.split(" | ")
            rows.append(summarize_cell(matrix[labels == label], {"analysis": analysis, "condition_a": left, "condition_b": right, "bin_rule": rule}))
    return pd.DataFrame(rows)


def fmt_rate(n: int, d: int) -> str:
    return f"{n}/{d} ({n/d:.3f})" if d else "NA"


def write_report(matrix: pd.DataFrame, metrics: dict[str, dict], unique: pd.DataFrame, bins: pd.DataFrame, twod: pd.DataFrame) -> None:
    alternatives = [name for name in RUNS if name != "b0"]
    b0_fn = matrix[matrix.b0_tp == 0].copy()
    rescue_lines = []
    for row in b0_fn.itertuples():
        states = ", ".join(f"{s}={'TP' if getattr(row, s + '_tp') else 'FN'}" for s in alternatives)
        rescue_lines.append(f"- `{row.stem}#{int(row.object_id)}`: {states}; rescue strategies={int(row.rescue_strategy_count)}")
    all_fail = matrix[matrix.all_fail]
    feature_gains = bins[bins.best_alternative_minus_b0_objects > 0].sort_values("best_alternative_minus_b0_objects", ascending=False)
    two_d_gains = twod[twod.best_alternative_minus_b0_objects > 0].sort_values("best_alternative_minus_b0_objects", ascending=False)
    lines = [
        "# Stage 6 Fold1 object-level strategy winner analysis", "",
        "## Scope and interpretation", "",
        "This is a one-fold exploratory diagnostic over the same 290 Fold1 TXT-GT objects. GT-derived size, contrast, background, image-edge, and inferred product-edge features are diagnostic only and are not deployable routing inputs. Product-edge distance is an inferred-boundary proxy. No causal, physical-density, material, thickness, or final-generalization claim is made.", "",
        "## 1. Join integrity", "",
        "All seven object tables contained exactly 290 unique and identical `(stem, object_id)` keys, all assigned to Fold1. The Stage1B feature join retained all 290 objects. TP uses reporting confidence 0.25 and IoU >= 0.50.", "",
        "## 2. Global metrics", "",
        "| Strategy | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50-95 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s, m in metrics.items():
        lines.append(f"| {s} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['ap50']:.4f} | {m['map50_95']:.4f} |")
    lines += ["", "B0 remains the global primary reference. M3 ties B0 at the fixed-threshold confusion counts but has lower AP50 and mAP50-95; equality of aggregate counts does not mean identical object outcomes.", "", "## 3. B0 FN rescue matrix", "", *rescue_lines, "",
              "## 4. Rescue overlap and unique rescue", ""]
    for r in unique.itertuples():
        lines.append(f"- **{r.strategy}:** rescued {r.total_b0_fn_rescue}/9 B0 FN; unique {r.unique_rescue_count}; shared {r.shared_rescue_count}.")
    lines += ["", "## 5. Feature-bin winners", ""]
    if feature_gains.empty:
        lines.append("No alternative exceeded B0 in any feature bin.")
    else:
        for r in feature_gains.itertuples():
            lines.append(f"- `{r.feature}` {r.bin} (n={r.n}): best={r.best_strategy}; best alternative vs B0={int(r.best_alternative_minus_b0_objects):+d} object(s); evidence={r.evidence_note}.")
    lines += ["", "## 6. Two-dimensional condition winners", ""]
    if two_d_gains.empty:
        lines.append("No alternative exceeded B0 in any 2D cell.")
    else:
        for r in two_d_gains.itertuples():
            lines.append(f"- `{r.analysis}` [{r.condition_a}, {r.condition_b}] (n={r.n}): best={r.best_strategy}; best alternative vs B0={int(r.best_alternative_minus_b0_objects):+d} object(s); evidence={r.evidence_note}.")
    lines += ["", "## 7. All-fail objects", "", f"All seven strategies failed on {len(all_fail)} objects."]
    for r in all_fail.itertuples():
        lines.append(f"- `{r.stem}#{int(r.object_id)}`: min-side={r.bbox_min_side_px:.3g}px, abs contrast={r.absolute_median_difference:.3g}, robust CNR-like={r.cnr_like_robust:.3g}, inferred product-edge norm={r.product_edge_distance_norm:.3g}.")
    lines += ["", "## 8. Conditional-specialist assessment", "",
              "- **B0 — global primary:** best overall stability; M3 ties its fixed-threshold counts but not its AP metrics.",
              "- **P192 — exploratory conditional specialist:** rescues 5/9 B0 FN and uniquely rescues two, but loses 20 B0 TP and is not a replacement detector.",
              "- **P256, M2, M3, M4 — no evidence of net or unique conditional benefit:** their rescues are shared and offset by losses; M3's equal global counts arise from different objects.",
              "- **M5 — insufficient specialist evidence:** one unique rescue is anecdotal. Its three localization failures are fewer than B0's four, but this is accompanied by ten low-confidence and two no-detection FN, so localization reduction is not a repeated condition-specific gain.",
              "No strategy is promoted from one-fold subgroup wins alone; small cells remain anecdotal.", "",
              "## 9. Size-only routing", "",
              "Size alone is not a deployable strategy selector here: it is GT-derived in this audit, the alternatives both rescue and lose objects within size strata, and one-fold counts are sparse. Any later routing design would require prediction-time observable signals and independent validation.", "",
              "## 10. Why this is not a deployable router", "",
              "The comparison uses GT boxes and Stage1B GT-local features, examines only Fold1, and does not account for routing errors, fusion calibration, latency, or external/negative-image behavior. It identifies hypotheses, not an operational decision rule.", "",
              "## 11. Next experiment", "",
              "Run M6 (mild CLAHE followed by bilateral filtering) on the same B0-matched Fold1 protocol. This completes the pre-registered order ablation against M5 without introducing a new factor; only afterward should any specialist claim or 4-fold expansion be considered.", ""]
    (EDA / "06_strategy_winner_analysis_fold1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    matrix, metrics = load_and_verify()
    overlap, unique = rescue_tables(matrix)
    bins = feature_bins(matrix)
    twod = two_dimensional(matrix)
    groups = global_summary(matrix, metrics)
    TABLES.mkdir(parents=True, exist_ok=True)
    EDA.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(TABLES / "06_strategy_object_matrix_fold1.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(TABLES / "06_strategy_rescue_overlap_fold1.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(TABLES / "06_strategy_feature_bins_fold1.csv", index=False, encoding="utf-8-sig")
    twod.to_csv(TABLES / "06_strategy_2d_conditions_fold1.csv", index=False, encoding="utf-8-sig")
    unique.to_csv(TABLES / "06_strategy_unique_rescues_fold1.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(TABLES / "06_strategy_group_summary_fold1.csv", index=False, encoding="utf-8-sig")
    write_report(matrix, metrics, unique, bins, twod)


if __name__ == "__main__":
    main()

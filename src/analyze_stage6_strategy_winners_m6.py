from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analyze_stage6_strategy_winners as base  # noqa: E402


M6 = "m6_clahe_bilateral"
base.RUNS[M6] = ROOT / "outputs/stage6_runs/stage6_m6_clahe_bilateral_yolov8n_640_fold1_b0matched_v1"
base.EXPECTED[M6] = (277, 21, 13)
base.EXPECTED_STAGE6_DETAIL[M6] = {
    "localization_failure_count": 2,
    "low_confidence_fn_count": 8,
    "no_detection_fn_count": 3,
    "small_tp": 73,
    "low_contrast_tp": 76,
    "small_low_contrast_tp": 25,
}


CLASSIFICATIONS = {
    "b0": ("A_GLOBAL_PRIMARY", "Best overall stability and the reference against which all alternatives are evaluated."),
    "p192": ("B_EXPLORATORY_CONDITIONAL_SPECIALIST", "Largest B0-FN rescue set (5/9) and one remaining unique rescue after M6, but 20 B0-TP losses."),
    "p256": ("C_NO_EVIDENCE_OF_BENEFIT", "Two shared rescues, no unique rescue, and seven B0-TP losses."),
    "m2_p2": ("C_NO_EVIDENCE_OF_BENEFIT", "Three shared rescues, no unique rescue, and seven B0-TP losses."),
    "m3_bilateral": ("C_NO_EVIDENCE_OF_BENEFIT", "Aggregate counts tie B0 but two rescues are offset by two losses; no unique rescue and lower AP."),
    "m4_clahe": ("C_NO_EVIDENCE_OF_BENEFIT", "Three shared rescues, no unique rescue, and eight B0-TP losses."),
    "m5_bilateral_clahe": ("C_NO_EVIDENCE_OF_BENEFIT", "One unique rescue remains anecdotal; global recall is lower and eight B0-TP objects are lost."),
    M6: ("C_NO_EVIDENCE_OF_BENEFIT", "Two shared rescues, no unique rescue, six B0-TP losses, and lower global F1; localization/small gains are not repeated specialist evidence."),
}


def group_summary(matrix: pd.DataFrame, metrics: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy, m in metrics.items():
        rows.append({"row_type": "GLOBAL_METRIC", "group": strategy, "n": 290, "tp": m["tp"], "fp": m["fp"],
                     "fn": m["fn"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "ap50": m["ap50"], "map50_95": m["map50_95"]})
    groups = ["all_success", "all_fail", "b0_advantage", "alternative_rescue"] + [f"{x}_rescue" for x in base.RUNS if x != "b0"]
    for group in groups:
        sub = matrix[matrix[group]]
        medians = {feature: (None if sub[feature].dropna().empty else float(sub[feature].median())) for feature in base.FEATURES}
        rows.append({"row_type": "OBJECT_GROUP", "group": group, "n": len(sub),
                     "feature_medians_json": json.dumps(medians, ensure_ascii=False),
                     "object_ids": ";".join(f"{r.stem}#{int(r.object_id)}" for r in sub.itertuples())})
    for strategy, (classification, reason) in CLASSIFICATIONS.items():
        rows.append({"row_type": "STRATEGY_CLASSIFICATION", "group": strategy, "n": 290,
                     "classification": classification, "basis": reason})
    return pd.DataFrame(rows)


def preprocessing_table(matrix: pd.DataFrame, metrics: dict[str, dict], unique: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "b0": "Conservative reference", "m3_bilateral": "Bilateral",
        "m4_clahe": "CLAHE", "m5_bilateral_clahe": "Bilateral->CLAHE", M6: "CLAHE->Bilateral",
    }
    unique_by = unique.set_index("strategy")
    rows = []
    for strategy, label in labels.items():
        m = metrics[strategy]
        rows.append({
            "strategy": strategy, "preprocessing": label, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "ap50": m["ap50"],
            "map50_95": m["map50_95"], "localization_failure": m.get("localization_failure_count", int((matrix[f"{strategy}_error_type"] == "LOCALIZATION_FAILURE").sum())),
            "low_confidence_fn": m.get("low_confidence_fn_count", int((matrix[f"{strategy}_error_type"] == "LOW_CONFIDENCE").sum())),
            "no_detection_fn": m.get("no_detection_fn_count", int((matrix[f"{strategy}_error_type"] == "NO_DETECTION").sum())),
            "small_recall": int(matrix.loc[matrix.small, f"{strategy}_tp"].sum()) / int(matrix.small.sum()),
            "low_contrast_recall": int(matrix.loc[matrix.low_contrast, f"{strategy}_tp"].sum()) / int(matrix.low_contrast.sum()),
            "small_low_contrast_recall": int(matrix.loc[matrix.small_low_contrast, f"{strategy}_tp"].sum()) / int(matrix.small_low_contrast.sum()),
            "b0_fn_rescue_count": 0 if strategy == "b0" else int(matrix[f"{strategy}_rescue"].sum()),
            "unique_rescue_count": 0 if strategy == "b0" else int(unique_by.loc[strategy, "unique_rescue_count"]),
            "b0_tp_loss_count": 0 if strategy == "b0" else int(((matrix.b0_tp == 1) & (matrix[f"{strategy}_tp"] == 0)).sum()),
        })
    return pd.DataFrame(rows)


def report(matrix: pd.DataFrame, metrics: dict[str, dict], unique: pd.DataFrame,
           bins: pd.DataFrame, twod: pd.DataFrame, preprocessing: pd.DataFrame) -> None:
    b0_fn = matrix[matrix.b0_tp == 0]
    all_fail = matrix[matrix.all_fail]
    m6_rescued = matrix[matrix.m6_clahe_bilateral_rescue]
    m5_m6_diff = matrix[matrix.m5_bilateral_clahe_tp != matrix.m6_clahe_bilateral_tp]
    m6_bin_wins = bins[bins.best_strategy.fillna("").str.split(";").apply(lambda xs: M6 in xs)]
    m6_2d_wins = twod[twod.best_strategy.fillna("").str.split(";").apply(lambda xs: M6 in xs)]
    lines = [
        "# Stage 6 Fold1 object-level strategy winner analysis — M6 update", "",
        "## Scope", "",
        "One-fold exploratory diagnostic over the same 290 Fold1 TXT-GT objects. GT-derived features are diagnostic only and are not deployable routing inputs. Product-edge distance remains an inferred-boundary proxy. No causal or physical material/density/thickness interpretation is made.", "",
        "## 1. Join integrity", "",
        "All eight strategies contain exactly 290 identical unique `(stem, object_id)` keys; duplicate and missing counts are zero. TP uses confidence 0.25 and IoU >= 0.50.", "",
        "## 2. Global metrics", "",
        "| Strategy | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50-95 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, m in metrics.items():
        lines.append(f"| {strategy} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['ap50']:.4f} | {m['map50_95']:.4f} |")
    lines += ["", "## 3. B0 FN rescue matrix", ""]
    alternatives = [s for s in base.RUNS if s != "b0"]
    for row in b0_fn.itertuples():
        status = ", ".join(f"{s}={'TP' if getattr(row, s + '_tp') else 'FN'}" for s in alternatives)
        lines.append(f"- `{row.stem}#{int(row.object_id)}`: {status}; rescue strategies={int(row.rescue_strategy_count)}")
    m6_unique = int(unique.set_index("strategy").loc[M6, "unique_rescue_count"])
    lines += ["", "## 4. M6 rescue and role", "",
              f"M6 rescues {len(m6_rescued)}/9 B0 FN: " + "; ".join(f"`{r.stem}#{int(r.object_id)}`" for r in m6_rescued.itertuples()) + ".",
              f"Unique rescue={m6_unique}; shared rescue={len(m6_rescued)-m6_unique}. Both M6 rescues overlap P192. The seven-alternative rescue union remains {int(matrix.alternative_rescue.sum())}/9 and all-fail remains {len(all_fail)}.",
              "One M6 rescue is small (8 px) and one is not (10 px), so rescue is not concentrated exclusively in small objects.",
              "M6 changes B0's four localization failures to two, but only one original localization failure is rescued; two remain localization failures and one becomes no-detection. This is a mixed error-type redistribution, not repeated localization-specialist evidence.", "",
              "## 5. Preprocessing order ablation", "",
              "| Strategy | TP/FP/FN | F1 | Loc/LowConf/NoDet | Small | Low contrast | Small+low | Rescue/Unique/Loss |", "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in preprocessing[preprocessing.strategy != "b0"].itertuples():
        lines.append(f"| {row.strategy} | {row.tp}/{row.fp}/{row.fn} | {row.f1:.4f} | {row.localization_failure}/{row.low_confidence_fn}/{row.no_detection_fn} | {row.small_recall:.4f} | {row.low_contrast_recall:.4f} | {row.small_low_contrast_recall:.4f} | {row.b0_fn_rescue_count}/{row.unique_rescue_count}/{row.b0_tp_loss_count} |")
    lines += ["", f"M5 and M6 differ on {len(m5_m6_diff)} objects: M6 gains 10 M5-FN objects and loses 8 M5-TP objects. Their B0-FN rescue sets are disjoint. Reversing the order raises TP by 2 and small recall by one object and reduces localization failures 3→2, but increases FP 14→21, FN confidence/no-detection burden, and lowers F1 .9499→.9422.", "",
              "## 6. M6 feature-bin results", ""]
    if m6_bin_wins.empty:
        lines.append("M6 is not a best strategy in any feature bin.")
    else:
        for r in m6_bin_wins.itertuples():
            delta = getattr(r, "m6_clahe_bilateral_minus_b0")
            rescue = getattr(r, "m6_clahe_bilateral_b0_fn_rescue")
            lines.append(f"- `{r.feature}` {r.bin} (n={r.n}): best={r.best_strategy}; M6-B0={delta:+.4f}; B0-FN rescue={rescue}; {r.evidence_note}.")
    lines += ["", "M6 small recall is 73/77 versus B0 72/77 because it rescues one small B0 FN and loses no small B0 TP. It has no low-contrast or small+low rescue; low-contrast recall is one object below B0.", "",
              "## 7. M6 two-dimensional condition results", ""]
    if m6_2d_wins.empty:
        lines.append("M6 is not a best strategy in any 2D condition cell.")
    else:
        for r in m6_2d_wins.itertuples():
            delta = getattr(r, "m6_clahe_bilateral_minus_b0")
            rescue = getattr(r, "m6_clahe_bilateral_b0_fn_rescue")
            lines.append(f"- `{r.analysis}` [{r.condition_a}, {r.condition_b}] (n={r.n}): best={r.best_strategy}; M6-B0={delta:+.4f}; rescue={rescue}; {r.evidence_note}.")
    lines += ["", "## 8. All-fail objects", "", f"All eight strategies fail on {len(all_fail)} objects; M6 rescues none of the prior three all-fail cases."]
    for r in all_fail.itertuples():
        lines.append(f"- `{r.stem}#{int(r.object_id)}`: min-side={r.bbox_min_side_px:.3g}px, abs contrast={r.absolute_median_difference:.3g}, background gradient={r.background_gradient_mean:.3g}, inferred product-edge norm={r.product_edge_distance_norm:.3g}.")
    lines += ["", "## 9. Updated strategy classification", ""]
    for strategy, (classification, reason) in CLASSIFICATIONS.items():
        lines.append(f"- **{strategy}: {classification}** — {reason}")
    lines += ["", "M6 shows an anecdotal localization/small-object signal but not a detection specialist pattern and no deployable expert evidence. No GT-derived condition here may be used directly as a router.", "",
              "## 10. Conclusion and next step", "",
              "B0 remains the global primary. P192 remains the only exploratory conditional specialist, with substantial replacement and compute costs. M6 does not expand the rescue union or reduce all-fail cases and is not promoted.",
              "No further Stage6 training is justified by the completed M1–M6 Fold1 screen. The next single step is to freeze the Stage6 exploratory conclusion and update the competition evidence/report table with B0 as primary and the documented limitations.", ""]
    (ROOT / "outputs/eda/06_strategy_winner_analysis_fold1_m6.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    matrix, metrics = base.load_and_verify()
    if len(matrix) != 290 or matrix[["stem", "object_id"]].duplicated().any() or matrix[f"{M6}_tp"].isna().any():
        raise RuntimeError("M6 join integrity failed")
    overlap, unique = base.rescue_tables(matrix)
    bins = base.feature_bins(matrix)
    twod = base.two_dimensional(matrix)
    groups = group_summary(matrix, metrics)
    preprocessing = preprocessing_table(matrix, metrics, unique)
    out = ROOT / "outputs/tables"
    matrix.to_csv(out / "06_strategy_object_matrix_fold1_m6.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(out / "06_strategy_rescue_overlap_fold1_m6.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(out / "06_strategy_feature_bins_fold1_m6.csv", index=False, encoding="utf-8-sig")
    twod.to_csv(out / "06_strategy_2d_conditions_fold1_m6.csv", index=False, encoding="utf-8-sig")
    unique.to_csv(out / "06_strategy_unique_rescues_fold1_m6.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(out / "06_strategy_group_summary_fold1_m6.csv", index=False, encoding="utf-8-sig")
    preprocessing.to_csv(out / "06_preprocessing_order_ablation_fold1.csv", index=False, encoding="utf-8-sig")
    report(matrix, metrics, unique, bins, twod, preprocessing)


if __name__ == "__main__":
    main()

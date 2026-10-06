"""Compare pre-model factors against B0 OOF detection failures.

This is a statistical error analysis. It does not train or run a detector and
does not modify the frozen validation assignment.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = {
    "bbox_min_side_px": "SIZE",
    "bbox_area_ratio": "SIZE",
    "bbox_center_x_norm": "POSITION",
    "bbox_center_y_norm": "POSITION",
    "image_edge_distance_norm": "POSITION",
    "bbox_aspect_ratio": "SHAPE",
    "signed_median_difference": "VISIBILITY_CONTRAST",
    "absolute_median_difference": "VISIBILITY_CONTRAST",
    "cnr_like_robust": "VISIBILITY_CONTRAST",
    "object_iqr": "BACKGROUND_HETEROGENEITY",
    "background_gradient_mean": "BACKGROUND_HETEROGENEITY",
    "background_iqr": "BACKGROUND_HETEROGENEITY",
    "product_edge_distance_norm": "PRODUCT_EDGE_PROXY",
}

MULTIVARIABLE_FEATURES = [
    "bbox_min_side_px",
    "image_edge_distance_norm",
    "bbox_aspect_ratio",
    "absolute_median_difference",
    "background_gradient_mean",
    "product_edge_distance_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def quartile_codes(values: pd.Series) -> tuple[pd.Series, list[float]]:
    clean = pd.to_numeric(values, errors="coerce")
    cuts = sorted(set(float(x) for x in clean.quantile([0.25, 0.5, 0.75]).dropna()))
    edges = [-np.inf, *cuts, np.inf]
    labels = [f"Q{i + 1}" for i in range(len(edges) - 1)]
    return pd.cut(clean, bins=edges, labels=labels, include_lowest=True), edges


def interval_text(edges: list[float], index: int) -> str:
    low, high = edges[index], edges[index + 1]
    low_text = "-inf" if np.isneginf(low) else f"{low:.8g}"
    high_text = "inf" if np.isposinf(high) else f"{high:.8g}"
    return f"({low_text},{high_text}]"


def corrected_rr(events_a: float, n_a: float, events_b: float, n_b: float) -> float:
    return ((events_a + 0.5) / (n_a + 1.0)) / ((events_b + 0.5) / (n_b + 1.0))


def corrected_or(events_a: float, n_a: float, events_b: float, n_b: float) -> float:
    odds_a = (events_a + 0.5) / (n_a - events_a + 0.5)
    odds_b = (events_b + 0.5) / (n_b - events_b + 0.5)
    return odds_a / odds_b


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return math.nan, math.nan
    return tuple(float(x) for x in np.quantile(finite, [0.025, 0.975]))


def component_bootstrap_effect(
    data: pd.DataFrame,
    bin_column: str,
    high_bin: str,
    low_bin: str,
    outcome: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    subset = data[data[bin_column].astype(str).isin([high_bin, low_bin])].copy()
    grouped = (
        subset.groupby(["final_group_component_id", bin_column], observed=True)[outcome]
        .agg(["size", "sum"])
        .reset_index()
    )
    components = sorted(subset["final_group_component_id"].unique())
    lookup = {
        (r["final_group_component_id"], str(r[bin_column])): (r["size"], r["sum"])
        for _, r in grouped.iterrows()
    }
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in [high_bin, low_bin]:
        n = np.array([lookup.get((g, label), (0, 0))[0] for g in components], dtype=float)
        e = np.array([lookup.get((g, label), (0, 0))[1] for g in components], dtype=float)
        arrays[label] = (n, e)
    draws = rng.integers(0, len(components), size=(replicates, len(components)))
    n_h = arrays[high_bin][0][draws].sum(axis=1)
    e_h = arrays[high_bin][1][draws].sum(axis=1)
    n_l = arrays[low_bin][0][draws].sum(axis=1)
    e_l = arrays[low_bin][1][draws].sum(axis=1)
    valid = (n_h > 0) & (n_l > 0)
    gap = np.full(replicates, np.nan)
    rr = np.full(replicates, np.nan)
    gap[valid] = e_h[valid] / n_h[valid] - e_l[valid] / n_l[valid]
    rr[valid] = ((e_h[valid] + 0.5) / (n_h[valid] + 1.0)) / (
        (e_l[valid] + 0.5) / (n_l[valid] + 1.0)
    )
    gap_low, gap_high = percentile_ci(gap)
    rr_low, rr_high = percentile_ci(rr)
    return {
        "gap_ci_low": gap_low,
        "gap_ci_high": gap_high,
        "risk_ratio_ci_low": rr_low,
        "risk_ratio_ci_high": rr_high,
        "valid_bootstrap_replicates": int(valid.sum()),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_l2_logistic(x: np.ndarray, y: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=float)
    penalty_matrix = np.diag([0.0] + [penalty] * x.shape[1])
    for _ in range(100):
        p = sigmoid(design @ beta)
        gradient = design.T @ (p - y) + penalty_matrix @ beta
        weights = np.clip(p * (1.0 - p), 1e-7, None)
        hessian = design.T @ (design * weights[:, None]) + penalty_matrix
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def multivariable_analysis(
    data: pd.DataFrame,
    outcome: str,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    columns = MULTIVARIABLE_FEATURES
    complete = data[[*columns, outcome, "final_group_component_id"]].dropna().copy()
    x_raw = complete[columns].to_numpy(dtype=float)
    y = complete[outcome].to_numpy(dtype=float)
    x, _, _ = standardize(x_raw)
    beta = fit_l2_logistic(x, y)

    component_indices = {
        group: np.flatnonzero(complete["final_group_component_id"].to_numpy() == group)
        for group in sorted(complete["final_group_component_id"].unique())
    }
    groups = list(component_indices)
    boot = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([component_indices[groups[i]] for i in sampled])
        y_b = y[indices]
        if np.unique(y_b).size < 2:
            continue
        x_b, _, _ = standardize(x_raw[indices])
        boot.append(fit_l2_logistic(x_b, y_b)[1:])
    boot_arr = np.asarray(boot, dtype=float)

    rows = []
    for j, feature in enumerate(columns):
        ci_low, ci_high = percentile_ci(boot_arr[:, j])
        coefficient = float(beta[j + 1])
        sign_stability = float(np.mean(np.sign(boot_arr[:, j]) == np.sign(coefficient)))
        rows.append(
            {
                "outcome": outcome,
                "feature": feature,
                "factor": FEATURES[feature],
                "n_objects": len(complete),
                "event_count": int(y.sum()),
                "standardized_l2_coefficient": coefficient,
                "odds_ratio_per_sd": float(np.exp(np.clip(coefficient, -20, 20))),
                "coefficient_ci_low": ci_low,
                "coefficient_ci_high": ci_high,
                "bootstrap_sign_stability": sign_stability,
                "bootstrap_replicates": len(boot_arr),
                "regularization": "L2 penalty=1; coefficient is association per within-sample SD",
                "notes": "Exploratory multivariable association, not causal. Acquisition categories omitted because events are sparse and machine/date/resolution are confounded.",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = root / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(root / "outputs/tables/02a_conservative_640_oof_v1/02a_object_predictions.csv")
    geometry = pd.read_csv(root / "outputs/tables/01a_object_geometry.csv").rename(columns={"image_id": "stem"})
    xray = pd.read_csv(root / "outputs/tables/01b_object_xray_features.csv")
    folds = pd.read_csv(root / "outputs/tables/01c_final_validation_folds.csv")[
        ["stem", "final_group_component_id", "machine", "date", "resolution"]
    ]

    geometry_columns = [
        "stem", "object_id", "bbox_min_side_px", "bbox_area_ratio", "bbox_center_x_norm",
        "bbox_center_y_norm", "image_edge_distance_norm", "bbox_aspect_ratio",
    ]
    xray_columns = [
        "stem", "object_id", "signed_median_difference", "absolute_median_difference",
        "cnr_like_robust", "object_iqr", "background_gradient_mean", "background_iqr",
        "product_edge_distance_norm",
    ]
    data = predictions.merge(geometry[geometry_columns], on=["stem", "object_id"], validate="one_to_one")
    data = data.merge(xray[xray_columns], on=["stem", "object_id"], validate="one_to_one")
    data = data.merge(folds, on="stem", validate="many_to_one", suffixes=("", "_fold"))
    if len(data) != 1147 or data[["stem", "object_id"]].duplicated().any():
        raise RuntimeError("B0 object/feature merge integrity failed")
    if set(data["status"]) != {"TP", "FN"}:
        raise RuntimeError("Unexpected B0 object outcomes")
    data["miss"] = (data["status"] == "FN").astype(int)
    data["localization_failure"] = (data["error_type"] == "LOCALIZATION_FAILURE").astype(int)

    rng = np.random.default_rng(args.seed)
    univariate_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    for feature, factor in FEATURES.items():
        bin_column = f"__bin_{feature}"
        bins, edges = quartile_codes(data[feature])
        data[bin_column] = bins
        feature_rows = []
        for i, label in enumerate(bins.cat.categories):
            part = data[data[bin_column] == label]
            tp = int((part["status"] == "TP").sum())
            fn = int((part["status"] == "FN").sum())
            loc = int(part["localization_failure"].sum())
            row = {
                "factor": factor,
                "feature": feature,
                "quartile_bin": str(label),
                "group_or_range": interval_text(edges, i),
                "feature_value_min": part[feature].min(),
                "feature_value_median": part[feature].median(),
                "feature_value_max": part[feature].max(),
                "object_count": len(part),
                "tp": tp,
                "fn": fn,
                "recall": tp / len(part),
                "fn_rate": fn / len(part),
                "localization_failure_count": loc,
                "localization_failure_rate": loc / len(part),
                "prediction_confidence_n": int(part["prediction_confidence"].notna().sum()),
                "prediction_confidence_median": part["prediction_confidence"].median(),
                "matched_iou_n": int(part["iou"].notna().sum()),
                "matched_iou_median": part["iou"].median(),
                "fold_coverage": part["fold_id"].nunique(),
                "notes": "Exploratory full-corpus quartile; not a permanent threshold.",
            }
            univariate_rows.append(row)
            feature_rows.append(row)

        high_miss = sorted(feature_rows, key=lambda r: (-r["fn_rate"], -r["object_count"], r["quartile_bin"]))[0]
        low_miss = sorted(feature_rows, key=lambda r: (r["fn_rate"], -r["object_count"], r["quartile_bin"]))[0]
        high_loc = sorted(feature_rows, key=lambda r: (-r["localization_failure_rate"], -r["object_count"], r["quartile_bin"]))[0]
        low_loc = sorted(feature_rows, key=lambda r: (r["localization_failure_rate"], -r["object_count"], r["quartile_bin"]))[0]

        miss_boot = component_bootstrap_effect(
            data, bin_column, high_miss["quartile_bin"], low_miss["quartile_bin"],
            "miss", args.bootstrap_replicates, rng,
        )
        loc_boot = component_bootstrap_effect(
            data, bin_column, high_loc["quartile_bin"], low_loc["quartile_bin"],
            "localization_failure", args.bootstrap_replicates, rng,
        )
        valid_conf = data[[feature, "prediction_confidence"]].dropna()
        valid_iou = data[[feature, "iou"]].dropna()
        effect_rows.append(
            {
                "factor": factor,
                "feature": feature,
                "miss_high_risk_bin": high_miss["quartile_bin"],
                "miss_high_risk_range": high_miss["group_or_range"],
                "miss_high_risk_n": high_miss["object_count"],
                "miss_high_risk_fn": high_miss["fn"],
                "miss_low_risk_bin": low_miss["quartile_bin"],
                "miss_low_risk_range": low_miss["group_or_range"],
                "miss_low_risk_n": low_miss["object_count"],
                "miss_low_risk_fn": low_miss["fn"],
                "recall_gap_low_minus_high": low_miss["recall"] - high_miss["recall"],
                "fn_rate_gap_high_minus_low": high_miss["fn_rate"] - low_miss["fn_rate"],
                "fn_rate_gap_ci_low": miss_boot["gap_ci_low"],
                "fn_rate_gap_ci_high": miss_boot["gap_ci_high"],
                "miss_risk_ratio_corrected": corrected_rr(
                    high_miss["fn"], high_miss["object_count"], low_miss["fn"], low_miss["object_count"]
                ),
                "miss_risk_ratio_ci_low": miss_boot["risk_ratio_ci_low"],
                "miss_risk_ratio_ci_high": miss_boot["risk_ratio_ci_high"],
                "miss_odds_ratio_corrected": corrected_or(
                    high_miss["fn"], high_miss["object_count"], low_miss["fn"], low_miss["object_count"]
                ),
                "localization_high_risk_bin": high_loc["quartile_bin"],
                "localization_high_risk_range": high_loc["group_or_range"],
                "localization_high_risk_n": high_loc["object_count"],
                "localization_high_risk_count": high_loc["localization_failure_count"],
                "localization_low_risk_bin": low_loc["quartile_bin"],
                "localization_low_risk_range": low_loc["group_or_range"],
                "localization_low_risk_n": low_loc["object_count"],
                "localization_low_risk_count": low_loc["localization_failure_count"],
                "localization_rate_gap_high_minus_low": high_loc["localization_failure_rate"] - low_loc["localization_failure_rate"],
                "localization_rate_gap_ci_low": loc_boot["gap_ci_low"],
                "localization_rate_gap_ci_high": loc_boot["gap_ci_high"],
                "localization_risk_ratio_corrected": corrected_rr(
                    high_loc["localization_failure_count"], high_loc["object_count"],
                    low_loc["localization_failure_count"], low_loc["object_count"],
                ),
                "localization_risk_ratio_ci_low": loc_boot["risk_ratio_ci_low"],
                "localization_risk_ratio_ci_high": loc_boot["risk_ratio_ci_high"],
                "spearman_with_prediction_confidence": valid_conf[feature].rank().corr(valid_conf["prediction_confidence"].rank()),
                "spearman_with_matched_iou": valid_iou[feature].rank().corr(valid_iou["iou"].rank()),
                "bootstrap_unit": "frozen 10-second/similarity final_group_component_id",
                "bootstrap_replicates": args.bootstrap_replicates,
                "notes": "High/low bins are observed quartiles with max/min event rate. CIs are conditional on those selected bins; corrected ratios use 0.5 continuity correction.",
            }
        )

    univariate = pd.DataFrame(univariate_rows)
    effects = pd.DataFrame(effect_rows)
    multi_rows = []
    multi_rows.extend(multivariable_analysis(data, "miss", args.bootstrap_replicates, rng))
    multi_rows.extend(multivariable_analysis(data, "localization_failure", args.bootstrap_replicates, rng))
    multivariable = pd.DataFrame(multi_rows)

    univariate.to_csv(output_dir / "04_factor_univariate_comparison.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(output_dir / "04_factor_effect_size.csv", index=False, encoding="utf-8-sig")
    multivariable.to_csv(output_dir / "04_factor_multivariable.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

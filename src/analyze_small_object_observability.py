"""Stage 5 B0 original-pixel observability analysis.

Uses existing OOF outcomes and Stage 1 object features only. No detector is run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "bbox_min_side_px", "bbox_width_px", "bbox_height_px", "bbox_area_px",
    "bbox_area_ratio", "absolute_median_difference", "signed_median_difference",
    "cnr_like_robust", "image_edge_distance_px", "product_edge_distance_px",
]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def outcome_group(row: pd.Series) -> str:
    if row["status"] == "TP":
        return "TP"
    mapping = {
        "LOW_CONFIDENCE": "LOW_CONFIDENCE_FN",
        "LOCALIZATION_FAILURE": "LOCALIZATION_FAILURE",
        "NO_DETECTION": "NO_DETECTION_FN",
    }
    return mapping[str(row["error_type"])]


def component_bootstrap_median_difference(
    data: pd.DataFrame,
    feature: str,
    group: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    components = sorted(data["final_group_component_id"].unique())
    indices = {
        component: np.flatnonzero(data["final_group_component_id"].to_numpy() == component)
        for component in components
    }
    values = data[feature].to_numpy(float)
    labels = data["outcome_group"].to_numpy(str)
    boot: list[float] = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(components), size=len(components))
        selected = np.concatenate([indices[components[i]] for i in sampled])
        a = values[selected][labels[selected] == group]
        b = values[selected][labels[selected] == "TP"]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) and len(b):
            boot.append(float(np.median(a) - np.median(b)))
    if not boot:
        return np.nan, np.nan, 0
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), len(boot)


def main() -> None:
    config = args()
    root = config.project_root.resolve()
    pred = pd.read_csv(root / "outputs/tables/02a_conservative_640_oof_v1/02a_object_predictions.csv")
    geo = pd.read_csv(root / "outputs/tables/01a_object_geometry.csv").rename(columns={"image_id": "stem"})
    xray = pd.read_csv(root / "outputs/tables/01b_object_xray_features.csv")
    folds = pd.read_csv(root / "outputs/tables/01c_final_validation_folds.csv")[
        ["stem", "final_group_component_id"]
    ]
    data = pred.merge(
        geo[["stem", "object_id", "bbox_width_px", "bbox_height_px"]],
        on=["stem", "object_id"], validate="one_to_one",
    ).merge(
        xray[["stem", "object_id", *[f for f in FEATURES if f not in {"bbox_width_px", "bbox_height_px"}]]],
        on=["stem", "object_id"], validate="one_to_one",
    ).merge(folds, on="stem", validate="many_to_one")
    if len(data) != 1147:
        raise RuntimeError("Expected 1,147 B0 OOF objects")
    data["outcome_group"] = data.apply(outcome_group, axis=1)
    data["is_small"] = data["bbox_min_side_px"] <= 8.0
    data["is_low_contrast"] = data["absolute_median_difference"] <= 4.0
    data["is_small_low_contrast"] = data["is_small"] & data["is_low_contrast"]

    rng = np.random.default_rng(config.seed)
    groups = ["TP", "FN_ALL", "LOW_CONFIDENCE_FN", "LOCALIZATION_FAILURE", "NO_DETECTION_FN"]
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        tp_median = float(data.loc[data["outcome_group"] == "TP", feature].median())
        for group in groups:
            part = data[data["status"] == "FN"] if group == "FN_ALL" else data[data["outcome_group"] == group]
            values = pd.to_numeric(part[feature], errors="coerce").dropna()
            if group == "TP":
                ci_low = ci_high = 0.0
                valid_boot = config.bootstrap_replicates
            else:
                bootstrap_group = group
                if group == "FN_ALL":
                    temp = data.copy()
                    temp.loc[temp["status"] == "FN", "outcome_group"] = "FN_ALL"
                    ci_low, ci_high, valid_boot = component_bootstrap_median_difference(
                        temp, feature, bootstrap_group, config.bootstrap_replicates, rng
                    )
                else:
                    ci_low, ci_high, valid_boot = component_bootstrap_median_difference(
                        data, feature, bootstrap_group, config.bootstrap_replicates, rng
                    )
            rows.append({
                "outcome_group": group,
                "feature": feature,
                "object_count": len(part),
                "valid_count": len(values),
                "q05": values.quantile(0.05),
                "q25": values.quantile(0.25),
                "median": values.median(),
                "q75": values.quantile(0.75),
                "q95": values.quantile(0.95),
                "iqr": values.quantile(0.75) - values.quantile(0.25),
                "median_difference_vs_tp": values.median() - tp_median,
                "median_difference_ci_low": ci_low,
                "median_difference_ci_high": ci_high,
                "bootstrap_unit": "frozen final_group_component_id",
                "bootstrap_valid_replicates": valid_boot,
                "notes": "Image-derived/annotation-derived feature comparison; no physical density interpretation.",
            })

    # Retain the pre-specified Stage 1B exploratory interaction thresholds.
    for label, mask in {
        "SMALL_ONLY": data["is_small"] & ~data["is_low_contrast"],
        "LOW_CONTRAST_ONLY": ~data["is_small"] & data["is_low_contrast"],
        "SMALL_LOW_CONTRAST": data["is_small_low_contrast"],
        "NEITHER": ~data["is_small"] & ~data["is_low_contrast"],
    }.items():
        part = data[mask]
        tp = int((part["status"] == "TP").sum())
        fn = int((part["status"] == "FN").sum())
        rows.append({
            "outcome_group": label,
            "feature": "PRE_SPECIFIED_INTERACTION",
            "object_count": len(part),
            "valid_count": len(part),
            "q05": np.nan, "q25": np.nan, "median": np.nan, "q75": np.nan, "q95": np.nan, "iqr": np.nan,
            "median_difference_vs_tp": np.nan, "median_difference_ci_low": np.nan, "median_difference_ci_high": np.nan,
            "bootstrap_unit": "not applicable",
            "bootstrap_valid_replicates": 0,
            "tp": tp, "fn": fn, "recall": tp / len(part) if len(part) else np.nan,
            "notes": "Stage 1B exploratory definitions: small=min-side<=8 px; low contrast=absolute median difference<=4.",
        })

    out = pd.DataFrame(rows)
    path = root / "outputs/tables/05_object_observability.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

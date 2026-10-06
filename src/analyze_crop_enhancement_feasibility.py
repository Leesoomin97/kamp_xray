"""Stage 5 crop/patch and local-enhancement feasibility EDA.

All calculations use approved development samples and Conservative Stage 1D
images. Transformations are analysis-only and are never written as training data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def imread_grayscale(path: Path) -> np.ndarray | None:
    """Read an image without relying on OpenCV's Windows Unicode path handling."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except (OSError, ValueError):
        return None


CROP_SIZES = (128, 192, 256, 320)
OVERLAPS = (0.0, 0.25, 0.5)
PIXEL_THRESHOLDS = (8, 12, 16, 24, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def integer_box(value: str, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = json.loads(value)
    ix0, iy0 = max(0, math.floor(x0)), max(0, math.floor(y0))
    ix1, iy1 = min(width, math.ceil(x1)), min(height, math.ceil(y1))
    return ix0, iy0, max(ix0 + 1, ix1), max(iy0 + 1, iy1)


def expanded_box(box: tuple[int, int, int, int], scale: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = (x1 - x0) * scale, (y1 - y0) * scale
    return (
        max(0, math.floor(cx - bw / 2)), max(0, math.floor(cy - bh / 2)),
        min(width, math.ceil(cx + bw / 2)), min(height, math.ceil(cy + bh / 2)),
    )


def gradients(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy)


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    ar = pd.Series(a).rank(method="average").to_numpy(float)
    br = pd.Series(b).rank(method="average").to_numpy(float)
    if np.std(ar) == 0 or np.std(br) == 0:
        return 1.0 if np.array_equal(a, b) else np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.array_equal(a, b) else np.nan
    return float(np.corrcoef(a, b)[0, 1])


def edge_centroid(gradient: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    weights = gradient[yy, xx]
    total = float(weights.sum())
    if total <= 1e-9:
        return None
    return float(np.sum(xx * weights) / total), float(np.sum(yy * weights) / total)


def local_metrics(image: np.ndarray, gradient: np.ndarray, object_mask: np.ndarray, background_mask: np.ndarray) -> dict[str, float]:
    obj = image[object_mask].astype(float)
    bg = image[background_mask].astype(float)
    signed = float(np.median(obj) - np.median(bg))
    bg_median = float(np.median(bg))
    mad = float(np.median(np.abs(bg - bg_median)))
    return {
        "signed_contrast": signed,
        "absolute_contrast": abs(signed),
        "robust_cnr_like": abs(signed) / (1.4826 * mad + 1e-6),
        "object_edge_strength": float(np.mean(gradient[object_mask])),
        "background_gradient_mean": float(np.mean(gradient[background_mask])),
        "object_texture_iqr": float(np.quantile(obj, 0.75) - np.quantile(obj, 0.25)),
    }


def enhance(image: np.ndarray, name: str) -> np.ndarray:
    if name == "IDENTITY":
        return image.copy()
    if name == "CLAHE_MILD":
        return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(image)
    if name == "MEDIAN_3":
        return cv2.medianBlur(image, 3)
    if name == "BILATERAL_MILD":
        return cv2.bilateralFilter(image, d=5, sigmaColor=10, sigmaSpace=3)
    if name == "UNSHARP_MILD":
        smooth = cv2.GaussianBlur(image, (0, 0), 1.0)
        return np.clip(image.astype(float) * 1.15 - smooth.astype(float) * 0.15, 0, 255).round().astype(np.uint8)
    if name == "GAUSSIAN_REFERENCE":
        return cv2.GaussianBlur(image, (3, 3), 0.5)
    raise ValueError(name)


def tile_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def subgroup_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ALL": pd.Series(True, index=data.index),
        "TP": data["status"].eq("TP"),
        "FN_ALL": data["status"].eq("FN"),
        "SMALL_ONLY": data["is_small"] & ~data["is_low_contrast"],
        "LOW_CONTRAST_ONLY": ~data["is_small"] & data["is_low_contrast"],
        "SMALL_LOW_CONTRAST": data["is_small"] & data["is_low_contrast"],
        "NEAR_PRODUCT_EDGE": data["is_near_product_edge"],
        "LOCALIZATION_FAILURE": data["error_type"].eq("LOCALIZATION_FAILURE"),
        "LOW_CONFIDENCE_FN": data["error_type"].eq("LOW_CONFIDENCE"),
    }


def main() -> None:
    config = parse_args()
    root = config.project_root.resolve()
    table_dir = root / "outputs/tables"
    figure_dir = root / "outputs/figures/stage5_feasibility"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(root / "outputs/tables/02a_conservative_640_oof_v1/02a_object_predictions.csv")
    xray = pd.read_csv(root / "outputs/tables/01b_object_xray_features.csv")
    manifest = pd.read_csv(root / "outputs/tables/01d_processed_image_manifest.csv")
    conservative = manifest[manifest["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE"].copy()
    grayscale = manifest[manifest["representation"] == "GRAYSCALE_DIRECT"].copy()
    image_paths = conservative.set_index("stem")["derived_path"].to_dict()
    grayscale_paths = grayscale.set_index("stem")["derived_path"].to_dict()
    data = pred.merge(xray, on=["stem", "object_id"], validate="one_to_one")
    if len(data) != 1147 or len(image_paths) != 500:
        raise RuntimeError("Stage 5 input integrity failed")
    data["is_small"] = data["bbox_min_side_px"] <= 8.0
    data["is_low_contrast"] = data["absolute_median_difference"] <= 4.0
    product_edge_q25 = float(data["product_edge_distance_norm"].quantile(0.25))
    data["is_near_product_edge"] = data["product_edge_distance_norm"] <= product_edge_q25

    # Crop scale simulation.
    crop = data[["stem", "object_id", "fold_id", "status", "error_type", "bbox_min_side_px", "bbox_area_ratio", "absolute_median_difference"]].copy()
    crop["full_image_640_min_side_px"] = data["bbox_min_side_px"] * np.minimum(
        640.0 / data["resolution"].str.split("x").str[0].astype(float),
        640.0 / data["resolution"].str.split("x").str[1].astype(float),
    )
    crop["is_small"] = data["is_small"]
    crop["is_low_contrast"] = data["is_low_contrast"]
    crop["is_small_low_contrast"] = data["is_small"] & data["is_low_contrast"]
    for size in CROP_SIZES:
        col = f"crop{size}_to640_min_side_px"
        crop[col] = data["bbox_min_side_px"] * 640.0 / size
        crop[f"crop{size}_magnification_vs_full"] = crop[col] / crop["full_image_640_min_side_px"]
        for threshold in PIXEL_THRESHOLDS:
            crop[f"crop{size}_ge{threshold}px"] = crop[col] >= threshold
    crop.to_csv(table_dir / "05_crop_scale_simulation.csv", index=False, encoding="utf-8-sig")

    # Parse boxes and calculate fixed-grid containment trade-offs.
    image_meta = conservative.set_index("stem")[["width", "height"]].to_dict("index")
    box_by_key: dict[tuple[str, int], tuple[int, int, int, int]] = {}
    for row in data.itertuples():
        meta = image_meta[row.stem]
        box_by_key[(row.stem, int(row.object_id))] = integer_box(row.gt_bbox_json, int(meta["width"]), int(meta["height"]))
    patch_rows = []
    subgroup = subgroup_masks(data)
    for patch in CROP_SIZES:
        for overlap in OVERLAPS:
            stride = max(1, int(round(patch * (1.0 - overlap))))
            counts = []
            patch_counts = {}
            for stem, meta in image_meta.items():
                xs = tile_starts(int(meta["width"]), patch, stride)
                ys = tile_starts(int(meta["height"]), patch, stride)
                tiles = [(x, y, x + patch, y + patch) for y in ys for x in xs]
                patch_counts[stem] = len(tiles)
                for object_id in data.loc[data["stem"] == stem, "object_id"]:
                    box = box_by_key[(stem, int(object_id))]
                    contained = sum(box[0] >= t[0] and box[1] >= t[1] and box[2] <= t[2] and box[3] <= t[3] for t in tiles)
                    counts.append((stem, int(object_id), contained))
            count_df = pd.DataFrame(counts, columns=["stem", "object_id", "full_coverage_count"])
            joined = data[["stem", "object_id"]].merge(count_df, on=["stem", "object_id"], validate="one_to_one")
            for name, mask in subgroup.items():
                part = joined[mask.to_numpy()]
                patch_rows.append({
                    "patch_size": patch,
                    "overlap_fraction": overlap,
                    "stride_px": stride,
                    "subgroup": name,
                    "object_count": len(part),
                    "object_fully_contained_rate": float((part["full_coverage_count"] >= 1).mean()),
                    "object_boundary_crossing_rate": float((part["full_coverage_count"] == 0).mean()),
                    "duplicate_coverage_rate": float((part["full_coverage_count"] >= 2).mean()),
                    "mean_full_coverage_count": float(part["full_coverage_count"].mean()),
                    "average_patches_per_image": float(np.mean(list(patch_counts.values()))),
                    "notes": "Boundary crossing means no grid patch fully contains the GT box. Last tiles are anchored to the image edge; padding is assumed when patch exceeds an image dimension.",
                })
    patch_tradeoff = pd.DataFrame(patch_rows)
    patch_tradeoff.to_csv(table_dir / "05_patch_boundary_tradeoff.csv", index=False, encoding="utf-8-sig")

    # Enhancement feature and preservation calculations.
    methods = ["IDENTITY", "CLAHE_MILD", "MEDIAN_3", "BILATERAL_MILD", "UNSHARP_MILD", "GAUSSIAN_REFERENCE"]
    enhancement_rows: list[dict[str, object]] = []
    for stem, stem_data in data.groupby("stem", sort=True):
        source = imread_grayscale(root / image_paths[stem])
        gray_direct = imread_grayscale(root / grayscale_paths[stem])
        if source is None or gray_direct is None:
            raise FileNotFoundError(stem)
        height, width = source.shape
        repair_mask = (source != gray_direct).astype(np.uint8)
        repair_neighborhood = cv2.dilate(repair_mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        boxes = {int(r.object_id): box_by_key[(stem, int(r.object_id))] for r in stem_data.itertuples()}
        transformed = {name: enhance(source, name) for name in methods}
        grad = {name: gradients(image) for name, image in transformed.items()}
        source_grad = grad["IDENTITY"]
        artifact_reference = float(source_grad[repair_neighborhood].mean()) if repair_neighborhood.any() else np.nan
        for row in stem_data.itertuples():
            box = boxes[int(row.object_id)]
            expanded = expanded_box(box, 2.0, width, height)
            object_mask = np.zeros((height, width), dtype=bool)
            object_mask[box[1]:box[3], box[0]:box[2]] = True
            background_mask = np.zeros((height, width), dtype=bool)
            background_mask[expanded[1]:expanded[3], expanded[0]:expanded[2]] = True
            background_mask[object_mask] = False
            for other_id, other in boxes.items():
                if other_id != int(row.object_id):
                    background_mask[other[1]:other[3], other[0]:other[2]] = False
            if object_mask.sum() == 0 or background_mask.sum() < 10:
                continue
            source_obj = source[object_mask].astype(float)
            source_centroid = edge_centroid(source_grad, object_mask)
            identity_metrics = local_metrics(source, source_grad, object_mask, background_mask)
            for name in methods:
                image = transformed[name]
                metric = local_metrics(image, grad[name], object_mask, background_mask)
                obj = image[object_mask].astype(float)
                centroid = edge_centroid(grad[name], object_mask)
                centroid_shift = np.nan
                if source_centroid is not None and centroid is not None:
                    centroid_shift = math.hypot(centroid[0] - source_centroid[0], centroid[1] - source_centroid[1])
                artifact_grad = float(grad[name][repair_neighborhood].mean()) if repair_neighborhood.any() else np.nan
                enhancement_rows.append({
                    "stem": stem, "object_id": int(row.object_id), "fold_id": int(row.fold_id),
                    "status": row.status, "error_type": row.error_type, "method": name,
                    "is_small": bool(row.is_small), "is_low_contrast": bool(row.is_low_contrast),
                    "is_small_low_contrast": bool(row.is_small and row.is_low_contrast),
                    "is_near_product_edge": bool(row.is_near_product_edge),
                    **metric,
                    "delta_absolute_contrast_vs_identity": metric["absolute_contrast"] - identity_metrics["absolute_contrast"],
                    "delta_robust_cnr_vs_identity": metric["robust_cnr_like"] - identity_metrics["robust_cnr_like"],
                    "delta_object_edge_vs_identity": metric["object_edge_strength"] - identity_metrics["object_edge_strength"],
                    "background_gradient_ratio_vs_identity": metric["background_gradient_mean"] / max(identity_metrics["background_gradient_mean"], 1e-6),
                    "object_intensity_rank_correlation": rank_correlation(source_obj, obj),
                    "bbox_intensity_correlation": correlation(source_obj, obj),
                    "object_edge_centroid_shift_px": centroid_shift,
                    "artifact_trace_gradient_ratio_vs_identity": artifact_grad / max(artifact_reference, 1e-6) if np.isfinite(artifact_reference) else np.nan,
                    "notes": "Analysis-only feature measurement on Conservative image; not detector performance. Product edge is inferred.",
                })
    enhancement = pd.DataFrame(enhancement_rows)
    enhancement.to_csv(table_dir / "05_enhancement_comparison.csv", index=False, encoding="utf-8-sig")

    # Aggregate signal-preservation and subgroup feasibility.
    e_masks = subgroup_masks(enhancement)
    signal_rows = []
    crop_indexed = crop.set_index(["stem", "object_id"])
    for crop_size in CROP_SIZES:
        column = f"crop{crop_size}_to640_min_side_px"
        enhancement[column] = [crop_indexed.loc[(s, int(o)), column] for s, o in zip(enhancement.stem, enhancement.object_id)]
    for method in methods:
        method_data = enhancement[enhancement["method"] == method]
        method_masks = subgroup_masks(method_data)
        for name, mask in method_masks.items():
            part = method_data[mask]
            signal_rows.append({
                "method": method, "subgroup": name, "object_count": len(part),
                "median_absolute_contrast": part["absolute_contrast"].median(),
                "median_contrast_delta": part["delta_absolute_contrast_vs_identity"].median(),
                "fraction_contrast_improved": float((part["delta_absolute_contrast_vs_identity"] > 0).mean()),
                "median_robust_cnr_like": part["robust_cnr_like"].median(),
                "median_cnr_delta": part["delta_robust_cnr_vs_identity"].median(),
                "fraction_cnr_improved": float((part["delta_robust_cnr_vs_identity"] > 0).mean()),
                "median_object_edge_delta": part["delta_object_edge_vs_identity"].median(),
                "median_background_gradient_ratio": part["background_gradient_ratio_vs_identity"].median(),
                "median_object_rank_correlation": part["object_intensity_rank_correlation"].median(),
                "median_bbox_intensity_correlation": part["bbox_intensity_correlation"].median(),
                "median_edge_centroid_shift_px": part["object_edge_centroid_shift_px"].median(),
                "median_artifact_trace_gradient_ratio": part["artifact_trace_gradient_ratio_vs_identity"].median(),
                **{
                    f"fraction_crop{crop_size}_ge24_and_contrast_improved": float(
                        ((part[f"crop{crop_size}_to640_min_side_px"] >= 24)
                         & (part["delta_absolute_contrast_vs_identity"] > 0)).mean()
                    )
                    for crop_size in CROP_SIZES
                },
                "notes": "Combined column is feature-level feasibility only, not projected detector performance.",
            })
    signal = pd.DataFrame(signal_rows)
    signal.to_csv(table_dir / "05_signal_preservation.csv", index=False, encoding="utf-8-sig")

    # Compact diagnostic figures.
    order = ["TP", "LOW_CONFIDENCE", "LOCALIZATION_FAILURE", "NO_DETECTION"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    values = [data.loc[data["error_type"].where(data["status"].eq("FN"), "TP") == name, "bbox_min_side_px"] for name in order]
    ax.boxplot(values, tick_labels=order, showfliers=False)
    ax.set_ylabel("Original bbox minimum side (px)"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(figure_dir / "05a_original_size_by_outcome.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Full 640", *[f"Crop {s}" for s in CROP_SIZES]]
    columns = ["full_image_640_min_side_px", *[f"crop{s}_to640_min_side_px" for s in CROP_SIZES]]
    for threshold in (8, 16, 24, 32):
        ax.plot(labels, [(crop[c] >= threshold).mean() for c in columns], marker="o", label=f">={threshold}px")
    ax.set_ylabel("Fraction of GT objects"); ax.set_ylim(0, 1.03); ax.legend(); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(figure_dir / "05b_crop_projected_size_coverage.png", dpi=160); plt.close(fig)

    all_signal = signal[signal["subgroup"] == "ALL"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(all_signal["method"], all_signal["median_contrast_delta"])
    ax.set_ylabel("Median absolute-contrast delta"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(figure_dir / "05c_enhancement_contrast_delta.png", dpi=160); plt.close(fig)

    overall_patch = patch_tradeoff[patch_tradeoff["subgroup"] == "ALL"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for overlap in OVERLAPS:
        part = overall_patch[overall_patch["overlap_fraction"] == overlap]
        ax.plot(part["patch_size"], part["object_boundary_crossing_rate"], marker="o", label=f"overlap {overlap:.0%}")
    ax.set_xlabel("Patch size"); ax.set_ylabel("Boundary-crossing rate"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "05d_patch_boundary_tradeoff.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()

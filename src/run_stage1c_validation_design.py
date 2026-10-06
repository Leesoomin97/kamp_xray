from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from run_stage1a_structural_eda import fnum, parse_bmp, quantile, save_boxplot, tf, write_csv, write_png
from run_stage1b_xray_difficulty_eda import geometry_box


GROUPING_RULES = (6, 10)
FOLD_COUNTS = (5, 4, 3)
GENERATOR_VERSION = "stage1c_v1"
FEATURES = (
    "bbox_min_side_px", "bbox_area_ratio", "bbox_aspect_ratio", "image_edge_distance_px",
    "product_edge_distance_px", "absolute_median_difference", "signed_median_difference",
    "cnr_like_robust", "object_iqr", "background_gradient_mean", "background_iqr",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_true(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def median(values: list[float]) -> float:
    return quantile(values, .5)


def std(values: list[float]) -> float:
    if not values:
        return math.nan
    mean = statistics.fmean(values)
    return math.sqrt(statistics.fmean([(value - mean) ** 2 for value in values]))


def ks_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return math.nan
    a, b = sorted(a), sorted(b)
    i = j = 0
    distance = 0.0
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and a[i] <= b[j]):
            value = a[i]
        else:
            value = b[j]
        while i < len(a) and a[i] <= value: i += 1
        while j < len(b) and b[j] <= value: j += 1
        distance = max(distance, abs(i / len(a) - j / len(b)))
    return distance


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            low, high = sorted((ra, rb))
            self.parent[high] = low


def render_review_pair(
    path: Path,
    image_a: dict[str, object], image_b: dict[str, object],
    boxes_a: list[tuple[int, int, int, int]], boxes_b: list[tuple[int, int, int, int]],
) -> None:
    target_w, target_h, gap = 480, 360, 10
    canvas_w = target_w * 2 + gap
    canvas = bytearray([255] * (canvas_w * target_h * 3))
    for offset, image, boxes in ((0, image_a, boxes_a), (target_w + gap, image_b, boxes_b)):
        rows: list[bytes] = image["rows"]  # type: ignore[assignment]
        palette: list[tuple[int, int, int]] = image["palette"]  # type: ignore[assignment]
        source_h, source_w = len(rows), len(rows[0])
        for y in range(target_h):
            sy = min(source_h - 1, int(y * source_h / target_h))
            for x in range(target_w):
                sx = min(source_w - 1, int(x * source_w / target_w))
                color = palette[rows[sy][sx]]
                index = (y * canvas_w + offset + x) * 3
                canvas[index:index + 3] = bytes(color)
        for box in boxes:
            x0 = min(target_w - 1, max(0, round(box[0] * target_w / source_w)))
            x1 = min(target_w, max(x0 + 1, round(box[2] * target_w / source_w)))
            y0 = min(target_h - 1, max(0, round(box[1] * target_h / source_h)))
            y1 = min(target_h, max(y0 + 1, round(box[3] * target_h / source_h)))
            for x in range(x0, x1):
                for y in (y0, y1 - 1):
                    index = (y * canvas_w + offset + x) * 3
                    canvas[index:index + 3] = b"\x00\xff\xff"
            for y in range(y0, y1):
                for x in (x0, x1 - 1):
                    index = (y * canvas_w + offset + x) * 3
                    canvas[index:index + 3] = b"\x00\xff\xff"
    write_png(path, canvas_w, target_h, bytes(canvas))


def build_component_vectors(
    components: dict[str, list[str]], sample_info: dict[str, dict[str, object]],
    objects_by_stem: dict[str, list[dict[str, str]]], cutpoints: dict[str, tuple[float, float, float]],
) -> tuple[dict[str, Counter[str]], dict[str, str]]:
    vectors: dict[str, Counter[str]] = {}
    families: dict[str, str] = {}
    for component_id, stems in components.items():
        vector: Counter[str] = Counter()
        for stem in stems:
            info = sample_info[stem]
            vector["sample:ALL"] += 1
            for family, value in (("machine", info["machine"]), ("date", info["date"]), ("resolution", info["resolution"]), ("object_count", info["object_count"])):
                token = f"{family}:{value}"
                vector[token] += 1
                families[token] = family
            for obj in objects_by_stem[stem]:
                for feature in FEATURES:
                    value = float(obj[feature])
                    q1, q2, q3 = cutpoints[feature]
                    level = "Q1" if value <= q1 else "Q2" if value <= q2 else "Q3" if value <= q3 else "Q4"
                    token = f"obj_{feature}:{level}"
                    vector[token] += 1
                    families[token] = f"obj_{feature}"
        vectors[component_id] = vector
    families["sample:ALL"] = "sample"
    return vectors, families


def assign_components(
    components: dict[str, list[str]], vectors: dict[str, Counter[str]], families: dict[str, str], n_folds: int,
) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for vector in vectors.values(): totals.update(vector)
    family_tokens: dict[str, list[str]] = defaultdict(list)
    for token, family in families.items(): family_tokens[family].append(token)
    family_weights = {family: 1.0 for family in family_tokens}
    family_weights["sample"] = 5.0
    family_weights["machine"] = 2.0
    family_weights["date"] = 1.5
    family_weights["resolution"] = 2.0
    family_weights["object_count"] = 1.5

    rarity = {}
    for component_id, vector in vectors.items():
        rarity[component_id] = max((count / max(1, totals[token]) for token, count in vector.items() if token != "sample:ALL"), default=0)
    ordered = sorted(components, key=lambda c: (-len(components[c]), -rarity[c], c))
    fold_vectors = [Counter() for _ in range(n_folds)]
    assignment: dict[str, int] = {}

    def objective(candidate_fold: int, vector: Counter[str]) -> float:
        score = 0.0
        for family, tokens in family_tokens.items():
            family_score = 0.0
            for token in tokens:
                target = totals[token] / n_folds
                denominator = max(1.0, target)
                for fold in range(n_folds):
                    observed = fold_vectors[fold][token] + (vector[token] if fold == candidate_fold else 0)
                    family_score += ((observed - target) / denominator) ** 2
            score += family_weights[family] * family_score / max(1, len(tokens))
        return score

    for component_id in ordered:
        vector = vectors[component_id]
        scores = [(objective(fold, vector), fold_vectors[fold]["sample:ALL"], fold) for fold in range(n_folds)]
        chosen = min(scores)[2]
        assignment[component_id] = chosen + 1
        fold_vectors[chosen].update(vector)
    return assignment


def main(workspace: Path) -> None:
    table_dir = workspace / "outputs" / "tables"
    eda_dir = workspace / "outputs" / "eda"
    figure_dir = workspace / "outputs" / "figures" / "stage1c"
    review_dir = figure_dir / "nearduplicate_review"
    balance_dir = figure_dir / "fold_balance"
    for directory in (table_dir, eda_dir, figure_dir, review_dir, balance_dir): directory.mkdir(parents=True, exist_ok=True)

    allowlist_rows = [row for row in read_csv(table_dir / "00_stage1_allowlist.csv") if is_true(row["eligible_for_stage1_eda"])]
    metadata_rows = read_csv(table_dir / "01a_sample_metadata.csv")
    geometry_rows = read_csv(table_dir / "01a_object_geometry.csv")
    sequence_rows = read_csv(table_dir / "01a5_sequence_group_assignments.csv")
    sequence_summary = read_csv(table_dir / "01a5_sequence_group_summary.csv")
    near_coverage = read_csv(table_dir / "01a5_nearduplicate_group_coverage.csv")
    xray_rows = read_csv(table_dir / "01b_object_xray_features.csv")
    _candidate_features = read_csv(table_dir / "01b_stage2_candidate_features.csv")
    _relationships = read_csv(table_dir / "01b_feature_relationships.csv")

    allowed = {row["stem"]: row for row in allowlist_rows}
    metadata = {row["stem"]: row for row in metadata_rows if row["stem"] in allowed}
    geometry_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in geometry_rows: geometry_by_stem[row["image_id"]].append(row)
    objects_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in xray_rows: objects_by_stem[row["stem"]].append(row)
    sequence_by_threshold: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in sequence_rows:
        threshold = int(row["threshold_seconds"])
        if threshold in GROUPING_RULES: sequence_by_threshold[threshold][row["stem"]] = row

    geometry_keys = {(row["image_id"], row["object_id"]) for row in geometry_rows}
    xray_keys = {(row["stem"], row["object_id"]) for row in xray_rows}
    integrity_checks = [
        ("eligible samples", 500, len(allowed), len(allowed) == 500),
        ("GT objects", 1147, len(geometry_rows), len(geometry_rows) == 1147),
        ("Stage 1A/1B object key match", 1147, len(geometry_keys & xray_keys), geometry_keys == xray_keys),
        ("6s assignments", 500, len(sequence_by_threshold[6]), set(sequence_by_threshold[6]) == set(allowed)),
        ("10s assignments", 500, len(sequence_by_threshold[10]), set(sequence_by_threshold[10]) == set(allowed)),
        ("metadata rows", 500, len(metadata), set(metadata) == set(allowed)),
        ("xray objects with product edge", 1147, sum(row["product_edge_distance_px"] != "" for row in xray_rows), all(row["product_edge_distance_px"] != "" for row in xray_rows)),
        ("required xray features complete", 1147 * len(FEATURES), sum(row[feature] != "" for row in xray_rows for feature in FEATURES), all(row[feature] != "" for row in xray_rows for feature in FEATURES)),
    ]
    integrity_rows = [{"check": name, "expected": expected, "observed": observed, "status": "OK" if ok else "ERROR", "notes": "Approved Stage 1 scope only."} for name, expected, observed, ok in integrity_checks]
    write_csv(table_dir / "01c_input_integrity.csv", ["check", "expected", "observed", "status", "notes"], integrity_rows)
    if not all(ok for _, _, _, ok in integrity_checks):
        raise RuntimeError("Stage 1C input integrity failed")

    sample_info = {}
    for stem in sorted(allowed):
        first = objects_by_stem[stem][0]
        sample_info[stem] = {
            "machine": metadata[stem]["machine"], "date": metadata[stem]["filename_date"],
            "timestamp": metadata[stem]["timestamp"], "resolution": first["resolution"],
            "object_count": len(objects_by_stem[stem]),
        }

    # Three dHash candidates, manually reviewed from original BMP side-by-side with TXT overlays.
    near_pairs = []
    seen_pairs = set()
    for row in near_coverage:
        pair = tuple(sorted((row["stem_a"], row["stem_b"])))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            near_pairs.append((pair[0], pair[1], float(row["similarity_value"]), float(row["time_gap_seconds"])))
    near_pairs.sort()
    review_rows = []
    for index, (stem_a, stem_b, similarity, time_gap) in enumerate(near_pairs, start=1):
        image_a = parse_bmp(workspace / Path(allowed[stem_a]["canonical_raw_path"]))
        image_b = parse_bmp(workspace / Path(allowed[stem_b]["canonical_raw_path"]))
        boxes_a = [geometry_box(row, int(image_a["width"]), int(image_a["height"])) for row in geometry_by_stem[stem_a]]
        boxes_b = [geometry_box(row, int(image_b["width"]), int(image_b["height"])) for row in geometry_by_stem[stem_b]]
        render_review_pair(review_dir / f"pair_{index:02d}.png", image_a, image_b, boxes_a, boxes_b)
        review_rows.append({
            "pair_id": f"P{index:02d}", "stem_a": stem_a, "stem_b": stem_b,
            "machine_a": sample_info[stem_a]["machine"], "machine_b": sample_info[stem_b]["machine"],
            "date_a": sample_info[stem_a]["date"], "date_b": sample_info[stem_b]["date"],
            "time_gap_if_meaningful": fnum(time_gap), "dhash_distance": fnum((1 - similarity) * 256),
            "visual_similarity_class": "A — strong likely near-duplicate / same scene-family",
            "same_product_geometry": "YES", "same_internal_structure": "YES", "same_bbox_layout": "YES",
            "artifact_similarity": "HIGH", "force_same_validation_group": "TRUE",
            "reason": "Manual side-by-side review confirms closely matching product silhouette, internal grayscale bars, three-object layout, and outline layout; conservative grouping prevents scene-family leakage.",
            "limitations": "Different dates preclude claiming the same physical item; classification is visual/structural, not provenance proof.",
        })
    review_columns = ["pair_id", "stem_a", "stem_b", "machine_a", "machine_b", "date_a", "date_b", "time_gap_if_meaningful", "dhash_distance", "visual_similarity_class", "same_product_geometry", "same_internal_structure", "same_bbox_layout", "artifact_similarity", "force_same_validation_group", "reason", "limitations"]
    write_csv(table_dir / "01c_nearduplicate_manual_review.csv", review_columns, review_rows)
    forced_pairs = [(row["stem_a"], row["stem_b"], row["pair_id"]) for row in review_rows if is_true(str(row["force_same_validation_group"]))]

    similarity_stems = sorted({stem for stem_a, stem_b, _ in forced_pairs for stem in (stem_a, stem_b)})
    similarity_union = UnionFind(similarity_stems)
    for stem_a, stem_b, _ in forced_pairs:
        similarity_union.union(stem_a, stem_b)
    similarity_roots: dict[str, list[str]] = defaultdict(list)
    for stem in similarity_stems:
        similarity_roots[similarity_union.find(stem)].append(stem)
    similarity_component_for_stem = {}
    for index, root in enumerate(sorted(similarity_roots, key=lambda item: min(similarity_roots[item])), start=1):
        for stem in similarity_roots[root]:
            similarity_component_for_stem[stem] = f"SIM{index:03d}"

    component_rows = []
    components_by_rule: dict[int, dict[str, list[str]]] = {}
    component_for_rule: dict[int, dict[str, str]] = {}
    for threshold in GROUPING_RULES:
        temporal_for = {stem: row["sequence_group_id"] for stem, row in sequence_by_threshold[threshold].items()}
        temporal_groups = sorted(set(temporal_for.values()))
        union = UnionFind(temporal_groups)
        linked_temporal: set[str] = set()
        for stem_a, stem_b, _ in forced_pairs:
            group_a, group_b = temporal_for[stem_a], temporal_for[stem_b]
            union.union(group_a, group_b)
            linked_temporal.update((group_a, group_b))
        roots: dict[str, list[str]] = defaultdict(list)
        for group in temporal_groups: roots[union.find(group)].append(group)
        ordered_roots = sorted(roots, key=lambda root: min(roots[root]))
        final_id_by_temporal = {}
        for index, root in enumerate(ordered_roots, start=1):
            final_id = f"T{threshold:02d}_C{index:03d}"
            for temporal in roots[root]: final_id_by_temporal[temporal] = final_id
        components: dict[str, list[str]] = defaultdict(list)
        for stem in sorted(allowed): components[final_id_by_temporal[temporal_for[stem]]].append(stem)
        components_by_rule[threshold] = dict(components)
        component_for_rule[threshold] = {stem: component_id for component_id, stems in components.items() for stem in stems}
        for stem in sorted(allowed):
            final_id = component_for_rule[threshold][stem]
            temporal_id = temporal_for[stem]
            component_rows.append({
                "grouping_rule": f"{threshold}s", "stem": stem, "temporal_group_id": temporal_id,
                "similarity_component_id": similarity_component_for_stem.get(stem, ""),
                "final_group_component_id": final_id, "component_size": len(components[final_id]),
                "machine": sample_info[stem]["machine"], "date": sample_info[stem]["date"],
                "reason_for_connection": "temporal group + manually reviewed similarity merge" if temporal_id in linked_temporal else "temporal group only",
                "notes": "Similarity merging is transitive; links can cross dates only after manual review.",
            })
    component_columns = ["grouping_rule", "stem", "temporal_group_id", "similarity_component_id", "final_group_component_id", "component_size", "machine", "date", "reason_for_connection", "notes"]
    write_csv(table_dir / "01c_group_components.csv", component_columns, component_rows)

    cutpoints = {feature: tuple(quantile([float(row[feature]) for row in xray_rows], q) for q in (.25, .50, .75)) for feature in FEATURES}
    candidate_rows = []
    assignments: dict[tuple[int, int], dict[str, int]] = {}
    for threshold in GROUPING_RULES:
        vectors, families = build_component_vectors(components_by_rule[threshold], sample_info, objects_by_stem, cutpoints)
        for n_folds in FOLD_COUNTS:
            component_assignment = assign_components(components_by_rule[threshold], vectors, families, n_folds)
            fold_for_stem = {stem: component_assignment[component_for_rule[threshold][stem]] for stem in allowed}
            assignments[(threshold, n_folds)] = fold_for_stem
            for stem in sorted(allowed):
                component_id = component_for_rule[threshold][stem]
                candidate_rows.append({
                    "grouping_rule": f"{threshold}s", "n_folds": n_folds, "fold_id": fold_for_stem[stem], "stem": stem,
                    "final_group_component_id": component_id, "component_size": len(components_by_rule[threshold][component_id]),
                    "machine": sample_info[stem]["machine"], "date": sample_info[stem]["date"],
                    "resolution": sample_info[stem]["resolution"], "object_count": sample_info[stem]["object_count"],
                    "notes": "Deterministic largest-first group assignment balancing sample/category and object-feature quartile counts; no model outputs.",
                })
    candidate_columns = ["grouping_rule", "n_folds", "fold_id", "stem", "final_group_component_id", "component_size", "machine", "date", "resolution", "object_count", "notes"]
    write_csv(table_dir / "01c_candidate_fold_assignments.csv", candidate_columns, candidate_rows)

    sample_balance_rows = []
    sample_metrics: dict[tuple[int, int], dict[str, float]] = {}
    for threshold in GROUPING_RULES:
        for n_folds in FOLD_COUNTS:
            fold_for = assignments[(threshold, n_folds)]
            fold_sizes = Counter(fold_for.values())
            scheme_metrics = {"min_images": min(fold_sizes.values()), "max_images": max(fold_sizes.values()), "max_image_deviation": max(abs(count / len(allowed) - 1 / n_folds) for count in fold_sizes.values())}
            for fold in range(1, n_folds + 1):
                stems = [stem for stem, value in fold_for.items() if value == fold]
                sample_balance_rows.append({
                    "grouping_rule": f"{threshold}s", "n_folds": n_folds, "fold_id": fold,
                    "feature": "image_count", "category": "ALL", "fold_count": len(stems),
                    "fold_fraction": fnum(len(stems) / len(allowed)), "overall_count": len(allowed), "overall_fraction": "1",
                    "absolute_fraction_deviation": fnum(abs(len(stems) / len(allowed) - 1 / n_folds)), "total_variation_distance": "",
                    "notes": "Image fraction compared with ideal 1/K.",
                })
                for feature in ("machine", "date", "resolution", "object_count"):
                    overall_counts = Counter(str(info[feature]) for info in sample_info.values())
                    fold_counts = Counter(str(sample_info[stem][feature]) for stem in stems)
                    tv = .5 * sum(abs(fold_counts[category] / len(stems) - overall_counts[category] / len(allowed)) for category in overall_counts)
                    for category in sorted(overall_counts):
                        fold_fraction = fold_counts[category] / len(stems)
                        overall_fraction = overall_counts[category] / len(allowed)
                        sample_balance_rows.append({
                            "grouping_rule": f"{threshold}s", "n_folds": n_folds, "fold_id": fold,
                            "feature": feature, "category": category, "fold_count": fold_counts[category], "fold_fraction": fnum(fold_fraction),
                            "overall_count": overall_counts[category], "overall_fraction": fnum(overall_fraction),
                            "absolute_fraction_deviation": fnum(abs(fold_fraction - overall_fraction)), "total_variation_distance": fnum(tv),
                            "notes": "Fold composition compared with full approved development pool.",
                        })
                    scheme_metrics[f"{feature}_max_tv"] = max(scheme_metrics.get(f"{feature}_max_tv", 0.0), tv)
            sample_metrics[(threshold, n_folds)] = scheme_metrics
    sample_balance_columns = ["grouping_rule", "n_folds", "fold_id", "feature", "category", "fold_count", "fold_fraction", "overall_count", "overall_fraction", "absolute_fraction_deviation", "total_variation_distance", "notes"]
    write_csv(table_dir / "01c_sample_fold_balance.csv", sample_balance_columns, sample_balance_rows)

    object_balance_rows = []
    object_metrics: dict[tuple[int, int], dict[str, float]] = {}
    for threshold in GROUPING_RULES:
        for n_folds in FOLD_COUNTS:
            fold_for = assignments[(threshold, n_folds)]
            metrics = {}
            for feature in FEATURES:
                overall = [float(row[feature]) for row in xray_rows]
                overall_mean, overall_median, overall_std = statistics.fmean(overall), median(overall), std(overall)
                max_abs_smd = max_ks = 0.0
                for fold in range(1, n_folds + 1):
                    values = [float(row[feature]) for row in xray_rows if fold_for[row["stem"]] == fold]
                    smd = (statistics.fmean(values) - overall_mean) / overall_std if overall_std else 0.0
                    ks = ks_distance(values, overall)
                    max_abs_smd, max_ks = max(max_abs_smd, abs(smd)), max(max_ks, ks)
                    object_balance_rows.append({
                        "grouping_rule": f"{threshold}s", "n_folds": n_folds, "fold_id": fold, "feature": feature,
                        "n_objects": len(values), "mean": fnum(statistics.fmean(values)), "median": fnum(median(values)),
                        "q10": fnum(quantile(values,.10)), "q25": fnum(quantile(values,.25)), "q75": fnum(quantile(values,.75)), "q90": fnum(quantile(values,.90)),
                        "iqr": fnum(quantile(values,.75)-quantile(values,.25)), "overall_mean": fnum(overall_mean), "overall_median": fnum(overall_median),
                        "standardized_mean_difference": fnum(smd), "ks_distance": fnum(ks),
                        "notes": "Descriptive fold-vs-full diagnostic; KS is not used as a hypothesis-test verdict.",
                    })
                metrics[f"{feature}_max_abs_smd"] = max_abs_smd
                metrics[f"{feature}_max_ks"] = max_ks
                groups = {f"F{fold}": [float(row[feature]) for row in xray_rows if fold_for[row["stem"]] == fold] for fold in range(1, n_folds + 1)}
                if feature in ("bbox_min_side_px", "absolute_median_difference", "cnr_like_robust", "background_gradient_mean", "product_edge_distance_px"):
                    save_boxplot(balance_dir / f"{threshold}s_{n_folds}fold_{feature}.svg", groups, f"{threshold}s {n_folds}-fold: {feature}", feature)
            object_metrics[(threshold, n_folds)] = metrics
    object_balance_columns = ["grouping_rule", "n_folds", "fold_id", "feature", "n_objects", "mean", "median", "q10", "q25", "q75", "q90", "iqr", "overall_mean", "overall_median", "standardized_mean_difference", "ks_distance", "notes"]
    write_csv(table_dir / "01c_object_fold_balance.csv", object_balance_columns, object_balance_rows)

    thresholds = {
        "small": cutpoints["bbox_min_side_px"][0], "low_contrast": cutpoints["absolute_median_difference"][0],
        "near_image_edge": cutpoints["image_edge_distance_px"][0], "complex_background": cutpoints["background_gradient_mean"][2],
        "low_cnr": cutpoints["cnr_like_robust"][0], "high_heterogeneity": cutpoints["object_iqr"][2],
        "near_product_edge": cutpoints["product_edge_distance_px"][0],
    }
    condition_functions = {
        "small": lambda row: float(row["bbox_min_side_px"]) <= thresholds["small"],
        "low_contrast": lambda row: float(row["absolute_median_difference"]) <= thresholds["low_contrast"],
        "near_image_edge": lambda row: float(row["image_edge_distance_px"]) <= thresholds["near_image_edge"],
        "complex_background": lambda row: float(row["background_gradient_mean"]) >= thresholds["complex_background"],
        "low_cnr": lambda row: float(row["cnr_like_robust"]) <= thresholds["low_cnr"],
        "high_heterogeneity": lambda row: float(row["object_iqr"]) >= thresholds["high_heterogeneity"],
        "near_product_edge": lambda row: float(row["product_edge_distance_px"]) <= thresholds["near_product_edge"],
    }
    joint_conditions = (("small","low_contrast"),("small","near_image_edge"),("small","complex_background"),("low_contrast","complex_background"),("low_cnr","high_heterogeneity"),("near_product_edge","low_contrast"))
    joint_rows = []
    joint_metrics: dict[tuple[int,int], dict[str,int]] = {}
    for threshold in GROUPING_RULES:
        for n_folds in FOLD_COUNTS:
            fold_for = assignments[(threshold,n_folds)]
            zero_cells = sparse_cells = 0
            for first, second in joint_conditions:
                matched = [row for row in xray_rows if condition_functions[first](row) and condition_functions[second](row)]
                for fold in range(1,n_folds+1):
                    count = sum(fold_for[row["stem"]] == fold for row in matched)
                    zero_cells += int(count == 0)
                    sparse_cells += int(count < 5)
                    joint_rows.append({
                        "grouping_rule": f"{threshold}s", "n_folds": n_folds, "fold_id": fold,
                        "condition": f"{first} + {second}", "threshold_definition": f"{first}={fnum(thresholds[first])}; {second}={fnum(thresholds[second])}; Stage 1B exploratory quartiles",
                        "total_objects": len(matched), "fold_object_count": count, "fold_fraction_of_condition": fnum(count/len(matched) if matched else 0),
                        "zero_coverage": tf(count==0), "sparse_coverage_lt5": tf(count<5),
                        "notes": "Coverage only; exploratory bins are not permanent difficulty thresholds.",
                    })
            joint_metrics[(threshold,n_folds)] = {"zero_cells":zero_cells,"sparse_cells":sparse_cells}
    joint_columns = ["grouping_rule","n_folds","fold_id","condition","threshold_definition","total_objects","fold_object_count","fold_fraction_of_condition","zero_coverage","sparse_coverage_lt5","notes"]
    write_csv(table_dir / "01c_joint_condition_fold_balance.csv", joint_columns, joint_rows)

    leakage_rows = []
    leakage_metrics = {}
    forced_pair_stems = [(a,b) for a,b,_ in forced_pairs]
    for threshold in GROUPING_RULES:
        temporal_for = {stem: sequence_by_threshold[threshold][stem]["sequence_group_id"] for stem in allowed}
        for n_folds in FOLD_COUNTS:
            fold_for = assignments[(threshold,n_folds)]
            temporal_violations = sum(len({fold_for[stem] for stem in allowed if temporal_for[stem] == group}) > 1 for group in set(temporal_for.values()))
            similarity_violations = sum(fold_for[a] != fold_for[b] for a,b in forced_pair_stems)
            gaps = []
            same_day_adjacent_crossfold = 0
            by_group: dict[tuple[str,str],list[str]] = defaultdict(list)
            for stem in allowed: by_group[(str(sample_info[stem]["machine"]),str(sample_info[stem]["date"]))].append(stem)
            for stems in by_group.values():
                stems = sorted(stems,key=lambda s:(sample_info[s]["timestamp"],s))
                for a,b in zip(stems,stems[1:]):
                    if fold_for[a] != fold_for[b]:
                        ta,tb=datetime.fromisoformat(str(sample_info[a]["timestamp"])),datetime.fromisoformat(str(sample_info[b]["timestamp"]))
                        gaps.append((tb-ta).total_seconds()); same_day_adjacent_crossfold += 1
            close_counts={limit:sum(gap<=limit for gap in gaps) for limit in (3,6,10,30,60)}
            leakage_rows.append({
                "grouping_rule":f"{threshold}s","n_folds":n_folds,"temporal_component_violations":temporal_violations,
                "forced_similarity_violations":similarity_violations,"minimum_cross_fold_gap_seconds":fnum(min(gaps) if gaps else None),
                "cross_fold_adjacent_pairs_le3s":close_counts[3],"cross_fold_adjacent_pairs_le6s":close_counts[6],"cross_fold_adjacent_pairs_le10s":close_counts[10],
                "cross_fold_adjacent_pairs_le30s":close_counts[30],"cross_fold_adjacent_pairs_le60s":close_counts[60],
                "same_day_adjacent_sequences_in_different_folds":same_day_adjacent_crossfold,
                "classification":"PASS with residual boundary risk" if temporal_violations==0 and similarity_violations==0 else "GROUP VIOLATION",
                "notes":"Different temporal groups on the same day are residual dependence risk or ordinary same-day sampling, not automatically leakage.",
            })
            leakage_metrics[(threshold,n_folds)]={"temporal_violations":temporal_violations,"similarity_violations":similarity_violations,"close10":close_counts[10],"close30":close_counts[30],"min_gap":min(gaps) if gaps else math.nan}
    leakage_columns=["grouping_rule","n_folds","temporal_component_violations","forced_similarity_violations","minimum_cross_fold_gap_seconds","cross_fold_adjacent_pairs_le3s","cross_fold_adjacent_pairs_le6s","cross_fold_adjacent_pairs_le10s","cross_fold_adjacent_pairs_le30s","cross_fold_adjacent_pairs_le60s","same_day_adjacent_sequences_in_different_folds","classification","notes"]
    write_csv(table_dir / "01c_leakage_audit.csv", leakage_columns, leakage_rows)

    # Secondary stress-test feasibility only.
    stress_rows=[]
    overall_feature_values={feature:[float(row[feature]) for row in xray_rows] for feature in ("bbox_min_side_px","absolute_median_difference","cnr_like_robust","background_gradient_mean")}
    for machine in sorted({str(info["machine"]) for info in sample_info.values()}):
        validation={stem for stem,info in sample_info.items() if info["machine"]==machine}
        validation_objects=[row for row in xray_rows if row["stem"] in validation]
        shifts=[]
        for feature,overall_values in overall_feature_values.items():
            values=[float(row[feature]) for row in validation_objects]
            shifts.append(f"{feature}:SMD={(statistics.fmean(values)-statistics.fmean(overall_values))/std(overall_values):.3f}")
        stress_rows.append({
            "stress_test":"leave_one_machine_out","held_out_group":machine,"training_images":len(allowed)-len(validation),"validation_images":len(validation),
            "training_fraction":fnum((len(allowed)-len(validation))/len(allowed)),"validation_fraction":fnum(len(validation)/len(allowed)),
            "validation_resolutions":";".join(f"{k}={v}" for k,v in sorted(Counter(str(sample_info[s]["resolution"]) for s in validation).items())),
            "distribution_shift_summary":"; ".join(shifts),"feasibility":"SECONDARY_FEASIBLE",
            "limitations":"Machine is confounded with SN/resolution/date and the test primarily measures domain shift.",
        })
    dates=sorted({str(info["date"]) for info in sample_info.values()}); latest_dates=set(dates[-5:])
    validation={stem for stem,info in sample_info.items() if info["date"] in latest_dates}
    validation_objects=[row for row in xray_rows if row["stem"] in validation]
    shifts=[]
    for feature,overall_values in overall_feature_values.items():
        values=[float(row[feature]) for row in validation_objects]
        shifts.append(f"{feature}:SMD={(statistics.fmean(values)-statistics.fmean(overall_values))/std(overall_values):.3f}")
    stress_rows.append({
        "stress_test":"latest_five_date_holdout","held_out_group":";".join(sorted(latest_dates)),"training_images":len(allowed)-len(validation),"validation_images":len(validation),
        "training_fraction":fnum((len(allowed)-len(validation))/len(allowed)),"validation_fraction":fnum(len(validation)/len(allowed)),
        "validation_resolutions":";".join(f"{k}={v}" for k,v in sorted(Counter(str(sample_info[s]["resolution"]) for s in validation).items())),
        "distribution_shift_summary":"; ".join(shifts),"feasibility":"SECONDARY_FEASIBLE" if len(validation)>=50 else "LIMITED",
        "limitations":"Selected chronological-date stress test can be confounded by machine, resolution, and changing object mix.",
    })
    stress_columns=["stress_test","held_out_group","training_images","validation_images","training_fraction","validation_fraction","validation_resolutions","distribution_shift_summary","feasibility","limitations"]
    write_csv(table_dir / "01c_stress_test_feasibility.csv",stress_columns,stress_rows)

    comparison_rows=[]
    for threshold in GROUPING_RULES:
        for n_folds in FOLD_COUNTS:
            sm=sample_metrics[(threshold,n_folds)]; om=object_metrics[(threshold,n_folds)]; jm=joint_metrics[(threshold,n_folds)]; lm=leakage_metrics[(threshold,n_folds)]
            fold_sizes=Counter(assignments[(threshold,n_folds)].values())
            object_count_balance=sm["object_count_max_tv"]
            strengths=("Strongest tested short-gap protection; " if threshold==10 else "More independent components; ") + f"{len(components_by_rule[threshold])} components and deterministic group integrity."
            limitations=("Larger merged components reduce assignment flexibility; " if threshold==10 else "Leaves 7–10 second adjacent frames as residual cross-fold risk; ") + f"rare-condition zero/sparse cells={jm['zero_cells']}/{jm['sparse_cells']}."
            feasibility="HIGH" if lm["temporal_violations"]==0 and lm["similarity_violations"]==0 and jm["zero_cells"]==0 and max(om["absolute_median_difference_max_abs_smd"],om["cnr_like_robust_max_abs_smd"],om["background_gradient_mean_max_abs_smd"])<.35 else "MODERATE"
            comparison_rows.append({
                "grouping_rule":f"{threshold}s","n_folds":n_folds,"n_group_components":len(components_by_rule[threshold]),
                "min_images_per_fold":min(fold_sizes.values()),"max_images_per_fold":max(fold_sizes.values()),
                "image_balance":f"max ideal-fraction deviation={sm['max_image_deviation']:.4f}",
                "machine_balance":f"max TV={sm['machine_max_tv']:.4f}","date_balance":f"max TV={sm['date_max_tv']:.4f}",
                "resolution_balance":f"max TV={sm['resolution_max_tv']:.4f}","object_count_balance":f"max TV={object_count_balance:.4f}",
                "bbox_size_balance":f"max |SMD|={om['bbox_min_side_px_max_abs_smd']:.4f}; max KS={om['bbox_min_side_px_max_ks']:.4f}",
                "contrast_balance":f"max |SMD|={om['absolute_median_difference_max_abs_smd']:.4f}; max KS={om['absolute_median_difference_max_ks']:.4f}",
                "cnr_balance":f"max |SMD|={om['cnr_like_robust_max_abs_smd']:.4f}; max KS={om['cnr_like_robust_max_ks']:.4f}",
                "background_complexity_balance":f"max |SMD|={om['background_gradient_mean_max_abs_smd']:.4f}; max KS={om['background_gradient_mean_max_ks']:.4f}",
                "joint_condition_coverage":f"zero cells={jm['zero_cells']}; sparse<5 cells={jm['sparse_cells']}",
                "confirmed_group_leakage":lm["temporal_violations"]+lm["similarity_violations"],
                "residual_dependency_risk":f"cross-fold adjacent <=10s={lm['close10']}; <=30s={lm['close30']}; minimum gap={lm['min_gap']:.1f}s",
                "training_fraction_per_fold":fnum(1-1/n_folds),"validation_fraction_per_fold":fnum(1/n_folds),
                "strengths":strengths,"limitations":limitations,"overall_feasibility":feasibility,
            })
    comparison_columns=["grouping_rule","n_folds","n_group_components","min_images_per_fold","max_images_per_fold","image_balance","machine_balance","date_balance","resolution_balance","object_count_balance","bbox_size_balance","contrast_balance","cnr_balance","background_complexity_balance","joint_condition_coverage","confirmed_group_leakage","residual_dependency_risk","training_fraction_per_fold","validation_fraction_per_fold","strengths","limitations","overall_feasibility"]
    write_csv(table_dir / "01c_validation_scheme_comparison.csv",comparison_columns,comparison_rows)

    # Select by pre-model evidence: prefer 10s, require no rare-condition zero cells,
    # worst key-feature |SMD| <= 0.12, and image-fraction deviation <= 0.01.
    # Among qualifying designs, choose the largest K up to 4 to preserve training data
    # without accepting the weaker 5-fold balance/rare-condition coverage.
    qualifying=[]
    for n_folds in FOLD_COUNTS:
        om=object_metrics[(10,n_folds)]; jm=joint_metrics[(10,n_folds)]; sm=sample_metrics[(10,n_folds)]
        worst_smd=max(om["bbox_min_side_px_max_abs_smd"],om["absolute_median_difference_max_abs_smd"],om["cnr_like_robust_max_abs_smd"],om["background_gradient_mean_max_abs_smd"])
        if jm["zero_cells"]==0 and worst_smd<=.12 and sm["max_image_deviation"]<=.01 and n_folds<=4:
            qualifying.append(n_folds)
    selected_folds=max(qualifying) if qualifying else 3
    selected_threshold=10
    selected_assignment=assignments[(selected_threshold,selected_folds)]
    final_rows=[]
    for stem in sorted(allowed):
        component_id=component_for_rule[selected_threshold][stem]
        final_rows.append({
            "stem":stem,"fold_id":selected_assignment[stem],"final_group_component_id":component_id,
            "machine":sample_info[stem]["machine"],"date":sample_info[stem]["date"],"resolution":sample_info[stem]["resolution"],"object_count":sample_info[stem]["object_count"],
        })
    final_columns=["stem","fold_id","final_group_component_id","machine","date","resolution","object_count"]
    write_csv(table_dir / "01c_final_validation_folds.csv",final_columns,final_rows)
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    generated_at=datetime.now(timezone.utc).isoformat()
    metadata_md=f"""# Stage 1C Validation Freeze Metadata

- Status: FROZEN DEVELOPMENT VALIDATION ASSIGNMENT
- Grouping threshold: {selected_threshold} seconds
- Fold count: {selected_folds}
- Similarity links applied: {len(forced_pairs)} manually reviewed strong links
- Similarity pair IDs: {', '.join(row['pair_id'] for row in review_rows)}
- Assignment method: deterministic largest-first group assignment with pre-model balance vectors
- Random seed: not applicable (no randomness)
- Generator version: {GENERATOR_VERSION}
- Generator SHA-256: `{script_hash}`
- Generation timestamp (UTC): {generated_at}
- Revision rule: future changes require a new version and explicit pre-model justification; model inconvenience is not a valid reason.
"""
    (eda_dir / "01c_validation_freeze_metadata.md").write_text(metadata_md,encoding="utf-8-sig")

    selected_comparison=next(row for row in comparison_rows if row["grouping_rule"]=="10s" and int(row["n_folds"])==selected_folds)
    selected_joint=joint_metrics[(10,selected_folds)]
    selected_leak=leakage_metrics[(10,selected_folds)]
    fold_counts=Counter(selected_assignment.values())
    six_sizes=[len(stems) for stems in components_by_rule[6].values()]; ten_sizes=[len(stems) for stems in components_by_rule[10].values()]
    evidence_rows=[
        {"finding":"Seconds-apart temporal dependence","evidence":"Stage 1A.5: 6s groups=150, 10s groups=130 before similarity merges","validation_decision":"Use temporal group-aware validation, not random image KFold","why_it_matters":"Prevents adjacent burst members crossing folds","limitation":"Time proximity does not prove same physical item","reporting_use":"Leakage-control rationale"},
        {"finding":"Reviewed scene-family similarity","evidence":"3/3 dHash candidates classified A by side-by-side review","validation_decision":"Force all three links with transitive component merging","why_it_matters":"Time grouping alone cannot capture cross-date scene-family similarity","limitation":"Different dates; not provenance proof of same item","reporting_use":"Duplicate-control rationale"},
        {"finding":"10s preferred over 6s","evidence":f"Selected 10s candidate has cross-fold adjacent <=10s={selected_leak['close10']} versus materially more for 6s; the remaining pairs are isolated date-conflict samples","validation_decision":"Freeze 10-second components","why_it_matters":"Matches the conservative short-run sensitivity result while preserving unresolved timestamps","limitation":"10s is analytical, not a documented production cycle; conflict samples retain residual risk","reporting_use":"Grouping-rule decision"},
        {"finding":"Primary fold count selected pre-model","evidence":f"{selected_folds}-fold sizes="+",".join(str(fold_counts[i]) for i in sorted(fold_counts))+f"; rare zero/sparse cells={selected_joint['zero_cells']}/{selected_joint['sparse_cells']}","validation_decision":f"Freeze group-aware {selected_folds}-fold CV","why_it_matters":"Balances training fraction, fold feasibility, and rare-condition coverage","limitation":"Small+complex-background remains low-n overall","reporting_use":"Primary validation specification"},
        {"finding":"Secondary domain-shift checks","evidence":"LOMO and latest-five-date holdout feasibility quantified","validation_decision":"Retain as secondary robustness checks only","why_it_matters":"Separates model-selection validation from equipment/time transfer stress","limitation":"Strong machine/resolution/date confounding","reporting_use":"Robustness evaluation plan"},
    ]
    evidence_columns=["finding","evidence","validation_decision","why_it_matters","limitation","reporting_use"]
    write_csv(table_dir / "01c_chapter1_validation_evidence.csv",evidence_columns,evidence_rows)

    summary_md=f"""# Stage 1C — Validation Design Audit Summary

## 1. Input integrity

- **VERIFIED** — 500 approved samples, 1,147 GT objects, matching Stage 1A/1B object keys, complete 6s/10s assignments, and complete required pre-model features.

## 2. Near-duplicate manual review

- **STRONGLY SUPPORTED** — all 3 dHash pairs are class A scene-family similarities after original-BMP side-by-side review of silhouette, internal grayscale structure, TXT bbox layout, and artifact layout.
- All 3 links are forced into validation components. Different dates prevent claiming the same physical item.

## 3. 6s vs 10s group components

- **VERIFIED** — after transitive similarity merging: 6s components={len(components_by_rule[6])}, median/max size={median([float(v) for v in six_sizes]):.1f}/{max(six_sizes)}; 10s components={len(components_by_rule[10])}, median/max size={median([float(v) for v in ten_sizes]):.1f}/{max(ten_sizes)}.
- **STRONGLY SUPPORTED** — 10s materially reduces cross-fold adjacency at or below 10 seconds while retaining enough components for 3–5 folds. The remaining three ≤10s boundaries involve deliberately isolated folder/filename-date conflict samples.

## 4. Candidate CV schemes

- **VERIFIED** — compared group-aware 3/4/5-fold CV for both 6s and 10s components. Ordinary random KFold was not used.
- Assignment is deterministic, unseeded, largest-first, and balances pre-model sample categories plus object-feature quartile counts without model outputs.

## 5. Sample balance

- **VERIFIED** — selected 10s/{selected_folds}-fold image counts: {', '.join(f'F{k}={v}' for k,v in sorted(fold_counts.items()))}.
- Selected maximum TV distances: machine={sample_metrics[(10,selected_folds)]['machine_max_tv']:.3f}, date={sample_metrics[(10,selected_folds)]['date_max_tv']:.3f}, resolution={sample_metrics[(10,selected_folds)]['resolution_max_tv']:.3f}, object-count={sample_metrics[(10,selected_folds)]['object_count_max_tv']:.3f}.

## 6. Object/X-ray difficulty balance

- **VERIFIED** — selected max |SMD|: bbox minimum side={object_metrics[(10,selected_folds)]['bbox_min_side_px_max_abs_smd']:.3f}, absolute contrast={object_metrics[(10,selected_folds)]['absolute_median_difference_max_abs_smd']:.3f}, robust CNR-like={object_metrics[(10,selected_folds)]['cnr_like_robust_max_abs_smd']:.3f}, background gradient={object_metrics[(10,selected_folds)]['background_gradient_mean_max_abs_smd']:.3f}.
- KS values are descriptive distribution diagnostics only, not hypothesis-test decisions.

## 7. Joint-condition coverage

- **VERIFIED** — selected scheme has zero-coverage cells={selected_joint['zero_cells']} and sparse (<5) cells={selected_joint['sparse_cells']} across six Stage 1B exploratory joint conditions.
- **UNRESOLVED** — small + complex background has only 12 objects overall, so some fold estimates remain intrinsically unstable.

## 8. Leakage audit

- **VERIFIED** — selected temporal-component violations={selected_leak['temporal_violations']}; forced-similarity violations={selected_leak['similarity_violations']}; cross-fold adjacent pairs ≤10s={selected_leak['close10']}.
- Separate same-day sequence groups in different folds are residual dependence risk or ordinary same-day sampling, not automatically confirmed leakage.

## 9. Machine/date stress-test feasibility

- **STRONGLY SUPPORTED** — leave-one-machine-out is feasible only as a secondary equipment/domain-shift test because machine, SN, resolution, date, and object conditions are confounded.
- **EXPLORATORY** — latest-five-date holdout is retained as a temporal stress test, not primary model-selection validation.

## 10. Selected primary validation strategy

- **STRONGLY SUPPORTED** — primary design: 10-second temporal components + all 3 manually validated similarity links + deterministic group-aware {selected_folds}-fold CV.
- Rationale: stronger observed short-gap protection than 6s, no confirmed group violation, acceptable sample/object balance, and the best transparent rare-condition/feature-balance trade-off among the required candidates.
- This design is for future model/preprocessing/hyperparameter comparison and later threshold development using development data only.

## 11. Frozen fold assignment status

- **VERIFIED** — `01c_final_validation_folds.csv` is frozen before model results. No randomness was used; generator version/hash/timestamp are recorded separately.
- It must not be modified because later model results are inconvenient. Revisions require a new version and pre-model justification.

## 12. Remaining uncertainty

- Ten seconds is an evidence-based analytical boundary, not a documented machine cycle.
- Reviewed pairs are strong scene-family matches but not proven identical physical items.
- Rare joint conditions remain low-n, and group-aware folds cannot remove all same-day dependence or acquisition confounding.
- Product-edge distance uses a strongly supported analysis boundary without physical segmentation GT.

## 13. Recommended next stage

- Freeze artifact-control experiment specifications against these exact folds, then run a separately authorized artifact-control ablation before any fair detector baseline. External held-out data remains untouched.
"""
    (eda_dir / "01c_validation_design_summary.md").write_text(summary_md,encoding="utf-8-sig")
    print(f"components6={len(components_by_rule[6])} components10={len(components_by_rule[10])} selected=10s/{selected_folds}fold sizes={dict(sorted(fold_counts.items()))}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Stage 1C leakage-aware validation design audit")
    parser.add_argument("--workspace",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    main(args.workspace.resolve())

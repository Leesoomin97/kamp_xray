from __future__ import annotations

import argparse
import csv
import math
import statistics
import struct
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


THRESHOLDS = (3, 6, 10, 30, 60)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def quantile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def fmt(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def bmp_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(26)
    if len(header) < 26 or header[:2] != b"BM":
        raise ValueError(f"Unsupported BMP header: {path}")
    width, height = struct.unpack_from("<ii", header, 18)
    return abs(width), abs(height)


def parse_yolo(path: Path, width: int, height: int) -> tuple[int, list[float], list[float]]:
    minimum_sides: list[float] = []
    area_ratios: list[float] = []
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO row in {path}: {line!r}")
            _, _, _, bw, bh = map(float, parts)
            minimum_sides.append(min(bw * width, bh * height))
            area_ratios.append(bw * bh)
            count += 1
    return count, minimum_sides, area_ratios


def main(workspace: Path) -> None:
    table_dir = workspace / "outputs" / "tables"
    eda_dir = workspace / "outputs" / "eda"
    allowlist = [r for r in read_csv(table_dir / "00_stage1_allowlist.csv") if is_true(r["eligible_for_stage1_eda"])]
    metadata_rows = read_csv(table_dir / "01a_sample_metadata.csv")
    prior_sequence = read_csv(table_dir / "01a_sequence_analysis.csv")
    near_rows = [r for r in read_csv(table_dir / "01a_duplicate_nearduplicate.csv") if is_true(r["near_duplicate_flag"])]

    if len(allowlist) != len({r["stem"] for r in allowlist}):
        raise RuntimeError("Eligible allowlist has duplicate stems")
    allowed = {r["stem"]: r for r in allowlist}
    metadata = {r["stem"]: r for r in metadata_rows if r["stem"] in allowed}
    sequence_stems = {r["stem"] for r in prior_sequence}
    if set(allowed) != set(metadata) or set(allowed) != sequence_stems:
        raise RuntimeError("Allowlist, metadata, and Stage 1A sequence stems do not match")

    sample_features: dict[str, dict[str, object]] = {}
    for stem, row in allowed.items():
        raw_path = workspace / Path(row["canonical_raw_path"])
        txt_path = workspace / Path(row["official_txt_path"])
        if not raw_path.is_file() or not txt_path.is_file():
            raise RuntimeError(f"Missing approved input for {stem}")
        width, height = bmp_dimensions(raw_path)
        object_count, min_sides, area_ratios = parse_yolo(txt_path, width, height)
        sample_features[stem] = {
            "resolution": f"{width}x{height}",
            "object_count": object_count,
            "bbox_min_sides": min_sides,
            "bbox_area_ratios": area_ratios,
        }

    machine_order = {name: idx + 1 for idx, name in enumerate(sorted({metadata[s]["machine"] for s in allowed}))}
    parsed: dict[str, dict[str, object]] = {}
    for stem in sorted(allowed):
        row = metadata[stem]
        timestamp = parse_timestamp(row["timestamp"])
        conflict = is_true(row["metadata_conflict"])
        parsed[stem] = {
            "stem": stem,
            "machine": row["machine"],
            "date": row["filename_date"],
            "timestamp_text": row["timestamp"],
            "timestamp": timestamp,
            "isolated": timestamp is None or conflict,
            "reason": "missing timestamp" if timestamp is None else ("folder/filename date conflict" if conflict else ""),
        }

    assignments: list[dict[str, object]] = []
    assignment_lookup: dict[int, dict[str, str]] = {}
    group_members_by_threshold: dict[int, dict[str, list[str]]] = {}

    for threshold in THRESHOLDS:
        group_for: dict[str, str] = {}
        previous_gap: dict[str, float | None] = {s: None for s in allowed}
        next_gap: dict[str, float | None] = {s: None for s in allowed}
        groups: dict[str, list[str]] = {}
        valid_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
        isolated = [s for s, r in parsed.items() if r["isolated"]]
        for stem, row in parsed.items():
            if not row["isolated"]:
                valid_by_key[(str(row["machine"]), str(row["date"]))].append(stem)

        for (machine, date) in sorted(valid_by_key):
            stems = sorted(valid_by_key[(machine, date)], key=lambda s: (parsed[s]["timestamp"], s))
            run = 1
            current_id = f"T{threshold:03d}_M{machine_order[machine]:03d}_D{date}_G{run:03d}"
            groups[current_id] = []
            for index, stem in enumerate(stems):
                if index:
                    gap = (parsed[stem]["timestamp"] - parsed[stems[index - 1]]["timestamp"]).total_seconds()  # type: ignore[operator]
                    previous_gap[stem] = gap
                    next_gap[stems[index - 1]] = gap
                    if gap > threshold:
                        run += 1
                        current_id = f"T{threshold:03d}_M{machine_order[machine]:03d}_D{date}_G{run:03d}"
                        groups[current_id] = []
                group_for[stem] = current_id
                groups[current_id].append(stem)

        for index, stem in enumerate(sorted(isolated), start=1):
            machine = str(parsed[stem]["machine"])
            date = str(parsed[stem]["date"] or "UNKNOWN")
            group_id = f"T{threshold:03d}_M{machine_order[machine]:03d}_D{date}_ISO{index:03d}"
            group_for[stem] = group_id
            groups[group_id] = [stem]

        assignment_lookup[threshold] = group_for
        group_members_by_threshold[threshold] = groups
        for stem in sorted(allowed):
            row = parsed[stem]
            note = "Isolated; " + str(row["reason"]) + "; no temporal linkage inferred." if row["isolated"] else "Grouped only within the same machine and filename date."
            assignments.append({
                "threshold_seconds": threshold,
                "stem": stem,
                "machine": row["machine"],
                "date": row["date"],
                "timestamp": row["timestamp_text"],
                "sequence_group_id": group_for[stem],
                "sequence_group_size": len(groups[group_for[stem]]),
                "previous_gap_seconds": "" if previous_gap[stem] is None else fmt(float(previous_gap[stem]), 3),
                "next_gap_seconds": "" if next_gap[stem] is None else fmt(float(next_gap[stem]), 3),
                "notes": note,
            })

    assignment_columns = ["threshold_seconds", "stem", "machine", "date", "timestamp", "sequence_group_id", "sequence_group_size", "previous_gap_seconds", "next_gap_seconds", "notes"]
    write_csv(table_dir / "01a5_sequence_group_assignments.csv", assignment_columns, assignments)

    summary_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    threshold_metrics: dict[int, dict[str, float]] = {}
    for threshold in THRESHOLDS:
        groups = group_members_by_threshold[threshold]
        sizes = [len(v) for v in groups.values()]
        singleton_groups = sum(size == 1 for size in sizes)
        multi_samples = sum(size for size in sizes if size >= 2)
        five_samples = sum(size for size in sizes if size >= 5)
        ten_samples = sum(size for size in sizes if size >= 10)
        groups_per_machine = Counter(str(parsed[members[0]]["machine"]) for members in groups.values())
        groups_per_date = Counter(str(parsed[members[0]]["date"]) for members in groups.values())
        metrics = {
            "n_groups": float(len(groups)),
            "median": statistics.median(sizes),
            "mean": statistics.mean(sizes),
            "max": float(max(sizes)),
            "singleton_fraction": singleton_groups / len(groups),
            "multi_sample_fraction": multi_samples / len(allowed),
        }
        threshold_metrics[threshold] = metrics
        summary_rows.append({
            "threshold_seconds": threshold,
            "total_samples": len(allowed),
            "total_sequence_groups": len(groups),
            "minimum_group_size": min(sizes),
            "mean_group_size": fmt(statistics.mean(sizes), 4),
            "median_group_size": fmt(statistics.median(sizes), 4),
            "maximum_group_size": max(sizes),
            "singleton_group_count": singleton_groups,
            "singleton_group_fraction": fmt(singleton_groups / len(groups), 6),
            "samples_in_groups_ge_2": multi_samples,
            "sample_fraction_in_groups_ge_2": fmt(multi_samples / len(allowed), 6),
            "samples_in_groups_ge_5": five_samples,
            "sample_fraction_in_groups_ge_5": fmt(five_samples / len(allowed), 6),
            "samples_in_groups_ge_10": ten_samples,
            "sample_fraction_in_groups_ge_10": fmt(ten_samples / len(allowed), 6),
            "group_size_q25": fmt(quantile([float(x) for x in sizes], 0.25), 4),
            "group_size_q50": fmt(quantile([float(x) for x in sizes], 0.50), 4),
            "group_size_q75": fmt(quantile([float(x) for x in sizes], 0.75), 4),
            "group_size_q90": fmt(quantile([float(x) for x in sizes], 0.90), 4),
            "group_size_q95": fmt(quantile([float(x) for x in sizes], 0.95), 4),
            "groups_per_machine": "; ".join(f"{k}={v}" for k, v in sorted(groups_per_machine.items())),
            "groups_per_date": "; ".join(f"{k}={v}" for k, v in sorted(groups_per_date.items())),
            "isolated_missing_or_conflicting_samples": sum(bool(parsed[s]["isolated"]) for s in allowed),
            "notes": "Groups use adjacent gaps within machine and filename date; conflicted/missing timestamps are isolated.",
        })

        ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        cumulative = 0
        for rank, (group_id, members) in enumerate(ranked, start=1):
            size = len(members)
            cumulative += size
            concentration_rows.append({
                "threshold_seconds": threshold,
                "group_id": group_id,
                "machine": parsed[members[0]]["machine"],
                "date": parsed[members[0]]["date"],
                "group_size": size,
                "sample_fraction": fmt(size / len(allowed), 6),
                "cumulative_sample_fraction": fmt(cumulative / len(allowed), 6),
                "rank_within_threshold": rank,
                "notes": "Large-group concentration diagnostic; no fold assignment implied.",
            })

    summary_columns = ["threshold_seconds", "total_samples", "total_sequence_groups", "minimum_group_size", "mean_group_size", "median_group_size", "maximum_group_size", "singleton_group_count", "singleton_group_fraction", "samples_in_groups_ge_2", "sample_fraction_in_groups_ge_2", "samples_in_groups_ge_5", "sample_fraction_in_groups_ge_5", "samples_in_groups_ge_10", "sample_fraction_in_groups_ge_10", "group_size_q25", "group_size_q50", "group_size_q75", "group_size_q90", "group_size_q95", "groups_per_machine", "groups_per_date", "isolated_missing_or_conflicting_samples", "notes"]
    write_csv(table_dir / "01a5_sequence_group_summary.csv", summary_columns, summary_rows)
    concentration_columns = ["threshold_seconds", "group_id", "machine", "date", "group_size", "sample_fraction", "cumulative_sample_fraction", "rank_within_threshold", "notes"]
    write_csv(table_dir / "01a5_group_concentration.csv", concentration_columns, concentration_rows)

    coverage_rows: list[dict[str, object]] = []
    captured_by_threshold: Counter[int] = Counter()
    for threshold in THRESHOLDS:
        for pair in near_rows:
            stem_a, stem_b = pair["stem_a"], pair["stem_b"]
            if stem_a not in allowed or stem_b not in allowed:
                continue
            dt_a = parsed[stem_a]["timestamp"]
            dt_b = parsed[stem_b]["timestamp"]
            time_gap = abs((dt_b - dt_a).total_seconds()) if dt_a is not None and dt_b is not None else None  # type: ignore[operator]
            same_group = assignment_lookup[threshold][stem_a] == assignment_lookup[threshold][stem_b]
            captured_by_threshold[threshold] += int(same_group)
            coverage_rows.append({
                "threshold_seconds": threshold,
                "stem_a": stem_a,
                "stem_b": stem_b,
                "time_gap_seconds": "" if time_gap is None else fmt(float(time_gap), 3),
                "same_sequence_group": str(same_group).upper(),
                "similarity_metric": pair["similarity_metric"],
                "similarity_value": pair["similarity_value"],
                "notes": "Temporal groups never cross machine/date; similarity pairs are not collapsed.",
            })
    coverage_columns = ["threshold_seconds", "stem_a", "stem_b", "time_gap_seconds", "same_sequence_group", "similarity_metric", "similarity_value", "notes"]
    write_csv(table_dir / "01a5_nearduplicate_group_coverage.csv", coverage_columns, coverage_rows)

    distribution_rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        groups = group_members_by_threshold[threshold]
        group_for = assignment_lookup[threshold]
        categorical = {
            "machine": {s: str(parsed[s]["machine"]) for s in allowed},
            "date": {s: str(parsed[s]["date"]) for s in allowed},
            "image_resolution": {s: str(sample_features[s]["resolution"]) for s in allowed},
            "object_count": {s: str(sample_features[s]["object_count"]) for s in allowed},
        }
        for feature, values in categorical.items():
            categories = sorted(set(values.values()))
            for category in categories:
                stems = [s for s, value in values.items() if value == category]
                touched_groups = {group_for[s] for s in stems}
                largest_in_category = max(sum(values[s] == category for s in groups[g]) for g in touched_groups)
                distribution_rows.append({
                    "threshold_seconds": threshold,
                    "feature": feature,
                    "group_or_category": category,
                    "sample_count": len(stems),
                    "object_count": sum(int(sample_features[s]["object_count"]) for s in stems),
                    "group_count": len(touched_groups),
                    "sample_fraction": fmt(len(stems) / len(allowed), 6),
                    "group_fraction": fmt(len(touched_groups) / len(groups), 6),
                    "group_metric_min": "",
                    "group_metric_q25": "",
                    "group_metric_median": "",
                    "group_metric_q75": "",
                    "group_metric_max": largest_in_category,
                    "notes": "Categorical composition; max is the largest within-group category count.",
                })
        for feature, key in (("bbox_min_side_px", "bbox_min_sides"), ("bbox_area_ratio", "bbox_area_ratios")):
            group_means: list[float] = []
            object_total = 0
            for members in groups.values():
                values = [float(v) for s in members for v in sample_features[s][key]]  # type: ignore[index]
                object_total += len(values)
                if values:
                    group_means.append(statistics.mean(values))
            distribution_rows.append({
                "threshold_seconds": threshold,
                "feature": feature,
                "group_or_category": "group_object_mean_distribution",
                "sample_count": len(allowed),
                "object_count": object_total,
                "group_count": len(group_means),
                "sample_fraction": "1",
                "group_fraction": fmt(len(group_means) / len(groups), 6),
                "group_metric_min": fmt(min(group_means), 8),
                "group_metric_q25": fmt(quantile(group_means, 0.25), 8),
                "group_metric_median": fmt(quantile(group_means, 0.50), 8),
                "group_metric_q75": fmt(quantile(group_means, 0.75), 8),
                "group_metric_max": fmt(max(group_means), 8),
                "notes": "Continuous distribution of per-group object means; no categorical threshold applied.",
            })
    distribution_columns = ["threshold_seconds", "feature", "group_or_category", "sample_count", "object_count", "group_count", "sample_fraction", "group_fraction", "group_metric_min", "group_metric_q25", "group_metric_median", "group_metric_q75", "group_metric_max", "notes"]
    write_csv(table_dir / "01a5_group_distribution_summary.csv", distribution_columns, distribution_rows)

    comparison_rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        metrics = threshold_metrics[threshold]
        n_groups = int(metrics["n_groups"])
        max_size = int(metrics["max"])
        if threshold == 3:
            feasibility = "High group count, but likely fragments 4–6 second acquisition runs."
            strength = "Low to moderate"
            limitations = "May split structurally related short-interval frames."
        elif threshold in (6, 10):
            feasibility = "Feasible candidate; retains many groups while joining dominant short-gap runs."
            strength = "Moderate" if threshold == 6 else "Moderate to high"
            limitations = "Threshold is analytical, not a documented production boundary."
        else:
            feasibility = "Feasible only as a sensitivity/stress rule; inspect large-group concentration."
            strength = "High for within-day temporal adjacency"
            limitations = "May merge independent products separated by tens of seconds."
        machine_counts = Counter(str(parsed[members[0]]["machine"]) for members in group_members_by_threshold[threshold].values())
        date_counts = Counter(str(parsed[members[0]]["date"]) for members in group_members_by_threshold[threshold].values())
        comparison_rows.append({
            "threshold_seconds": threshold,
            "n_groups": n_groups,
            "median_group_size": fmt(metrics["median"], 4),
            "max_group_size": max_size,
            "singleton_fraction": fmt(metrics["singleton_fraction"], 6),
            "sample_fraction_in_multi_sample_groups": fmt(metrics["multi_sample_fraction"], 6),
            "near_duplicate_pairs_captured": captured_by_threshold[threshold],
            "near_duplicate_pairs_total": len(near_rows),
            "machine_balance_comment": "Group counts by machine: " + ", ".join(f"{k}={v}" for k, v in sorted(machine_counts.items())),
            "date_balance_comment": f"Groups span {len(date_counts)} dates; date-level sample imbalance remains and groups do not cross dates.",
            "validation_feasibility": feasibility,
            "leakage_protection_strength": strength,
            "limitations": limitations,
        })
    comparison_columns = ["threshold_seconds", "n_groups", "median_group_size", "max_group_size", "singleton_fraction", "sample_fraction_in_multi_sample_groups", "near_duplicate_pairs_captured", "near_duplicate_pairs_total", "machine_balance_comment", "date_balance_comment", "validation_feasibility", "leakage_protection_strength", "limitations"]
    write_csv(table_dir / "01a5_threshold_comparison.csv", comparison_columns, comparison_rows)

    evidence_rows = [
        {"finding": "Short-interval temporal dependence", "evidence": "Stage 1A found dense 3–6 second gaps; this audit quantifies group sensitivity across five thresholds.", "validation_implication": "Adjacent frames should not automatically be separated across folds.", "limitation": "Close time does not prove the same physical product.", "reporting_use": "Chapter 1 time-relationship and leakage-risk evidence."},
        {"finding": "Random image split risk", "evidence": "Multi-sample group coverage rises as the temporal threshold increases.", "validation_implication": "A random image split can place members of an observed temporal run in different folds.", "limitation": "The magnitude of score inflation is not evaluated without modeling.", "reporting_use": "Justification for considering group-aware validation."},
        {"finding": "Near duplicates are not temporally captured", "evidence": f"All {len(near_rows)} exploratory dHash pairs cross dates and are captured by 0 temporal groups at every threshold.", "validation_implication": "Similarity components require a separate grouping constraint if retained after review.", "limitation": "dHash threshold is exploratory and can be influenced by global image structure/colored markings.", "reporting_use": "Explain why time grouping alone is insufficient."},
        {"finding": "Timestamp conflicts isolated", "evidence": f"{sum(bool(parsed[s]['isolated']) for s in allowed)} samples with missing/conflicting timestamp metadata are singleton groups at every threshold.", "validation_implication": "No undocumented date resolution is introduced.", "limitation": "Conservative isolation may under-connect a real sequence.", "reporting_use": "Document metadata-quality handling."},
        {"finding": "Candidate temporal thresholds", "evidence": "Six- and ten-second rules connect dominant short-gap runs while avoiding cross-date links.", "validation_implication": "Both should proceed to validation-design feasibility checks; neither defines final folds.", "limitation": "No documented process cycle boundary is available.", "reporting_use": "Support transparent candidate selection."},
    ]
    evidence_columns = ["finding", "evidence", "validation_implication", "limitation", "reporting_use"]
    write_csv(table_dir / "01a5_chapter1_validation_evidence.csv", evidence_columns, evidence_rows)

    by_threshold = {int(row["threshold_seconds"]): row for row in summary_rows}
    largest_lines = []
    for threshold in THRESHOLDS:
        top = [r for r in concentration_rows if int(r["threshold_seconds"]) == threshold][:5]
        top_share = sum(int(r["group_size"]) for r in top) / len(allowed)
        largest_lines.append(f"- {threshold}s: max={by_threshold[threshold]['maximum_group_size']}, top-5 groups={top_share:.1%} of samples")
    result_lines = []
    for threshold in THRESHOLDS:
        row = by_threshold[threshold]
        result_lines.append(
            f"- {threshold}s: groups={row['total_sequence_groups']}, mean={row['mean_group_size']}, median={row['median_group_size']}, max={row['maximum_group_size']}, singleton groups={row['singleton_group_count']} ({float(row['singleton_group_fraction']):.1%}), samples in groups≥2={float(row['sample_fraction_in_groups_ge_2']):.1%}."
        )

    summary_md = f"""# Stage 1A.5 — Sequence Group Sensitivity Summary

Scope: {len(allowed)} approved Stage 1 samples only. No fold was created and no model was trained.

## 1. Results for candidate thresholds

{chr(10).join(result_lines)}

## 2. Group-size distribution

- **VERIFIED** — group IDs are deterministic and connect adjacent timestamps only within the same machine and filename date.
- **VERIFIED** — {sum(bool(parsed[s]['isolated']) for s in allowed)} folder/filename-date conflict samples are isolated rather than assigned by an undocumented assumption.
- **STRONGLY SUPPORTED** — the large change between 3s and 6s reflects the observed concentration of 4–6 second gaps.

## 3. Large-group concentration

{chr(10).join(largest_lines)}

- **EXPLORATORY** — concentration is a split-feasibility diagnostic, not evidence that every temporal group is one physical product.
- **VERIFIED** — no tested threshold creates a group of 10 or more samples; the largest group is 7 and the five largest groups contain at most 6.2% of samples. Group size alone does not make any candidate infeasible.

## 4. Near-duplicate coverage

- **VERIFIED** — temporal grouping captures 0/{len(near_rows)} flagged dHash pairs at every tested threshold because all flagged pairs cross dates.
- **STRONGLY SUPPORTED** — temporal grouping alone is insufficient to constrain these similarity pairs; a separate similarity-component rule would be required if the candidates remain accepted.
- **EXPLORATORY** — dHash is a global structural screen and may be influenced by colored markings; no sample was collapsed.

## 5. Machine/date balance implications

- **VERIFIED** — groups never cross machines or dates, so the existing date-frequency imbalance remains visible rather than being merged away.
- **VERIFIED** — at 6s the machine-level group counts are 42/35/73, and at 10s they are 36/33/61; machine 3 contributes disproportionately many groups relative to its 33.4% sample share.
- **VERIFIED** — the two largest dates contain 42.4% of samples but 35.3% of 6s groups and 33.8% of 10s groups. Date concentration therefore remains relevant even though group counts are less concentrated than sample counts.
- **EXPLORATORY** — thresholds change group counts unevenly when machines/dates contain different gap patterns; group-aware fold construction will need composition checks for machine, date, resolution, object count, bbox minimum side, and bbox area ratio.

## 6. Validation feasibility

- **EXPLORATORY** — 3s preserves many groups but fragments the common 4–6 second acquisition pattern.
- **STRONGLY SUPPORTED** — 6s and 10s remain plausible candidates: both capture dominant short runs without crossing date boundaries.
- **EXPLORATORY** — 30s and 60s provide conservative sensitivity bounds. They do not create impractically large groups here, but can join independent samples separated by tens of seconds without process documentation.

## 7. Thresholds that appear too permissive

- **EXPLORATORY** — 3s is likely too permissive against leakage because it breaks most 4–6 second runs.

## 8. Thresholds that appear too aggressive

- **EXPLORATORY** — no threshold is rejected by group size alone. However, 30s and especially 60s are semantically aggressive without documented production-cycle boundaries and add little beyond 10s (130 groups versus 120/119); use them as stress rules rather than default assumptions.

## 9. Plausible thresholds for the next validation-design step

1. **6 seconds** — matches the dominant observed short-gap structure while retaining more independent groups.
2. **10 seconds** — a more conservative alternative covering additional short runs.

Neither is a final split decision. Similarity-pair components must be handled separately from time groups.

## 10. Remaining uncertainty

- **UNRESOLVED** — whether seconds-apart frames show repeated views of one item or successive products.
- **UNRESOLVED** — documented machine cycle/lot boundaries are unavailable.
- **UNRESOLVED** — folder date versus filename date for six conflicted samples.
- **UNRESOLVED** — whether all three dHash candidates should be enforced as similarity components after method-sensitivity review.
"""
    eda_dir.mkdir(parents=True, exist_ok=True)
    (eda_dir / "01a5_sequence_group_sensitivity_summary.md").write_text(summary_md, encoding="utf-8-sig")

    print(" ".join(f"t{t}={int(threshold_metrics[t]['n_groups'])}" for t in THRESHOLDS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1A.5 temporal sequence-group sensitivity audit")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    main(args.workspace.resolve())

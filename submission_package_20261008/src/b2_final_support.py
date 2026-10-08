from __future__ import annotations

from pathlib import Path
from typing import Any

from stage2a_common import EXPECTED_FOLD_SHA256, read_csv, verify_frozen_folds, write_csv


OVERSAMPLING_RULES = {
    "none": "No oversampling; use the frozen train list once per image.",
    "small_low_2x": (
        "Within the training partition only, duplicate once each image containing at least one GT object with "
        "bbox_min_side_px <= 8 and absolute_median_difference <= 4. Validation images are never duplicated."
    ),
}


def _read_split_paths(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_oversampled_training_split(
    *, project_root: Path, base_yaml: Path, representation: str, fold: int, mode: str
) -> tuple[Path, Path | None, dict[str, Any]]:
    """Create a deterministic train-only duplicated-path split for B2 hard-case exposure.

    Duplicate image paths are intentionally retained by Ultralytics' text-list loader. Labels and source images are
    unchanged, and validation membership is copied byte-for-byte from the frozen runtime split.
    """
    if mode not in OVERSAMPLING_RULES:
        raise ValueError(f"Unsupported oversampling mode: {mode}")
    if mode == "none":
        return base_yaml, None, {
            "mode": mode,
            "rule": OVERSAMPLING_RULES[mode],
            "eligible_train_images": 0,
            "base_train_images": None,
            "effective_train_entries": None,
        }

    project_root = project_root.resolve()
    folds, fold_by_stem = verify_frozen_folds(project_root)
    if len(folds) != 500:
        raise RuntimeError("Frozen fold validation failed")
    train_stems = {stem for stem, assigned in fold_by_stem.items() if assigned != str(fold)}
    val_stems = {stem for stem, assigned in fold_by_stem.items() if assigned == str(fold)}

    feature_path = project_root / "outputs" / "tables" / "01b_object_xray_features.csv"
    feature_rows = read_csv(feature_path)
    eligible_all = {
        row["stem"]
        for row in feature_rows
        if float(row["bbox_min_side_px"]) <= 8.0 and float(row["absolute_median_difference"]) <= 4.0
    }
    eligible_train = eligible_all & train_stems
    if eligible_train & val_stems:
        raise RuntimeError("Validation stems entered the oversampling eligibility set")

    split_dir = base_yaml.parent
    base_train_file = split_dir / f"fold_{fold}_train.txt"
    base_val_file = split_dir / f"fold_{fold}_val.txt"
    train_paths = _read_split_paths(base_train_file)
    val_paths = _read_split_paths(base_val_file)
    train_by_stem = {path.stem: path for path in train_paths}
    if set(train_by_stem) != train_stems or len(train_paths) != len(train_stems):
        raise RuntimeError("Base runtime train list differs from frozen fold membership")
    if {path.stem for path in val_paths} != val_stems or len(val_paths) != len(val_stems):
        raise RuntimeError("Base runtime validation list differs from frozen fold membership")
    if not eligible_train <= set(train_by_stem):
        raise RuntimeError("An eligible training stem is missing from the runtime train list")

    output_dir = project_root / "outputs" / "cache" / "b2_final_splits" / representation / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / f"fold_{fold}_train.txt"
    val_file = output_dir / f"fold_{fold}_val.txt"
    yaml_path = output_dir / f"fold_{fold}.yaml"
    expanded_paths = list(train_paths) + [train_by_stem[stem] for stem in sorted(eligible_train)]
    train_file.write_text("\n".join(path.resolve().as_posix() for path in expanded_paths) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(path.resolve().as_posix() for path in val_paths) + "\n", encoding="utf-8")
    yaml_path.write_text(
        f"train: {train_file.resolve().as_posix()}\n"
        f"val: {val_file.resolve().as_posix()}\n"
        "nc: 1\n"
        "names:\n  0: foreign_object\n",
        encoding="utf-8",
    )

    manifest_path = project_root / "outputs" / "tables" / f"08_hardcase_oversampling_manifest_fold{fold}.csv"
    manifest_rows = [
        {
            "stem": stem,
            "fold_id": fold_by_stem[stem],
            "partition": "train",
            "small_low_object_present": stem in eligible_train,
            "base_occurrences": 1,
            "extra_occurrences": 1 if stem in eligible_train else 0,
            "total_occurrences": 2 if stem in eligible_train else 1,
            "oversampling_mode": mode,
            "oversampling_rule": OVERSAMPLING_RULES[mode],
            "frozen_fold_sha256": EXPECTED_FOLD_SHA256,
        }
        for stem in sorted(train_stems)
    ] + [
        {
            "stem": stem,
            "fold_id": fold_by_stem[stem],
            "partition": "validation",
            "small_low_object_present": stem in eligible_all,
            "base_occurrences": 1,
            "extra_occurrences": 0,
            "total_occurrences": 1,
            "oversampling_mode": mode,
            "oversampling_rule": "Validation is never oversampled.",
            "frozen_fold_sha256": EXPECTED_FOLD_SHA256,
        }
        for stem in sorted(val_stems)
    ]
    fields = list(manifest_rows[0])
    write_csv(manifest_path, fields, manifest_rows)

    with train_file.open("r", encoding="utf-8") as handle:
        written_entries = sum(1 for line in handle if line.strip())
    if written_entries != len(train_stems) + len(eligible_train):
        raise RuntimeError("Oversampled train entry count is inconsistent")
    if any(row["partition"] == "validation" and int(row["extra_occurrences"]) != 0 for row in manifest_rows):
        raise RuntimeError("Validation oversampling invariant failed")

    return yaml_path, manifest_path, {
        "mode": mode,
        "rule": OVERSAMPLING_RULES[mode],
        "eligible_train_images": len(eligible_train),
        "base_train_images": len(train_stems),
        "effective_train_entries": written_entries,
        "validation_images": len(val_stems),
        "validation_oversampled": False,
    }

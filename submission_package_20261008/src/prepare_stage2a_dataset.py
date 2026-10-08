from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path


EXPECTED_FOLD_SHA256 = "963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8"
REPRESENTATIONS = {
    "conservative": "ARTIFACT_INPAINT_CONSERVATIVE",
    "local": "ARTIFACT_LOCAL_INTERPOLATION",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) == sha256(source):
            return "existing"
        if source.suffix.lower() != ".png" or target.suffix.lower() != ".png":
            raise RuntimeError(f"Existing target content mismatch: {target}")
        # PNG byte streams can differ across zlib/platform versions while decoding
        # to identical pixels. Accept only exact pixel equivalence, then replace the
        # packaged encoding with the just-generated source so the manifest hash and
        # downstream integrity table remain exact.
        import cv2
        import numpy as np

        source_image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        target_image = cv2.imdecode(np.fromfile(str(target), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if source_image is None or target_image is None or not np.array_equal(source_image, target_image):
            raise RuntimeError(f"Existing target pixel mismatch: {target}")
        target.unlink()
        try:
            os.link(source, target)
            return "replaced_pixel_equivalent_hardlink"
        except OSError:
            shutil.copy2(source, target)
            return "replaced_pixel_equivalent_copy"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main(workspace: Path) -> None:
    tables = workspace / "outputs" / "tables"
    fold_path = tables / "01c_final_validation_folds.csv"
    fold_hash = sha256(fold_path)
    if fold_hash != EXPECTED_FOLD_SHA256:
        raise RuntimeError(f"Frozen fold hash changed: {fold_hash}")

    allow_rows = [r for r in read_csv(tables / "00_stage1_allowlist.csv") if r["eligible_for_stage1_eda"].strip().lower() == "true"]
    folds = read_csv(fold_path)
    manifest = read_csv(tables / "01d_processed_image_manifest.csv")
    if len(allow_rows) != 500 or len({r["stem"] for r in allow_rows}) != 500:
        raise RuntimeError("Allowlist is not 500 unique eligible stems")
    if len(folds) != 500 or {r["stem"] for r in folds} != {r["stem"] for r in allow_rows}:
        raise RuntimeError("Frozen folds do not match allowlist")
    fold_counts = {fold: sum(r["fold_id"] == fold for r in folds) for fold in ("1", "2", "3", "4")}
    if [fold_counts[str(i)] for i in range(1, 5)] != [124, 125, 126, 125]:
        raise RuntimeError(f"Unexpected fold counts: {fold_counts}")

    allow = {r["stem"]: r for r in allow_rows}
    fold_by_stem = {r["stem"]: r["fold_id"] for r in folds}
    manifest_by_key = {(r["stem"], r["representation"]): r for r in manifest}
    root = workspace / "outputs" / "processed_data" / "stage2a"
    integrity: list[dict[str, object]] = []

    for short_name, representation in REPRESENTATIONS.items():
        image_dir, label_dir = root / short_name / "images", root / short_name / "labels"
        image_paths: dict[str, Path] = {}
        for stem in sorted(allow):
            item = manifest_by_key.get((stem, representation))
            if item is None:
                raise RuntimeError(f"Missing manifest row: {stem} {representation}")
            source_image = workspace / Path(item["derived_path"])
            source_label = workspace / Path(allow[stem]["official_txt_path"])
            if not source_image.is_file() or not source_label.is_file():
                raise RuntimeError(f"Missing source image/label: {stem}")
            if sha256(source_image) != item["derived_hash"]:
                raise RuntimeError(f"Derived image hash mismatch: {stem} {representation}")
            target_image, target_label = image_dir / f"{stem}.png", label_dir / f"{stem}.txt"
            image_method = link_or_copy(source_image, target_image)
            label_method = link_or_copy(source_label, target_label)
            image_paths[stem] = target_image.resolve()
            integrity.append({
                "stem": stem, "representation": representation, "fold_id": fold_by_stem[stem],
                "image_path": target_image.relative_to(workspace).as_posix(), "label_path": target_label.relative_to(workspace).as_posix(),
                "image_exists": target_image.is_file(), "label_exists": target_label.is_file(),
                "image_hash_matches_manifest": sha256(target_image) == item["derived_hash"],
                "image_materialization": image_method, "label_materialization": label_method,
                "status": "OK",
            })

        split_dir = root / "splits" / short_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for held_out in range(1, 5):
            train_paths = [str(image_paths[s]).replace("\\", "/") for s in sorted(allow) if fold_by_stem[s] != str(held_out)]
            val_paths = [str(image_paths[s]).replace("\\", "/") for s in sorted(allow) if fold_by_stem[s] == str(held_out)]
            train_file, val_file = split_dir / f"fold_{held_out}_train.txt", split_dir / f"fold_{held_out}_val.txt"
            train_file.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
            val_file.write_text("\n".join(val_paths) + "\n", encoding="utf-8")
            yaml_path = split_dir / f"fold_{held_out}.yaml"
            yaml_path.write_text(
                f"train: {str(train_file.resolve()).replace(chr(92), '/')}\n"
                f"val: {str(val_file.resolve()).replace(chr(92), '/')}\n"
                "nc: 1\n"
                "names:\n  0: foreign_object\n",
                encoding="utf-8",
            )

    fields = ["stem", "representation", "fold_id", "image_path", "label_path", "image_exists", "label_exists", "image_hash_matches_manifest", "image_materialization", "label_materialization", "status"]
    write_csv(tables / "02a_experiment_integrity.csv", fields, integrity)
    print(f"integrity_rows={len(integrity)} fold_hash={fold_hash} fold_counts={fold_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    main(args.workspace.resolve())

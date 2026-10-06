from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from stage2a_common import (
    print_output_manifest, project_root_from_script, read_csv, save_run_manifest,
    sha256, verify_frozen_folds, write_csv,
)


METHODS = {
    "bilateral": {"d": 5, "sigma_color": 10.0, "sigma_space": 3.0},
    "clahe": {"clip_limit": 1.5, "tile_grid_size": [8, 8]},
    "bilateral_clahe": {
        "order": ["bilateral", "clahe"],
        "bilateral": {"d": 5, "sigma_color": 10.0, "sigma_space": 3.0},
        "clahe": {"clip_limit": 1.5, "tile_grid_size": [8, 8]},
    },
    "clahe_bilateral": {
        "order": ["clahe", "bilateral"],
        "clahe": {"clip_limit": 1.5, "tile_grid_size": [8, 8]},
        "bilateral": {"d": 5, "sigma_color": 10.0, "sigma_space": 3.0},
    },
}


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def bilateral(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, d=5, sigmaColor=10.0, sigmaSpace=3.0)


def clahe(image: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(image)


TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "bilateral": bilateral,
    "clahe": clahe,
    "bilateral_clahe": lambda image: clahe(bilateral(image)),
    "clahe_bilateral": lambda image: bilateral(clahe(image)),
}


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != sha256(source):
            raise RuntimeError(f"Existing label differs from source: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main(project_root: Path, materialize: bool) -> None:
    project_root = project_root.resolve()
    _, fold_by_stem = verify_frozen_folds(project_root)
    integrity = [
        row
        for row in read_csv(project_root / "outputs" / "tables" / "02a_experiment_integrity.csv")
        if row["representation"] == "ARTIFACT_INPAINT_CONSERVATIVE"
    ]
    if len(integrity) != 500 or {row["stem"] for row in integrity} != set(fold_by_stem):
        raise RuntimeError("Conservative Stage 6 source coverage differs from frozen corpus")
    rows = []
    for method, parameters in METHODS.items():
        target_root = project_root / "outputs" / "processed_data" / "stage6" / method
        for item in sorted(integrity, key=lambda row: row["stem"]):
            source = project_root / Path(item["image_path"])
            source_label = project_root / Path(item["label_path"])
            output = target_root / "images" / f"{item['stem']}.png"
            output_label = target_root / "labels" / f"{item['stem']}.txt"
            status, output_hash, label_method = "PLANNED", "", "PLANNED"
            if materialize:
                write_png(output, TRANSFORMS[method](read_grayscale(source)))
                label_method = link_or_copy(source_label, output_label)
                output_hash = sha256(output)
                status = "COMPLETE"
            rows.append({
                "stem": item["stem"],
                "fold_id": fold_by_stem[item["stem"]],
                "preprocessing_type": method,
                "parameters_json": json.dumps(parameters, separators=(",", ":")),
                "source_path": source.relative_to(project_root).as_posix(),
                "source_label_path": source_label.relative_to(project_root).as_posix(),
                "output_path": output.relative_to(project_root).as_posix(),
                "output_label_path": output_label.relative_to(project_root).as_posix(),
                "source_sha256": sha256(source),
                "output_sha256": output_hash,
                "label_materialization": label_method,
                "materialization_status": status,
                "dtype": "uint8",
                "format": "lossless PNG",
            })
    manifest_table = project_root / "outputs" / "tables" / "06_preprocessing_manifest.csv"
    write_csv(manifest_table, list(rows[0]), rows)
    if materialize:
        run_name = "stage6_preprocessing_dataset_preparation"
        run_manifest = save_run_manifest(project_root, run_name, {
            "status": "COMPLETE", "stage": "Stage 6 preprocessing dataset preparation", "run_name": run_name,
            "methods": list(METHODS), "source_sample_count": len(fold_by_stem), "derived_image_count": len(rows),
            "generated_metric_paths": [str(manifest_table.resolve())],
        })
        directories = [project_root / "outputs" / "processed_data" / "stage6" / method for method in METHODS]
        print_output_manifest(stage="Stage 6 preprocessing dataset preparation", run_name=run_name, project_root=project_root,
            files=[manifest_table, run_manifest], directories=directories, downloads=[manifest_table, run_manifest],
            next_action="Review checksums and run only one explicitly approved Stage 6 fold experiment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan or materialize Stage 6 mild preprocessing variants.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    main(args.project_root, args.materialize)

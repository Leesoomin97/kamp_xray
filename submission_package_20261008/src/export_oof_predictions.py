from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> None:
    source = args.oof_dir.resolve() / "02a_image_predictions.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    images = read_csv(source)
    if len(images) != 500 or len({row["stem"] for row in images}) != 500:
        raise RuntimeError("OOF image table must contain 500 unique images")

    boxes: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for row in sorted(images, key=lambda item: item["stem"]):
        predictions = json.loads(row["prediction_boxes_json"])
        kept = [item for item in predictions if float(item["confidence"]) >= args.confidence]
        for prediction_id, item in enumerate(kept):
            x1, y1, x2, y2 = map(float, item["bbox"])
            boxes.append({
                "image_id": row["stem"], "fold_id": int(row["fold_id"]),
                "prediction_id": prediction_id, "x1": x1, "y1": y1,
                "x2": x2, "y2": y2, "confidence": float(item["confidence"]),
                "class_id": 0, "class_name": "foreign_object",
            })
        summaries.append({
            "image_id": row["stem"], "fold_id": int(row["fold_id"]),
            "max_confidence": max((float(item["confidence"]) for item in predictions), default=""),
            "prediction_count": len(kept), "reporting_confidence": args.confidence,
        })

    write_csv(args.bbox_output.resolve(),
              ["image_id", "fold_id", "prediction_id", "x1", "y1", "x2", "y2",
               "confidence", "class_id", "class_name"], boxes)
    write_csv(args.image_output.resolve(),
              ["image_id", "fold_id", "max_confidence", "prediction_count", "reporting_confidence"], summaries)
    print(f"bbox_rows={len(boxes)} image_rows={len(summaries)} confidence={args.confidence}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a clearly labelled frozen-fold OOF prediction artifact.")
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--bbox-output", type=Path, required=True)
    parser.add_argument("--image-output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    main(parser.parse_args())

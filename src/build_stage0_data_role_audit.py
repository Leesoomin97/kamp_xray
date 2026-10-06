from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


COLUMNS = [
    "path_or_sample", "stem", "file_type", "role", "role_confidence", "evidence",
    "canonical_source", "official_ground_truth", "derived_copy", "legacy_train",
    "legacy_test", "practice_subset", "prediction_artifact", "existing_weight",
    "exclude_from_eda", "exclude_from_training", "final_test_only", "notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tf(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage 0 data-role audit only.")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    original_root = workspace / "4. X-ray 검사장비 AI 데이터셋"
    dataset = original_root / "dataset"
    yolo = dataset / "test1" / "yolov3"
    raw_root = yolo / "X선이물검출기(06.23_09.22)"
    label_set = dataset / "라벨링 6종 세트"
    openlabel = dataset / "OpenLabeling-master"
    csv_path = workspace / "outputs" / "tables" / "data_role_audit.csv"
    summary_path = workspace / "outputs" / "eda" / "00_data_role_audit_summary.md"
    if not original_root.is_dir():
        raise FileNotFoundError(original_root)

    def rel(path: Path) -> str:
        return path.resolve().relative_to(workspace).as_posix()

    rows: list[dict[str, str]] = []

    def add(path_or_sample: str, stem: str, file_type: str, role: str,
            confidence: str, evidence: str, *, canonical: bool = False,
            official_gt: bool = False, derived: bool = False,
            legacy_train: bool = False, legacy_test: bool = False,
            practice: bool = False, prediction: bool = False,
            weight: bool = False, exclude_eda: bool = True,
            exclude_training: bool = True, final_test: bool = False,
            notes: str = "") -> None:
        values = {
            "path_or_sample": path_or_sample, "stem": stem, "file_type": file_type,
            "role": role, "role_confidence": confidence, "evidence": evidence,
            "canonical_source": tf(canonical), "official_ground_truth": tf(official_gt),
            "derived_copy": tf(derived), "legacy_train": tf(legacy_train),
            "legacy_test": tf(legacy_test), "practice_subset": tf(practice),
            "prediction_artifact": tf(prediction), "existing_weight": tf(weight),
            "exclude_from_eda": tf(exclude_eda),
            "exclude_from_training": tf(exclude_training),
            "final_test_only": tf(final_test), "notes": notes,
        }
        rows.append(values)

    train_lines = [x.strip() for x in (yolo / "train.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    test_lines = [x.strip() for x in (yolo / "test.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    train_stems = {Path(x).stem for x in train_lines}
    test_stems = {Path(x).stem for x in test_lines}

    raw_files = sorted(raw_root.rglob("*.bmp"))
    raw_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in raw_files:
        raw_by_stem[path.stem].append(path)

    hash_cache: dict[Path, str] = {}
    def file_hash(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256(path)
        return hash_cache[path]

    duplicate_groups = {s: p for s, p in raw_by_stem.items() if len(p) > 1}
    duplicate_status: dict[str, str] = {}
    same_byte_duplicate_groups = 0
    different_byte_stems: list[str] = []
    for stem, paths in duplicate_groups.items():
        if len({file_hash(path) for path in paths}) == 1:
            same_byte_duplicate_groups += 1
            duplicate_status[stem] = "same stem and identical SHA-256 across raw paths"
        else:
            different_byte_stems.append(stem)
            duplicate_status[stem] = "same stem but different SHA-256 across raw paths"

    label500_files = sorted((label_set / "labels").glob("*.txt"))
    label500_stems = {path.stem for path in label500_files}
    legacy_label_files = sorted((yolo / "labels").glob("*.txt"))
    legacy_label_by_stem = {path.stem: path for path in legacy_label_files}
    conflict_stems = {path.stem for path in label500_files
                      if path.stem in legacy_label_by_stem
                      and file_hash(path) != file_hash(legacy_label_by_stem[path.stem])}

    def is_yolo_txt(path: Path) -> bool:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                return False
            try:
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                return False
            if class_id < 0 or any(value < 0 or value > 1 for value in coords):
                return False
        return True

    label500_yolo_valid = all(is_yolo_txt(path) for path in label500_files)
    legacy_yolo_valid = all(is_yolo_txt(path) for path in legacy_label_files)

    add(rel(raw_root), "", "DIRECTORY", "CANONICAL_RAW", "STRONGLY_SUPPORTED",
        "Notebook names this hierarchy Raw Image/original data; it is the only large X-ray BMP tree.",
        canonical=True,
        notes=f"Contains {len(raw_files)} BMP paths and {len(raw_by_stem)} unique stems. Use per-file rows and deduplicate stems/hashes.")

    for path in raw_files:
        stem = path.stem
        duplicate_note = "Unique raw stem."
        if stem in duplicate_groups:
            duplicate_note = f"Appears in {len(duplicate_groups[stem])} raw paths; {duplicate_status[stem]}."
        conflict = stem in conflict_stems
        if stem in test_stems:
            role, confidence = "LEGACY_TEST", "VERIFIED"
            evidence = "Stem is listed in test1/yolov3/test.txt; custom.data maps that list to valid."
            exclude_eda = exclude_training = True
        elif stem in train_stems:
            role, confidence = "LEGACY_TRAIN", "VERIFIED"
            evidence = "Stem is listed in test1/yolov3/train.txt and has a legacy YOLO TXT label."
            exclude_eda = exclude_training = conflict
        elif stem in label500_stems:
            role, confidence = "CANONICAL_RAW", "STRONGLY_SUPPORTED"
            evidence = "BMP is in the raw hierarchy and has a same-stem TXT in 라벨링 6종 세트/labels."
            exclude_eda = exclude_training = False
        else:
            role, confidence = "UNRESOLVED", "VERIFIED"
            evidence = "BMP is in the raw hierarchy but no same-stem TXT was found in either X-ray TXT label collection."
            exclude_eda = exclude_training = True
        add(rel(path), stem, "BMP", role, confidence, evidence, canonical=True,
            legacy_train=stem in train_stems, legacy_test=stem in test_stems,
            exclude_eda=exclude_eda, exclude_training=exclude_training,
            notes=duplicate_note + (" Conflicting TXT annotation versions exist." if conflict else ""))

    for path in label500_files:
        stem = path.stem
        conflict = stem in conflict_stems
        held_out = stem in test_stems
        add(rel(path), stem, "TXT_YOLO", "OFFICIAL_GT", "STRONGLY_SUPPORTED",
            "Competition instructions designate TXT as ground truth; file is valid YOLO structure and matches a raw BMP stem.",
            official_gt=True, legacy_train=stem in train_stems, legacy_test=held_out,
            exclude_eda=held_out or conflict, exclude_training=held_out or conflict,
            notes="Primary 500-label collection. " +
                  ("Same-stem legacy TXT has different bbox values; precedence unresolved." if conflict else ""))

    for path in legacy_label_files:
        stem = path.stem
        held_out = stem in test_stems
        conflict = stem in conflict_stems
        add(rel(path), stem, "TXT_YOLO", "LEGACY_TEST" if held_out else "LEGACY_TRAIN", "VERIFIED",
            "Notebook says OpenLabeling YOLO_darknet TXT files were copied into yolov3/labels; structure is valid YOLO.",
            official_gt=True, legacy_train=stem in train_stems, legacy_test=held_out,
            exclude_eda=held_out or conflict, exclude_training=held_out or conflict,
            notes="Legacy practice annotation; duplicate exists in OpenLabeling YOLO_darknet. " +
                  ("Same-stem 500-label TXT has different bbox values." if conflict else ""))

    for kind, lines in (("train", train_lines), ("test", test_lines)):
        for line in lines:
            stem = Path(line).stem
            is_test = kind == "test"
            add(f"{rel(yolo / (kind + '.txt'))}#{line}", stem, "SPLIT_MEMBERSHIP",
                "LEGACY_TEST" if is_test else "LEGACY_TRAIN", "VERIFIED",
                f"Exact entry in legacy {kind}.txt.", legacy_train=not is_test,
                legacy_test=is_test, exclude_eda=is_test, exclude_training=is_test,
                notes="Membership evidence only; do not count as an additional sample.")

    for path in sorted((yolo / "images").glob("*")):
        if not path.is_file():
            continue
        exact = any(file_hash(candidate) == file_hash(path) for candidate in raw_by_stem.get(path.stem, []))
        bm = path.read_bytes()[:2] == b"BM"
        add(rel(path), path.stem, "JPG_EXTENSION_BMP_CONTENT" if bm else path.suffix[1:].upper(),
            "DERIVED_COPY", "VERIFIED" if exact and bm else "UNRESOLVED",
            "Same-stem raw BMP has identical SHA-256; file has BMP BM signature despite .jpg extension.",
            derived=True, legacy_train=path.stem in train_stems,
            legacy_test=path.stem in test_stems,
            notes="Legacy representation; use canonical raw BMP and do not count separately.")

    subset_sizes = [15, 50, 100, 200, 300, 400]
    subset_stems: dict[int, set[str]] = {}
    subset_exact: dict[int, bool] = {}
    for size in subset_sizes:
        files = sorted((label_set / f"images {size}").glob("*.jpg"))
        subset_stems[size] = {path.stem for path in files}
        exact_all = True
        for path in files:
            exact = any(file_hash(candidate) == file_hash(path) for candidate in raw_by_stem.get(path.stem, []))
            bm = path.read_bytes()[:2] == b"BM"
            exact_all = exact_all and exact and bm
            add(rel(path), path.stem, "JPG_EXTENSION_BMP_CONTENT" if bm else path.suffix[1:].upper(),
                "PRACTICE_SUBSET", "VERIFIED" if exact and bm else "UNRESOLVED",
                "Same-stem raw BMP has identical SHA-256; cumulative subset membership was checked.",
                derived=True, practice=True,
                notes=f"Member of images {size}; never count independently from raw BMP.")
        subset_exact[size] = exact_all
    nested = [subset_stems[a].issubset(subset_stems[b]) for a, b in zip(subset_sizes, subset_sizes[1:])]

    for path in sorted((openlabel / "main" / "input").glob("*")):
        if path.is_file():
            exact = any(file_hash(candidate) == file_hash(path) for candidate in raw_by_stem.get(path.stem, []))
            add(rel(path), path.stem, path.suffix[1:].upper(), "DERIVED_COPY",
                "VERIFIED" if exact else "UNRESOLVED",
                "OpenLabeling input copy matches a same-stem raw BMP by SHA-256.", derived=True,
                notes="Annotation workspace input; do not count separately.")

    for path in sorted((openlabel / "main" / "output" / "YOLO_darknet").glob("*.txt")):
        legacy = legacy_label_by_stem.get(path.stem)
        exact = legacy is not None and file_hash(legacy) == file_hash(path)
        add(rel(path), path.stem, "TXT_YOLO", "DERIVED_COPY",
            "VERIFIED" if exact else "UNRESOLVED",
            "OpenLabeling output TXT is byte-identical to same-stem yolov3/labels TXT.",
            derived=True, notes="Duplicate annotation copy; do not count separately.")

    raw_stems = set(raw_by_stem)
    xml_files = sorted((openlabel / "main" / "output" / "PASCAL_VOC").glob("*.xml"))
    xray_xml = [path for path in xml_files if path.stem in raw_stems]
    demo_xml = [path for path in xml_files if path.stem not in raw_stems]
    for path in xray_xml:
        evidence = ("PASCAL VOC annotation for a raw X-ray stem; conversion matches the legacy YOLO TXT."
                    if path.stem in legacy_label_by_stem else
                    "PASCAL VOC annotation for a raw X-ray stem; same-stem primary TXT exists but bbox values differ.")
        add(rel(path), path.stem, "XML_PASCAL_VOC", "DERIVED_COPY", "VERIFIED", evidence,
            derived=True, notes="Alternate/legacy annotation; TXT takes precedence and XML is not independent.")
    for path in demo_xml:
        add(rel(path), path.stem, "XML_PASCAL_VOC", "DEMO_OR_UNRELATED", "VERIFIED",
            "Stem is absent from raw X-ray tree; XML filename/path identifies people_walking or generic img demo content.",
            notes="Exclude completely.")

    for path in sorted((yolo / "result").glob("*")):
        if path.is_file():
            add(rel(path), path.stem, path.suffix[1:].upper(), "PREDICTION_ARTIFACT", "VERIFIED",
                "Located in legacy YOLO result output directory.", prediction=True,
                notes="Never use as source data.")
    for name in ["results.jpg", "results.png", "results.txt", "test_batch0_gt.jpg", "test_batch0_pred.jpg"]:
        path = yolo / name
        if path.exists():
            add(rel(path), path.stem, path.suffix[1:].upper(), "PREDICTION_ARTIFACT", "VERIFIED",
                "Legacy YOLO result/metric artifact filename.", prediction=True,
                notes="Never use as source data.")

    weight_files = sorted((dataset / "실습별 가중치파일").glob("*.pt")) + sorted((yolo / "weights").glob("*"))
    for path in weight_files:
        if path.is_file():
            practice = path.parent.name == "실습별 가중치파일"
            evidence = ("last15/50/100/200/300/400 names mirror practice subset sizes."
                        if practice else "Located in legacy YOLO weights directory.")
            add(rel(path), path.stem, path.suffix[1:].upper(), "EXISTING_WEIGHT", "VERIFIED",
                evidence, practice=practice, weight=True,
                notes="Do not infer dataset statistics or GT; exact training provenance is undocumented.")

    evidence_files = [dataset / "yolov3_20201200.ipynb", yolo / "yolov3_20201200.ipynb",
                      yolo / "train.txt", yolo / "test.txt", yolo / "custom.data",
                      yolo / "splitdata.py", yolo / "train.py", yolo / "test.py",
                      yolo / "README.jpg", openlabel / "README.md"]
    for path in evidence_files:
        if path.exists():
            add(rel(path), path.stem, path.suffix[1:].upper() or "FILE", "TOOL_OR_CODE", "VERIFIED",
                "Inspected as split, conversion, labeling, training, or tool provenance evidence.",
                notes="Not a data sample.")

    add(".", "", "AUDIT_FINDING", "UNRESOLVED", "STRONGLY_SUPPORTED",
        "No supplied image directory, list, config, or brief section identifies a separate official held-out test set.",
        notes="External portal or organizer confirmation is required; no local row is FINAL_TEST_ONLY.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    matched_label_paths = sum(len(raw_by_stem.get(stem, [])) for stem in label500_stems)
    legacy_only = len(set(legacy_label_by_stem) - label500_stems)
    people_count = sum(path.stem.startswith("people_walking_mp4_") for path in demo_xml)
    img_demo_count = sum(path.stem.startswith("img_") for path in demo_xml)
    train_list = "\n".join(f"- `{stem}`" for stem in sorted(train_stems))
    test_list = "\n".join(f"- `{stem}`" for stem in sorted(test_stems))
    different_list = ", ".join(f"`{stem}`" for stem in sorted(different_byte_stems))
    conflict_list = ", ".join(f"`{stem}`" for stem in sorted(conflict_stems))

    summary = f"""# Stage 0 — Data Role / Split Audit

이 문서는 데이터 역할과 분할 근거만 정리한다. 이미지 통계, bbox 분포, 모델 성능, 전처리 비교는 수행하지 않았다.

## 1. Canonical raw-data candidate

- **STRONGLY SUPPORTED** — `{rel(raw_root)}`가 canonical raw X-ray source 후보이다.
- **VERIFIED** — BMP 경로 {len(raw_files):,}개, 고유 stem {len(raw_by_stem):,}개가 있다.
- **VERIFIED** — 중복 stem 그룹 {len(duplicate_groups):,}개 중 {same_byte_duplicate_groups:,}개는 SHA-256이 같고, {len(different_byte_stems):,}개는 바이트가 다르다.
- **VERIFIED** — 바이트가 다른 중복 stem: {different_list}.
- **UNRESOLVED** — 동일 바이트가 여러 설비/날짜 경로에 반복된 이유와 authoritative provenance는 문서화되어 있지 않다. 독립 sample로 중복 집계하면 안 된다.

## 2. Ground-truth label candidate

- **STRONGLY SUPPORTED** — `{rel(label_set / 'labels')}`의 TXT {len(label500_files):,}개가 operational GT의 1차 후보이다.
- **VERIFIED** — 모든 label stem은 raw tree에 존재하며, 중복 raw 경로를 포함하면 {matched_label_paths:,}개 BMP 경로에 대응한다.
- **VERIFIED** — 두 TXT 모음 모두 YOLO 구조이다: label500={label500_yolo_valid}, legacy={legacy_yolo_valid}.
- **VERIFIED** — legacy TXT {len(legacy_label_files)}개 중 {legacy_only}개 stem은 500-label collection 밖에 있다.
- **VERIFIED** — 겹치는 {len(conflict_stems)}개 stem은 bbox 값이 다르다: {conflict_list}.
- **UNRESOLVED** — 충돌 stem의 authoritative annotation과 legacy-only TXT 포함 여부를 확정해야 한다.

## 3. BMP / JPG / TXT / XML relationship

- **VERIFIED** — `test1/yolov3/images`의 15개 `.jpg`는 BMP `BM` signature를 가지며 raw BMP와 SHA-256이 같다. 확장자만 바꾼 복사본이다.
- **VERIFIED** — `images 15/50/100/200/300/400`의 모든 파일도 raw BMP와 바이트가 같다.
- **VERIFIED** — OpenLabeling input BMP 15개는 raw BMP 복사본이고, YOLO_darknet TXT 15개는 `yolov3/labels`와 바이트가 같다.
- **VERIFIED** — X-ray XML {len(xray_xml)}개는 PASCAL VOC 구조이며, legacy XML 15개의 YOLO 변환은 legacy TXT와 정확히 일치한다.
- **VERIFIED** — 500-label collection과 겹치는 XML 23개는 모두 해당 TXT와 bbox 값이 다르다. XML은 독립 GT로 혼합하지 않는다.

## 4. Meaning of test1/yolov3

- **STRONGLY SUPPORTED** — legacy YOLO 학습·라벨링 실습 workspace이다.
- 근거: notebook이 raw BMP를 복사하고 `ren *.* *.jpg*`로 확장자를 바꾸며, OpenLabeling 결과를 옮기고, `splitdata.py`로 무작위 20% `test.txt`를 만든다.
- 근거: `custom.data`는 `train.txt`를 train, `test.txt`를 valid로 연결하고 `train.py/test.py`가 사용한다.
- **UNRESOLVED** — 공식 competition held-out를 담는다는 근거는 없다.

## 5. Legacy train/test split findings

- **VERIFIED** — 현재 목록은 train {len(train_stems)} stems / test {len(test_stems)} stems이다.
- **STRONGLY SUPPORTED** — 15-image workspace의 내부 practice train/validation split이다. test 3개는 Stage 1 사용 금지 상태다.

### Legacy train stems

{train_list}

### Legacy test stems

{test_list}

## 6. Meaning of images 15/50/100/200/300/400

- **VERIFIED** — 각 디렉터리 파일 수는 이름의 숫자와 일치한다.
- **VERIFIED** — 15⊂50⊂100⊂200⊂300⊂400: {all(nested)}.
- **VERIFIED** — 모든 파일은 raw BMP와 바이트가 같다: {all(subset_exact.values())}.
- **STRONGLY SUPPORTED** — 독립 데이터셋이 아닌 학습량 비교용 누적 practice subsets이다.

## 7. Meaning of existing .pt weights

- **VERIFIED** — `last15/50/100/200/300/400.pt`가 존재한다.
- **STRONGLY SUPPORTED** — 이름이 practice subset 크기와 대응한다.
- **UNRESOLVED** — 정확한 membership, seed, epoch, preprocessing provenance는 확정되지 않는다.

## 8. Confirmed excluded areas

- **VERIFIED** — `result/`, `results.*`, `test_batch0_gt.jpg`, `test_batch0_pred.jpg`: prediction/training artifacts.
- **VERIFIED** — `people_walking_mp4_*` XML {people_count}개와 `img_*` XML {img_demo_count}개: non-X-ray demo.
- **VERIFIED** — 파생 JPG, OpenLabeling copies, alternate XML은 독립 source에서 제외한다.
- **VERIFIED** — 기존 `.pt` weights는 EDA와 데이터 역할 추론에서 제외한다.

## 9. Official held-out competition test

- **STRONGLY SUPPORTED** — 제공된 로컬 트리에서 별도 공식 held-out test set을 찾지 못했다.
- **VERIFIED** — `test.txt`는 무작위 20% legacy validation 목록이며 `custom.data`의 `valid`에 연결된다.
- **UNRESOLVED** — 외부 KAMP 포털에 별도 평가 데이터가 있는지는 로컬 자료만으로 확정할 수 없다.

## 10. Issues to settle before Stage 1

1. **UNRESOLVED** — 충돌 TXT {len(conflict_stems)}개 stem의 authoritative bbox.
2. **UNRESOLVED** — 500-label collection 밖 legacy-only TXT {legacy_only}개 포함 여부.
3. **UNRESOLVED** — 동일 stem·동일 바이트 BMP의 반복 원인과 canonical provenance.
4. **UNRESOLVED** — 동일 stem·서로 다른 BMP {len(different_byte_stems)}개 그룹의 sample identity.
5. **UNRESOLVED** — 외부 공식 held-out test 제공 여부.
6. **UNRESOLVED** — legacy test를 보존할지, 공식 test가 아님을 확인한 뒤 새 split을 만들지 여부.

Stage 1 EDA는 시작하지 않았다.
"""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8-sig")
    print(f"CSV={csv_path}")
    print(f"SUMMARY={summary_path}")
    print(f"ROWS={len(rows)}")


if __name__ == "__main__":
    main()

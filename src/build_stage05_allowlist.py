from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tf(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the Stage 1 labeled-data allowlist only.")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    original_root = workspace / "4. X-ray 검사장비 AI 데이터셋"
    dataset = original_root / "dataset"
    yolo = dataset / "test1" / "yolov3"
    raw_root = yolo / "X선이물검출기(06.23_09.22)"
    primary_label_dir = dataset / "라벨링 6종 세트" / "labels"
    subset_root = dataset / "라벨링 6종 세트"
    legacy_label_dir = yolo / "labels"
    xml_dir = dataset / "OpenLabeling-master" / "main" / "output" / "PASCAL_VOC"
    stage0_table = workspace / "outputs" / "tables" / "data_role_audit.csv"
    stage0_summary = workspace / "outputs" / "eda" / "00_data_role_audit_summary.md"

    # Stage 0 outputs are required inputs to this stage.
    with stage0_table.open("r", encoding="utf-8-sig", newline="") as handle:
        stage0_rows = list(csv.DictReader(handle))
    stage0_summary_text = stage0_summary.read_text(encoding="utf-8-sig")
    if not stage0_rows or "Stage 0" not in stage0_summary_text:
        raise RuntimeError("Stage 0 outputs are missing or invalid")

    def rel(path: Path) -> str:
        return path.resolve().relative_to(workspace).as_posix()

    hash_cache: dict[Path, str] = {}
    def file_hash(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256(path)
        return hash_cache[path]

    raw_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(raw_root.rglob("*.bmp")):
        raw_by_stem[path.stem].append(path)

    primary_labels = {path.stem: path for path in sorted(primary_label_dir.glob("*.txt"))}
    legacy_labels = {path.stem: path for path in sorted(legacy_label_dir.glob("*.txt"))}
    conflict_stems = sorted(
        stem for stem in primary_labels.keys() & legacy_labels.keys()
        if file_hash(primary_labels[stem]) != file_hash(legacy_labels[stem])
    )
    legacy_only_stems = sorted(legacy_labels.keys() - primary_labels.keys())
    train_stems = {
        Path(line.strip()).stem for line in (yolo / "train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    test_stems = {
        Path(line.strip()).stem for line in (yolo / "test.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    subset_membership: dict[str, list[int]] = defaultdict(list)
    for size in [15, 50, 100, 200, 300, 400]:
        for path in (subset_root / f"images {size}").glob("*.jpg"):
            subset_membership[path.stem].append(size)

    candidate_columns = [
        "stem", "txt_path", "raw_match_count", "canonical_raw_path",
        "duplicate_raw_paths", "raw_byte_identical", "label_status",
        "eligibility_status", "notes",
    ]
    candidate_rows: list[dict[str, str]] = []
    duplicate_columns = [
        "stem", "raw_path_count", "exact_duplicate", "byte_hashes",
        "candidate_paths", "logical_sample_count", "identity_status", "notes",
    ]
    duplicate_rows: list[dict[str, str]] = []
    allowlist_columns = [
        "stem", "canonical_raw_path", "official_txt_path", "eligible_for_stage1_eda",
        "reason", "duplicate_group", "machine", "date",
        "sequence_or_timestamp_if_available", "label_provenance", "identity_status", "notes",
    ]
    allowlist_rows: list[dict[str, str]] = []

    exact_duplicate_candidate_count = 0
    for stem, txt_path in sorted(primary_labels.items()):
        matches = sorted(raw_by_stem.get(stem, []), key=lambda value: rel(value))
        hashes = sorted({file_hash(path) for path in matches})
        byte_identical = len(matches) > 0 and len(hashes) == 1
        if not matches:
            canonical = ""
            duplicates = ""
            identity_status = "MISSING_RAW"
            label_status = "PRIMARY_GT_RAW_MISSING"
            eligibility = "BLOCKED"
            logical_count = 0
            notes = "No same-stem raw BMP was found."
        elif len(hashes) > 1:
            canonical = ""
            duplicates = ";".join(rel(path) for path in matches)
            identity_status = "SAME_STEM_BYTE_DIFFERENT_UNRESOLVED"
            label_status = "PRIMARY_GT_RAW_IDENTITY_UNRESOLVED"
            eligibility = "BLOCKED"
            logical_count = len(hashes)
            notes = "No canonical raw path selected because same-stem BMP content differs."
        else:
            canonical = rel(matches[0])
            duplicates = ";".join(rel(path) for path in matches[1:])
            identity_status = "EXACT_DUPLICATE_COLLAPSED" if len(matches) > 1 else "UNIQUE_RAW_MATCH"
            if len(matches) > 1:
                exact_duplicate_candidate_count += 1
            label_status = "PRIMARY_GT_CONFLICT_RESOLVED" if stem in conflict_stems else "PRIMARY_GT"
            eligibility = "ELIGIBLE"
            logical_count = 1
            notes = (
                "Byte-identical raw paths represent one logical sample; canonical path is the deterministic lexicographic representative."
                if len(matches) > 1 else "Single matching raw BMP."
            )
            if stem in conflict_stems:
                notes += " Primary 500-label TXT selected over legacy TXT with STRONGLY SUPPORTED confidence."

        candidate_rows.append({
            "stem": stem,
            "txt_path": rel(txt_path),
            "raw_match_count": str(len(matches)),
            "canonical_raw_path": canonical,
            "duplicate_raw_paths": duplicates,
            "raw_byte_identical": (
                tf(byte_identical) if len(matches) > 1 else "NOT_APPLICABLE"
            ),
            "label_status": label_status,
            "eligibility_status": eligibility,
            "notes": notes,
        })
        duplicate_rows.append({
            "stem": stem,
            "raw_path_count": str(len(matches)),
            "exact_duplicate": tf(len(matches) > 1 and byte_identical),
            "byte_hashes": ";".join(hashes),
            "candidate_paths": ";".join(rel(path) for path in matches),
            "logical_sample_count": str(logical_count),
            "identity_status": identity_status,
            "notes": notes,
        })

        machines: list[str] = []
        for path in matches:
            for part in path.parts:
                if re.match(r"^\d+호기\(", part) and part not in machines:
                    machines.append(part)
        match = re.match(r"^\d{3}_(\d{8})_(\d{6})\(([^)]*)\)$", stem)
        date = match.group(1) if match else ""
        timestamp = match.group(2) if match else ""
        conflict = stem in conflict_stems
        reason = (
            "Primary 500-label TXT selected by consolidated-collection, subset-membership, and later-file provenance; raw identity resolved."
            if conflict else
            "Primary 500-label TXT has a resolved one-logical-sample raw match."
        )
        allowlist_rows.append({
            "stem": stem,
            "canonical_raw_path": canonical,
            "official_txt_path": rel(txt_path),
            "eligible_for_stage1_eda": tf(eligibility == "ELIGIBLE"),
            "reason": reason if eligibility == "ELIGIBLE" else notes,
            "duplicate_group": f"{stem}|sha256:{hashes[0]}" if len(hashes) == 1 else stem,
            "machine": ";".join(machines),
            "date": date,
            "sequence_or_timestamp_if_available": timestamp,
            "label_provenance": (
                "PRIMARY_500_LABEL_COLLECTION_SELECTED_OVER_LEGACY"
                if conflict else "PRIMARY_500_LABEL_COLLECTION"
            ),
            "identity_status": identity_status,
            "notes": (
                ("Legacy train/test membership is obsolete and does not restrict this sample. "
                 if stem in train_stems or stem in test_stems else "No legacy train/test membership. ")
                + ("Parenthetical filename suffix meaning is unresolved." if match else "Filename metadata pattern unresolved.")
            ),
        })

    conflict_columns = [
        "stem", "primary_txt_path", "primary_bbox_values", "legacy_txt_path",
        "legacy_bbox_values", "related_xml", "xml_bbox_values", "subset_membership",
        "primary_txt_mtime", "legacy_txt_mtime", "xml_mtime", "provenance_evidence",
        "recommended_authoritative_txt", "confidence", "resolution_status", "notes",
    ]
    conflict_rows: list[dict[str, str]] = []
    for stem in conflict_stems:
        primary = primary_labels[stem]
        legacy = legacy_labels[stem]
        xml_path = xml_dir / f"{stem}.xml"
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        xml_values = []
        for obj in tree.getroot().findall("object"):
            box = obj.find("bndbox")
            xml_values.append(
                f"{obj.findtext('name')} xmin={box.findtext('xmin')} ymin={box.findtext('ymin')} "
                f"xmax={box.findtext('xmax')} ymax={box.findtext('ymax')}"
            )
        evidence = (
            "The XML converts exactly to the legacy TXT, so XML and legacy TXT are one annotation lineage, not independent votes. "
            "The primary TXT belongs to the dedicated 500-label collection, its image appears in cumulative practice subsets "
            f"{subset_membership.get(stem, [])}, and its file timestamp is later than the legacy TXT/XML. "
            "This supports the primary collection as the later consolidated annotation source; timestamps are supporting "
            "file metadata and are not treated alone as verified annotation chronology."
        )
        conflict_rows.append({
            "stem": stem,
            "primary_txt_path": rel(primary),
            "primary_bbox_values": " | ".join(line.strip() for line in primary.read_text(encoding="utf-8").splitlines() if line.strip()),
            "legacy_txt_path": rel(legacy),
            "legacy_bbox_values": " | ".join(line.strip() for line in legacy.read_text(encoding="utf-8").splitlines() if line.strip()),
            "related_xml": rel(xml_path),
            "xml_bbox_values": " | ".join(xml_values),
            "subset_membership": ";".join(str(size) for size in subset_membership.get(stem, [])),
            "primary_txt_mtime": datetime.fromtimestamp(primary.stat().st_mtime).astimezone().isoformat(),
            "legacy_txt_mtime": datetime.fromtimestamp(legacy.stat().st_mtime).astimezone().isoformat(),
            "xml_mtime": datetime.fromtimestamp(xml_path.stat().st_mtime).astimezone().isoformat(),
            "provenance_evidence": evidence,
            "recommended_authoritative_txt": rel(primary),
            "confidence": "STRONGLY SUPPORTED",
            "resolution_status": "RESOLVED_FOR_STAGE1_ALLOWLIST",
            "notes": "Original TXT/XML files remain unchanged; this is an operational provenance decision, not a claim of visual correctness.",
        })

    legacy_columns = [
        "stem", "legacy_txt_path", "raw_match", "raw_path", "related_xml",
        "evidence", "recommended_status", "confidence", "notes",
    ]
    legacy_rows: list[dict[str, str]] = []
    for stem in legacy_only_stems:
        label = legacy_labels[stem]
        matches = sorted(raw_by_stem.get(stem, []), key=lambda value: rel(value))
        xml_path = xml_dir / f"{stem}.xml"
        exact = len({file_hash(path) for path in matches}) <= 1
        evidence = (
            "Raw BMP exists; a PASCAL VOC XML exists and converts to this legacy TXT. "
            "Notebook places the TXT in the 15-image OpenLabeling/YOLO practice workflow. "
            "Stem is absent from the dedicated 500-label collection and all cumulative images 15/50/100/200/300/400 subsets."
        )
        legacy_rows.append({
            "stem": stem,
            "legacy_txt_path": rel(label),
            "raw_match": tf(bool(matches)),
            "raw_path": ";".join(rel(path) for path in matches),
            "related_xml": rel(xml_path) if xml_path.exists() else "",
            "evidence": evidence,
            "recommended_status": "EXCLUDE_FROM_AUTHORITATIVE_DEVELOPMENT",
            "confidence": "STRONGLY SUPPORTED",
            "notes": (
                "Reason for omission from the 500-label collection is undocumented. Excluded conservatively; not asserted to be an invalid annotation. "
                + ("Multiple raw paths are byte-identical." if len(matches) > 1 and exact else "")
            ),
        })

    outputs = workspace / "outputs" / "tables"
    write_csv(outputs / "00_labeled_development_candidates.csv", candidate_columns, candidate_rows)
    write_csv(outputs / "00_label_conflicts.csv", conflict_columns, conflict_rows)
    write_csv(outputs / "00_legacy_only_labels.csv", legacy_columns, legacy_rows)
    write_csv(outputs / "00_duplicate_identity_audit.csv", duplicate_columns, duplicate_rows)
    write_csv(outputs / "00_stage1_allowlist.csv", allowlist_columns, allowlist_rows)

    eligible_count = sum(row["eligible_for_stage1_eda"] == "TRUE" for row in allowlist_rows)
    blocked_primary_count = len(allowlist_rows) - eligible_count
    unresolved_primary_count = sum("UNRESOLVED" in row["identity_status"] for row in allowlist_rows)
    duplicate_path_count = sum(int(row["raw_path_count"]) - 1 for row in duplicate_rows)
    hash_to_stems: dict[str, list[str]] = defaultdict(list)
    for row in duplicate_rows:
        if row["logical_sample_count"] == "1" and ";" not in row["byte_hashes"]:
            hash_to_stems[row["byte_hashes"]].append(row["stem"])
    cross_stem_hash_groups = [stems for stems in hash_to_stems.values() if len(stems) > 1]
    conflict_list = ", ".join(f"`{stem}`" for stem in conflict_stems)
    legacy_list = ", ".join(f"`{stem}`" for stem in legacy_only_stems)

    summary = f"""# Stage 0.5 — Stage 1 Allowlist Finalization

이 단계는 labeled development membership과 identity만 확정한다. 이미지 통계, bbox 분포, 시각화, 모델링, 최종 split 생성은 수행하지 않았다.

## 1. Final authoritative labeled development sample count

- **STRONGLY SUPPORTED** — Stage 1 허용 logical samples는 **{eligible_count}개**이다.
- **VERIFIED** — 500-label collection의 모든 stem이 raw BMP에 대응하며, byte-different raw ambiguity는 이 500개 안에 없다.

## 2. Excluded and unresolved sample counts

- **STRONGLY SUPPORTED** — TXT-labeled stem 전체 후보 512개 중 primary collection 500개는 허용하고 legacy-only 12개는 제외한다.
- **VERIFIED** — excluded samples: **{len(legacy_rows)}개**. Primary 500개 중 blocked: **{blocked_primary_count}개**.
- **VERIFIED** — Primary 500개 중 unresolved identity/label samples: **{unresolved_primary_count}개**.

## 3. Label conflict resolution

- **STRONGLY SUPPORTED** — 충돌 3개 모두 `라벨링 6종 세트/labels` TXT를 operational authoritative version으로 선택했다: {conflict_list}.
- XML은 legacy TXT로 정확히 변환되므로 독립적인 두 번째 근거가 아니다. Primary TXT는 전용 500-label collection, 누적 subset membership, 더 늦은 file timestamp로 뒷받침된다. Timestamp는 보조 metadata이며 단독으로 annotation 생성 순서를 증명하지 않는다.
- 원본 TXT/XML은 수정하지 않았다. 결정은 provenance 기반이며 bbox의 시각적 정확성을 새로 주장하지 않는다.

## 4. Legacy-only TXT decision

- **STRONGLY SUPPORTED** — legacy-only 12개를 authoritative development corpus에서 제외한다: {legacy_list}.
- **VERIFIED** — 모두 raw BMP 및 legacy XML과 대응하지만, notebook의 15-image practice workflow 소속이고 primary 500-label collection 및 누적 practice subsets에는 없다.
- **UNRESOLVED** — primary collection에서 빠진 구체적인 이유는 문서화되어 있지 않다.

## 5. Legacy train/test split decision

- **STRONGLY SUPPORTED** — `train.txt/test.txt`는 **B. legacy practice/internal split**이다.
- **VERIFIED** — `splitdata.py`와 notebook이 15개 이미지에서 무작위 20% test를 만들고, `custom.data`가 이를 `valid`로 연결한다.
- 기존 split은 프로젝트 기준으로 obsolete이다. Old test membership만으로 샘플을 특별 test로 보존하지 않는다.
- Primary collection에 속한 old-test 2개는 allowlist에 포함되고, legacy-only old-test 1개는 legacy-only 정책으로 제외된다.

## 6. Duplicate and sample-identity findings

- **VERIFIED** — Primary 500 stems 중 raw 경로가 반복되는 그룹은 **{exact_duplicate_candidate_count}개**이며 모두 그룹 내부 SHA-256이 같다.
- **VERIFIED** — 이 반복 경로로 생기는 추가 physical paths는 **{duplicate_path_count}개**이며 각각 한 logical sample로 축약했다.
- **VERIFIED** — Primary 500개에는 same-stem/byte-different 그룹이 없다.
- **VERIFIED** — Primary 500개 canonical content hash 간 cross-stem exact duplicate 그룹은 **{len(cross_stem_hash_groups)}개**이다.
- Canonical raw path는 byte-identical 그룹 안에서 재현 가능한 lexicographic representative로 기록했다. 다른 경로는 audit table에 보존했다.

## 7. Exact Stage 1 allowlist scope

- **VERIFIED** — `00_stage1_allowlist.csv`의 `eligible_for_stage1_eda=TRUE`인 500행만 Stage 1 대상이다.
- Canonical input은 raw BMP이고 official label은 `라벨링 6종 세트/labels` TXT이다.
- Derived JPG, XML, OpenLabeling copies, result artifacts, existing weights, legacy-only TXT는 범위 밖이다.
- 이 allowlist는 train/validation/test split이 아니다.

## 8. Samples still blocked from Stage 1

- **STRONGLY SUPPORTED** — legacy-only TXT 12개는 authoritative membership 근거 부족으로 차단했다.
- **VERIFIED** — raw tree의 unlabeled BMP와 primary 500 밖의 same-stem/byte-different 세 그룹은 labeled Stage 1 allowlist에 포함되지 않는다.

## 9. Official held-out competition test

- **STRONGLY SUPPORTED** — 제공된 로컬 자료에서 별도 official held-out set은 발견되지 않았다.
- **UNRESOLVED** — 외부 KAMP 포털 또는 운영기관이 별도 test data를 제공하는지는 확인이 필요하다.

## 10. Remaining unresolved issues before Stage 1

1. **UNRESOLVED** — 외부 official held-out 제공 여부.
2. **UNRESOLVED** — byte-identical BMP가 여러 설비 경로에 반복된 원래의 복제 경위. 현재는 hash 기반으로 한 logical sample로 안전하게 처리했다.
3. **UNRESOLVED** — filename 괄호 값의 공식 의미. Allowlist에서는 의미를 추정하지 않았다.
4. **UNRESOLVED** — legacy-only 12개가 primary collection에서 빠진 원래 사유. 현재 Stage 1 범위에서는 제외했다.

## 11. Recommended next step

- 사용자가 이 500-sample allowlist와 provenance 결정을 승인한 뒤에만 Stage 1 EDA를 별도 요청으로 시작한다.
- 최종 train/validation/test split은 아직 만들지 않는다.
"""
    summary_path = workspace / "outputs" / "eda" / "00_stage1_allowlist_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8-sig")

    print(f"eligible={eligible_count}")
    print(f"excluded_legacy_only={len(legacy_rows)}")
    print(f"conflicts={len(conflict_rows)}")
    print(f"duplicate_groups={exact_duplicate_candidate_count}")


if __name__ == "__main__":
    main()

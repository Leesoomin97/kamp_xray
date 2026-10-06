# Stage 0.5 — Stage 1 Allowlist Finalization

이 단계는 labeled development membership과 identity만 확정한다. 이미지 통계, bbox 분포, 시각화, 모델링, 최종 split 생성은 수행하지 않았다.

## 1. Final authoritative labeled development sample count

- **STRONGLY SUPPORTED** — Stage 1 허용 logical samples는 **500개**이다.
- **VERIFIED** — 500-label collection의 모든 stem이 raw BMP에 대응하며, byte-different raw ambiguity는 이 500개 안에 없다.

## 2. Excluded and unresolved sample counts

- **STRONGLY SUPPORTED** — TXT-labeled stem 전체 후보 512개 중 primary collection 500개는 허용하고 legacy-only 12개는 제외한다.
- **VERIFIED** — excluded samples: **12개**. Primary 500개 중 blocked: **0개**.
- **VERIFIED** — Primary 500개 중 unresolved identity/label samples: **0개**.

## 3. Label conflict resolution

- **STRONGLY SUPPORTED** — 충돌 3개 모두 `라벨링 6종 세트/labels` TXT를 operational authoritative version으로 선택했다: `001_20200622_203312(6)`, `001_20200623_003516(8)`, `001_20200922_163332(2)`.
- XML은 legacy TXT로 정확히 변환되므로 독립적인 두 번째 근거가 아니다. Primary TXT는 전용 500-label collection, 누적 subset membership, 더 늦은 file timestamp로 뒷받침된다. Timestamp는 보조 metadata이며 단독으로 annotation 생성 순서를 증명하지 않는다.
- 원본 TXT/XML은 수정하지 않았다. 결정은 provenance 기반이며 bbox의 시각적 정확성을 새로 주장하지 않는다.

## 4. Legacy-only TXT decision

- **STRONGLY SUPPORTED** — legacy-only 12개를 authoritative development corpus에서 제외한다: `001_20200901_003429(1)`, `001_20200922_163329(0)`, `002_20200714_043029(1)`, `002_20200715_042416(9)`, `002_20200831_083545(0)`, `002_20200831_123134(5)`, `002_20200908_043137(8)`, `002_20200915_003459(4)`, `002_20200917_043110(7)`, `002_20200917_202952(6)`, `002_20200917_203001(5)`, `002_20200922_083747(7)`.
- **VERIFIED** — 모두 raw BMP 및 legacy XML과 대응하지만, notebook의 15-image practice workflow 소속이고 primary 500-label collection 및 누적 practice subsets에는 없다.
- **UNRESOLVED** — primary collection에서 빠진 구체적인 이유는 문서화되어 있지 않다.

## 5. Legacy train/test split decision

- **STRONGLY SUPPORTED** — `train.txt/test.txt`는 **B. legacy practice/internal split**이다.
- **VERIFIED** — `splitdata.py`와 notebook이 15개 이미지에서 무작위 20% test를 만들고, `custom.data`가 이를 `valid`로 연결한다.
- 기존 split은 프로젝트 기준으로 obsolete이다. Old test membership만으로 샘플을 특별 test로 보존하지 않는다.
- Primary collection에 속한 old-test 2개는 allowlist에 포함되고, legacy-only old-test 1개는 legacy-only 정책으로 제외된다.

## 6. Duplicate and sample-identity findings

- **VERIFIED** — Primary 500 stems 중 raw 경로가 반복되는 그룹은 **65개**이며 모두 그룹 내부 SHA-256이 같다.
- **VERIFIED** — 이 반복 경로로 생기는 추가 physical paths는 **65개**이며 각각 한 logical sample로 축약했다.
- **VERIFIED** — Primary 500개에는 same-stem/byte-different 그룹이 없다.
- **VERIFIED** — Primary 500개 canonical content hash 간 cross-stem exact duplicate 그룹은 **0개**이다.
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

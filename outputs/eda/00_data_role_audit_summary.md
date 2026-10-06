# Stage 0 — Data Role / Split Audit

이 문서는 데이터 역할과 분할 근거만 정리한다. 이미지 통계, bbox 분포, 모델 성능, 전처리 비교는 수행하지 않았다.

## 1. Canonical raw-data candidate

- **STRONGLY SUPPORTED** — `4. X-ray 검사장비 AI 데이터셋/dataset/test1/yolov3/X선이물검출기(06.23_09.22)`가 canonical raw X-ray source 후보이다.
- **VERIFIED** — BMP 경로 2,809개, 고유 stem 2,529개가 있다.
- **VERIFIED** — 중복 stem 그룹 280개 중 277개는 SHA-256이 같고, 3개는 바이트가 다르다.
- **VERIFIED** — 바이트가 다른 중복 stem: `002_20200916_122650(1)`, `002_20200918_043338(2)`, `002_20200922_163053(8)`.
- **UNRESOLVED** — 동일 바이트가 여러 설비/날짜 경로에 반복된 이유와 authoritative provenance는 문서화되어 있지 않다. 독립 sample로 중복 집계하면 안 된다.

## 2. Ground-truth label candidate

- **STRONGLY SUPPORTED** — `4. X-ray 검사장비 AI 데이터셋/dataset/라벨링 6종 세트/labels`의 TXT 500개가 operational GT의 1차 후보이다.
- **VERIFIED** — 모든 label stem은 raw tree에 존재하며, 중복 raw 경로를 포함하면 565개 BMP 경로에 대응한다.
- **VERIFIED** — 두 TXT 모음 모두 YOLO 구조이다: label500=True, legacy=True.
- **VERIFIED** — legacy TXT 15개 중 12개 stem은 500-label collection 밖에 있다.
- **VERIFIED** — 겹치는 3개 stem은 bbox 값이 다르다: `001_20200622_203312(6)`, `001_20200623_003516(8)`, `001_20200922_163332(2)`.
- **UNRESOLVED** — 충돌 stem의 authoritative annotation과 legacy-only TXT 포함 여부를 확정해야 한다.

## 3. BMP / JPG / TXT / XML relationship

- **VERIFIED** — `test1/yolov3/images`의 15개 `.jpg`는 BMP `BM` signature를 가지며 raw BMP와 SHA-256이 같다. 확장자만 바꾼 복사본이다.
- **VERIFIED** — `images 15/50/100/200/300/400`의 모든 파일도 raw BMP와 바이트가 같다.
- **VERIFIED** — OpenLabeling input BMP 15개는 raw BMP 복사본이고, YOLO_darknet TXT 15개는 `yolov3/labels`와 바이트가 같다.
- **VERIFIED** — X-ray XML 35개는 PASCAL VOC 구조이며, legacy XML 15개의 YOLO 변환은 legacy TXT와 정확히 일치한다.
- **VERIFIED** — 500-label collection과 겹치는 XML 23개는 모두 해당 TXT와 bbox 값이 다르다. XML은 독립 GT로 혼합하지 않는다.

## 4. Meaning of test1/yolov3

- **STRONGLY SUPPORTED** — legacy YOLO 학습·라벨링 실습 workspace이다.
- 근거: notebook이 raw BMP를 복사하고 `ren *.* *.jpg*`로 확장자를 바꾸며, OpenLabeling 결과를 옮기고, `splitdata.py`로 무작위 20% `test.txt`를 만든다.
- 근거: `custom.data`는 `train.txt`를 train, `test.txt`를 valid로 연결하고 `train.py/test.py`가 사용한다.
- **UNRESOLVED** — 공식 competition held-out를 담는다는 근거는 없다.

## 5. Legacy train/test split findings

- **VERIFIED** — 현재 목록은 train 12 stems / test 3 stems이다.
- **STRONGLY SUPPORTED** — 15-image workspace의 내부 practice train/validation split이다. test 3개는 Stage 1 사용 금지 상태다.

### Legacy train stems

- `001_20200901_003429(1)`
- `001_20200922_163329(0)`
- `001_20200922_163332(2)`
- `002_20200714_043029(1)`
- `002_20200715_042416(9)`
- `002_20200831_083545(0)`
- `002_20200831_123134(5)`
- `002_20200908_043137(8)`
- `002_20200917_043110(7)`
- `002_20200917_202952(6)`
- `002_20200917_203001(5)`
- `002_20200922_083747(7)`

### Legacy test stems

- `001_20200622_203312(6)`
- `001_20200623_003516(8)`
- `002_20200915_003459(4)`

## 6. Meaning of images 15/50/100/200/300/400

- **VERIFIED** — 각 디렉터리 파일 수는 이름의 숫자와 일치한다.
- **VERIFIED** — 15⊂50⊂100⊂200⊂300⊂400: True.
- **VERIFIED** — 모든 파일은 raw BMP와 바이트가 같다: True.
- **STRONGLY SUPPORTED** — 독립 데이터셋이 아닌 학습량 비교용 누적 practice subsets이다.

## 7. Meaning of existing .pt weights

- **VERIFIED** — `last15/50/100/200/300/400.pt`가 존재한다.
- **STRONGLY SUPPORTED** — 이름이 practice subset 크기와 대응한다.
- **UNRESOLVED** — 정확한 membership, seed, epoch, preprocessing provenance는 확정되지 않는다.

## 8. Confirmed excluded areas

- **VERIFIED** — `result/`, `results.*`, `test_batch0_gt.jpg`, `test_batch0_pred.jpg`: prediction/training artifacts.
- **VERIFIED** — `people_walking_mp4_*` XML 795개와 `img_*` XML 3개: non-X-ray demo.
- **VERIFIED** — 파생 JPG, OpenLabeling copies, alternate XML은 독립 source에서 제외한다.
- **VERIFIED** — 기존 `.pt` weights는 EDA와 데이터 역할 추론에서 제외한다.

## 9. Official held-out competition test

- **STRONGLY SUPPORTED** — 제공된 로컬 트리에서 별도 공식 held-out test set을 찾지 못했다.
- **VERIFIED** — `test.txt`는 무작위 20% legacy validation 목록이며 `custom.data`의 `valid`에 연결된다.
- **UNRESOLVED** — 외부 KAMP 포털에 별도 평가 데이터가 있는지는 로컬 자료만으로 확정할 수 없다.

## 10. Issues to settle before Stage 1

1. **UNRESOLVED** — 충돌 TXT 3개 stem의 authoritative bbox.
2. **UNRESOLVED** — 500-label collection 밖 legacy-only TXT 12개 포함 여부.
3. **UNRESOLVED** — 동일 stem·동일 바이트 BMP의 반복 원인과 canonical provenance.
4. **UNRESOLVED** — 동일 stem·서로 다른 BMP 3개 그룹의 sample identity.
5. **UNRESOLVED** — 외부 공식 held-out test 제공 여부.
6. **UNRESOLVED** — legacy test를 보존할지, 공식 test가 아님을 확인한 뒤 새 split을 만들지 여부.

Stage 1 EDA는 시작하지 않았다.

# Project Instructions

## Project

K-인공지능 제조데이터 분석 경진대회

Task:
X-ray 영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석

Project root:
`D:\KAMP제조대회\`

Original dataset root:
`D:\KAMP제조대회\4. X-ray 검사장비 AI 데이터셋\`

---

## Core Principles

- 모든 분석과 모델링은 데이터에서 확인된 사실과 실험 결과를 근거로 한다.
- 데이터 의미가 확인되지 않은 상태에서 파일명, 폴더명, 색상, 클래스, 가중치의 의미를 임의로 추정하지 않는다.
- 관찰된 사실, 계산된 지표, 해석/가설을 명확히 구분한다.
- 재현 가능한 분석과 모델링을 우선한다.
- 원본 데이터와 파생 데이터를 혼동하지 않는다.
- 파일 수를 곧바로 독립 샘플 수로 간주하지 않는다.
- 불확실한 데이터 의미나 구조는 임의로 해석하지 않고 unresolved issue로 남긴다.

---

## Dataset Protection Rules

The original dataset directory is READ-ONLY.

- Original dataset root 및 그 하위의 모든 파일과 폴더는 읽기 전용으로 취급한다.
- 원본 데이터셋 내부에 새로운 파일이나 폴더를 생성하지 않는다.
- 원본 파일을 수정, 덮어쓰기, 이름 변경, 이동, 삭제하지 않는다.
- 원본 TXT, XML, BMP, JPG, notebook, script, weight 파일을 변경하지 않는다.
- Original dataset root 및 그 하위 디렉터리를 working directory로 사용하지 않는다.
- 라이브러리가 자동 생성하는 cache, runs, labels.cache, temporary file, prediction output 등을 원본 데이터셋 내부에 만들지 않는다.
- 전처리된 이미지가 필요하면 원본을 복사해 프로젝트의 별도 출력 폴더에 생성한다.
- 파일 저장 전 항상 대상 경로가 Original dataset root 내부가 아닌지 확인한다.

Generated files must be saved only outside the Original dataset root.

Preferred locations:
- `notebooks/` : 탐색, EDA, 실험용 notebook
- `src/` : 최종 재현 가능한 Python source code
- `outputs/tables/` : CSV 및 분석 테이블
- `outputs/figures/` : 그래프 및 시각화
- `outputs/eda/` : EDA 산출물
- `outputs/processed_data/` : 전처리된 이미지 및 파생 데이터
- `outputs/predictions/` : prediction 결과
- `models/` : 새로 학습한 모델 가중치

---

## Data Role and Leakage Rules

- 공식 test / held-out test 데이터는 EDA에 사용하지 않는다.
- 공식 test / held-out test 데이터는 feature engineering, preprocessing 선택, threshold 설정, hyperparameter tuning, model selection에 사용하지 않는다.
- 공식 test / held-out test 데이터는 최종 inference 전까지 봉인한다.
- 공식 test / held-out test 데이터는 분포 분석, 시각적 inspection, 방법 선택을 위한 qualitative review에도 사용하지 않는다.
- 폴더명이 `test`, `test1`이라고 해서 자동으로 공식 test set이라고 단정하지 않는다.
- 실제 역할은 `train.txt`, `test.txt`, config, notebook, script, README 등 파일 내용을 확인하여 판단한다.
- 기존 `train.txt`, `test.txt`가 legacy practice split인지 공식 competition split인지 구분한다.
- 역할이 확인되지 않은 데이터는 분석/학습에 사용하지 않고 unresolved 상태로 남긴다.
- `result/`, `results.*`, `test_batch*_gt`, `test_batch*_pred` 등 기존 예측 산출물은 source data로 사용하지 않는다.
- 기존 `.pt` weight는 데이터 통계나 ground truth로 사용하지 않는다.
- `images 15/50/100/200/300/400` 등 제공된 subset은 역할이 확인되기 전까지 독립적인 EDA 샘플이나 학습 데이터로 사용하지 않는다.
- 동일 sample의 BMP/JPG/XML/TXT 등 파생 표현은 서로 다른 독립 sample로 세지 않는다.
- 동일 sample이 여러 폴더에 반복 존재할 경우 하나의 sample로 관리한다.
- 파생 JPG가 원본 BMP와 동일 sample임이 확인되면 원본 BMP를 canonical source로 우선한다.
- Stage 0에서 EDA/학습/검증에 사용할 수 있는 데이터 범위를 명시적으로 확정한 뒤에만 Stage 1로 진행한다.

---

## Ground Truth Rules

- TXT label을 이물 위치 및 개수의 공식 ground truth로 사용한다.
- TXT와 BMP의 색상 사각형 표시가 불일치할 경우 TXT를 우선한다.
- BMP에 포함된 색상 사각형은 이물질의 고유 특성이 아니므로 predictive feature로 직접 사용하지 않는다.
- 색상 사각형의 의미를 임의로 추정하지 않는다.
- red/green 색상의 의미를 근거 없이 class, 정상/불량, detection status 등으로 해석하지 않는다.
- 색상 artifact는 제거 전에 먼저 존재 여부, 위치, TXT bbox와의 관계를 분석한다.
- 색상 artifact 제거 시 bbox 내부 전체를 지우지 않는다.
- artifact 처리 방식과 영향은 반드시 문서화한다.
- TXT label의 format이 명확하지 않을 경우 몇 개의 실제 파일을 확인한 후 형식을 판단한다.
- label 품질에 의문이 있는 경우 임의로 수정하지 않고 annotation quality issue로 기록한다.

---

## X-ray Interpretation Rules

- 실제 물리적 밀도, 재질, 두께, 3D 방향을 이미지에서 직접 알 수 있다고 가정하지 않는다.
- intensity, contrast, CNR, entropy, texture 등은 image-derived proxy로만 해석한다.
- bbox aspect ratio를 실제 이물의 물리적 형상이라고 부르지 않는다.
- bbox 내부 통계는 실제 이물 자체가 아니라 background가 포함된 bbox-region 통계임을 명시한다.
- product edge와 image edge를 구분한다.
- 제품 종류가 명시적 metadata나 label로 존재하지 않는 경우 이미지 외관만 보고 음식 종류를 임의 분류하지 않는다.
- X-ray intensity 차이를 실제 밀도 차이로 직접 해석하지 않는다.
- single projection image만으로 실제 두께, 원자번호, 3D 구조를 복원할 수 있다고 주장하지 않는다.

---

## EDA Rules

EDA는 단계적으로 수행한다.

### Stage 0: Data Role / Split Audit

목적:
- 어떤 데이터가 canonical source인지 확정한다.
- 어떤 데이터가 EDA/학습/검증에 사용 가능한지 확정한다.
- 공식 test / held-out test가 존재하는지 확인한다.

확인 항목:
- canonical raw source
- official TXT ground truth 위치
- legacy train/test split
- derived/practice/result artifact
- duplicated sample
- existing model weight
- official held-out test 여부

규칙:
- 이 단계에서는 본격적인 통계 EDA를 수행하지 않는다.
- image intensity distribution, bbox distribution, contrast/CNR, preprocessing comparison 등은 수행하지 않는다.
- Stage 0 결과로 데이터 사용 범위를 명시적으로 확정한 뒤에만 Stage 1로 진행한다.

### Stage 1: Pre-model EDA

분석 대상:
- image resolution
- image aspect ratio
- channel
- dtype
- bit depth
- intensity distribution
- clipping / saturation
- bbox size
- bbox relative area
- bbox minimum side
- bbox position
- image-edge distance
- bbox shape proxy
- object count per image
- negative / zero-object image 여부
- multi-object image 여부
- local contrast
- normalized contrast
- CNR-like metric
- bbox-region heterogeneity
- background complexity
- product-edge extraction feasibility
- colored artifact
- machine / date / sequence
- duplicate / near-duplicate
- feature relationships / confounding
- X-ray representation exploration

규칙:
- 이 단계에서는 model Recall, FN, FP 등 모델 성능에 관한 결론을 내리지 않는다.
- small / medium / large 같은 임의 threshold를 먼저 정하지 않는다.
- 연속형 분포를 먼저 확인한 뒤 필요 시 data-driven threshold를 정의한다.
- correlation을 causation으로 해석하지 않는다.
- product-edge distance는 신뢰 가능한 product mask를 만들 수 있을 때만 사용한다.

### Stage 2: Baseline Error Analysis

Baseline model 이후 분석 대상:
- overall detection metrics
- size vs Recall/FN
- edge distance vs Recall/FN
- contrast/CNR vs Recall/FN
- background complexity vs Recall/FN
- heterogeneity vs Recall/FN
- machine/date subgroup performance
- FN / FP case analysis
- localization error
- no-detection vs low-confidence vs localization-error 구분
- interaction analysis
  - size × edge
  - size × contrast
  - edge × contrast
  - size × background complexity
  - contrast × background complexity

규칙:
- 조건별 성능에는 sample count를 함께 제시한다.
- subgroup sample 수가 매우 적은 경우 탐색적 결과로만 해석한다.
- 필요 시 bootstrap confidence interval 또는 다른 uncertainty estimate를 사용한다.
- 단일 조건만 보고 원인을 단정하지 않는다.

### Stage 3: Safety Analysis

개선 모델 이후 분석 대상:
- remaining failure condition
- empirical detection limit
- size threshold
- edge threshold
- contrast threshold
- confidence threshold
- FN / FP trade-off
- confidence calibration
- high-risk condition 정의
- PASS / DETECT / REINSPECT 기준 제안
- 재검사 우선순위 기준

규칙:
- threshold는 단순 F1 maximum만으로 결정하지 않는다.
- safety context에서는 FN risk를 별도로 고려한다.
- reinspection rule은 실제 FN 집중 조건 및 model confidence를 근거로 설계한다.
- detection limit은 universal physical limit이 아니라 해당 데이터/모델의 empirical operating range로 표현한다.

---

## Modeling Rules

- 모델 학습 전 validation strategy를 먼저 확정한다.
- random split을 자동으로 사용하지 않는다.
- machine, date, timestamp, sequence, duplicate 여부를 확인한 후 split 방법을 결정한다.
- 동일하거나 매우 유사한 sample이 train/validation에 동시에 포함되지 않도록 한다.
- 연속 촬영 이미지의 유사성이 높을 경우 random image split으로 leakage가 발생할 수 있음을 고려한다.
- 제공된 기존 weight를 baseline으로 사용할 경우 학습 데이터 출처가 확인되어야 한다.
- 출처를 확인할 수 없는 기존 weight는 공정한 baseline 비교에 사용하지 않는다.
- 최소 2개 이상의 모델 또는 모델링 접근을 비교한다.
- baseline과 개선 모델의 차이는 EDA 또는 error analysis에서 발견된 문제를 근거로 설명한다.
- 성능 개선을 위해 불필요하게 복잡한 구조를 추가하지 않는다.
- 모델 변경보다 input representation, resolution, tiling, preprocessing, threshold 등이 더 적절한 경우 해당 방법을 우선 검토한다.
- 개선 모델의 평가는 overall metric뿐 아니라 취약 subgroup metric도 함께 비교한다.

---

## Preprocessing Rules

- 전처리는 먼저 적용하고 이유를 나중에 만드는 방식으로 진행하지 않는다.
- 모든 전처리는 EDA 또는 baseline error analysis의 관찰 결과를 근거로 선택한다.
- CLAHE, windowing, local residual, gradient, denoising, sharpening 등을 무조건 적용하지 않는다.
- 전처리 전후의 foreign-object signal separation과 background artifact 증가를 함께 평가한다.
- 보기 좋은 이미지가 아니라 detection에 유리한 representation인지 평가한다.
- resize, padding, letterbox가 small object와 bbox geometry에 미치는 영향을 확인한다.
- 전처리로 원본 정보를 손실하거나 새로운 shortcut artifact를 만들지 않도록 주의한다.
- colored artifact 제거 시 실제 X-ray signal을 과도하게 훼손하지 않는다.
- 전처리 방법을 선택할 때 raw representation과 반드시 비교한다.

---

## Validation and Evaluation Rules

- validation set은 train set과 독립적으로 유지한다.
- validation set은 preprocessing, model selection, threshold tuning에 사용할 수 있다.
- final held-out test set은 어떠한 개발 의사결정에도 사용하지 않는다.
- object detection 평가는 IoU 기준을 명시한다.
- 완전 미탐과 localization error를 가능하면 구분한다.
- FP 분석에서는 정상 구조, product edge, background complexity 등 false-positive 발생 조건을 확인한다.
- threshold 비교 시 Precision, Recall, F1, FN, FP를 함께 본다.
- safety-oriented 분석에서는 Recall/FN을 별도로 강조한다.

---

## Reproducibility Rules

- 코드 실행 결과가 동일 환경에서 재현 가능해야 한다.
- random seed를 사용하는 경우 고정한다.
- package dependency는 `requirements.txt` 또는 `environment.yml`에 정리한다.
- hard-coded absolute path는 최소화하고 project root 기준 상대경로를 우선한다.
- 반복되는 로직은 reusable function으로 분리한다.
- 최종 재현 가능한 코드는 `src/`에 정리한다.
- notebook은 탐색/분석 용도로 사용하고, 최종 pipeline은 source code로 정리한다.
- preprocessing → training → inference → result generation 흐름이 최종적으로 재현 가능해야 한다.
- 실행에 필요한 주요 parameter는 코드 내부에 숨기지 말고 명시적으로 관리한다.

---

## Output Rules

- 모든 생성물은 Original dataset root 밖에 저장한다.
- CSV/table은 `outputs/tables/`
- plot/image는 `outputs/figures/`
- EDA 관련 산출물은 `outputs/eda/`
- processed data는 `outputs/processed_data/`
- prediction은 `outputs/predictions/`
- model weight는 `models/`
- notebook은 `notebooks/`
- final source code는 `src/`

파일명은 목적을 알 수 있도록 명확하게 작성한다.

가능하면 stage prefix를 사용한다.

예:
- `00_data_role_audit.csv`
- `01_image_features.csv`
- `01_object_features.csv`
- `02_baseline_predictions.csv`
- `03_threshold_analysis.csv`

---

## Reporting Rules

분석 결과에서 다음을 명확히 구분한다:

1. Observed fact
2. Derived metric
3. Hypothesis / interpretation

추가 규칙:
- 데이터로 확인되지 않은 내용을 사실처럼 작성하지 않는다.
- correlation을 causation으로 해석하지 않는다.
- sample size가 작은 subgroup에서는 강한 결론을 피한다.
- 중요한 subgroup 성능을 보고할 때 sample count를 함께 제시한다.
- 가능하면 중요한 Recall/FN 비교에 uncertainty 또는 confidence interval을 사용한다.
- 모델의 한계와 현재 데이터로 확인할 수 없는 사항을 명시한다.
- EDA 결과와 modeling decision 사이의 연결 근거를 명확히 기록한다.
- "왜 이 방법을 선택했는가"를 데이터 또는 실험 결과로 설명할 수 있어야 한다.

---

## Codex Work Style

- 작업 전에 관련 파일과 기존 코드를 먼저 확인한다.
- 기존 파일을 불필요하게 다시 작성하지 않는다.
- 큰 작업은 단계별로 수행한다.
- 각 단계가 끝나면 다음 단계로 자동 확장하지 않는다.
- 요청받지 않은 모델 학습이나 대규모 실험을 임의로 시작하지 않는다.
- 오류나 애매한 데이터 의미를 임의로 해결하지 말고 unresolved issue로 남긴다.
- 어떤 파일을 수정하거나 새로 만들기 전에 저장 위치가 Original dataset root 밖인지 확인한다.
- 제공 코드나 notebook의 의미를 확인하기 위해 읽는 것은 가능하지만 원본 파일은 수정하지 않는다.
- 불필요한 package 설치를 피한다.
- 장시간 실행되는 실험을 임의로 시작하지 않는다.
- 현재 stage의 목표와 직접 관련 없는 작업으로 확장하지 않는다.

최종 응답에서는 작업 과정, shell command log, 중간 명령어, 긴 디버깅 설명을 출력하지 않는다.

최종 응답에는 다음만 간결하게 보고한다:
- 생성/수정한 파일
- 확인된 주요 사실
- unresolved issue
- 다음 권장 작업

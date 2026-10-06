# KAMP 제조경진대회 — Stage 0~3 EDA 정리

> 상태 표기  
> **VERIFIED**: 현재 대화에서 실제 실행 출력/업로드 결과로 수치가 확인됨  
> **SUPPORTED**: 프로젝트 산출물/설계 문서에서 구조·의도가 확인되지만, 현재 대화에서 해당 최종 파일 원문 전체를 다시 대조하지 못함  
> **LOCAL CHECK**: 과거 정리·기억에 기반한 항목으로, 제출 전 로컬 산출물과 반드시 재검증 필요

| Stage | 핵심 질문 | 무엇을 분석했나 | 주요 발견 | 의미 | 검증상태 |
|---|---|---|---|---|---|
| Stage 0 — 데이터 역할 감사 | 이 데이터셋을 그대로 믿고 써도 되는가? | 원본 BMP 경로, 중복 stem, primary/legacy TXT, practice/test 폴더, 기존 결과·weights의 역할 구분 | BMP 후보 2,809개, unique stem 2,529개, 중복 stem 280개로 정리. primary GT는 500 TXT. legacy 15개는 primary와 겹치거나 primary 밖 샘플을 포함 | 개발 corpus·GT 기준을 먼저 고정하지 않으면 중복·누수·잘못된 평가가 생길 수 있음 | LOCAL CHECK |
| Stage 0.5 — 개발 corpus 확정 | 실제로 어떤 샘플을 개발에 쓸 것인가? | primary TXT와 canonical BMP 연결, 중복 경로 정리, legacy-only 제외 | 최종 500개 labeled logical sample을 개발 corpus로 확정 | 이후 Stage 1~3의 분석 단위를 500개 logical sample로 고정 | SUPPORTED |
| Stage 1A — 기본 데이터 구조 | 이물과 영상의 기본 분포는 어떤가? | 이미지 해상도, 장비/날짜, GT 수, bbox 크기·면적비, resize 후 projected size | 500장 / 1,147 GT. 1개 이물 176장, 2개 1장, 3개 323장. bbox 최소변 중앙값 약 10 px | 작은 객체 탐지가 핵심 난점이며 입력 해상도 선택이 중요 | SUPPORTED |
| Stage 1A.5/1C — 검증 구조 | 랜덤 split을 써도 되는가? | 시간·시퀀스·유사 샘플 관계, component grouping | group-aware 4-fold를 동결하고 fold 크기 124/125/126/125로 유지 | 모델 비교에서 유사 샘플 누수를 줄이고 동일 검증 조건을 유지 | SUPPORTED |
| Stage 1B — 가시성·배경 EDA | 어떤 이물이 영상상 더 어려운가? | 크기, 절대/부호 대비, robust CNR-like proxy, background gradient/IQR, image edge, inferred product edge, interaction | `small + low contrast`가 대표 위험조건. 위치 효과는 단순 '가장자리일수록 나쁨'으로 설명되지 않음 | Stage 2에서 실제 모델 miss와 연결해 검증할 위험조건을 정의 | SUPPORTED |
| Stage 1B — shortcut audit | 모델이 진짜 이물을 보는가, 색 사각형을 외울 위험이 있는가? | chromatic artifact와 GT 위치 관계 | 500/500 이미지에 chromatic artifact가 있고, GT 1,147개 중 1,124개가 artifact component 근처 | raw RGB/gray를 그대로 쓰면 shortcut learning 위험이 큼 | LOCAL CHECK |
| Stage 1D — 입력 표현 결정 | artifact 제거 후 어떤 입력을 쓸 것인가? | Conservative inpainting vs Local interpolation | Conservative inpainting을 primary representation, Local interpolation을 secondary ablation으로 사용 | 모델 입력에서 artifact shortcut 위험을 줄이고 표현 방식의 영향을 분리 | SUPPORTED |
| Stage 2A — baseline detector | baseline detector 성능은 어느 정도인가? | YOLOv8n, 640 입력, frozen 4-fold OOF, IoU 0.50 기준 | B0 baseline OOF: TP 1112, FP 83, FN 35, P 0.9305, R 0.9695, F1 0.9496 | baseline은 강하지만 남은 FN/FP의 조건 분석이 필요 | SUPPORTED |
| Stage 2A — baseline 오류 EDA | baseline은 어떤 조건에서 틀리는가? | size/area, position, contrast/CNR, shape/background, interaction, acquisition group | small+low contrast: n=101, FN=10, Recall≈0.901. image-edge distance는 단순 near-edge가 최악이 아니었음. acquisition group별 FP 차이가 있으나 machine/date/resolution은 confounded | 위험조건은 단일 요인보다 interaction과 취득조건을 함께 봐야 함 | VERIFIED/SUPPORTED |
| Stage 2B — 개선 실험 | 어떤 개입이 실제 OOF 오류를 줄이는가? | B0~B6 비교: mosaic ablation, weak value perturbation, combined, representation, capacity, weak Gaussian blur | B2 visibility가 현재 최고: TP 1121, FP 51, FN 26, P 0.9565, R 0.9773, F1 0.9668, AP50 0.9604, mAP50-95 0.3808 | 현재 조건에서는 weak value/brightness perturbation이 가장 안정적인 개선 | VERIFIED |
| Stage 2B — residual error EDA | B2에서도 남는 FN은 왜 생기는가? | FN error type, projected size, contrast/CNR, background, position, acquisition group | FN 26 = LOW_CONFIDENCE 13 + LOCALIZATION_FAILURE 11 + NO_DETECTION 2. residual FN은 '저대비' 하나로 설명되지 않음 | 잔여 오류를 confidence 문제와 localization 문제로 분리해야 함 | VERIFIED |
| Stage 2B — pixel-size EDA | 특정 px 이하를 detection limit으로 말할 수 있는가? | projected min-side bin별 Recall/FN | 약 11.34 px 부근 변화가 보이지만 비단조적이고 20 px 이상 FN도 존재 | 11 px를 고정 detection limit으로 쓰면 안 됨 | VERIFIED |
| Stage 3 — low-confidence OOF sweep | confidence를 바꾸면 FN과 FP가 어떻게 변하는가? | B2 4-fold raw prediction을 conf floor 0.001까지 재평가 | conf 0.25→0.40에서 TP/FN은 1121/26으로 동일, FP 51→30. conf 0.001에서는 TP 1134, FP 3688, FN 13 | 0.40은 DETECT 후보로 효율적이지만, threshold를 극단적으로 낮추면 FP가 폭증 | VERIFIED |
| Stage 3 — image-level safety | 객체 기준이 아니라 이미지/제품 기준으로 보면 어떤가? | threshold별 최소 1개 GT 검출, 모든 GT 검출, partial/zero detection | conf 0.40에서 최소 1개 정확 검출 495/500=99.0%, 모든 GT 검출 475/500=95.0%, zero-correct 5장 | 제품 배제 관점과 모든 이물 위치 검출 관점은 구분해야 함 | VERIFIED |
| Stage 3 — single-object safety | 99%가 다중 이물 효과로 과대평가된 것은 아닌가? | GT 1개인 176장만 별도 분석 | conf 0.40에서 171/176=97.16%, 5장 miss | 다중 이물의 '다른 객체가 대신 검출되는 효과'를 제거해도 잔여 제품 실패가 존재 | VERIFIED |
| Stage 3 — residual 5-case review | zero-correct 5장은 왜 실패했는가? | GT diagnostics와 high-confidence wrong prediction 연결 | 5장 모두 single-object. 3건 LOCALIZATION_FAILURE, 2건 LOW_CONFIDENCE. 동시에 5장 모두 conf≥0.40의 localization false positive 존재 | threshold 조정만으로는 잔여 실패를 모두 해결할 수 없음 | VERIFIED |
| Stage 3 — audit | Stage 3 산출물 숫자들이 서로 일치하는가? | object/image/single-object/trade-off/case tables 상호검증 | 18 checks PASS, 0 FAIL | Stage 3 EDA 수치 정합성 확인 | VERIFIED |

## 전체 흐름 요약

Stage 0에서는 **데이터와 GT를 신뢰 가능한 개발 corpus로 정리**했다.  
Stage 1에서는 **작은 객체·가시성·배경·위치 등 잠재 위험조건과 검증 구조**를 정의했다.  
Stage 2에서는 **frozen 4-fold OOF로 baseline 오류를 검증하고 B0~B6 개선 실험을 비교한 뒤, B2의 잔여 오류를 다시 분해**했다.  
Stage 3에서는 **confidence threshold에 따른 FN 회수와 FP 증가의 trade-off를 확인하고, 이미지 수준 안전성과 마지막 zero-correct 사례의 실패 메커니즘을 분석**했다.

## 제출 전 필수 로컬 검증 항목

다음 항목은 현재 대화만으로는 로컬 원본 산출물과 다시 대조하지 못했으므로 반드시 확인한다.

- Stage 0의 `2,809 / 2,529 / 280` 수치
- legacy TXT `15`, primary와 겹치는 `3`, primary 밖 `12`
- chromatic artifact `500/500`, GT 인접 `1,124/1,147`
- Stage 1A의 bbox 최소변 중앙값 및 resize-risk 세부 수치
- Stage 1B의 contrast/CNR quartile 및 상관계수 전체
- Stage 1D의 Conservative/Local interpolation 결정 근거
- Stage 2A baseline OOF 집계값과 input-size 640 vs 1024 결과
- B0~B6 전체 비교표의 AP50/mAP50-95 및 fold별 수치

이 문서는 **EDA 구조를 정리한 검토용 문서**이며, `LOCAL CHECK` 항목을 실제 로컬 산출물과 대조하기 전에는 최종 제출 근거문서로 사용하지 않는다.

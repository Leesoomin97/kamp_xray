# KAMP 제조 X-ray 이물질 탐지 — B2-640

식품 완제품 X-ray 영상에서 이물질 위치를 탐지하고, 미탐·오탐 조건과 분포 변화에 따른 성능 저하를 분석한 프로젝트이다. 공식 TXT annotation을 ground truth로 사용하며, 원본 BMP의 색상 사각형은 annotation artifact로 취급해 predictive feature로 사용하지 않도록 통제했다.

## 저장소와 재현 패키지의 구분

이 `main` 브랜치는 포트폴리오와 코드 검토를 위한 경량 저장소이다. source code, configuration, 소형 metadata와 분석 evidence를 포함하지만 원본/처리 영상, `*.pt` weight와 제출 ZIP은 포함하지 않는다.

데이터·최종 Fold1~4 weight·고정 pretrained asset까지 포함한 전체 재현 패키지는 [GitHub Release](https://github.com/Leesoomin97/kamp_xray/releases/tag/submission-2026-kamp-final)에서 제공한다. 아래 FULL/QUICK 명령은 해당 패키지를 내려받아 압축 해제한 루트에서 실행하는 절차이다.

## 최종 모델과 성능

최종 선택 모델은 **B2-640 YOLOv8n**이다.

| 항목 | 설정 |
|---|---|
| Representation | conservative artifact control |
| Input / epochs / batch | 640 / 30 / 8 |
| Augmentation | visibility, `hsv_v=0.05`, mosaic 0.5 |
| Optimizer | AdamW, `lr0=0.01`, `lrf=0.1`, weight decay 0.0005 |
| Loss weights | `box=7.5`, `dfl=1.5`, `cls=0.5` |
| Seed / freeze / oversampling | 42 / 0 / none |

Reporting confidence 0.25, GT matching IoU 0.50의 고정 4-fold development OOF 결과는 다음과 같다.

| Images | GT objects | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1,147 | 1,121 | 51 | 26 | 0.9564846416382252 | 0.977332170880558 | 0.966796032772747 | 약 0.96039 | 약 0.38077 |

이는 frozen development corpus의 OOF 결과이며 official external held-out test 성능이 아니다.

## 주요 디렉터리

- `src/`: 전처리, 학습, 평가, OOF 집계, prediction export 및 분석 코드
- `configs/`: 모델/configuration 파일
- `metadata/`: 최종 B2 설정과 재현 범위 metadata
- `outputs/tables/`: CSV/JSON evidence
- `outputs/eda/`: 분석 및 의사결정 문서
- `outputs/run_manifests/`: 실행 이력과 provenance manifest
- `metadata/`와 `outputs/`: portable config, 집계 metric 및 분석 evidence를 보존

## 검증 환경

검증된 KAMP 환경은 Python 3.11.9, PyTorch 2.5.0+cu121, torchvision 0.20.0+cu121, Ultralytics 8.4.158, Tesla V100-SXM2-32GB이다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
which python
python --version
python -m pip --version
```

## 전체 재현(FULL reproduction)

Release의 전체 재현 패키지 루트에서 다음 명령은 전처리 → 고정 fold materialization → 4-fold 학습 → held-out-fold 추론/평가 → OOF 집계 → prediction export를 순차 수행한다.

```bash
python src/run_submission_pipeline.py full --tag kamp_full_reproduction_v1 --device 0 --workers 2
```

새 checkpoint는 `models/stage2a/kamp_full_reproduction_v1_fold1/weights/best.pt`부터 `fold4`까지 생성되고, 평가·OOF·prediction 결과는 `reproduction_runs/kamp_full_reproduction_v1/`에 저장된다.

학습 초기화 asset은 전체 패키지의 `models/stage2a/pretrained/yolov8n.pt`이며 크기는 6,549,796 bytes, SHA-256은 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`이다.

## 빠른 검증(QUICK verification)

QUICK mode는 학습하지 않고 전체 패키지에 포함된 최종 Fold1~4 weight로 inference/evaluation/OOF aggregation을 검증한다. FULL reproduction을 대체하지 않는다.

```bash
python src/run_submission_pipeline.py quick --tag quick_verification_v1 --device 0
```

## Prediction artifact

전체 패키지에는 다음 frozen 4-fold development OOF 결과가 포함된다.

- `predictions/b2_640_frozen_4fold_oof_bbox_conf025.csv`: 1,172행, 500 image IDs
- `predictions/b2_640_frozen_4fold_oof_image_summary_conf025.csv`: 500행

각 이미지는 자신의 fold를 제외하고 학습된 모델로 예측했다. 주최 측에서 별도 external test dataset과 고정 prediction schema를 제공하지 않아 이를 external test prediction이라고 부르지 않는다.

## 재현성과 한계

- 고정 fold SHA-256: `963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8`
- Fold1은 동일 KAMP 환경·seed·fold·recipe 재실행에서 TP 281, FP 19, FN 9, Precision 0.9366666667, Recall 0.9689655172, F1 0.9525423729, AP50 0.9569807375, mAP50-95 0.3736790631을 재현했다.
- 서로 다른 GPU 간 bitwise identity는 주장하지 않는다.
- 500개 development image가 모두 GT-positive이므로 specificity, true-negative 성능, automatic PASS safety와 정상제품 false-alarm workload는 검증하지 않았다.
- NO DETECTION은 PASS를 의미하지 않는다.
- Confidence 0.25, 0.40, 0.45는 development comparison point이며 범용 operating threshold가 아니다. 현장 적용 전 정상·이물 자료를 이용한 site-specific validation이 필요하다.
- W&B는 optional tracking layer이며 local CSV/JSON/YAML과 frozen-fold metadata가 source of truth이다.

## 제출 자산 보존

최종 제출 ZIP은 Git repository에 commit하지 않고 Release asset으로 분리했다. 제출본 SHA-256은 `44ce84ab7ffcd2f176f41814352621bd9e835344898acf4f5d712def51e6394d`이다.

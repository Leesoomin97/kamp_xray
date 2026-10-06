# KAMP 제조 X-ray 이물질 탐지

## 프로젝트 개요

이 프로젝트는 X-ray 완제품 영상에서 이물질을 검출하고, 모델 미탐과 연관된 영상 조건을 분석한 K-인공지능 제조데이터 분석 경진대회 작업물이다.

- 공식 ground truth는 제공된 TXT bounding-box label이다.
- 원본 BMP에 포함된 색상 사각형은 annotation 과정에서 생긴 chromatic artifact로 취급한다.
- 색상 사각형 자체가 이물질 검출 shortcut으로 사용되지 않도록 Conservative artifact-control representation을 사용한다.
- 원본 데이터는 읽기 전용이며, 파생 영상과 실험 산출물은 프로젝트의 `outputs/` 및 `models/` 아래에 분리한다.

## 최종 모델

최종 선택 모델은 **B2-640**이다.

| 항목 | 값 |
|---|---|
| Detector | YOLOv8n |
| Representation | Conservative artifact control |
| Input size | 640 |
| Augmentation profile | `visibility` |
| Training-only `hsv_v` | 0.05 |
| Epochs | 30 |
| Batch | 8 |
| Freeze | 0, full fine-tuning |
| Seed | 42 |
| Box / DFL / CLS loss weights | 7.5 / 1.5 / 0.5 |
| Oversampling | none |
| Initialization | Official Ultralytics COCO `yolov8n.pt` |

W&B는 선택적인 학습 추적 계층일 뿐이다. 실험의 source of truth는 로컬/KAMP CSV, JSON, YAML, frozen-fold metadata와 checkpoint hash이다.

## 검증 방식

- 10초 temporal component와 수동 승인 similarity link를 유지한 deterministic group-aware 4-fold CV를 사용한다.
- frozen assignment: `outputs/tables/01c_final_validation_folds.csv`
- 개발 OOF 범위: 500 images, 1,147 GT objects
- 동일 fold/seed/config 재현 run이 존재한다.
- 이 결과는 동일 frozen development corpus를 이용한 model-development 결과이며, untouched external held-out test 성능으로 표현하지 않는다.

## 최종 OOF 성능

Reporting confidence 0.25 및 GT matching IoU 0.50 기준:

| TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1121 | 51 | 26 | 0.9564846416382252 | 0.977332170880558 | 0.966796032772747 | 0.96039 | 0.38077 |

`0.25`는 성능 보고용 threshold이며 운영 안전 threshold가 아니다.

## Stage 3 운영 후보

개발 OOF에서 검토한 image-level 3-way routing 후보는 다음과 같다.

- **DETECT:** image-level maximum raw confidence ≥ 0.45
- **REINSPECT:** 0.25 ≤ maximum raw confidence < 0.45
- **PASS candidate:** maximum raw confidence < 0.25

GT-positive 개발 영상 500장에서는 DETECT 499, REINSPECT 1, PASS candidate 0이었고 DETECT+REINSPECT가 500/500을 포착했다. 그러나 정상 제품 영상이 없으므로 specificity, true-negative PASS safety 및 실제 현장 workload는 검증되지 않았다. 0.25/0.45는 운영 확정값이 아니라 development OOF 후보값이다.

## 주요 디렉터리

- `src/`: 재현 가능한 전처리, 학습, 평가, 집계 및 분석 코드
- `configs/`: Stage 6 등 구조 설정
- `outputs/tables/`: CSV/JSON evidence
- `outputs/eda/`: 분석 및 의사결정 문서
- `outputs/run_manifests/`: 실행 이력과 생성 파일 manifest
- `models/`: Git에 포함 가능한 args, metadata, results; `*.pt`는 제외
- `final_b2_weights_backup/`: 로컬 별도 checkpoint 보관소이며 Git에서 제외

원본 dataset, `outputs/processed_data/`, cache, W&B local cache, archive chunk와 model weight는 Git source snapshot에 포함하지 않는다.

## 환경 설치

Python 환경에서 다음을 실행한다.

```bash
python -m pip install -r requirements.txt
```

PyTorch GPU 빌드는 CUDA 및 실행 플랫폼에 맞는 공식 PyTorch 설치 방법이 필요할 수 있다. 최종 B2 KAMP run은 Python 3.11.9, PyTorch 2.5.0+cu121, Ultralytics 8.4.158 환경에서 수행됐으며, 로컬 CPU 복구 환경의 직접 dependency pins는 `requirements.txt`와 `outputs/eda/environment_audit.md`에 기록되어 있다. W&B는 `--use-wandb`를 지정한 학습에서만 lazy import되며, 제출용 evaluation/inference에는 인증이나 인터넷 연결이 필요하지 않다.

## 데이터 준비

원본 dataset은 저장소에 포함되지 않는다. 승인된 Stage 1D Conservative 파생 영상과 frozen fold metadata가 준비된 프로젝트 루트에서 다음 명령으로 Stage 2A 학습용 image/label/split 구조를 검증·구성한다.

```bash
python src/prepare_stage2a_dataset.py --workspace "/path/to/project"
```

## B2-640 단일 fold 학습 재현

다음은 fold 1 단일 run 예시다. 실행 전 승인된 `yolov8n.pt`를 `models/stage2a/pretrained/yolov8n.pt`에 두어야 하며, runner는 기록된 SHA-256을 검증한다. 기존 run과 충돌하지 않는 새 `--run-name`을 사용한다.

```bash
python src/run_stage2a_training.py \
  --project-root "/path/to/project" \
  --representation conservative \
  --model yolov8n \
  --augmentation-profile visibility \
  --fold 1 \
  --imgsz 640 \
  --epochs 30 \
  --batch 8 \
  --device 0 \
  --freeze 0 \
  --seed 42 \
  --workers 2 \
  --box 7.5 \
  --dfl 1.5 \
  --cls 0.5 \
  --oversampling-mode none \
  --run-name b2_640_fold1_reproduction
```

W&B 추적이 필요할 때만 `--use-wandb`, `--wandb-entity`, `--wandb-project`, `--wandb-group`을 추가한다. W&B를 사용하지 않아도 학습 metadata와 run manifest는 로컬에 저장된다.

## 단일 fold 평가 재현

```bash
python src/evaluate_stage2a_run.py \
  --project-root "/path/to/project" \
  --run-dir models/stage2a/b2_640_fold1_reproduction \
  --output-dir outputs/stage2a_runs/b2_640_fold1_reproduction \
  --device 0 \
  --conf-floor 0.001 \
  --report-confidence 0.25 \
  --nms-iou 0.7 \
  --max-det 300
```

평가 결과는 `metrics.json`, `metrics.csv`, `confidence_sweep.csv`, image/object/FP prediction CSV 및 evaluation metadata로 저장된다.

## 4-fold OOF aggregation

네 fold 평가가 모두 완료된 뒤 한 번만 집계한다. 출력 디렉터리는 비어 있거나 존재하지 않아야 한다.

```bash
python src/aggregate_stage2a_oof.py \
  --project-root "/path/to/project" \
  --fold-results \
    outputs/stage2a_runs/b2_640_fold1_reproduction \
    outputs/stage2a_runs/b2_640_fold2_reproduction \
    outputs/stage2a_runs/b2_640_fold3_reproduction \
    outputs/stage2a_runs/b2_640_fold4_reproduction \
  --output-dir outputs/tables/b2_640_reproduction_oof \
  --run-name b2_640_reproduction_oof
```

## Manifest 역할

`stage2b_b2_visibility_yolov8n_conservative_640_fold*_e30_v1_original_evaluation_manifest.json`은 KAMP에서 수행한 원래 Stage 2 평가 경로를 보존한다. 대응하는 `*_stage3_reanalysis_manifest.json`은 이후 로컬 Stage 3 저신뢰 재평가 경로를 보존한다. 같은 run의 서로 다른 평가 단계를 나타내므로 한쪽을 다른 쪽으로 덮어쓰지 않는다.

## 향후 test inference

최종 test inference entry point와 제출 prediction 파일 형식은 추후 확정 예정이다. 현재 저장소의 development evaluation script를 official held-out test용 코드라고 간주하지 않는다.

## 한계

- 500개 development image가 모두 GT-positive이므로 specificity, true-negative 성능과 PASS 안전성을 검증할 수 없다.
- 외부 untouched held-out 최종 평가는 수행되지 않았다.
- 실제 물질 밀도, 재질, 두께에 대한 GT가 없으며 이를 영상에서 추론하지 않는다.
- contrast와 CNR-like 값은 image-derived proxy이다.
- product boundary는 GT segmentation이 아닌 inferred proxy이다.
- bbox shape evidence는 제한적이다.
- confidence 0.25, 0.40, 0.45는 development candidates이며 운영 확정 threshold가 아니다.
- 색상 사각형 shortcut 위험을 줄이기 위해 Conservative artifact control을 적용했지만 실제 X-ray signal이 완벽히 보존됐다고 단정하지 않는다.
- Stage 6 Fold 1 실험은 exploratory screening이며 4-fold 최종 성능으로 해석하지 않는다.

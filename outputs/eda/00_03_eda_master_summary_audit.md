# `00_03_eda_master_summary_draft.md` 검증 로그

## 감사 범위와 결과

- 검증 단위: 최종 master table의 **25개 항목**
- 상태 분포: **VERIFIED 18 / SUPPORTED 4 / UNRESOLVED 3**
- 실질 수정 항목: **15개**
- 초안은 보존했으며 검증본을 별도 생성했다.
- 우선순위는 실제 CSV → 실제 MD → run manifest → frozen metadata → source code → 기존 초안 순으로 적용했다.

## 수정 내역

| # | 수정 항목 | 초안 값/표현 | 검증된 값/표현 | 수정 이유 | 근거 파일 |
|---:|---|---|---|---|---|
| 1 | 문서 상태 체계 | `LOCAL CHECK` 포함 | VERIFIED / SUPPORTED / UNRESOLVED만 사용 | 요청된 최종 상태 체계로 통일 | `outputs/eda/00_03_eda_master_summary_draft.md` |
| 2 | Stage 0 GT 관계 | legacy 15개가 overlap 또는 primary 밖이라고만 표현 | legacy 15, overlap-conflict 3, legacy-only 12로 명시 | 실제 conflict/legacy-only 표에서 직접 확인 | `outputs/tables/00_label_conflicts.csv`<br>`outputs/tables/00_legacy_only_labels.csv` |
| 3 | Stage 0 `test1/yolov3` 역할과 external official test 분리 | `test1`을 legacy/practice split으로 요약했으나 직접 확인된 라벨링·학습 workflow가 충분히 명시되지 않았고, 외부 official test 불확실성과 혼동될 여지가 있었음 | OpenLabeling으로 X-ray bbox annotation을 생성하고 images/labels를 구성한 뒤 15 samples를 train 12 / validation 3으로 나누어 YOLOv3를 학습·평가한 실습형 workspace로 VERIFIED 처리. Stage 0 구조 감사에는 포함했지만 Stage 0.5 이후 500-sample corpus의 Stage 1~3 분석·학습·OOF에는 이 split을 사용하지 않았음을 명시. 외부/비공개 official held-out 존재 여부는 별도 UNRESOLVED 이슈로 유지 | OpenLabeling README·submodule, 두 notebook, train/test 목록, `custom.data`, 실제 `results.txt` epoch log가 역할을 직접 뒷받침함 | `4. X-ray 검사장비 AI 데이터셋/dataset/OpenLabeling-master/README.md`<br>`4. X-ray 검사장비 AI 데이터셋/dataset/OpenLabeling-master/.gitmodules`<br>`4. X-ray 검사장비 AI 데이터셋/dataset/yolov3_20201200.ipynb`<br>`4. X-ray 검사장비 AI 데이터셋/dataset/test1/yolov3/yolov3_20201200.ipynb`<br>`4. X-ray 검사장비 AI 데이터셋/dataset/test1/yolov3/train.txt`<br>`4. X-ray 검사장비 AI 데이터셋/dataset/test1/yolov3/test.txt`<br>`4. X-ray 검사장비 AI 데이터셋/dataset/test1/yolov3/custom.data`<br>`4. X-ray 검사장비 AI 데이터셋/dataset/test1/yolov3/results.txt` |
| 4 | Stage 0.5 duplicate identity | 최종 500만 제시 | 반복 raw-path 그룹 65개, 모두 byte-identical, primary blocked 0 추가 | logical sample 확정 근거 보강 | `outputs/tables/00_labeled_development_candidates.csv`<br>`outputs/tables/00_duplicate_identity_audit.csv` |
| 5 | Stage 1A image/object profile | 일부 핵심 수치만 제시 | 3 resolutions와 counts, 3 machines/3 SN/24 dates, 1/2/3-object=176/1/323, GT=1,147 명시 | 실제 구조표와 label table로 검증 | `outputs/tables/01a_image_features.csv`<br>`outputs/tables/01a_label_integrity.csv` |
| 6 | Stage 1A resize | “resize projected size”만 서술 | 640 min/median=6.67/15.42 px, `<8`=0.17%, `<16`=53.88%; 320·1024 비교 추가 | 실제 resize table에서 재계산 | `outputs/tables/01a_resize_risk.csv` |
| 7 | Stage 1A.5/1C grouping | group-aware 4-fold만 제시 | 10-second components + 승인 similarity links 3 + deterministic 4-fold 및 124/125/126/125 명시 | 최종 freeze rule의 핵심 누락 보완 | `outputs/eda/01c_validation_freeze_metadata.md`<br>`outputs/tables/01c_final_validation_folds.csv` |
| 8 | Artifact spatial association | `500/500`, `1,124/1,147`가 LOCAL CHECK | 같은 수치를 실제 artifact CSV로 VERIFIED 처리하고 493/500 component-count agreement 추가 | 원표에서 직접 확인됨 | `outputs/tables/01b_artifact_image_summary.csv`<br>`outputs/tables/01b_artifact_object_relationship.csv` |
| 9 | Stage 2A baseline | TP/FP/FN/P/R/F1만 제시 | AP50=0.9506, mAP50-95=0.3740 및 정확한 recipe 추가 | OOF summary·manifest에서 직접 확인 | `outputs/tables/02a_conservative_640_oof_v1/02a_oof_summary.csv`<br>`outputs/run_manifests/stage2a_conservative_640_e30_fullft_oof_v1.json` |
| 10 | 640 vs 1024 | 초안의 로컬 검증 목록에만 존재 | fold1 640=281/15/9, 1024=284/104/6과 F1/AP를 본문에 추가 | 1024는 4-fold OOF가 아니라 fold1 비교임을 명확화 | `outputs/stage2a_runs/kamp_conservative_640_fold1_e30_fullft_v1/metrics.csv`<br>`outputs/stage2a_runs/kamp_conservative_1024_fold1_e30_fullft_v1/metrics.csv` |
| 11 | B1~B6 결과 | B2만 수치 제시 | B0~B6 TP/FP/FN/P/R/F1/AP50/mAP50-95 전체 표 추가 | 실험 간 혼합 방지와 완전성 확보 | 각 `outputs/tables/02*_oof_v1/02a_oof_summary.csv` |
| 12 | B2R seed123 | 명시 부족 | fold1 replication(TP/FP/FN=282/9/8)으로만 분리, B2 4-fold OOF 미포함 명시 | seed123을 B2 OOF와 섞지 않기 위함 | `outputs/run_manifests/stage2b_b2r_visibility_seed123_yolov8n_conservative_640_fold1_e30_v1.json`<br>`outputs/stage2a_runs/stage2b_b2r_visibility_seed123_yolov8n_conservative_640_fold1_e30_v1/metrics.csv` |
| 13 | B2 residual error | FN 분해만 제시 | FN 13/11/2와 FP 30/17/4 분해, B0→B2 주요 interaction 변화 추가 | object error와 unmatched prediction error를 구분 | `outputs/tables/02b_b2_visibility_error_analysis_v1/02a_error_type_distribution.csv`<br>`outputs/tables/02b_b2_visibility_error_analysis_v1/02a_interaction_errors.csv` |
| 14 | projected 11.34 px | “변화가 보이나 비단조” | 두 인접 bin의 n/FN/Recall과 20+ px bin 저하까지 명시 | sample count와 비단조 근거 없이 limit처럼 읽힐 위험 제거 | `outputs/tables/02b_b2_visibility_error_analysis_v1/02b_pixel_size_detectability.csv`<br>`outputs/tables/02b_b2_visibility_error_analysis_v1/02b_pixel_size_transition_scan.csv` |
| 15 | Stage 3 conf 0.40 FP | **FP 51→0** | **FP 51→30**; TP/FN은 1121/26 유지 | 초안의 명백한 수치 오류 | `outputs/tables/03_stage3_safety_analysis_v1/03e_lowconf_oof_threshold_sweep.csv` |
| 16 | Stage 3 conf 0.001 | TP/FP/FN만 제시 | P=0.2352, R=0.9887, F1=0.3800 추가 | 요청된 전체 지표를 실제 low-confidence OOF 표에서 확인 | `outputs/tables/03_stage3_safety_analysis_v1/03e_lowconf_oof_threshold_sweep.csv` |
| 17 | image-level safety 용어 | product rejection 의미가 any/all과 혼재 가능 | any-GT 495/500, all-GT 475/500, partial 20, zero 5를 분리 | object-level과 image-level, any와 all 혼동 방지 | `outputs/tables/03_stage3_safety_analysis_v1/03j_image_level_safety_sweep.csv` |
| 18 | zero-correct 5 cases | 3 localization + 2 low confidence, 모두 high-conf localization FP | 값은 유지하되 low-confidence GT match와 별도 high-confidence wrong box가 공존함을 명시 | 서로 다른 prediction 역할을 혼동하지 않기 위함 | `outputs/tables/03_stage3_safety_analysis_v1/03o_zero_correct_detection_gt_cases.csv`<br>`outputs/tables/03_stage3_safety_analysis_v1/03p_zero_correct_detection_highconf_predictions.csv` |

위 18개 로그 중 용어·구조 보완만 한 3개를 제외하고, 수치·범위·Stage 분류·해석을 실질적으로 바꾼 항목을 15개로 집계했다.

## 초안에서 확인되어 유지한 핵심 값

- raw BMP 2,809 / unique stem 2,529 / duplicate-stem groups 280
- final logical labeled samples 500 / GT objects 1,147
- 1/2/3-object images 176/1/323, bbox min-side median 10 px
- chromatic artifact 500/500, near-GT 1,124/1,147
- frozen folds 124/125/126/125
- B0 TP/FP/FN=1,112/83/35
- B2 TP/FP/FN=1,121/51/26
- B2 residual FN=13 low-confidence + 11 localization + 2 no-detection
- image-level conf 0.40 any/all/zero=495/475/5
- Stage 3 audit PASS 18 / FAIL 0

## UNRESOLVED 검증 항목

1. 외부 KAMP 포털 또는 운영기관이 별도 official held-out test를 제공하는지 여부.
2. product/lot identity, filename 괄호 값, seconds-apart frames의 실제 생산 단위.
3. inferred product boundary의 물리적 정확도: segmentation GT가 없어 proxy로만 사용할 수 있다.

정상/true-negative corpus가 없어 specificity, true PASS rate, 실제 생산 reinspection workload를 산출할 수 없다는 점은 **UNRESOLVED 검증 행이 아니라 실제 corpus에서 확인된 데이터 한계**이다.

## 품질 점검

- 최종 표의 모든 수치와 상대경로를 실제 파일과 대조했다.
- B0~B6와 B2R을 분리했다.
- confidence 0.25 reporting point와 confidence 0.40 candidate를 분리했다.
- object-level TP/FP/FN과 image-level any/all detection을 분리했다.
- artifact를 GT로 표현하지 않았고 TXT를 GT로 유지했다.
- association을 causality로 바꾸지 않았으며 physical density/material/thickness를 추론하지 않았다.
- draft 파일은 수정하지 않았다.

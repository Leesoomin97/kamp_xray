# Final Report Evidence Map

## 1. 데이터 이해/진단

- **핵심 수치:** 500 logical images, 1,147 GT objects, all GT-positive; colored artifact in 500/500 images; frozen grouped folds 124/125/126/125.
- **표/파일:** `outputs/eda/00_03_eda_master_summary_verified.md`; `outputs/eda/01d_artifact_control_summary.md`; `outputs/tables/01c_final_validation_folds.csv`.
- **핵심 주장:** 원본/파생/실습 자료와 TXT GT를 구분하고 temporal/similarity group 및 artifact shortcut을 모델링 전에 통제했다.
- **제한:** 외부 official held-out이 확인·사용되지 않았고, product/lot identity와 물리적 product boundary는 미확정이다.

## 2. AI 예측모델 개발

- **핵심 수치:** B2-640 four-fold OOF TP 1121 / FP 51 / FN 26; Precision 0.95648; Recall 0.97733; F1 0.96680; AP50 0.96039; mAP50-95 0.38077.
- **표/파일:** `outputs/tables/final_model_experiment_comparison.csv`; `outputs/tables/07_final_model_comparison.csv`; `outputs/tables/02b_b2_visibility_oof_v1/02a_oof_summary.json`.
- **핵심 주장:** 동일한 frozen-fold 조건의 B0-B6 비교로 B2를 선택했고, 추가 Fold1 1024/HardOS/Box9 실험은 FN과 취약군 안정성을 우선해 기각했다.
- **제한:** 새로운 네 Fold1 run은 evaluation evidence만 로컬에 있고 training args/metadata/checkpoints/run manifests가 누락되어 loss-weight 실행 이력은 완전 검증되지 않았다. Fold1 screening을 four-fold 성능처럼 표현하면 안 된다.

## 3. 영향요인/오류분석

- **핵심 수치:** small+low-contrast n=101, FN=10, recall 0.90099; confidence 0.40 FN 26 = low-confidence 15 + localization failure 11; small FN 12, low-contrast FN 8, small+low FN 7.
- **표/파일:** `outputs/tables/07_failure_condition_evidence.csv`; `outputs/tables/09_b2_oof_threshold040_analysis/fn_type_summary.csv`; `outputs/tables/09_b2_oof_threshold040_analysis/fn_subgroup_summary.csv`; `outputs/eda/04_detection_failure_factor_summary.md`.
- **핵심 주장:** 작은 크기와 낮은 영상상 가시성은 주요 미탐 연관 조건이며, confidence miss와 localization miss는 별도 오류 메커니즘이다.
- **제한:** FN 수가 작아 CI가 넓고, 연관성을 인과로 해석할 수 없다. Contrast/CNR-like는 image-derived proxy이며 physical density/material/thickness가 아니다. Shape evidence는 제한적이다.

## 4. 현장 활용방안

- **핵심 수치:** candidate routing 0.45/0.25에서 GT-positive 500장 중 DETECT 499, REINSPECT 1, PASS candidate 0, positive-set capture 500/500. Confidence 0.25→0.40에서 FP 51→30, TP/FN 유지.
- **표/파일:** `outputs/tables/final_threshold_comparison.csv`; `outputs/tables/final_stage3_routing_summary.csv`; `outputs/tables/10_stage3_positive_routing_simulation/routing_threshold_grid.csv`.
- **핵심 주장:** 고신뢰 결과는 DETECT하고 중간 신뢰 결과는 REINSPECT로 보내는 안전 지향 후보 workflow를 제시한다.
- **제한:** 0.45/0.25는 개발 OOF 후보값이다. 정상 제품이 없어 specificity, PASS safety, 실제 재검률 및 현장 workload는 검증되지 않았다.

## 5. 창의성/차별성

- **핵심 수치:** Conservative artifact control; grouped validation; B2 vulnerable-subgroup analysis; HardOS rescues 3/losses 4 with target subgroup degradation; Stage6 negative patch/routing/fusion evidence.
- **표/파일:** `outputs/eda/01d_artifact_control_summary.md`; `outputs/tables/08_b2_hardos_fold1_paired/summary.csv`; `outputs/eda/07_final_evidence_freeze.md`.
- **핵심 주장:** 색상 사각형 shortcut을 제거하고, 단순 평균 성능이 아니라 미탐·취약군·배포 가능성을 연결해 모델을 선택했으며 부정적 실험도 근거로 보존했다.
- **제한:** Oracle complementarity와 Fold1 exploratory signal을 실제 deployable gain으로 표현하지 않는다.

## 6. 코드/재현성

- **핵심 수치:** 기존 B2 Fold1과 reproduction run의 핵심 평가 산출물 7개 SHA-256 identical; frozen fold SHA-256 `963116078893679B859C2D5150A1BA90700AD207C54AE385A1B6ACAC255B98A8`.
- **표/파일:** `src/run_stage2a_training.py`; `outputs/eda/environment_audit.md`; `outputs/eda/01c_validation_freeze_metadata.md`; `outputs/stage2a_runs/b2_640_repro_fold1_v1/`.
- **핵심 주장:** 현재 KAMP 코드/환경에서 동일 seed/fold/config의 Fold1 평가 결과가 완전히 재현되었다. Local evidence가 source of truth이며 W&B는 보조 tracking 계층이다.
- **제한:** 다른 하드웨어/GPU/driver/library 조합의 bitwise identity는 보장하지 않는다. 신규 네 run의 training metadata/checkpoints/run manifests는 로컬 evidence package에 추가 확보가 필요하다.

# 실험 노트북

본 학습은 루트의 `aimers_tuned_ensemble.ipynb` 하나뿐이다 (v5, 리더보드 990.9528).
여기 있는 것들은 전부 **진단·스크리닝용**이며, 본 학습 노트북에서 전처리 셀만 떼어내고
뒤에 실험 셀 하나를 붙인 구조다. 따라서 피처 파이프라인은 항상 동일하다.

판정 근거와 배경은 루트 `CLAUDE.md` 를 볼 것. 여기서는 **무엇을 재서 무슨 결론이 났는지**만
한 줄씩 남긴다.

## `capacity/` — 용량 축 (전부 소진됨)

| 노트북 | 잰 것 | 결론 |
|---|---|---|
| `aimers_v7.ipynb` | 700회 학습 + `ntree_end` 350/500/700 세 벌 제출 | ❌ 리더보드 986.21 / 967.89. **반복수는 1000 유지** |
| `aimers_converge.ipynb` | 반복수 수렴 곡선 (모델 1개) | ⚠️ 모델 1개라 결론이 뒤집힌 원인. 참고용 |
| `aimers_loss.ipynb` | 손실함수(Logloss vs RMSE) × 반복수 | ❌ 정점 795 로 완전 동점. RMSE 전환 무의미 |
| `aimers_topt.ipynb` | 2023 홀드아웃을 목적함수로 Optuna | ❌ 전 trial 음수. **2023 은 튜닝에 못 쓴다** |

`aimers_v7.ipynb` 는 기각됐지만 **지점별 isotonic + 다중 zip 산출 구조**는 재사용 가치가 있다.
학습 한 판으로 제출본 여러 개를 뽑을 때 Cell 6b/6c/7 을 참고할 것.

## `features/` — 피처 축

| 노트북 | 잰 것 | 결론 |
|---|---|---|
| `aimers_baseline.ipynb` | 원본 컬럼 vs ID 범주형 vs 현행 파이프라인 | 파이프라인 +35, ID 범주형 -120. ⚠️ v1 은 3000회로 돌려 무효 |
| `aimers_asof_screen.ipynb` | `asof_*` 를 최근 N구 이동평균으로 재설계 | ❌ fresh -217 / frozen -5~-27 |
| `aimers_tmdyn.ipynb` | 트랙맨 릴리스 동역학 12개 (v6) | ❌ 리더보드 988.47 |
| `aimers_bat.ipynb` | 타자측·포수대리·맞대결 조건부통계 | ❌ +1 / +3±9 / +6±4. **`cond_p` 가 +27 인데 타자판은 +1** |
| `aimers_pct.ipynb` | 수준 피처를 시즌 내 백분위로 변환 | ❌ +0(±9). 트리가 `season` 분기로 이미 학습 중이었다 |
| `aimers_oracle.ipynb` | **보정 축의 천장** — Murphy 분해 + 오라클/실행가능 상한 | ⛔ 천장 51점, 실행 가능한 접근은 -48~-566. **보정 축 종료** |
| `aimers_bias.ipynb` | 편향 b_Y 를 네 시즌 실측해 2025 로 외삽 | ✅ **b ~ 하락폭** 이 정답(b_2024 를 0.0006 오차로 예측). 다만 현행 상수가 이미 최적(+1) |

`aimers_oracle.ipynb` 는 재사용 가치가 높다. **새 보정 아이디어가 떠오르면 먼저 여기에
넣어서 오라클 상한부터 재라** — 천장이 51점이라는 걸 알면 며칠을 아낀다.

`aimers_bias.ipynb` 의 **검증 셀**(한 점을 빼고 맞혀보게 하는 것)이 224점짜리 실수를 잡았다.
외삽 모형을 세울 때는 반드시 이 구조를 넣을 것.

## `models/` — 모델 계열 축

| 노트북 | 잰 것 | 결론 |
|---|---|---|
| `aimers_fam.ipynb` | CatBoost/Lossguide/LightGBM/XGBoost 비교 v1 | ⚠️ 전부 1000회 고정이라 lgbm/xgb 과소평가. **계열 비교는 각자의 최적 반복수에서** |
| `aimers_fam2.ipynb` | 계열별 반복수 곡선 + 최적점 블렌드 | 최적점에서 lgbm 739 / xgb 724, 블렌드 +8 |
| `aimers_fam3.ipynb` | 블렌드 이득 × 앙상블 규모 (k=1,2,3,5) | ❌ +9→+7 로 오히려 감소. **앙상블 규모 가설도 기각** |
| `aimers_nn.ipynb` | NN 엔티티 임베딩 1차 (대용량) | ❌ 단독 365. 단 cb 상관 **0.67** — 트리 계열(0.92~0.97)이 못 넘던 벽 |
| `aimers_nn3.ipynb` | NN 2차 (소용량 + 강한 정규화) | ❌ 단독 519, 상관 0.78, 블렌드 +1. **좋아질수록 트리와 닮아간다** |

## `archive/`

| 노트북 | 내용 |
|---|---|
| `aimers_skfold.ipynb` | 932.96 버전. 현행 `aimers_tuned_ensemble` 의 기반 |
| `aimers_offset.ipynb` | v4 에서 오프셋 측정만 떼어낸 축소 커널. 학습 뒤쪽 셀이 크래시했을 때 산출물만 회수해 보완하는 용도 |

## 실행 방법

전부 캐글 커널로 돌렸다. `kernel-metadata.json` 의 `id`/`code_file` 을 바꿔 push 한다.

```bash
PYTHONUTF8=1 kaggle kernels push -p .
PYTHONUTF8=1 kaggle kernels output <owner>/<kernel> -p ./out
```

**Windows 에서는 `PYTHONUTF8=1` 을 반드시 붙일 것** — 안 붙이면 노트북의 한글을 cp949 로
디코딩하려다 크래시난다.

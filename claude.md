# CLAUDE.md — KBO 제구 성공 예측 프로젝트

이 문서는 Claude Code가 이 프로젝트에서 작업할 때 참고할 컨텍스트입니다.
데이콘 해커톤: 투구별 제구 성공 확률(`control_success`) 예측, 평가지표는 Brier Skill Score.

```
Score = max(0, 100000 × (1 - Brier / (r(1-r))))   # r = 평가 데이터 실제 평균 성공률
```

**현재 상태**: 리더보드 최고점 **959.4258** (`aimers_tuned_ensemble.ipynb`, 2026-08-18 확정).
932.96 대비 +26.47점. 목표: 최소 1100점. **마감 9/2.**

---

## 1. 데이터 구조

```
data/
  train.csv                 # 1,475,092행 x 49컬럼. control_success 포함
  test.csv                  # 평가용 (배포본엔 5행 샘플만, 실제 평가 시 서버가 교체)
  sample_submission.csv     # row_id, control_success 2컬럼
  data/data_description.md  # 주최측에서 제공한 데이터 설명서
  trackman_history.csv      # 2019~2024년 투구 로그, 1,793,078행 x 30컬럼
                            # train/test와 1:1 결합 안 됨. pitcher_trackman_id 기반 매핑 필요
                            # 2025년(=test 시즌) 데이터는 없음 — 중요, 아래 4-4 참고


                            # 이하 항목은 사용자가 만든 것으로 주최측 공식 제공은 아님.
pitcher_id_mapping.csv      # pitcher_id <-> pitcher_trackman_id 
batter_id_mapping.csv       # batter_id <-> batter_trackman_id  (타자측 트랙맨 피처용)
```

### train.csv 핵심 컬럼

- 식별자/상황: `row_id, season, game_month, game_dayofweek, inning, top_bottom, game_type`
- 카운트/점수: `balls_before, strikes_before, outs_before, score_diff_pitcher_team` 등
- 주자: `runner_on_1b/2b/3b, base_state(___/1__/_2_/__3/12_/1_3/_23/123), li`
- 선수: `pitcher_id, batter_id, pitcher_hand, batter_hand, pitcher_team_id, batter_team_id`
- **`asof_*` 컬럼**: 해당 투구 **직전까지**의 누적 통계 (leak-free, 운영 측 공식 제공).
  `asof_pitcher_success_rate`, `asof_pitcher_n`, `asof_pitcher_prev{1,3,5}_game_success_rate`,
  `asof_batter_success_rate`, `asof_pitcher_{fastball,breaking,offspeed}_rate` 등
- 타겟: `control_success` (0/1). 가운데로 몰림 / 크게 벗어남 / 포수 요구 반대방향 = 실패

### 규정상 금지 사항 (중요)

- test.csv의 다른 행을 이용한 통계(누적/빈도/target encoding/rolling) 금지
- 현재 투구 이후 확정 정보, 실제 위치/판정/구종 사용 금지
- trackman_history.csv에 2025년 데이터 없음 → test 시점 투수의 최신 트랙맨 값은 **2024년 값으로 근사**할 수밖에 없음

---

## 2. 전처리 파이프라인 (`step1` ~ `step21`)

**설계 원칙**: 학습 노트북과 제출용 `script.py`가 **완전히 동일한 소스**를 써야 한다.
`STEPS_SRC`라는 문자열 하나에 `step1~14`를 정의하고, 노트북은 `exec(STEPS_SRC)`로 로드,
`script.py`는 이 문자열을 그대로 텍스트 삽입한다. (이 원칙이 깨져서 대형 버그가 여러 번
발생했음 — 4장 참고.)

| 함수 | 내용 |
|---|---|
| `step1_basic_features` | 주말 낮경기, 한여름 폭염경기 플래그 |
| `step2_pitcher_role_features` | 순수선발/롱릴리프/구원 플래그 |
| `step3_matchup_features` | 투타 손 일치 여부 |
| `step4_refined_count_features` | 볼카운트 유불리(`count_advantage`: Pitcher/Batter/Neutral/**None**), 풀카운트/초구 플래그 |
| `step5_pitches_per_inning` | 이닝당 투구수 |
| `step6_combined_runner_features` | RISP, 승계주자, 도루위협 등 |
| `step7_bayesian_smoothing` | `smoothed_pitcher_success_rate = (n*curr + C*prior_mean)/(n+C)`, C=50 |
| `step8_batter_toughness_features` | 타자 까다로움 지수 |
| `step9_garbage_time_features` | 가비지타임 플래그/지수 |
| `step10_recent_form_momentum` | 최근 1/3/5경기 폼 변화량 |
| `step11_veteran_and_pressure_features` | 루키/베테랑 + 압박지수(li) 결합 |
| `step12_first_pitch_tendency` | 초구 직구 스트라이크 확률 지수 |
| `step13_sac_fly_threat` | 희생플라이 위협 상황 |
| `step14_convert_to_category` | 범주형 컬럼 dtype 변환 (CatBoost/LightGBM용) |
| `step15_prep_trackman_data` | 트랙맨 원본 + `pitcher_id_mapping` 병합, `count_advantage` 부여 |
| `step16_calc_expected_difficulty` | 투수가 상황별로 상대해온 구질 비율 × 릴리스포인트 변동성 = `expected_control_difficulty` |
| `step17_calc_pitch_speed` | 투수의 과거 평균 패스트볼 구속(`past_fb_speed_mean`) + **체감구속**(`past_fb_zone_speed_mean`, `zone_speed` 기반) |
| `step18_calc_pitch_consistency_by_group` | 구종군별 릴리스 지표 표준편차 (18개 `past_*_std` 피처) |

**현재 제출 파이프라인은 `step1~18`까지만 쓴다. `zone_speed` 기반 체감구속 피처는
포함되지 않는다 — 아래 참고.**

> ✅ **원인 분리 완료 (2026-08-18): `zone_speed` 자체가 원인이었다, 타자 피처가 아니었다.**
> `past_fb_zone_speed_mean`을 타자 피처 없이 `step17`에만(=`rel_speed`와 완전히 동일한
> shift(1).expanding() 패턴) 단독으로 추가해 Kaggle GPU로 재학습·제출한 결과 **929.6060점**
> (932.96 대비 **-3.35점**). 이전에 타자 피처 6개 + zone_speed를 함께 넣었을 때가 930.00
> (-2.96점)이었던 것과 거의 같은 낙폭 — 타자 피처를 뺐는데도 낙폭이 줄지 않았으므로,
> **원흉은 zone_speed였다.** `step17`은 원상복구했고(`past_fb_speed_mean`만 유지),
> **zone_speed는 이 방식(월 단위 집계)으로는 다시 시도하지 말 것.**
> (추측: `rel_speed`와 상관이 높아 중복 신호이거나, 월 단위로 뭉개진 집계라 이미 노이즈가 커서
> 피처를 하나 더 얹을수록 과적합 위험만 커졌을 가능성 — `asof_*` 재설계(최근 N구 이동평균)로
> 넘어가면 이 문제 자체가 달라질 수 있음.)

> ⚠️ **`step19~21`(타자측 트랙맨: `step19_prep_batter_trackman`,
> `step20_calc_batter_faced_profile`, `step21_calc_batter_zone_speed_faced`)은 여전히 폐기 상태.**
> 코드 자체는 실험 기록으로 어딘가 남아있을 수 있지만, 현재 학습/제출 파이프라인에는 포함되지 않는다.
> 다시 시도하려면 실패 원인(타자측 표본 희소 추정)부터 재확인할 것 — zone_speed와는 무관하게
> 독립적으로 재검증 필요.

**모든 `step16~18`은 `shift(1).expanding().mean()` 패턴으로 leak-free 누적 계산.**

### 트랙맨 병합 방식 — `TRACKMAN_MODE` 설정

- **`asof`(채택, 933점)**: `time_idx = season*100+game_month` 기준 `pd.merge_asof(direction='backward')`.
  2025 test 행은 해당 투수의 **가장 최근(2024) 값**을 받는다. 학습/추론 규칙이 동일해서 원칙적으로 타당.
- **`exact`(897점)**: `(season, game_month)` 완전일치 + `fillna(0)`. 900점 원본 버전과 동일 동작.
  트랙맨에 2025가 없으므로 **test에서는 전부 0**이 된다.
- `asof`가 `exact`보다 +36점 → 트랙맨 피처가 실제 신호를 담고 있다는 증거.

---

## 3. 모델링

### 현재 채택 구성 (959.4258점, `aimers_tuned_ensemble.ipynb`)

```python
# Cell 6a: Optuna 40 trial (전체의 약 33% 서브샘플, 3-fold)로 하이퍼파라미터 탐색
# 찾은 최적값 (932.96 버전의 depth=6,lr=0.05 기본값보다 훨씬 깊고 느리게 학습):
#   learning_rate≈0.0247, depth=9, l2_leaf_reg≈9.23,
#   bagging_temperature≈0.79, random_strength≈0.78

# Cell 6b: 위 파라미터로 전체 데이터 학습
for seed in [42, 202, 2024]:
    StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    CatBoostClassifier(iterations=1000, **optuna_best_params,
                        eval_metric='Logloss', task_type='GPU', early_stopping_rounds=50)
    # fold별 CalibratedClassifierCV(method='isotonic', cv='prefit')로 보정
# 10-fold x 3-seed = 30개 모델 평균 (cb_fold_{seed}_{fold}.cbm 네이밍)
```

- **피처 파이프라인은 932.96 버전과 완전히 동일** (`step1~18`, zone_speed 없음) — 차이는
  하이퍼파라미터 튜닝 + fold 5→10 + seed 1→3뿐. **"같은 정보를 더 안정적으로 뽑아내는" 개선이
  실제로 +26.47점**이라는 큰 폭으로 먹혔다. 새 피처 추가 시도(zone_speed 계열)가 전부 마이너스였던
  것과 대조적 — 지금 단계에서는 신규 피처보다 기존 신호를 잘 뽑아내는 쪽이 더 검증된 전략.
- **`TimeSeriesSplit`이 아니라 `StratifiedKFold`를 쓰는 이유**: `TimeSeriesSplit`은 fold마다
  학습 데이터량이 다르고(17%~83%), 초기 fold가 최근 시즌 예측에 취약해서(실측 skill -0.6~-2.0%)
  균등평균이 오히려 naive baseline보다 나빠짐(818점 사태). `StratifiedKFold`는 각 fold가
  전체의 80%~90%를 균등하게 학습해 fold 간 실력이 대등하고, 합집합이 데이터 100%를 커버한다.
- **CatBoost 단독** (LightGBM은 현재 미사용 — TimeSeriesSplit 시절 불리한 조건에서 비교했던 결과라
  StratifiedKFold 조건에서 재검토 여지 있음)
- **모델 저장은 네이티브 포맷**: `model.save_model("x.cbm")` (joblib pickle 금지 — 버전 의존성 문제로
  과거 크래시 발생). Isotonic 보정기만 `joblib.dump` (단순 배열 기반이라 안전).

### 시도했던 것들

| 시도 | 결과 |
|---|---|
| **Optuna 튜닝 + fold10 + seed3 앙상블** | **959.4258 (932.96 대비 +26.47점) → 채택, 현재 최고점** |
| LightGBM 85:15 앙상블 | TimeSeriesSplit 시절 실험, 재검토 필요 |
| 타자측 트랙맨 피처 6개 + zone_speed 체감구속 (동시) | 930.00 (932.96 대비 -2.96점) → 롤백 |
| zone_speed 체감구속 **단독** (타자 피처 없이) | 929.6060 (932.96 대비 -3.35점) → 롤백. 원인이 zone_speed였음을 확정 — 두 실험의 낙폭이 거의 같아서 타자 피처는 결백 |

### 노트북 계보 (혼동 방지용)

| 노트북 | 내용 | 리포 포함 여부 | 점수 |
|---|---|---|---|
| `aimers_tuned_ensemble.ipynb` (현재 최고) | 932.96 피처 기준 + Optuna + fold10 + seed3 | ✅ 커밋됨 | **959.4258 (현재 채택)** |
| `aimers_skfold.ipynb` | CatBoost + StratifiedKFold(5) + asof 트랙맨 (zone_speed 없음) | ✅ 커밋됨 | 932.96 (이전 최고, 이제 tuned_ensemble의 기반) |
| `aimers_batter_trackman.ipynb` | 932.96 기준 + zone_speed + 타자 피처 6개 | ❌ 없음 | 930.00 → 롤백 |

**Kaggle 커널 실행 참고**: `homekeggle/aimers-skfold`(→`aimers_skfold.ipynb`)와
`homekeggle/aimers-tuned-ensemble`(→`aimers_tuned_ensemble.ipynb`) 두 커널로 API push/pull
워크플로가 구축됨 (`kernel-metadata.json`, 리포 루트 — 현재는 skfold를 가리킴, 다른 노트북 돌릴
땐 `code_file`/`id` 바꿔서 push). `kaggle kernels push -p .` → 상태 폴링 →
`kaggle kernels output <kernel> -p ./kaggle_output`로 결과 회수.
**Windows(cp949 로케일)에서는 반드시 `PYTHONUTF8=1` 환경변수를 붙여서 실행할 것** — 안 붙이면
노트북의 한글/특수문자(em dash 등)를 cp949로 디코딩하려다 크래시남
(`'cp949' codec can't decode byte ... illegal multibyte sequence`).

**GPU 종류(P100 vs T4 x2) API로 선택 불가**: 공식 `machine_shape` 필드가 존재하긴 하는데
(GitHub `Kaggle/kaggle-api` `main` 브랜치 문서 기준) PyPI에 릴리스된 `kaggle` 패키지(현재 최신
1.7.4.5 포함)엔 아직 구현이 안 들어가 있음 — 웹 에디터에서 "Save & Run All"로 인터랙티브하게
돌릴 때만 T4 x2 선택 가능, API push는 항상 계정 기본값(P100)으로 감. CatBoost는 `devices`
미지정 시 GPU를 전부 자동으로 쓰므로, T4 x2를 쓰면 코드 수정 없이 더 빨라질 여지는 있음
(단, 이번엔 P100 기준으로 학습해서 959.43을 얻었으므로 성능 자체엔 문제없음).

---

## 4. 겪었던 핵심 버그들 (재발 방지용 — 반드시 숙지)

### 4-1. `prior_mean` 미저장
`step7_bayesian_smoothing`의 `prior_mean`은 학습 시 `train.csv`에서 계산한 값을
`model/train_constants.json`에 **반드시 저장**해야 한다. 저장 안 하면 추론이 하드코딩
기본값으로 폴백 → 핵심 피처(중요도 상위권)가 학습/추론 간 어긋남.
`asof_pitcher_success_rate.mean()`이 아니라 **`control_success.mean()`**을 써야 함
(전자는 이미 노이즈 낀 파생지표라 순환적).

### 4-2. `.dropna()`로 트랙맨 테이블 가공 금지
`step16~18` 출력을 `dropna()`해서 저장하면, 학습 때 NaN이던 자리가 추론 때
merge_asof로 "더 과거의 다른 값"을 끌어와 채워버림. **원본 그대로 저장.**

### 4-3. `count_advantage`의 `'None'` → NaN 자동 변환
`'None'`은 0-0/3-2 카운트를 뜻하는 **실제 문자열**(전체의 ~25%)인데, `pd.read_csv` 기본
설정이 이를 NaN으로 읽어버려 `by=['pitcher_id','count_advantage']` merge가 전량 실패한다.

```python
_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']  # 'None' 제외!
pd.read_csv(path, keep_default_na=False, na_values=_NA)
```

### 4-4. CatBoost 범주형 NaN 처리
CatBoost는 `cat_features`에 실제 float NaN이 있으면 에러를 던진다.
`astype(str).astype('category')`로 NaN을 문자열 `'nan'`으로 변환해야 함 (버그 아니라 필수 처리).

### 4-5. LightGBM 네이티브 포맷 관련 (LightGBM 재도입 시 주의)
- `lgb.Booster.feature_name()` 호출이 환경에 따라 access violation(세그폴트급 크래시)을 일으킨 적 있음 → 호출 자체를 피하고 `selected_features.json` 순서를 그대로 신뢰할 것.
- `.txt` 네이티브 저장은 `pandas_categorical` 매핑을 이미 자동으로 포함/복원함 — 수동으로 덮어쓰지 말 것 (덮어쓰다가 오히려 크래시 유발한 적 있음).
- 두 라이브러리(CatBoost+LightGBM)를 같은 프로세스에서 같이 쓸 때 Windows에서 OpenMP 충돌로 크래시 나는 사례 있었음 (`KMP_DUPLICATE_LIB_OK=TRUE`로 완화 시도, 리눅스 서버에서는 미재현 — 로컬 Windows 리허설 결과를 실제 서버 결과와 혼동하지 말 것).

### 4-6. 모델이 안 불러와지면 조용히 0으로 폴백 금지
```python
if len(preds) == 0:
    raise RuntimeError(f"모델 로드 실패. model 폴더 내용: {os.listdir('model')}")
```
과거에 zip에 모델 파일이 누락됐는데도 크래시 없이 `final_preds=0.01`로 채워져
"정상 종료 + 사실상 0점"이라는, 원인 특정이 가장 어려운 실패 유형을 겪은 적 있음.

### 4-7. 검증 방법론 자체의 함정
- `TimeSeriesSplit` 홀드아웃 검증은 정직하지만, "최근 데이터를 더 많이 학습에 쓸수록 좋다"는
  사실은 과소평가한다 (최근 데이터를 검증용으로 떼어두게 되므로).
- **`StratifiedKFold` 채택 이후 로컬 홀드아웃 검증은 완전히 무효화됨** (모델이 검증 데이터를
  이미 학습에 사용). 로컬 검증 셀은 "파이프라인이 안 죽는지, 피처가 상식적인지" 확인 용도로만
  쓰고, 설정 간 우열은 **리더보드로만** 판단할 것.
- 단, 하이퍼파라미터 튜닝(Optuna)은 예외 — "이 파라미터가 데이터를 잘 맞히는가"는
  `StratifiedKFold`로도 정직하게 측정 가능.
- 로컬 검증 시 "최근 데이터의 X%"를 행 개수 기준으로 자르면, 연도별 데이터량 편차 때문에
  실제로는 최근 시즌의 극히 일부만 떼어지는 함정이 있음 (로컬 1054 vs 실제 810으로 샌 사례).
  **반드시 시즌 단위로 통째로 분할할 것.**
- **로컬 검증 시 학습/추론 경로를 각각 따로 실행해서 "평균값"만 비교하지 말 것.**
  행이 뒤섞여도 평균은 비슷하게 나올 수 있다. **반드시 같은 `row_id` 기준으로 행 단위(1:1) 비교.**

### 4-8. Optuna로 찾은 `best_params`에 `cat_features` 누락 (`aimers_tuned_ensemble.ipynb`)
`study.best_params`는 `trial.suggest_*`로 탐색한 값만 담고 있다. 최종 학습용 파라미터 딕셔너리를
`BEST_PARAMS = dict(study.best_params)` + 수동으로 `iterations`/`eval_metric`/`task_type`/
`early_stopping_rounds`만 추가해서 만들면 **`cat_features`가 빠진다.** CatBoost는 pandas
`category` dtype 컬럼이 있는데 `cat_features`가 없으면 자동 인식이 아니라 즉시
`CatBoostError`를 던진다 (직접 재현 확인). Optuna 탐색용 `objective()` 내부에서는 명시적으로
`cat_features`를 넘겨서 멀쩡히 돌아가므로, **탐색 단계는 무사통과하고 최종 학습(30개 모델) 첫
fold에서 크래시** — 방치하면 GPU 쿼터만 태우고 아무 것도 못 건진다.
**수정**: `BEST_PARAMS["cat_features"] = cat_features`를 반드시 추가.

### 4-9. `StratifiedKFold.split()`의 반환 순서 착각으로 서브샘플 비율이 의도보다 커짐
`skf.split(X, y)`는 `(train_idx, test_idx)` 순서로 반환한다. "30%만 빠르게 서브샘플링"하려고
```python
_sub_idx, _ = next(StratifiedKFold(n_splits=3, ...).split(X_full, y_full))  # 버그: train_idx(약 67%)를 씀
```
처럼 **첫 번째 원소**를 받으면 의도한 33%가 아니라 **약 67%**가 뽑힌다. 크래시는 안 나서
발견하기 어렵고, Optuna 40 trial × 3-fold가 조용히 2배 이상 느려진다.
**수정**: `_, _sub_idx = next(...)` (두 번째 원소가 test_idx ≈ 33%).

---

## 5. 제출 파이프라인 (`script.py`) 필수 구조

```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 최상단, import보다 먼저

# STEPS_SRC 텍스트 그대로 삽입 (학습 노트북과 동일 소스 공유)

def main():
    # 1. test.csv 로드 (data/ 또는 open/ 후보 경로 탐색)
    # 2. model/train_constants.json에서 prior_mean 로드 (없으면 RuntimeError)
    # 3. step1~13 피처 엔지니어링
    # 4. model/feat_diff.csv, feat_speed.csv, feat_rp.csv와 merge_asof
    #    (na_values 명시! count_advantage 'None' 보존 필수)
    # 5. step14 범주형 변환 + astype(str) NaN 처리
    # 6. model/cb_fold_*.cbm 을 glob으로 스캔해서 전부 로드 + isotonic 보정 + 평균
    #    (개수 하드코딩 금지 -> N_SPLITS/SEEDS 바뀌어도 script.py 수정 불필요)
    # 7. 모델 0개면 RuntimeError로 명시적 실패 (조용히 0.01 폴백 금지)
    # 8. output/submission.csv 저장

if __name__ == "__main__":
    try:
        main()
    except Exception:
        os.makedirs("output", exist_ok=True)
        with open("output/error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        raise
```

### 필수 제출 zip 구성
```
script.py, requirements.txt
model/cb_fold_{seed}_{fold}.cbm  (N_SPLITS x len(SEEDS)개)
model/isotonic_fold_{seed}_{fold}.pkl  (동일 개수)
model/selected_features.json, model/train_constants.json, model/best_params.json
model/feat_diff.csv, model/feat_speed.csv, model/feat_rp.csv
```

### 제출 전 필수 검증
학습 노트북 안에서 `script.py`를 **실제로 서브프로세스 실행**해서, 정답을 아는 홀드아웃
시즌으로 Brier/Skill을 직접 채점하는 검증 셀을 반드시 통과시킬 것. "안 죽는지"만 확인하는 건
불충분함 — 과거에 크래시 없이 정상 종료되지만 skill이 naive보다 나쁜(≈0점) 사고가 여러 번 있었음.

### 평가 서버 환경 (주최측 공지, 2026-08-18 확인)

인터넷 접속 불가(외부 다운로드 코드/모델 동작 안 함). 아래 패키지가 **버전 고정된 채로 이미
설치**되어 있음 — `requirements.txt`에 다시 적지 말 것 (버전이 어긋나면 설치 에러 발생):

```
pandas==2.0.3, numpy==1.26.4, scipy==1.15.3, scikit-learn==1.8.0, joblib==1.5.3,
threadpoolctl==3.6.0, torch==2.7.1+cu128, transformers==4.46.3 등
```

현재 `requirements.txt`는 `catboost` 한 줄만 있음 — 목록에 없으므로 맞는 구성.
**LightGBM 재도입 시 `lightgbm`을 여기 추가해야 함** (사전 설치 목록에 없음).

**제출 오류는 두 종류이며 일일 제출 횟수 반영 여부가 다르다:**
- **설치 오류** (zip 구조 불일치, 패키지 설치 실패) → 일일 제출 횟수에 **반영 안 됨**
- **제출 오류** (`script.py` 실행 중 발생하는 모든 오류) → 일일 제출 횟수에 **반영됨**

즉 런타임 버그로 인한 실패는 그날의 제출 기회를 실제로 태운다 — 위 "제출 전 필수 검증"이
단순 권장이 아니라 **일일 제출 횟수를 아끼기 위한 필수 절차**인 이유.

---

## 6. 로드맵 (마감 9/2)

| 기간 | 작업 |
|---|---|
| ~8/18 | ✅ Optuna 튜닝 + fold10 + seed3 앙상블 — **완료, 959.4258로 채택** |
| 8/19~8/27 | `asof_*` 피처 직접 재설계 (카운트/구종/좌우별 세분화, 경기 단위 아닌 최근 N구 이동평균) — 최우선 승부처 |
| 8/28~8/31 | 여유 시: `middle_rate`/`reverse_rate` 분해 기반 보조모델, LightGBM 재도입 |
| 9/1~9/2 | **버퍼 — 신규 시도 금지.** 그 시점 최고 검증 버전 고정, 최종 제출만 |
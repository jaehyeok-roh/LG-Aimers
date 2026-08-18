# CLAUDE.md — KBO 제구 성공 예측 프로젝트

이 문서는 Claude Code가 이 프로젝트에서 작업할 때 참고할 컨텍스트입니다.
데이콘 해커톤: 투구별 제구 성공 확률(`control_success`) 예측, 평가지표는 Brier Skill Score.

```
Score = max(0, 100000 × (1 - Brier / (r(1-r))))   # r = 평가 데이터 실제 평균 성공률
```

**현재 상태**: 리더보드 최고점 **990.9528** (`aimers_tuned_ensemble.ipynb` v5, 2026-08-19 확정).
932.96 → 959.43(Optuna+fold10+seed3) → 987.39(조건부통계+재중심화) → 990.95(트랙맨 매핑 v2).
목표: 최소 1100점. **마감 9/2.**

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
                            # ⚠️ 1군 전용이 아니다. 2024 기준 38%가 2군(퓨처스, MIN_* 팀코드)이고
                            #    올스타(KBO_*)/기타(ACE_*)도 섞여 있다. 2장 '트랙맨 매핑' 참고


                            # 이하 항목은 사용자가 만든 것으로 주최측 공식 제공은 아님.
pitcher_id_mapping.csv      # pitcher_id <-> pitcher_trackman_id  ⚠️ 약 91% 틀림 — 2장 참고
pitcher_id_mapping_v2.csv   # 재구축본(1,686행). 2026-08-19 생성, 이걸 쓸 것
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

### ⚠️ 데이터의 두 가지 핵심 성질 (2026-08-18 분석)

**1) train은 2019~2024만, test는 2025.** 시즌이 완전히 분리돼 있다 →
train으로 만든 통계를 test에 쓰는 건 전부 "과거 정보"라 규정상 안전하다.
동시에 **평가는 항상 "미래 시즌 예측"**이므로, 피처 선별은 반드시 **2024 시즌 홀드아웃**
(train=2019~2023 → valid=2024)으로 할 것. 랜덤 shuffle CV는 이 난이도를 반영하지 못한다.

**2) 성공률이 매 시즌 단조 하락한다 (드리프트).**

| 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|
| .5647 | .5327 | .5328 | .5289 | .5000 | .4861 |

- **선수 구성 변화가 아니라 리그 전체 현상**: 양 시즌 500구+ 던진 동일 투수도 2023→24에 평균
  -0.0107(60%가 하락). Oaxaca 분해로 2022→23은 내부효과 102%, 2023→24는 내부 52%/구성 48%.
- 등판량 티어별로는 **핵심 1군(2000구+)이 -0.0051로 덜 떨어지고** 나머지가 -0.010~-0.013.
- **운영측 `asof_pitcher_success_rate`는 커리어 누적이라 이 하락을 못 따라가서 2024에 +0.0253
  과대평가** 중이다. 가장 중요한 피처가 최근 시즌일수록 위로 편향돼 있다.
- 추세 외삽은 신뢰할 만함: 2019~23만으로 2024를 예측하면 오차 0.0016(최근3년 선형).
  같은 방식으로 **2025 예상 베이스레이트는 0.462~0.475.**
- 단, 전체 피처를 쓰면 모델이 스스로 상당 부분 따라간다(2024 평균예측 0.4940 vs 실제 0.4861).
  그래서 재중심화 이득은 크지 않다(+18점). 자세한 건 4-11 참고.

**3) `asof_pitcher_n`은 경기내가 아니라 커리어 누적** (중앙값 1,776 / 최대 15,449).
이걸 경기내 투구수로 오해해서 만든 피처들이 죽어 있다 — 아래 2장 참고.
경기 ID도 날짜도 없고 test 내부 행 순서 사용은 금지라 **경기내 피로도는 복원 불가능**하다.

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
| **조건부 투수통계** (`attach_cond_features`) | `cond_p/pc/ph/phc` — pitcher, ×카운트, ×타자좌우, ×둘다. **시즌 디트렌드 + 0으로 shrink** |

**현재 제출 파이프라인은 `step1~18` + 조건부 투수통계를 쓴다. `zone_speed` 기반 체감구속
피처는 포함되지 않는다 — 아래 참고.**

### 죽은 피처 (제거됨 — `DEAD_FEATURES`)

`asof_pitcher_n`이 커리어 누적인데 경기내 투구수로 가정해서 만들어져 의도대로 동작하지 않는다:

| 피처 | 실제 동작 |
|---|---|
| `is_long_relief` | 전체의 **86%**(이닝>1 중 97%) → 사실상 `inning>1`과 동일 |
| `is_short_relief` | 2.4% |
| `is_strict_inherited_runner` | **0.05%** → 상수나 다름없음 |
| `pitches_per_inning` | 커리어투구수/이닝 (중앙값 383) → 의미 붕괴 |

제거 효과는 +4점(±2)으로 미미하지만 코드 정합성 차원에서 뺐다.
`is_rookie`/`is_veteran`(684/3725 기준)은 **커리어 기준이 맞으므로 유효**하다.

### 조건부 투수통계 설계 (2026-08-18 추가)

원시 성공률을 그대로 집계하면 과거 시즌의 높은 수준이 섞여 들어오므로,
**"그 시즌 리그평균 대비 편차"로 디트렌드**한 뒤 `sum/(count+C)`로 0(리그평균)에 shrink한다.
표본이 적은 조합일수록 자동으로 0에 수렴해 콜드스타트가 자연히 처리된다.

- 학습: 각 행을 **그 시즌보다 과거** 데이터로만 인코딩 (leak-free, 2019 행은 NaN)
- 추론: 전 시즌으로 만든 룩업 테이블(`model/cond_*.csv`)을 merge
- C값: pitcher 200 / ×카운트 100 / ×좌우 100 / ×셋 50
- **`count_advantage`의 `'None'`이 키에 들어가므로 4-3 라운드트립 버그 주의** (검증 통과 확인)
- ⚠️ 2024 투수의 **19.9%가 직전 5년에 없던 신규 투수** — 2025도 비슷할 것이므로 NaN 폴백 필수

검증(2024 홀드아웃, 3-seed 짝지어): 기준선 대비 **+20(원본)/+27(재중심화)**, 3 seed 전부 우세.
부수 효과로 seed 분산이 ±18 → ±6으로 감소.

### 재중심화 (2026-08-18 추가)

시즌 드리프트로 모델 평균예측이 위로 뜨는 걸 로짓 공간 상수 시프트로 보정한다.
**규정 주의**: 추론 시점에 "test 예측 평균 = 목표값"이 되도록 푸는 방식은 평가 데이터 전체를
보고 만든 사후 보정값에 해당할 소지가 있다. 그래서 **학습 시점에 홀드아웃
(~2023 학습 → 2024 예측 → 정답 아는 2024 평균에 맞춤)으로 오프셋을 측정해
`train_constants.json`의 `recenter_offset`에 상수로 박고**, 추론은 그 상수를 더하기만 한다
(Cell 6c). 구조가 "Y까지 학습 → Y+1 예측"으로 최종 모델과 동일하다. 효과 +18점.

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
- `asof`가 `exact`보다 +36점. ~~트랙맨에 신호가 있다는 증거~~ → **2026-08-19 정정**:
  `exact`는 test에서 트랙맨 20개 피처가 전부 `fillna(0)`이 되는데 모델은 실제값으로 학습된 상태다.
  즉 +36은 **학습/추론 분포 불일치를 없앤 효과**이지 트랙맨의 기여가 아니다.
  트랙맨 자체 기여는 매핑을 고쳐도 +9(±2) 수준이다 (바로 아래).

### 트랙맨 매핑 재구축 (2026-08-19) — 기존 매핑이 약 91% 틀려 있었다

**증상**: 트랙맨 21개 피처를 통째로 빼도 2024 홀드아웃 점수가 안 변했다
(3-seed 짝지어 +1±15). 신호가 없다는 뜻이다.

**원인 두 겹**

1. **`pitcher_id_mapping.csv`가 대부분 틀렸다.** 총 965행뿐이고(2022~24는 시즌당 ~100명,
   투구 커버리지 **28%**), `distance` 컬럼(0.015~0.029)이 말해주듯 **구종비율 하나로만**
   매칭돼 있었다. 순환되지 않는 지표로 재보면 무너진다:
   - 시즌 간 일관성 **1.9%** (같은 trackman_id가 시즌마다 다른 pitcher_id로 감)
   - train 투구량 증분 vs 트랙맨 투구수 상관 **0.20** (무작위 셔플은 0.02)
2. **트랙맨의 38%가 2군 경기**인데 구분 없이 한 투수 평균으로 뭉개지고 있었다
   (906명 중 430명이 1군·2군 혼재).

**재구축 방법 (팀 → 투수 2단계)**

- 팀: `(월 × 요일 × 공수)` 63차원 투구량 프로파일로 헝가리안 매칭.
  **10개 팀이 6시즌 내내 동일 프랜차이즈로 대응** → 사실상 정답.
  `12:두산 13:LG 14:키움 15:롯데 16:KIA 17:한화 18:삼성 19:NC 20:KT 21:SSG(SK)`
  (⚠️ 월 단위 9차원으로는 실패한다 — 팀별 월간 분포가 거의 동일해서 비용이 평평해지고
  헝가리안이 무작위 배정을 한다. 일치율 10%였다.)
- 투수: 팀-시즌 안에서 등판 프로파일 + 이닝 분포 + 구종배합 + 총투구량, 손은 하드제약.
  ⚠️ **손 코딩이 다르다**: train은 `1=Left, 2=Right` 정수, 트랙맨은 `'Left'/'Right'` 문자열.
- 마지막에 시즌 간 다수결로 교정 (1,686쌍 중 44쌍).

| | 시즌간 일관성 | 2024 커버리지 | 기존과 일치 |
|---|---|---|---|
| 기존 `pitcher_id_mapping.csv` | 1.9% | 28.2% | — |
| 신규 `pitcher_id_mapping_v2.csv` | **90.9%** (고신뢰 구간 100%) | **94.0%** | 9.1% |

**효과 (2024 홀드아웃, 3-seed 짝지어, 기준 = 트랙맨 없음)**

| 설정 | 원본 | 재중심화 |
|---|---|---|
| 현행 매핑 | +8(±13) | +8(±12) |
| **새 매핑, 2군 필터 없음** | **+9(±5)** | **+9(±2)** |
| 새 매핑, 1군만 | +9(±18) | +5(±20) |
| 새 매핑 + 2군 등판비율 피처 | +10(±19) | -0(±22) |

평균이 아니라 **편차**가 핵심이다. 현행 매핑의 +8은 ±12라 "0일 수도" 있는 반면
새 매핑은 3 seed 전부 일관되게 이긴다. 예상과 어긋난 것 두 가지:
- **2군을 걸러내면 오히려 나빠진다** (결측 9.7%→14.1%). 정확도보다 커버리지가 중요하다.
  과거 'season 매핑'(정확하지만 커버리지 38%) 실험이 실패한 것도 같은 이유.
- **2군 등판 비율 피처는 무효**. train에 없는 새 정보라 기대했으나 재중심화 기준 -0±22.

**주의**: 이 홀드아웃은 트랙맨에 유리하다. 2024 검증행은 최신 트랙맨을 받지만
실제 2025 test는 1년 묵은 2024 값을 받는다. 실전 이득은 +9보다 작을 수 있다.

**리더보드 실측 (v5, 2026-08-19): +3.56 (987.3936 → 990.9528).**
예측 +9의 약 40%다. 위 '주의'가 그대로 실현된 셈이니, **앞으로 트랙맨 관련 홀드아웃
수치는 절반쯤 할인해서 읽을 것.** 참고로 누수 없는 중간 지표(오프셋 측정 셀의
~2023 학습 → 2024 예측)는 v4 777 → v5 784 로 +7 이었고, OOF 는 Brier 0.24401 → 0.24395.
둘 다 방향은 맞았으나 리더보드보다 낙관적이었다.

**폐기**: 트랙맨 추가 파생(속도변화율, 구종 엔트로피, 무브먼트 일관성 추세 등)은
밑에 깔린 집계가 pitcher×month 평균이라 이미 뭉개져 있다. zone_speed가 실패한 것과 같은 이유.

---

## 3. 모델링

### 현재 채택 구성 (990.9528점, `aimers_tuned_ensemble.ipynb` v5)

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
| **v5: 트랙맨 투수 매핑 재구축** | **990.9528 (987.39 대비 +3.56점) → 채택, 현재 최고점** |
| v4: 조건부 투수통계 + 재중심화 + 죽은 피처 제거 | 987.3936 (959.43 대비 +27.97점) |
| Optuna 튜닝 + fold10 + seed3 앙상블 | 959.4258 (932.96 대비 +26.47점) → v4의 기반 |
| `prior_mean` 3안 비교 (현행 0.5352 / `control_success.mean()` 0.5238 / 시즌별) | 전부 오차범위 안 → 현행 유지. 4-1 참고 |
| LightGBM 85:15 앙상블 | TimeSeriesSplit 시절 실험, 재검토 필요 |
| 타자측 트랙맨 피처 6개 + zone_speed 체감구속 (동시) | 930.00 (932.96 대비 -2.96점) → 롤백 |
| zone_speed 체감구속 **단독** (타자 피처 없이) | 929.6060 (932.96 대비 -3.35점) → 롤백. 원인이 zone_speed였음을 확정 — 두 실험의 낙폭이 거의 같아서 타자 피처는 결백 |

### 2024 홀드아웃 스크리닝 결과 모음 (3-seed 짝지어, 트랙맨 제외 300iter 단일모델 기준)

절대값이 아니라 **기준선 대비 차이**만 의미가 있다. 단일 seed는 ±32점 노이즈이므로 4-10 참고.

| 시도 | 기준선 대비 | 판정 |
|---|---|---|
| **손실함수 RMSE** (Logloss 대신) | **원본 +15(±6) / 재중심화 +13(±10)** | ✅ 채택 후보. 3seed 전부 우세, 편차 ±1로 매우 안정 |
| 조건부 투수통계 | 원본 +20(±15) / 재중심화 +27(±17) | ✅ 채택 (v4 반영) |
| 재중심화 | 전 설정 +11~22 (평균 +18) | ✅ 채택 (v4 반영) |
| 죽은 피처 제거 | +4(±2) | ➖ 미미하나 코드 정합성 차원에서 채택 |
| `asof_*` 시즌 디트렌드 | 원본 **-13(±10)** / 재중심화 +5(±13) | ❌ 기각. 조건부 통계와 중복으로 추정 |
| 최근가중 ×1.5 / ×2 학습 | -33 / -118 | ❌ 기각 |
| 2021~23만 / 2022~23만 학습 | -62 / -128 | ❌ 기각. 데이터 양이 최신성을 이긴다 |

**RMSE 도입 시 필요한 변경**: `CatBoostClassifier` → `CatBoostRegressor(loss_function='RMSE')`,
`predict_proba`→`predict`+클리핑, **Optuna 재탐색 필수**(현 `best_params`는 Logloss 기준),
isotonic 보정이 회귀 출력에도 유효한지 재확인.

### 실행 불가능으로 확인된 것 (로드맵에서 제외)

- **`middle_rate`/`reverse_rate` 분해 보조모델**: 투구별 실패유형 라벨이 없다.
  운영측은 `asof_pitcher_middle_rate` 등 **누적 비율만** 제공하고 타겟은 `control_success` 하나뿐이라
  보조 라벨을 만들 수 없다.
- **경기내 피로도(투구수/이닝)**: 경기 ID도 날짜도 없고, test 내부 행 순서 사용은 규정 금지.

### 노트북 계보 (혼동 방지용)

| 노트북 | 내용 | 리포 포함 여부 | 점수 |
|---|---|---|---|
| `aimers_tuned_ensemble.ipynb` (현재 최고) | v5 = v4 + 트랙맨 매핑 재구축 | ✅ 커밋됨 | **990.9528 (현재 채택)** |
| ↑ 같은 노트북 v4 | Optuna + fold10 + seed3 + 조건부통계 + 재중심화 | (히스토리) | 987.3936 |
| ↑ 같은 노트북 v3 | 조건부통계·재중심화 없던 버전 | (히스토리) | 959.4258 |
| `aimers_offset.ipynb` | v4에서 오프셋 측정만 떼어낸 축소 커널 (4-12 참고) | ✅ 커밋됨 | — |
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
~~`asof_pitcher_success_rate.mean()`이 아니라 `control_success.mean()`을 써야 함~~
→ **2026-08-19 측정 결과 무관함이 확인돼 정정.** 3안을 2024 홀드아웃 3-seed 짝지어 비교:

| prior | 값 | 기준선(현행) 대비 |
|---|---|---|
| A 현행 `asof_pitcher_success_rate.mean()` | 0.535228 | (기준) |
| B `control_success.mean()` | 0.523766 | 원본 +5(±9) / 재중심화 **-2**(±13) |
| C 시즌별 리그평균 (2024는 외삽 0.4877) | 시즌마다 | 원본 +13(±22) / 재중심화 +5(±20) |

셋 다 노이즈 안이고 B는 부호조차 안 맞는다. 이유: `smoothed = (n*curr + 50*prior)/(n+50)`
인데 `asof_pitcher_n < 200`인 행이 **9.0%**뿐이라 prior 가 힘을 쓰는 구간이 좁다.
A와 B는 값 차이가 0.011이라 최대 변화폭 0.0112로 사실상 같은 피처다.
C는 최대 0.0466까지 벌어지는데도 효과가 없는데, 전체 피처를 쓰면 모델이 드리프트를
다른 경로로 이미 흡수하기 때문으로 보인다 (4-11과 같은 패턴).
**현행 A를 유지한다. 다만 "반드시 저장해야 한다"는 본문은 그대로 유효하다.**

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

### 4-10. ⚠️ 단일 seed 홀드아웃 비교는 ±32점 노이즈 — 피처 판정에 쓰면 안 됨
2024 홀드아웃에서 **완전히 동일한 설정을 seed만 바꿔 돌리면 685 vs 717로 32점이 움직인다.**
이것 때문에 조건부 투수통계를 단일 seed로 평가했을 때 "효과 ±0"으로 잘못 판정할 뻔했다
(하필 기준선이 가장 잘 나온 seed와 붙었음). **같은 seed로 짝지어(paired) 최소 3회 반복**하고
차이의 평균과 표준편차를 함께 볼 것. 짝지으면 seed 효과가 상쇄돼 검정력이 크게 올라간다.

### 4-11. ⚠️ 피처를 뺀 축약 모델로 설계 결정을 내리지 말 것 (두 번 오도당함)
`step1~13` 파생피처 없이 원시 컬럼만으로 돌린 축약 실험이 **두 가지 결론을 모두 반대로** 만들었다.

| 항목 | 축약 모델 결론 | 실제(전체 피처) |
|---|---|---|
| 재중심화 이득 | +224~301점 | **+18점** |
| 최근 시즌 편중 학습 | +224점 (좋음) | **-30~-120점 (나쁨)** |

이유: 축약 모델은 `smoothed_pitcher_success_rate`/모멘텀 피처가 없어 드리프트를 못 따라가므로,
평균예측이 0.5107로 크게 뜨고(전체 피처는 0.4940) "최근 데이터만 쓰기"가 그 결함의 임시방편으로
작동했다. 전체 피처가 있으면 데이터 양이 최신성을 이긴다. **속도 때문에 피처를 빼는 순간
결론의 방향 자체가 뒤집힐 수 있다.**

### 4-12. 학습 뒤쪽 셀의 사소한 NameError로 2시간 GPU를 날릴 뻔함 (+ 회수 방법)

v4 실행에서 재중심화 셀이 `RANDOM_SEED`(정의된 적 없는 이름)를 참조해
**30개 모델 학습이 다 끝난 직후** `NameError`로 죽었다. 캐글 커널은 ERROR로 종료됐다.

- **교훈 1**: 학습(1시간+) *뒤에* 오는 셀은 반드시 먼저 문법·이름 검사를 하고 push할 것.
  `python -c "import ast; ast.parse(open('x.py').read())"`는 NameError를 못 잡는다.
  셀 안에서 쓰는 전역 이름이 앞 셀에 실제로 있는지 눈으로 확인하는 수밖에 없다.
  (이 노트북의 seed 변수명은 `SEEDS` 리스트다. 단일 seed가 필요하면 `SEEDS[0]`.)
- **교훈 2 (회수 방법)**: 크래시해도 **`kaggle kernels output`으로 산출물은 회수된다.**
  모델 30개 + isotonic 30개 + 룩업 테이블이 전부 남아 있었다. 전체 재실행(2시간) 대신
  **부족한 것만 만드는 축소 커널**을 따로 돌리는 게 맞다:
  Optuna(70분)는 저장된 `best_params.json`을 그대로 읽어 생략, 최종 학습(56분)도 생략,
  전처리(83초) + 오프셋 3 fold(4분)만 → **5분 30초**에 끝났다 (`aimers_offset.ipynb`).
  전처리가 결정적이라 룩업 테이블 7개가 원본과 **md5까지 동일**함을 확인하고 합쳤다.
- **교훈 3**: 로컬에서 제출 zip을 조립할 때는 오프셋이 `script.py`까지 실제로 전달되는지
  **오프셋 0으로 바꾼 대조 실행과 평균예측을 비교**해 확인할 것. 상수가 조용히 무시돼도
  크래시는 안 난다 (4-6과 같은 유형).

### 4-13. 노트북을 bash heredoc 으로 편집하지 말 것

`python - <<'PY' ... PY` 로 노트북 셀을 패치했더니 **백슬래시가 한 겹 벗겨져
`print("\n=== ...")` 의 `\n` 이 실제 줄바꿈이 됐고, 셀이 문법 오류로 깨졌다.**
따옴표로 감싼 heredoc(`<<'PY'`)인데도 이 환경에서는 리터럴이 보장되지 않는다.

**패치 스크립트는 반드시 파일로 써서 실행할 것** (`python scratchpad/patch.py`).
그리고 노트북을 건드린 뒤에는 항상 전 셀 문법 검사를 돌린다:

```python
import ast, json
nb = json.load(open('x.ipynb', encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']))
```

### 4-14. 변경 하나를 측정할 땐 Optuna 를 끌 것 (`RUN_OPTUNA`)

v4 는 Optuna 를 재탐색해서 파라미터가 같이 바뀌었다(depth 9→8, lr 0.0247→0.0228).
그래서 987.39 라는 결과가 "세 가지 피처 변경" 때문인지 "파라미터 변화" 때문인지
구분할 수 없었다. v5 부터는 설정 셀의 `RUN_OPTUNA = False` 로 v4 파라미터를 고정한다
(`V4_BEST_PARAMS`). 덕분에 v5 의 +3.56 은 트랙맨 매핑 하나의 효과로 귀결된다.
부수 효과로 실행시간이 **128분 → 62.7분**으로 줄었다 (Optuna 가 70분을 먹고 있었다).

Optuna 를 다시 켤 때는 그것 하나만 바꿔서 돌릴 것.

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
| ~8/18 | ✅ Optuna 튜닝 + fold10 + seed3 앙상블 — 959.4258 |
| 8/19 | ✅ v4(조건부통계+재중심화+죽은피처제거) 987.3936 / ✅ v5(트랙맨 매핑 v2) — **990.9528로 채택** |
| 8/20~8/27 | 남은 카드 2장: **RMSE 손실**(+15±6이나 300iter/depth6 축약 조건 — 실제 용량에서 재확인 필수, 4-11 참고), **`asof_*` 재설계**(카운트/구종/좌우별 세분화, 최근 N구 이동평균) — 후자가 가장 큰 미개척지 |
| 8/28~8/31 | 여유 시: `middle_rate`/`reverse_rate` 분해 기반 보조모델, LightGBM 재도입 |
| 9/1~9/2 | **버퍼 — 신규 시도 금지.** 그 시점 최고 검증 버전 고정, 최종 제출만 |
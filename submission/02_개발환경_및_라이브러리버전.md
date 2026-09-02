# 개발 환경 및 라이브러리 버전

Private Score **1,153.6031** (`v10wtor`) 를 낸 구성이다.
학습·추론·분석이 서로 다른 세 환경에서 돌았으므로 나눠 적는다.

---

## 1. 학습 환경 — Kaggle GPU 커널

| 항목 | 값 |
|---|---|
| OS | Linux (Kaggle 기본 컨테이너 이미지) |
| Python | 3.12 |
| GPU | NVIDIA Tesla P100 16GB (커널 기본 할당) |
| 주 라이브러리 | `catboost` (커널 이미지 기본 설치, GPU 빌드) |
| 학습 시간 | 30모델 약 63분 |
| 학습 완료 시각 | `2026-08-31T07:58:39Z` (모델 파일 `train_finish_time` 메타데이터) |

학습 노트북: **`experiments/v10w/aimers_v10wtor.ipynb`**

⚠️ CatBoost 는 **처리 장치에 따라 기본값이 다르다.** 같은 파라미터라도 CPU 에서는
`border_count=254` / `bootstrap_type=MVS`, GPU 에서는 `128` / `Bayesian` 으로 해석된다.
이 프로젝트에서 두 환경의 홀드아웃 점수가 26점 차이 났고 원인이 이것이었다.
**재현하려면 반드시 GPU(`task_type='GPU'`)로 학습할 것.**

⚠️ CatBoost GPU 학습은 실행 간 비결정적이다. 같은 코드·같은 seed 로 다시 돌려도
점수가 ±1.5 정도 움직인다. 정확히 같은 예측이 필요하면 학습을 다시 하지 말고
저장된 모델 파일(`model/cb_fold_*.cbm`)을 쓸 것.

### 모델 구성

```
CatBoost MultiClass (5분류)   10-fold x 3-seed = 30 모델
seeds [42, 202, 2024] | iterations 1000 | depth 8 | learning_rate 0.0228
폴드별 IsotonicRegression 보정 (이진 y 기준)
재중심화 오프셋 -0.0741 (학습 시점 홀드아웃에서 측정한 상수)
피처 130개 | OOF 스킬 2.222%
```

---

## 2. 추론 환경 — 주최측 평가 서버

인터넷 접속이 불가하고 아래 패키지가 **버전 고정된 채로 사전 설치**되어 있다.
그래서 `requirements.txt` 에는 사전 설치 목록에 없는 것만 적는다.

```
requirements.txt
  catboost
```

사전 설치되어 있어 **다시 적지 않은** 것 (적으면 버전 충돌로 설치 오류가 난다):

```
pandas 2.0.3 · numpy 1.26.4 · scipy 1.15.3 · scikit-learn 1.8.0
joblib 1.5.3 · threadpoolctl 3.6.0 · torch 2.7.1+cu128 · transformers 4.46.3
```

제출 zip 구성 (파일 80개):

```
script.py, requirements.txt
model/cb_fold_{seed}_{fold}.cbm          30개
model/isotonic_fold_{seed}_{fold}.pkl    30개
model/aux_rev_{fold}.cbm                  3개   (보조 타겟 교차적합 모델)
model/selected_features.json, train_constants.json, best_params.json
model/cond_*.csv, feat_*.csv, ws_*.csv    룩업 테이블
```

`script.py` 는 학습 노트북과 **같은 전처리 소스 문자열**(`STEPS_SRC`)을 그대로
삽입해서 만든다. 학습 경로와 추론 경로가 갈라지는 것이 이 프로젝트에서 가장 비쌌던
버그 유형이라(트랙맨 병합 방식 불일치로 36점), 소스를 하나로 묶었다.

---

## 3. 분석·검증 환경 — 로컬

```
Windows 11 (10.0.26200) · AMD64 · Python 3.10.10

numpy        1.26.4
pandas       2.1.4
scikit-learn 1.3.2
scipy        1.15.3
joblib       1.3.2
catboost     1.2.10
matplotlib   (그림 생성용)
Pillow       (그림 검증용)
```

`tools/` 아래 스크립트가 전부 이 환경에서 돈다. GPU 를 쓰지 않는다.

---

## 4. 재현 절차

```bash
# 1) 데이터 배치
data/train.csv  data/test.csv  data/sample_submission.csv  data/trackman_history.csv

# 2) 학습 (Kaggle GPU 커널에서 experiments/v10w/aimers_v10wtor.ipynb 실행)
#    노트북이 끝나면 제출 zip 까지 스스로 조립한다

# 3) 제출 전 검증 세 가지 — 노트북 안에서 자동으로 돈다
python tools/audit_independence.py   # 평가 데이터 행 독립성 (주최측 규정)
python tools/verify_ws_live.py       # 상수가 추론 경로까지 살아서 전달되는지
#    + 노트북 마지막 셀이 script.py 를 서브프로세스로 실제 실행해 채점
```

### 검증 실측치 (`v10wtor`)

| 검증 | 결과 |
|---|---|
| 행 독립성 (같은 행 단독 예측 vs 전체와 함께) | 최대 차이 `1.11e-16` |
| 라벨 복원 `success` vs 정답 (147만 행) | 일치율 `1.000000` |
| `ws_*` 상수 생존 (대조군 방식) | 행별 차이 0 이 아닌 행 `100.0%` |
| `script.py` 홀드아웃 실채점 | 정상 |

---

## 5. CAE 지표 재현 (솔루션 PPT 8~11쪽)

발표 자료의 지표는 **대회 제출 모델과 다른 기대값**을 쓴다. 8쪽에 적어둔 대로,
제출 모델은 입력에 투수 누적 통계를 갖고 있어 기대값에 실력이 이미 들어가 있다.

```bash
python tools/mk_cae.py        # 홀드아웃 예측 커널 생성 (Kaggle GPU, 약 11분)
python tools/cae_selftest.py  # ★ 구현 교정. 이걸 통과 못 하면 아래 숫자는 무의미하다
python tools/cae.py           # 지표 + 검증 3종 -> out/cae_metrics.json
python tools/cae_fig.py       # 그림 5장 -> deck/fig/
```

`cae_selftest.py` 는 반쪽 분할 신뢰도 구현을 **이미 아는 값으로 교정**한다.
`0.637` / `-0.008` / `0.227` / `-0.089` 를 소수 셋째 자리까지 재현해야 통과한다.
두 번째는 **0 이 나와야 정상인 대조군**이고, 첫 구현은 거기서 `+0.743` 을 내며
스스로 틀렸음을 알렸다.

발표 자료 빌드:

```bash
chrome --headless=new --no-pdf-header-footer --virtual-time-budget=25000 \
       --print-to-pdf=deck/solution.pdf deck/index.html
```

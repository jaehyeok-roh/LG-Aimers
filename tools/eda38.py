# EDA 38 — 재중심화 오프셋이 2025 에도 맞는가 (현행 107 피처로 재측정).
#
# eda37 에서 나온 예상 밖의 숫자:  재중심화 없음 760 -> 있음 894.  **+134** 다.
# claude.md 의 '+18' 은 v5 피처 기준(편향 +0.008)이고 지금은 편향이 +0.0182 다.
#
# 오프셋 오차 δ 의 대가 = δ²/U × 1e5 ≈ δ² × 400,000
#   δ=0.005 -> 10점 | 0.010 -> 40점 | 0.015 -> 90점 | 0.018 -> 130점
#
# 배포본은 b_2024 를 그대로 2025 에 쓴다. 2026-08-21 에 'b ~ 하락폭' 으로 검증해
# '차이 1점' 이라 결론냈는데 그건 **옛 피처**로 잰 b 이고 하락폭도 **R/F 혼합**이었다.
# 8/23 에 그 서사가 깨졌다:  2022->23 전체 -0.0290  vs  R 전용 -0.0006.
# 하락폭 정의에 따라 b_2025 예측이 갈리므로 네 점을 다시 잰다.
#
# ⚠️ claude.md 규칙: 외삽 모형은 **반드시 한 점을 빼고 맞혀보게 할 것**
#    (8/21 에 이 검증이 224점짜리 실수를 잡았다).
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'tools')
import importlib.util
spec = importlib.util.spec_from_file_location('sc', 'tools/screen.py')
sc = importlib.util.module_from_spec(spec)
sys.modules['sc'] = sc
spec.loader.exec_module(sc)

CACHE = 'cache'
ITERS = 1000
YEARS = [int(v) for v in os.environ.get('E38_YEARS', '2021,2022,2023,2024').split(',')]

X = pd.read_pickle(f'{CACHE}/X.pkl')
y = np.load(f'{CACHE}/y.npy')
season = np.load(f'{CACHE}/season.npy')
meta = json.load(open(f'{CACHE}/meta.json', encoding='utf-8'))
for c in X.columns:
    if X[c].dtype == np.float64:
        X[c] = X[c].astype(np.float32)

gt = X['game_type'].astype(str).to_numpy()
isR = gt == 'R'
params = dict(meta['best_params'])
params.update(iterations=ITERS, cat_features=meta['cat_features'], task_type='CPU',
              thread_count=-1, verbose=0, allow_writing_files=False)

# ---- 리그 평균과 하락폭: 전체 vs R 전용 ----
print('=' * 74)
print('시즌 리그평균  (b 를 설명할 후보 두 가지)')
print(f'{"시즌":<8}{"전체":>10}{"R 전용":>10}{"F 비중":>9}{"전체 낙폭":>11}{"R 낙폭":>10}')
print('-' * 74)
lg_all, lg_R = {}, {}
for s in sorted(set(season.tolist())):
    m = season == s
    lg_all[s] = float(y[m].mean())
    lg_R[s] = float(y[m & isR].mean())
    fs = float((~isR[m]).mean())
    da = lg_all[s] - lg_all.get(s - 1, np.nan)
    dr = lg_R[s] - lg_R.get(s - 1, np.nan)
    print(f'{s:<8}{lg_all[s]:>10.4f}{lg_R[s]:>10.4f}{fs:>9.1%}'
          f'{da:>11.4f}{dr:>10.4f}' if s > min(lg_all) else
          f'{s:<8}{lg_all[s]:>10.4f}{lg_R[s]:>10.4f}{fs:>9.1%}{"—":>11}{"—":>10}')

# ---- 피처 (HOLDOUT 과 무관하게 한 번만 만든다) ----
W, B = sc._wseason5_cols(), sc._wsbat_cols()
for D in (W, B):
    for k, v in D.items():
        X[k] = np.asarray(v, dtype=np.float32)
print(f'\n피처 {X.shape[1]}개 | 행 {len(X):,}\n', flush=True)

# ---- 각 해의 편향 b 를 배포와 같은 구조로 잰다 ----
rows = []
for Y in YEARS:
    mh, mv = season <= Y - 1, season == Y
    Xh, Xv = X[mh].reset_index(drop=True), X[mv].reset_index(drop=True)
    yh, yv = y[mh], y[mv]
    p = sc.cv_predict(Xh, yh, Xv, params, meta['cat_features'])
    rv = isR[mv]
    b_all = float(p.mean() - yv.mean())
    b_R = float(p[rv].mean() - yv[rv].mean())
    b_F = float(p[~rv].mean() - yv[~rv].mean()) if (~rv).any() else np.nan
    rows.append(dict(Y=Y, n_tr=int(mh.sum()), n_va=int(mv.sum()),
                     b_all=b_all, b_R=b_R, b_F=b_F,
                     d_all=lg_all[Y] - lg_all[Y - 1], d_R=lg_R[Y] - lg_R[Y - 1]))
    r = float(yv.mean())
    loss = b_all ** 2 / (r * (1 - r)) * 1e5
    print(f'  Y={Y}  학습 {mh.sum():>9,} -> 검증 {mv.sum():>8,} | '
          f'b_all {b_all:+.4f}  b_R {b_R:+.4f}  b_F {b_F:+.4f} | '
          f'무보정 손실 {loss:>6,.0f}점', flush=True)

d = pd.DataFrame(rows)
print('\n' + '=' * 74)
print('편향 b 와 하락폭')
print(f'{"Y":<7}{"b(전체)":>10}{"b(R)":>10}{"b(F)":>10}{"낙폭(전체)":>12}{"낙폭(R)":>11}')
print('-' * 74)
for _, r in d.iterrows():
    print(f'{int(r.Y):<7}{r.b_all:>+10.4f}{r.b_R:>+10.4f}{r.b_F:>+10.4f}'
          f'{r.d_all:>+12.4f}{r.d_R:>+11.4f}')


def fit_pred(xs, ys, x_new):
    """1차 회귀. 점이 2개면 직선, 1개면 상수."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) >= 2:
        a, c = np.polyfit(xs, ys, 1)
        return float(a * x_new + c)
    return float(ys.mean())


print('\n' + '=' * 74)
print('★ 한 점 빼고 맞히기 (외삽 모형 검증 — 8/21 에 이게 224점을 잡았다)')
print(f'{"모형":<26}{"대상":>7}{"예측 b":>10}{"실측 b":>10}{"오차":>9}{"손실":>8}')
print('-' * 74)
MODELS = [('A: b ~ 연도', 'Y', 'b_all'),
          ('B: b ~ 낙폭(전체)', 'd_all', 'b_all'),
          ('C: b ~ 낙폭(R)', 'd_R', 'b_all'),
          ('D: 직전 b 재사용 (현행)', None, 'b_all')]
score = {}
for lab, xc, yc in MODELS:
    tot = 0.0
    for i in range(1, len(d)):                     # 앞 i 점으로 i 번째를 맞힌다
        tgt = d.iloc[i]
        prev, act = d.iloc[:i], float(tgt[yc])
        pr = float(prev.iloc[-1][yc]) if xc is None else \
            fit_pred(prev[xc], prev[yc], float(tgt[xc]))
        r = lg_all[int(tgt.Y)]
        loss = (pr - act) ** 2 / (r * (1 - r)) * 1e5
        tot += loss
        print(f'{lab:<26}{int(tgt.Y):>7}{pr:>+10.4f}{act:>+10.4f}'
              f'{pr-act:>+9.4f}{loss:>8,.0f}')
    score[lab] = tot
    print(f'{"":<26}{"합계":>7}{"":>10}{"":>10}{"":>9}{tot:>8,.0f}')
    print('-' * 74)
best = min(score, key=score.get)
print(f'가장 잘 맞히는 모형: **{best}**  (누적 손실 {score[best]:,.0f}점)')

# ---- 2025 외삽 ----
print('\n' + '=' * 74)
print('2025 외삽 — 하락폭 정의가 갈리므로 양쪽 다 낸다')
print('-' * 74)
ys_all = np.array([lg_all[s] for s in sorted(lg_all)][-3:])
ys_R = np.array([lg_R[s] for s in sorted(lg_R)][-3:])
xs = np.array(sorted(lg_all)[-3:], float)
n25_all = float(np.polyval(np.polyfit(xs, ys_all, 1), 2025))
n25_R = float(np.polyval(np.polyfit(xs, ys_R, 1), 2025))
d25_all, d25_R = n25_all - lg_all[2024], n25_R - lg_R[2024]
print(f'  2025 리그평균 외삽(최근3년 선형)   전체 {n25_all:.4f} (낙폭 {d25_all:+.4f})'
      f'   R 전용 {n25_R:.4f} (낙폭 {d25_R:+.4f})')

cur = float(d[d.Y == 2024]['b_all'].iloc[0])
print(f'\n  현행 배포 = b_2024 재사용 = {cur:+.4f}')
print(f'{"모형":<26}{"b_2025 예측":>13}{"현행과 차이":>13}{"현행이 틀렸을 때 손실":>22}')
print('-' * 74)
r25 = n25_all
for lab, xc, _ in MODELS:
    if xc is None:
        pr = cur
    elif xc == 'Y':
        pr = fit_pred(d['Y'], d['b_all'], 2025)
    else:
        pr = fit_pred(d[xc], d['b_all'], d25_all if xc == 'd_all' else d25_R)
    loss = (cur - pr) ** 2 / (r25 * (1 - r25)) * 1e5
    print(f'{lab:<26}{pr:>+13.4f}{pr-cur:>+13.4f}{loss:>22,.0f}')

print("""
읽는 법:
  '한 점 빼고 맞히기' 에서 **현행(D)이 이기면** 오프셋은 그대로 둔다.
  다른 모형이 뚜렷이 이기고 그 모형의 b_2025 가 현행과 크게 다르면 -> 바꿀 값어치.
⚠️ 하락폭 정의(전체 vs R)가 갈리는 것이 이 표의 핵심이다. 8/23 에 전체 하락폭
   서사가 F 체제 변경 때문이었음이 밝혀졌으므로 C 가 이론적으로 옳다.
   하지만 이론이 이긴 적이 없다 (대리지표 3연패) — 검증표로만 판정한다.""")

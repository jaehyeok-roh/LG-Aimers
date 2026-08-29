# 우리 점수가 **어디서** 나오는지 쪼갠다. 이 프로젝트가 한 번도 안 한 것이다.
#
# claude.md 의 모든 수치는 전체 평균이다. 그런데 평가 데이터의 11.8% 가
# game_type=F(퓨처스)이고, F 는 2023 에 라벨 체제가 통째로 바뀌어 2023 홀드아웃을
# **쓸 수 없게** 만들었던 구간이다. 그 슬라이스가 스킬 0 이거나 음수면
# 후보를 더 얹는 것보다 그걸 고치는 쪽이 훨씬 크다.
#
# 분해 방식 (합이 정확히 전체 점수가 되게):
#   점수 = 100000 x (1 - Brier / (r(1-r)))     r = 전체 실제 평균
#   세그먼트 기여 = 비중 x (1 - MSE_seg / (r(1-r))) x 100000
#   -> 전 세그먼트 합 = 전체 점수. MSE_seg > r(1-r) 인 구간은 **음수 기여**다.
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('SCREEN_HOLDOUT', '2024')
# 캐글 커널에서는 mk_kscreen 이 이 파일을 screen.py 라는 이름으로 풀어 실행하므로
# `import screen` 이 자기 자신을 가리킨다. 부속 파일로 실은 tools/scr.py 를 쓴다.
if os.path.isdir('/kaggle'):
    sys.path.insert(0, 'tools')
    import scr as S  # noqa: E402
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import screen as S  # noqa: E402

CAND = os.environ.get('EDA42_CAND', 'wsboth')

X, y, season, meta = S.load()
params = dict(meta['best_params'])
params.update(iterations=int(os.environ.get('EDA42_ITERS', '1000')),
              cat_features=meta['cat_features'], task_type='CPU',
              thread_count=-1, verbose=0)
mh, mv = season <= 2023, season == 2024
ctx = {'Xh': X[mh].reset_index(drop=True), 'yh': y[mh],
       'Xv': X[mv].reset_index(drop=True), 'yv': y[mv],
       'sh': season[mh], 'params': params, 'cat': meta['cat_features'],
       'score_mask': None}
ctx['target'] = float(ctx['yv'].mean())
print(f'학습 {mh.sum():,}행 -> 검증 {mv.sum():,}행 | 후보 {CAND}', flush=True)

p = S.recenter(S.CANDS[CAND](ctx), ctx['target'])
yv = ctx['yv'].astype(np.float64)
r = float(yv.mean())
U = r * (1 - r)
se = (np.clip(p, 1e-6, 1 - 1e-6) - yv) ** 2
print(f'\n전체 {S.skill(p, yv):,.0f}점 | r={r:.4f} | U={U:.5f}\n', flush=True)

V = ctx['Xv']
segs = {}
gt = V['game_type'].astype(str).to_numpy()
segs['game_type'] = pd.Series(gt, name='game_type')

# 신규 투수: 커리어 누적이 당해 시즌분과 거의 같다 (직전 시즌 이력 없음)
if 'w_n' in V.columns:
    wn = V['w_n'].to_numpy(dtype='float64')
    an = V['asof_pitcher_n'].to_numpy(dtype='float64')
    segs['투수이력'] = pd.Series(np.where(an - wn < 50, '신규', '경력'))
    segs['당해투구수'] = pd.cut(wn, [-1, 60, 300, 800, 1e9],
                            labels=['0-60', '60-300', '300-800', '800+'])
segs['커리어투구수'] = pd.cut(V['asof_pitcher_n'].to_numpy(dtype='float64'),
                        [-1, 500, 2000, 5000, 1e9],
                        labels=['~500', '500-2k', '2k-5k', '5k+'])
segs['카운트'] = V['count_advantage'].astype(str)
segs['이닝'] = pd.cut(V['inning'].to_numpy(dtype='float64'), [0, 3, 6, 99],
                    labels=['1-3', '4-6', '7+'])

for name, s in segs.items():
    s = pd.Series(np.asarray(s)).fillna('NA').astype(str)
    print(f'--- {name} ---')
    print(f'{"구간":<12}{"비중":>8}{"MSE":>10}{"자체스킬":>10}{"기여점수":>10}')
    tot = 0.0
    for v in sorted(s.unique()):
        m = (s == v).to_numpy()
        frac = m.mean()
        mse = se[m].mean()
        own = (1 - mse / (yv[m].mean() * (1 - yv[m].mean()))) * 100000
        contrib = frac * (1 - mse / U) * 100000
        tot += contrib
        print(f'{v:<12}{frac:>7.1%}{mse:>10.5f}{own:>10,.0f}{contrib:>10,.0f}')
    print(f'{"합":<12}{"":>7}{"":>10}{"":>10}{tot:>10,.0f}\n', flush=True)

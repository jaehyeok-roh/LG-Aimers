# eda43 의 네 해 확장 — 세그먼트별 재중심화 상수가 **해를 넘겨 전이되는가**.
#
# eda43 은 2023->2024 한 쌍으로만 판정한다. eda38 에서 배운 것이 정확히 그 위험이다:
# 한 점으로 외삽했으면 224점을 잃었고, 네 점을 재고 한 점씩 빼서 맞혀본 표가 그걸 잡았다.
# 여기서는 2021~2024 네 해를 각각 '그 전 해까지 학습 -> 그 해 예측' 으로 만들고,
# 전이 쌍 세 개에서 오라클(천장)과 실행가능(전 해 상수를 그대로 적용)을 나란히 본다.
#
# 규정: 세그먼트는 행 A 자기 컬럼(game_type / count_advantage / inning / w_n)으로만
#       정해지고 상수는 train 으로만 만든다 -> 행 독립 시험을 자명하게 통과한다.
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('SCREEN_HOLDOUT', '2024')
if os.path.isdir('/kaggle'):
    # mk_kscreen 이 이 파일을 screen.py 라는 이름으로 푼다 -> 자기 자신을 import 하게 된다.
    sys.path.insert(0, 'tools')
    import scr as S  # noqa: E402
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import screen as S  # noqa: E402

ITERS = int(os.environ.get('EDA43_ITERS', '1000'))
YEARS = [int(v) for v in os.environ.get('EDA44_YEARS', '2021,2022,2023,2024').split(',')]
AXES = ['game_type', 'count_adv', 'inning', 'wn', 'newpit']
PAIRS = ['game_type|count_adv', 'count_adv|wn']


def logit(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def shift_to(lo, target):
    """로짓에 더할 상수 하나를 찾아 평균 예측을 target 에 맞춘다."""
    off = 0.0
    for _ in range(300):
        e = sig(lo + off).mean() - target
        if abs(e) < 1e-10:
            break
        off -= e * 4.0
    return off


def score(p, yv, U):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yv) ** 2).mean() / U) * 100000


def seg_offsets(lo, yv, key):
    """구간마다 '예측 평균 = 실제 평균' 이 되는 로짓 상수."""
    return {v: shift_to(lo[key == v], float(yv[key == v].mean()))
            for v in pd.unique(key)}


def apply_off(lo, key, off, target):
    z = lo + np.array([off.get(k, 0.0) for k in key])
    return sig(z + shift_to(z, target))   # 전역 평균은 원래대로 되돌린다


def run_year(Y, X, y, season, meta):
    S.HOLDOUT = Y
    S.DROP_F = False
    params = dict(meta['best_params'])
    params.update(iterations=ITERS, cat_features=meta['cat_features'],
                  task_type='CPU', thread_count=-1, verbose=0)
    mh, mv = season <= Y - 1, season == Y
    ctx = {'Xh': X[mh].reset_index(drop=True), 'yh': y[mh],
           'Xv': X[mv].reset_index(drop=True), 'yv': y[mv],
           'sh': season[mh], 'params': params, 'cat': meta['cat_features'],
           'score_mask': None}
    print('\n===== %d 홀드아웃: 학습 %s -> 검증 %s =====' % (Y, f'{mh.sum():,}', f'{mv.sum():,}'),
          flush=True)
    W5, WB = S._wseason5_cols(), S._wsbat_cols()
    raw = S._add(ctx, W5, WB)
    yv = ctx['yv'].astype(np.float64)
    lo = logit(raw)
    lo = lo + shift_to(lo, float(yv.mean()))     # 전역 재중심화 (배포와 동일)

    V = ctx['Xv']
    wn = W5['w5_n'][mv].astype(np.float64)
    an = V['asof_pitcher_n'].to_numpy(dtype='float64')
    seg = pd.DataFrame({
        'game_type': V['game_type'].astype(str).to_numpy(),
        'count_adv': V['count_advantage'].astype(str).to_numpy(),
        'inning': pd.cut(V['inning'].to_numpy(float), [0, 3, 6, 99],
                         labels=['1-3', '4-6', '7+']).astype(str),
        'wn': pd.cut(wn, [-1, 60, 300, 800, 1e9],
                     labels=['0-60', '60-300', '300-800', '800+']).astype(str),
        'newpit': np.where(an - wn < 50, 'new', 'vet'),
    })
    print('    %d 전역 재중심화 후 %s점' % (
        Y, f'{score(sig(lo), yv, float(yv.mean() * (1 - yv.mean()))):,.0f}'), flush=True)
    return lo, yv, seg


_W5CHK = S._wseason5_cols()
assert 'w5_n' in _W5CHK, sorted(_W5CHK)[:12]
print('키 검증 OK: w5_n 존재', flush=True)

X, y, season, meta = S.load()
R = {Y: run_year(Y, X, y, season, meta) for Y in YEARS}

print('\n\n' + '=' * 78)
print('세그먼트 편향 (예측평균 - 실제평균), 해마다')
print('=' * 78)
for col in AXES:
    vs = sorted(set().union(*[set(pd.unique(R[Y][2][col])) for Y in YEARS]))
    print('\n[%s]  %s%9s' % (col, ''.join('%10d' % Y for Y in YEARS), '비중'))
    for v in vs:
        row = ''
        for Y in YEARS:
            lo, yv, seg = R[Y]
            m = (seg[col].to_numpy() == v)
            row += '%10.4f' % (float(sig(lo[m]).mean() - yv[m].mean()) if m.sum()
                               else float('nan'))
        m4 = (R[YEARS[-1]][2][col].to_numpy() == v)
        print('%-12s%s%8.1f%%' % (v, row, 100 * m4.mean()))

print('\n\n' + '=' * 78)
print('세그먼트별 재중심화 이득 — 오라클(천장) vs 실행가능(전 해 상수 적용)')
print('=' * 78)
for i in range(1, len(YEARS)):
    Yp, Y = YEARS[i - 1], YEARS[i]
    lo, yv, seg = R[Y]
    lop, yvp, segp = R[Yp]
    U = float(yv.mean() * (1 - yv.mean()))
    tgt = float(yv.mean())
    base = score(sig(lo), yv, U)
    print('\n%d -> %d   기준 %s점' % (Yp, Y, f'{base:,.0f}'))
    print('%-24s%10s%11s' % ('축', '오라클', '실행가능'))
    for col in AXES + PAIRS:
        if '|' in col:
            a, b = col.split('|')
            k = (seg[a] + '|' + seg[b]).to_numpy()
            kp = (segp[a] + '|' + segp[b]).to_numpy()
        else:
            k, kp = seg[col].to_numpy(), segp[col].to_numpy()
        orc = score(apply_off(lo, k, seg_offsets(lo, yv, k), tgt), yv, U) - base
        exe = score(apply_off(lo, k, seg_offsets(lop, yvp, kp), tgt), yv, U) - base
        print('%-24s%+10.1f%+11.1f' % (col, orc, exe))

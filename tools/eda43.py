# 세그먼트별 재중심화 — 상수를 하나가 아니라 구간마다 두면 얼마나 버는가.
#
# eda42 결과: 새는 슬라이스는 없었다 (F 는 오히려 행당 32% 초과 기여).
# 그런데 eda42 는 MSE 만 쟀고 **편향은 안 쟀다.** 상수로 고칠 수 있는 건 편향이다.
# 지금 재중심화는 전 구간에 로짓 상수 **하나**를 민다 (그것만으로 홀드아웃 +126).
#
# 규정: 세그먼트는 행 A 자기 컬럼(game_type / count_advantage / inning / w_n)으로만
#       정해지고 상수는 train 으로만 만든다 -> 행 독립 시험을 자명하게 통과한다.
#
# ⚠️ 반드시 '한 시즌 앞' 구조로 검증한다. eda38 에서 '이론적으로 옳은' 판이
#    최악(276)이었고, 검증표가 224점/46점짜리 실수를 두 번 잡았다.
#      오라클   2024 에서 상수를 뽑아 2024 에 적용   -> 천장
#      실행가능 2023 에서 뽑아 2024 에 적용          -> 실제로 얻는 값
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('SCREEN_HOLDOUT', '2024')
import screen as S  # noqa: E402

ITERS = int(os.environ.get('EDA43_ITERS', '1000'))


def logit(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def shift_to(lo, target, w=None):
    """로짓에 더할 상수 하나를 찾아 (가중)평균 예측을 target 에 맞춘다."""
    off = 0.0
    for _ in range(300):
        cur = sig(lo + off)
        e = (cur.mean() if w is None else np.average(cur, weights=w)) - target
        if abs(e) < 1e-10:
            break
        off -= e * 4.0
    return off


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
    print(f'\n===== {Y} 홀드아웃: 학습 {mh.sum():,} -> 검증 {mv.sum():,} =====', flush=True)
    W5, WB = S._wseason5_cols(), S._wsbat_cols()
    raw = S._add(ctx, W5, WB)
    yv = ctx['yv'].astype(np.float64)
    lo = logit(raw) + shift_to(logit(raw), float(yv.mean()))   # 전역 재중심화(배포와 동일)

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
        'newpit': np.where(an - wn < 50, '신규', '경력'),
    })
    return lo, yv, seg


def seg_offsets(lo, yv, key):
    """구간마다 '예측 평균 = 실제 평균' 이 되는 로짓 상수."""
    out = {}
    for v in pd.unique(key):
        m = (key == v)
        out[v] = shift_to(lo[m], float(yv[m].mean()))
    return out


def apply_off(lo, key, off, yv_target):
    z = lo + np.array([off.get(k, 0.0) for k in key])
    return sig(z + shift_to(z, yv_target))   # 전역 평균은 원래대로 되돌린다


def score(p, yv, U):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yv) ** 2).mean() / U) * 100000


_W5CHK = S._wseason5_cols()
assert 'w5_n' in _W5CHK, sorted(_W5CHK)[:12]
print('키 검증 OK: w5_n 존재', flush=True)

X, y, season, meta = S.load()
lo23, yv23, seg23 = run_year(2023, X, y, season, meta)
lo24, yv24, seg24 = run_year(2024, X, y, season, meta)

U = float(yv24.mean() * (1 - yv24.mean()))
base = score(sig(lo24), yv24, U)
print(f'\n\n전역 재중심화만: {base:,.0f}점  (r={yv24.mean():.4f})\n')

print(f'{"축":<12}{"구간":<12}{"비중":>7}{"편향2023":>10}{"편향2024":>10}{"차이":>9}')
for col in seg24.columns:
    for v in sorted(pd.unique(seg24[col])):
        m4 = (seg24[col].to_numpy() == v)
        m3 = (seg23[col].to_numpy() == v)
        b4 = float(sig(lo24[m4]).mean() - yv24[m4].mean())
        b3 = float(sig(lo23[m3]).mean() - yv23[m3].mean()) if m3.sum() else float('nan')
        print(f'{col:<12}{v:<12}{m4.mean():>6.1%}{b3:>10.4f}{b4:>10.4f}{b4-b3:>9.4f}')
    print()

print(f'\n{"축":<14}{"오라클(천장)":>14}{"실행가능":>12}')
tgt = float(yv24.mean())
for col in seg24.columns:
    k4, k3 = seg24[col].to_numpy(), seg23[col].to_numpy()
    orc = score(apply_off(lo24, k4, seg_offsets(lo24, yv24, k4), tgt), yv24, U) - base
    exe = score(apply_off(lo24, k4, seg_offsets(lo23, yv23, k3), tgt), yv24, U) - base
    print(f'{col:<14}{orc:>+14,.1f}{exe:>+12,.1f}')

# 두 축을 곱한 격자도 하나 본다 (칸이 잘게 쪼개지면 오라클은 커지고 실행가능은 무너진다)
for a, b in [('game_type', 'count_adv'), ('count_adv', 'wn')]:
    k4 = (seg24[a] + '|' + seg24[b]).to_numpy()
    k3 = (seg23[a] + '|' + seg23[b]).to_numpy()
    orc = score(apply_off(lo24, k4, seg_offsets(lo24, yv24, k4), tgt), yv24, U) - base
    exe = score(apply_off(lo24, k4, seg_offsets(lo23, yv23, k3), tgt), yv24, U) - base
    print(f'{a}x{b:<6}{orc:>+14,.1f}{exe:>+12,.1f}')

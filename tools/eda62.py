# 오차가 어디에 몰리는가 — 구간별 **기여**와 꼬리·희소 범주. 학습 0 (eda61 예측 재사용).
#
#   python tools/eda61.py   먼저 (cache/err24.npz 를 만든다)
#   python tools/eda62.py
#
# eda61 은 **편향**(신뢰도)을 봤다. 여기는 **분해능** — 같은 구간에서 모델이
# 상수 예측보다 얼마나 나은가다. 둘은 다른 축이고 처방도 다르다:
#   편향  -> 그 구간을 설명하는 피처가 빠졌거나 값이 틀렸다
#   기여  -> 그 구간 안에서 행을 **구분**하지 못한다 (새 축이 필요하다)
#
# ⚠️ 채점과 일치하려면 분모는 **전역 r(1-r)** 이어야 한다 (claude.md eda42):
#     기여_s = w_s x (1 - MSE_s / r(1-r)) x 100000,   합 = 전체 점수
#    구간별 자체 베이스레이트로 나누면 구간끼리 비교가 안 된다. F 가 자체 스킬
#    544 로 나빠 보이지만 행당 기여는 R 보다 32% 높았던 것이 그 예다.
#
# ⚠️ 2024 는 학습에 들어가 있다. 암기가 균일하지 않을 수 있으므로(투구 많은 투수일수록
#    외우기 쉽다) **절대값이 아니라 구간 간 순서**만 읽는다.
import sys

import numpy as np
import pandas as pd

Z = np.load('cache/err24.npz', allow_pickle=True)
p, y, rid = Z['p'], Z['y'], Z['row_id']
_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
tr = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
d = pd.DataFrame({'row_id': rid}).merge(tr, on='row_id', how='left')
assert len(d) == len(p) and d.control_success.to_numpy().astype(float).tolist() == y.tolist()
R = float(y.mean())
U = R * (1 - R)
contrib = (1 - (p - y) ** 2 / U) * 100000        # 행당 기여. 평균이 곧 전체 점수
print('2024 %s행 | 실제 %.4f | 전체 점수(이 하네스) %.0f'
      % (format(len(d), ','), R, contrib.mean()))

# ── 당해 시즌 진행분 w_n 복원 (그 투수의 2024 첫 행 asof 를 뺀다) ──
an = d.asof_pitcher_n.to_numpy(dtype='float64')
pid = d.pitcher_id.to_numpy()
base = pd.Series(an).groupby(pid).transform('min').to_numpy()
w_n = an - base
n_2024 = pd.Series(1, index=range(len(d))).groupby(pid).transform('size').to_numpy()
print('  w_n 복원: 중앙 %.0f | 그 투수의 2024 표본 중앙 %.0f'
      % (np.median(w_n), np.median(n_2024)))

sd = d.score_diff_pitcher_team.to_numpy()


def q(v, n=5):
    return pd.qcut(pd.Series(v).astype('float64'), n,
                   labels=['q%d' % (i + 1) for i in range(n)],
                   duplicates='drop').astype(str).to_numpy()


segs = {
    'game_type': d.game_type.astype(str).to_numpy(),
    'cnt12': (d.balls_before.astype(str) + '-' + d.strikes_before.astype(str)).to_numpy(),
    'inning': pd.cut(d.inning, [0, 1, 3, 6, 8, 99],
                     labels=['1', '2-3', '4-6', '7-8', '9+']).astype(str).to_numpy(),
    'outs': d.outs_before.astype(str).to_numpy(),
    'base_state': d.base_state.astype(str).to_numpy(),
    'hand4': (d.pitcher_hand.astype(str) + d.batter_hand.astype(str)).to_numpy(),
    'score_diff': np.asarray(pd.cut(sd, [-99, -6, -3, -1, 1, 3, 6, 99]).astype(str)),
    'month': d.game_month.astype(str).to_numpy(),
    'pitcher_team': d.pitcher_team_id.astype(str).to_numpy(),
    'li': q(d.li, 5),
    '★ w_n (당해 진행분)': np.asarray(pd.cut(w_n, [-1, 30, 100, 300, 800, 1e9],
                            labels=['0-30', '30-100', '100-300', '300-800', '800+']
                            ).astype(str)),
    '★ asof_n (커리어)': np.asarray(pd.cut(an, [-1, 100, 500, 1500, 4000, 1e9],
                            labels=['0-100', '100-500', '500-1.5k', '1.5k-4k', '4k+']
                            ).astype(str)),
    '★ 그 투수의 2024 표본': np.asarray(pd.cut(n_2024, [-1, 50, 200, 600, 1500, 1e9],
                             labels=['~50', '50-200', '200-600', '600-1.5k', '1.5k+']
                             ).astype(str)),
    'asof_batter_n': q(d.asof_batter_n, 5),
    '예측 분위': q(p, 10),
}

print('\n=== 구간별 행당 기여 (전역 r 기준). 평균 %.0f 보다 낮으면 오차가 몰린 곳 ==='
      % contrib.mean())
rows = []
for name, s in segs.items():
    vals = sorted(pd.unique(s))
    print('\n%s' % name)
    print('  %-14s%10s%8s%11s%11s%11s%11s'
          % ('값', 'n', '비중', '행당기여', '평균대비', '총손실', '자체스킬'))
    for v in vals:
        k = s == v
        if k.sum() < 500:
            continue
        w = float(k.mean())
        c = float(contrib[k].mean())
        gap = c - contrib.mean()
        # ⚠️ 행당 기여는 **그 구간의 실제 비율이 전역 r 에서 얼마나 떨어져 있나**에
        #    기계적으로 묶인다 (p ≈ r 이면 기여는 정의상 0). 그래서 '기여가 낮다' 가
        #    '모델이 거기서 못한다' 를 뜻하지 않는다. 자체 베이스레이트로 나눈
        #    **자체 스킬**을 같이 내야 그 둘이 갈린다 (합산은 못 하지만 진단은 이쪽).
        _rs = float(y[k].mean())
        _us = _rs * (1 - _rs)
        selfsk = (1 - ((p[k] - y[k]) ** 2).mean() / _us) * 100000 if _us > 1e-6 else np.nan
        print('  %-14s%10s%7.1f%%%11.0f%+11.0f%+11.1f%11.0f'
              % (v[:13], format(int(k.sum()), ','), 100 * w, c, gap, w * gap, selfsk))
        rows.append((w * gap, name, v, int(k.sum()), w, c, gap, selfsk))

rows.sort()
print('\n=== 가장 크게 새는 구간 (비중 x 평균대비) ===')
print('%-22s%-14s%10s%8s%11s%11s%11s'
      % ('축', '값', 'n', '비중', '행당기여', '총손실', '자체스킬'))
for tot, name, v, n, w, c, gap, sk in rows[:14]:
    print('%-22s%-14s%10s%7.1f%%%11.0f%+11.1f%11.0f'
          % (name, v[:13], format(n, ','), 100 * w, c, tot, sk))

# ★ 진짜 진단: 자체 스킬이 낮은 구간 = 그 안에서 행을 구분하지 못하는 곳
print('\n=== ★ 자체 스킬이 낮은 구간 (비중 1% 이상) — 모델이 실제로 못하는 곳 ===')
print('%-22s%-14s%10s%8s%11s%11s'
      % ('축', '값', 'n', '비중', '자체스킬', '행당기여'))
# ⚠️ '예측 분위' 축은 여기서 제외한다. p 로 조건을 걸면 그 안에는 설명할 변동이
#    거의 안 남아 자체 스킬이 퇴화한다 (분위 내 미세 오보정이 그대로 음수로 나온다).
_bysk = sorted([r for r in rows if r[4] >= 0.01 and np.isfinite(r[7])
                and not r[1].startswith('예측')], key=lambda r: r[7])
for tot, name, v, n, w, c, gap, sk in _bysk[:14]:
    print('%-22s%-14s%10s%7.1f%%%11.0f%11.0f'
          % (name, v[:13], format(n, ','), 100 * w, sk, c))

print('''
읽는 법: '총손실' = 그 구간이 평균만큼만 해줬다면 더 벌었을 점수다.
         행당기여가 0 근처면 그 구간에서 모델이 **상수 예측과 다를 바 없다**.
         음수면 상수보다 나쁘다.''')

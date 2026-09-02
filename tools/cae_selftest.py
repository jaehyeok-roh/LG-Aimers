# cae.py 의 두 구현을 **알려진 값으로 교정한다**. 모델 예측이 필요 없다.
#
#   python tools/cae_selftest.py
#
# 새 지표를 만들 때 가장 위험한 것은 구현이 조용히 틀리는 것이다. 다행히 이
# 프로젝트에는 같은 방식으로 잰 값이 이미 있으므로, 그 값을 재현하는지로 교정한다.
#
#   투수 커리어 축 반쪽 분할 신뢰도   0.637   (claude.md eda33)
#   batter x season                 -0.008   (같은 표 — 0 이 나와야 정상인 대조군)
#   라벨 복원 success 일치율          1.000000 (eda40/eda63, 147만 행)
#
# 마지막 것이 특히 중요하다. 행 정렬이 틀리면 조용히 잘못된 라벨이 나오는데,
# success 는 정답 컬럼이 있어 즉시 잡힌다.
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relia

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
RNG = np.random.default_rng(0)
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
print('%s행' % format(len(df), ','))

# ── ① 라벨 복원 검산 ──
n = df['asof_pitcher_n'].to_numpy(dtype='float64')
g = df['pitcher_id'].to_numpy()
o = np.lexsort((n, g))
ns, gs = n[o], g[o]
ok = np.r_[(gs[:-1] == gs[1:]) & (ns[1:] - ns[:-1] == 1), False]
lab = {}
for k in ('success', 'middle', 'reverse'):
    cum = np.round(df['asof_pitcher_%s_rate' % k].fillna(0)
                   .to_numpy(dtype='float64')[o] * ns)
    d = np.r_[cum[1:] - cum[:-1], np.nan]
    v = np.where(ok & np.isin(d, [0.0, 1.0]), d, np.nan)
    b = np.full(len(df), np.nan)
    b[o] = v
    lab[k] = b
m = np.isfinite(lab['success'])
acc = float((lab['success'][m] == df.control_success.to_numpy()[m]).mean())
print('\n① 라벨 복원 success 일치율  %.6f   (기대 1.000000)  %s'
      % (acc, 'OK' if acc >= 0.9999 else '**실패**'))
assert acc >= 0.9999, acc


# ③④ 도 cae.py 와 **같은 함수**로 잰다. 검사가 다른 구현을 보면 의미가 없다.
half_split = relia.reliab_arr


# ── ② 신뢰도 구현 교정 ──
# ⚠️ **시즌 리그평균을 먼저 빼야 한다.** 성공률이 .5647 -> .4861 로 크게 드리프트하므로
#    원시값으로 재면 같은 셀의 두 조각이 '같은 시즌' 을 공유한다는 이유만으로 상관이
#    생긴다. batter x season 이 +0.74 로 나오면(기대 -0.008) 그 함정에 빠진 것이다.
y_raw = df.control_success.to_numpy(dtype='float64')
y = y_raw - df.groupby('season')['control_success'].transform('mean').to_numpy()
half = RNG.random(len(df)) < 0.5


def dt(v):
    """시즌 디트렌드. 라벨마다 시즌 평균이 다르므로 반드시 빼고 잰다."""
    t = pd.DataFrame({'v': v, 's': df.season.to_numpy()})
    return v - t.groupby('s')['v'].transform('mean').to_numpy()
print('\n② 반쪽 분할 신뢰도 — 알려진 값과 대조 (tools/relia.py)')
print('%-24s%9s%11s%9s%13s' % ('축', '신뢰도', '신호std', '셀', '기대(eda33)'))
relia.add_dev(df)
# eda33 과 **같은 난수**여야 한다. RNG 는 위에서 이미 소비됐으므로 새로 만든다
df['h'] = np.random.default_rng(0).integers(0, 2, len(df))
CAL = (('pitcher (커리어)', ['pitcher_id'], None, 0.637),
       ('batter x season', ['batter_id', 'season'], ['batter_id'], -0.008),
       ('batter (커리어)', ['batter_id'], None, 0.227))
got = {}
for nm, keys, par, exp in CAL:
    r, n, _o, sg, _c = relia.reliab(df, keys, par)
    got[nm] = r
    print('%-24s%9.3f%11.4f%9d%13.3f' % (nm, r, sg, n, exp))

print('\n판정')
bad = 0
for nm, _k, _p, exp in CAL:
    o = abs(got[nm] - exp) < 0.08
    bad += 0 if o else 1
    print('  %-18s%7.3f vs %6.3f   %s' % (nm, got[nm], exp, 'OK' if o else '**어긋남**'))
print('''
  두 번째가 대조군이다. batter x season 은 0 근처여야 정상이고, 여기가 크게
  양수로 나오면 구현이 표본 잡음을 실력으로 착각하고 있다는 뜻이다.''')

# ── ③ 실패 유형 신뢰도 (eda41 대조) ──
print('\n③ 실패 유형 신뢰도 — eda41 대조 (무작위 반쪽 분할)')
df = df.assign(_i=np.arange(len(df)))
rk = df.groupby('pitcher_id').cumcount()
tot = df.groupby('pitcher_id')['_i'].transform('size')
first = (rk < tot / 2).to_numpy()
fin = np.isfinite(lab['middle']) & np.isfinite(lab['reverse'])
mi, re_ = lab['middle'] == 1, lab['reverse'] == 1
print('%-16s%10s%14s' % ('타겟', '신뢰도', '기대(eda41)'))
TGT = (('success', y_raw, '0.911'),
       ('reverse', np.where(fin, re_, np.nan).astype('float64'), '0.941'),
       ('middle', np.where(fin, mi, np.nan).astype('float64'), '0.807'))
for nm, v, exp in TGT:
    r, sg, n = half_split(dt(v), df.pitcher_id.to_numpy(), half, min_n=100)
    print('%-16s%10.3f%14s' % (nm, r, exp))
print('''
  eda41 은 "양쪽 100구+" 조건이었다. 값이 정확히 같을 필요는 없고,
  **reverse >= success 라는 순서**가 재현되면 핵심 주장이 살아 있다.''')

# ── ④ "최근 폼" 축 — 발표에서 가장 세게 주장할 값이라 따로 교정한다 ──
# 코치의 네 질문 중 "오늘 등판시키는 게 맞나"(최근 흐름)에 대한 답이 여기 달려 있다.
# 커리어 전/후반이 아니라 **시즌 안의 전후반**이어야 하고, 투수x시즌 주효과를
# 뺀 뒤여야 한다. 그래야 '그 시즌 그 투수' 안에서의 변동만 남는다.
df['half_season'] = np.where(df.game_month <= 6, 'H1', 'H2')
r_f, n_f, _o, _s, cov_f = relia.reliab(
    df, ['pitcher_id', 'season', 'half_season'], ['pitcher_id', 'season'])
print('\n④ "최근 폼" — pitcher x season x 전후반 (투수x시즌 효과 제거 후)')
print('   신뢰도 %.3f   셀 %s   커버 %.1f%%   기대(eda33) -0.089  %s'
      % (r_f, format(n_f, ','), cov_f * 100, 'OK' if abs(r_f + 0.089) < 0.08
         else '**어긋남**'))
print("""
  음수는 '두 반쪽이 서로 반대로 움직인다' 가 아니라 **공통 신호가 0** 이라는 뜻이다.
  즉 시즌 내 폼 변동은 측정되는 것이 없고 전부 표본 잡음이다.
  이것이 최근성 가중이 리더보드에서 5연패한 이유와 같은 사실이다.""")

# 대조: 같은 지표를 커리어 전/후반으로 가르면 크게 양수가 나온다.
# 그건 '폼' 이 아니라 시즌 간 성장·보직 변화라 위 질문의 답이 아니다.
print('\n⑤ (대조) 무작위 분할 vs **커리어** 전/후반 분할')
print('%-16s%12s%12s%12s' % ('타겟', '무작위', '시간 순', '차이'))
for nm, v, _ in TGT:
    ra, _s, _n = half_split(dt(v), df.pitcher_id.to_numpy(), half, min_n=100)
    rt, _s, _n = half_split(dt(v), df.pitcher_id.to_numpy(), first, min_n=100)
    print('%-16s%12.3f%12.3f%12.3f' % (nm, ra, rt, ra - rt))
print('''
  차이는 시즌 내 변동이 아니라 **시즌을 건너뛴 변화**(성장·보직·부상)다.
  ④ 가 0 인데 여기가 크다는 것이 곧 "폼은 해 단위로 바뀌지 날 단위로 안 바뀐다" 이다.''')

# EDA 11 — 당해 시즌 비율의 **shrink 기준**을 고치면 얼마나 더 버는가.
#
# group_oof 로 밝혀진 것: 우리가 잃던 것은 시즌 드리프트가 아니라 '그 투수가 올해
# 어떤지' 를 모르는 비용이다 (A 1092 vs B 791, 차이 301). wseason 이 그중 ~100 을
# 회수했다. 남은 200 을 더 캐려면 **당해 시즌 추정 자체를 더 정확히** 해야 한다.
#
# 현행: w_rate = (ws + 리그평균 x C) / (wn + C),  C=100
#   -> 표본이 적을 때 **리그 평균**으로 당긴다.
#   그런데 5시즌 이력이 있는 투수라면 더 나은 사전값은 **그 투수 자신의 과거 수준**이다.
#   경험적 베이즈에서 당연한 이야기인데 지금은 안 하고 있다.
#
# 여기서 세 가지를 비교한다 (2024, 그 행 이전 정보만 = 합법):
#   L : 리그 평균으로 shrink        (현행)
#   P : 그 투수의 과거 시즌 수준으로 shrink   (디트렌드해서 옮긴다)
#   B : 표본 크기에 따라 둘을 섞음
# 그리고 C 를 훑는다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
SE = 2024
lg = df.groupby('season')['control_success'].mean()

# --- 과거 시즌 수준 (시즌 디트렌드 + shrink) = cond_p 와 같은 방식 ---
past = df[df.season < SE].copy()
past['dev'] = past['control_success'] - past['season'].map(lg)
pg = past.groupby('pitcher_id')['dev'].agg(psum='sum', pn='size')

d = df[df.season == SE].join(pg, on='pitcher_id').reset_index(drop=True)
d[['psum', 'pn']] = d[['psum', 'pn']].fillna(0)
y = d['control_success'].to_numpy(dtype='float64')
r = y.mean()
U = r * (1 - r)

# --- 당해 시즌 복원 ---
n = d['asof_pitcher_n'].to_numpy(dtype='float64')
s = (d['asof_pitcher_success_rate'].fillna(0).to_numpy(dtype='float64') * n).round()
pn0 = df[df.season < SE].groupby('pitcher_id').size().reindex(d['pitcher_id']).fillna(0).to_numpy()
ps0 = df[df.season < SE].groupby('pitcher_id')['control_success'].sum().reindex(
    d['pitcher_id']).fillna(0).to_numpy()
wn = np.maximum(n - pn0, 0.0)
ws = np.clip(s - ps0, 0.0, wn)


def skill(p, m=None):
    yy = y if m is None else y[m]
    rr = float(yy.mean())
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yy) ** 2).mean() / (rr * (1 - rr))) * 100000


# 과거 수준을 2024 스케일로: 리그평균 + (디트렌드 편차를 shrink 한 값)
Cp = 200.0
prior_dev = d['psum'].to_numpy() / (d['pn'].to_numpy() + Cp)
prior_lvl = r + prior_dev            # 그 투수의 '2024 기준' 기대 수준

print(f'{SE} {len(d):,}행 | 리그 {r:.4f}')
print(f'당해 시즌 투구수: ' + ' '.join(
    f'{q}%={int(np.quantile(wn, q/100))}' for q in (10, 25, 50, 75, 90)))
print(f'과거 이력 있는 행 {(d["pn"] > 0).mean():.1%} | '
      f'과거 수준 범위 [{prior_lvl.min():.3f}, {prior_lvl.max():.3f}]\n')

print('=' * 76)
print('shrink 기준별 단독 예측 스킬 (C 를 훑는다)')
print(f'{"C":>6}{"L 리그평균":>13}{"P 그 투수 과거":>15}{"B 혼합":>11}')
best = {}
for C in (25, 50, 100, 200, 400, 800):
    pL = (ws + r * C) / (wn + C)
    pP = (ws + prior_lvl * C) / (wn + C)
    # 혼합: 과거 표본이 많을수록 P 를 믿는다
    w = d['pn'].to_numpy() / (d['pn'].to_numpy() + 400.0)
    tgt = r + w * prior_dev
    pB = (ws + tgt * C) / (wn + C)
    row = [skill(pL), skill(pP), skill(pB)]
    for k, v in zip('LPB', row):
        if v > best.get(k, (-9e9,))[0]:
            best[k] = (v, C)
    print(f'{C:>6}{row[0]:>13,.0f}{row[1]:>15,.0f}{row[2]:>11,.0f}')
print(f'\n최적: ' + '  '.join(f'{k}={v[0]:,.0f}(C={v[1]})' for k, v in best.items()))

print('\n' + '=' * 76)
print('당해 시즌 표본 크기별 — 최적 C 에서 L vs P')
CL, CP = best['L'][1], best['P'][1]
pL = (ws + r * CL) / (wn + CL)
pP = (ws + prior_lvl * CP) / (wn + CP)
print(f'{"당해 투구수":<16}{"행수":>9}{"비중":>7}{"L":>9}{"P":>9}{"차이":>9}')
for lo, hi in [(0, 50), (50, 150), (150, 400), (400, 900), (900, 99999)]:
    m = (wn >= lo) & (wn < hi)
    if m.sum() < 3000:
        continue
    a, b = skill(pL[m], m), skill(pP[m], m)
    print(f'{f"{lo}~{hi}":<16}{m.sum():>9,}{m.mean():>6.1%}{a:>9,.0f}{b:>9,.0f}{b-a:>+9,.0f}')

print('\n' + '=' * 76)
print('과거 이력 유무로 나눠서')
seen = (d['pn'] > 0).to_numpy()
for lab, m in [('이력 있는 투수', seen), ('신규 투수', ~seen)]:
    a, b = skill(pL[m], m), skill(pP[m], m)
    print(f'  {lab:<16}행 {m.sum():>7,} ({m.mean():>5.1%})  L {a:>7,.0f}  P {b:>7,.0f}'
          f'  차이 {b-a:>+7,.0f}   전체환산 {(b-a)*m.mean():>+7,.0f}')

print("""
읽는 법: P 가 L 보다 크게 높으면 shrink 기준을 그 투수 과거 수준으로 바꿔야 한다.
  특히 당해 투구수가 적은 구간(시즌 초, 저등판 투수)에서 차이가 나야 말이 된다.
  ⚠️ 단독 스킬이므로 실제 모델 증분은 이보다 작다 — 모델에는 cond_p 가 이미 있어서
     트리가 두 값을 부분적으로 결합할 수 있다.""")

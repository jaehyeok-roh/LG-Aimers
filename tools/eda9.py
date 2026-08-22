# EDA 9 — eda8 재작성. 가법 결합의 **계수를 적합**해서 증분을 정직하게 잰다.
#
# eda8 은 각 항의 로짓 계수를 1.0 으로 고정해서 넣었다. 그래서 과거 시즌 항이 과대
# 반영됐고, 순서를 바꾸면 증분이 -456 이 나오는 말이 안 되는 표가 나왔다. 폐기한다.
#
# 여기서는 각 항을 피처로 놓고 로지스틱 회귀로 **계수를 적합**한다.
# 항이 5개뿐이고 행이 25만이라 과적합은 무시할 수준이다.
# 자기누수는 항 만들 때 이미 뺐다 (과거항은 <2024 만, 당해항은 asof = 그 투구 이전만,
# 카운트/타자/손은 leave-one-out).
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
SE = 2024
lg = df.groupby('season')['control_success'].mean()

past = df[df.season < SE].copy()
past['dev'] = past['control_success'] - past['season'].map(lg)
pg = past.groupby('pitcher_id')['dev'].agg(psum='sum', pn='size')
d = df[df.season == SE].join(pg, on='pitcher_id').reset_index(drop=True)
d[['psum', 'pn']] = d[['psum', 'pn']].fillna(0)
y = d['control_success'].to_numpy(dtype='float64')
r = y.mean()
U = r * (1 - r)


def skill(p, m=None):
    yy = y if m is None else y[m]
    rr = float(yy.mean())
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yy) ** 2).mean() / (rr * (1 - rr))) * 100000


n = d['asof_pitcher_n'].to_numpy(dtype='float64')
s = (d['asof_pitcher_success_rate'].fillna(0).to_numpy(dtype='float64') * n).round()
prevn = df[df.season < SE].groupby('pitcher_id').size().reindex(d['pitcher_id']).fillna(0).to_numpy()
prevs = df[df.season < SE].groupby('pitcher_id')['control_success'].sum().reindex(
    d['pitcher_id']).fillna(0).to_numpy()
wn = np.maximum(n - prevn, 0.0)
ws = np.clip(s - prevs, 0.0, wn)


def loo_dev(keys, C):
    e = y - r
    g = pd.DataFrame({'k': keys, 'e': e})
    sm = g.groupby('k')['e'].transform('sum').to_numpy()
    ct = g.groupby('k')['e'].transform('size').to_numpy()
    return (sm - e) / ((ct - 1) + C)


d['cs'] = d['balls_before'].astype(str) + '-' + d['strikes_before'].astype(str)
d['hd'] = d['pitcher_hand'].astype(str) + 'v' + d['batter_hand'].astype(str)

T = {
    'past': d['psum'].to_numpy() / (d['pn'].to_numpy() + 200.0),     # 과거 시즌 디트렌드
    'wcur': (ws - r * wn) / (wn + 100.0),                            # ★ 당해 시즌 복원
    'cnt': loo_dev(d['cs'].to_numpy(), 500.0),
    'bat': loo_dev(d['batter_id'].astype(str).to_numpy(), 400.0),
    'hnd': loo_dev(d['hd'].to_numpy(), 500.0),
    'career': (d['asof_pitcher_success_rate'].fillna(r).to_numpy() - r),  # 현행 asof (수준 오염)
}
seen = (d['pn'] > 0).to_numpy()
print(f'{SE} {len(d):,}행 | 리그 {r:.4f} | 이력 있는 투수 행 {seen.mean():.1%}')
print(f'당해 시즌 투구수 중앙값 {np.median(wn):.0f} / 과거 누적 중앙값 {np.median(d["pn"]):.0f}\n')


def fit(names):
    Z = np.column_stack([T[k] for k in names])
    m = LogisticRegression(C=1e6, max_iter=400).fit(Z, y)
    return m.predict_proba(Z)[:, 1], dict(zip(names, m.coef_[0].round(3)))


print('=' * 78)
print('계수를 적합한 뒤의 스킬 — 항을 하나씩 더한다')
print(f'{"구성":<46}{"스킬":>9}{"증분":>9}')
seq = [
    (['career'], '현행 asof (커리어, 수준 오염)'),
    (['career', 'past'], '+ 과거 시즌 디트렌드 (cond_p 대역)'),
    (['career', 'past', 'cnt', 'bat', 'hnd'], '+ 카운트 + 타자 + 손'),
    (['career', 'past', 'cnt', 'bat', 'hnd', 'wcur'], '+ 당해 시즌 복원  ★'),
]
prev = None
last = None
for names, lab in seq:
    p, co = fit(names)
    v = skill(p)
    print(f'{lab:<46}{v:>9,.0f}' + ('' if prev is None else f'{v-prev:>9,.0f}'))
    prev, last = v, (names, p)

print('\n' + '=' * 78)
print('마지막 구성의 계수 (정규화 안 한 원시 항이라 크기는 항의 스케일에 의존)')
_, co = fit(seq[-1][0])
for k, v in co.items():
    print(f'  {k:<10}{v:>10.3f}')

print('\n' + '=' * 78)
print('★ 당해 시즌 항만 빼면 얼마나 떨어지는가 (leave-one-out 방식 증분)')
full = seq[-1][0]
pf, _ = fit(full)
base_s = skill(pf)
print(f'{"뺀 항":<16}{"스킬":>10}{"손실":>10}')
for k in full:
    rest = [x for x in full if x != k]
    pr, _ = fit(rest)
    print(f'{k:<16}{skill(pr):>10,.0f}{skill(pr)-base_s:>10,.0f}')

print('\n' + '=' * 78)
print('조각별 — 당해 시즌 항의 증분')
no_w = [x for x in full if x != 'wcur']
p_no, _ = fit(no_w)
for lab, m in [('전체', np.ones(len(d), bool)), ('이력 있는 투수', seen), ('신규 투수', ~seen)]:
    a, b = skill(p_no[m], m), skill(pf[m], m)
    print(f'  {lab:<16}행 {m.sum():>7,} ({m.mean():>5.1%})   {a:>7,.0f} -> {b:>7,.0f}'
          f'   증분 {b-a:>+7,.0f}   전체환산 {(b-a)*m.mean():>+7,.0f}')

print("""
⚠️ 읽는 법
  이 5~6개 항짜리 선형 모형은 실제 96피처 CatBoost 가 아니다. 실제 모델은
  smoothed / prev1,3,5게임 / 트랙맨 등 부분적으로 겹치는 피처를 더 갖고 있으므로
  증분은 여기보다 **작다**. 그리고 홀드아웃 -> 리더보드 전달률이 0.3 이다.""")

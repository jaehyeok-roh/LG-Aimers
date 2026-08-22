# EDA 10 — 분해 가능한 asof 컬럼이 다섯 개가 아니라 **열 개**다.
#
# 지금 wseason5 는 투수 결과 비율 5개(success/reverse/middle/ball/strike)만 분해한다.
# 그런데 같은 구조(커리어 누적 비율 x 누적 개수)를 가진 컬럼이 5개 더 있다:
#   구종 비율 3개  fastball/breaking/offspeed  (분모가 asof_pitcher_pitchmix_n 로 다름)
#   타자 비율 2개  asof_batter_success/middle  (분모가 asof_batter_n)
#
# 여기서 각 묶음의 **증분**을 잰다 (eda9 처럼 계수를 적합한 선형 대리모형).
# 단독 스킬이 아니라 증분이어야 실제 모델에서의 기대와 맞는다.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
SE = 2024

GROUPS = {
    'P결과5': ('asof_pitcher_n', 'pitcher_id',
               ['success', 'reverse', 'middle', 'ball', 'strike'], 'asof_pitcher_%s_rate'),
    '구종3': ('asof_pitcher_pitchmix_n', 'pitcher_id',
              ['fastball', 'breaking', 'offspeed'], 'asof_pitcher_%s_rate'),
    '타자2': ('asof_batter_n', 'batter_id',
              ['success', 'middle'], 'asof_batter_%s_rate'),
}

# (키, 시즌) 첫 행 = 그 시즌 시작 시점 커리어 상태
built = {}
for gname, (ncol, key, rates, fmt) in GROUPS.items():
    o = np.argsort(df[ncol].fillna(0).to_numpy(dtype='float64'), kind='stable')
    first = df.iloc[o].groupby([key, 'season'], sort=False).head(1).set_index([key, 'season'])
    idx = pd.MultiIndex.from_arrays([df[key], df['season']])
    n = df[ncol].fillna(0).to_numpy(dtype='float64')
    n0 = first[ncol].fillna(0).reindex(idx).to_numpy(dtype='float64')
    wn = np.maximum(n - n0, 0.0)
    cols = {}
    for r in rates:
        c = fmt % r
        x = (df[c].fillna(0).to_numpy(dtype='float64') * n).round()
        x0 = (first[c].fillna(0).reindex(idx).to_numpy(dtype='float64') * n0).round()
        wx = np.clip(x - x0, 0.0, wn)
        rate = (wx + 0.0) / np.maximum(wn, 1.0)
        rate = np.where(wn > 0, rate, np.nan)
        cols[f'{gname}:{r}'] = rate
    built[gname] = (wn, cols, rates, fmt)
    ok = wn > 0
    print(f'{gname:<8} 분모 {ncol:<26} 당해시즌 표본 중앙값 {np.median(wn[ok]):>6.0f} '
          f'| 유효행 {ok.mean():.1%}')

m24 = (df['season'] == SE).to_numpy()
y = df.loc[m24, 'control_success'].to_numpy(dtype='float64')
r = y.mean()
U = r * (1 - r)
print(f'\n{SE} {m24.sum():,}행 | 리그 {r:.4f}\n')


def skill(p):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / U) * 100000


# ---- 기존 모델이 이미 가진 것 (커리어 원본 + 최근경기 + 상황) ----
d = df[m24].reset_index(drop=True)
T = {}
for c in ['asof_pitcher_success_rate', 'asof_pitcher_middle_rate', 'asof_pitcher_reverse_rate',
          'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate', 'asof_batter_success_rate',
          'asof_batter_middle_rate', 'asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate',
          'asof_pitcher_offspeed_rate']:
    v = d[c].to_numpy(dtype='float64')
    T['c_' + c] = np.nan_to_num(v - np.nanmean(v))
for k in (1, 3, 5):
    for w in ('success', 'middle'):
        c = f'asof_pitcher_prev{k}_game_{w}_rate'
        v = d[c].to_numpy(dtype='float64')
        T[f'g{k}{w}'] = np.nan_to_num(v - np.nanmean(v))
        T[f'g{k}{w}m'] = np.isnan(v).astype(float)


def loo(keys, C):
    e = y - r
    g = pd.DataFrame({'k': keys, 'e': e})
    sm = g.groupby('k')['e'].transform('sum').to_numpy()
    ct = g.groupby('k')['e'].transform('size').to_numpy()
    return (sm - e) / ((ct - 1) + C)


d['cs'] = d['balls_before'].astype(str) + '-' + d['strikes_before'].astype(str)
d['hd'] = d['pitcher_hand'].astype(str) + 'v' + d['batter_hand'].astype(str)
T['cnt'] = loo(d['cs'].to_numpy(), 500.0)
T['bat'] = loo(d['batter_id'].astype(str).to_numpy(), 400.0)
T['hnd'] = loo(d['hd'].to_numpy(), 500.0)

BASE = list(T)

# ---- 당해 시즌 복원 묶음들 ----
for gname, (wn, cols, rates, fmt) in built.items():
    w24 = wn[m24]
    T[f'W{gname}:n'] = np.log1p(w24)
    for cn, v in cols.items():
        vv = v[m24]
        mu = np.nanmean(vv)
        T['W' + cn] = np.nan_to_num(vv - mu)
        T['W' + cn + ':m'] = np.isnan(vv).astype(float)


def fit(names):
    Z = np.column_stack([T[k] for k in names])
    return LogisticRegression(C=1e6, max_iter=500).fit(Z, y).predict_proba(Z)[:, 1]


def grp(gname):
    return [k for k in T if k.startswith('W' + gname)]


print('=' * 74)
print('묶음별 증분 (계수 적합, base = 커리어 원본 + 최근경기 + 카운트/타자/손)')
print(f'{"구성":<44}{"스킬":>10}{"증분":>10}')
b = skill(fit(BASE))
print(f'{"base (지금 모델이 가진 것)":<44}{b:>10,.0f}')
cur = BASE + grp('P결과5')
v1 = skill(fit(cur))
print(f'{"+ 당해시즌 투수 결과 5개  (= 지금 후보)":<44}{v1:>10,.0f}{v1-b:>10,.0f}')
for extra, lab in [('구종3', '+ 당해시즌 구종 3개'), ('타자2', '+ 당해시즌 타자 2개')]:
    v = skill(fit(cur + grp(extra)))
    print(f'{lab:<44}{v:>10,.0f}{v-v1:>10,.0f}')
v_all = skill(fit(cur + grp('구종3') + grp('타자2')))
print(f'{"+ 둘 다 (열 개 전부)":<44}{v_all:>10,.0f}{v_all-v1:>10,.0f}')

print('\n' + '=' * 74)
print('투수 결과 5개 안에서 — 하나씩 빼면 (전체 10개 구성 기준)')
full = cur + grp('구종3') + grp('타자2')
base_s = skill(fit(full))
for rname in ['success', 'reverse', 'middle', 'ball', 'strike']:
    rest = [k for k in full if not k.startswith(f'WP결과5:{rname}')]
    print(f'  {rname:<12}{skill(fit(rest)):>10,.0f}{skill(fit(rest))-base_s:>10,.0f}')

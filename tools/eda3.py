# EDA 3 — 신호가 어디에 있는가 / 데이터에 우리가 못 본 구조가 있는가.
#
# eda1: 상황 변수의 조건부 구조는 안정적 (잔차 변화폭 0.004~0.02)
# eda2: 주요 피처의 관계 모양도 안정적 (전후상관 0.97~0.98)
# -> 그럼 신호는 어디 있고, 우리가 안 쓰는 구조가 남아 있는가?
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
y = df['control_success']
S = sorted(df['season'].unique())

print('=' * 78)
print('1) 분산 분해 — 성공률의 변동이 어디서 오는가 (2024 시즌)')
d24 = df[df.season == 2024]
tot = d24['control_success'].var()
for key, lab in [('pitcher_id', '투수'), ('batter_id', '타자'),
                 ('count_str', '볼카운트'), ('base_state', '주자'),
                 ('pitcher_team_id', '투수팀'), ('inning', '이닝')]:
    if key == 'count_str':
        k = d24['balls_before'].astype(str) + '-' + d24['strikes_before'].astype(str)
    else:
        k = d24[key]
    g = d24.groupby(k)['control_success'].agg(['mean', 'size'])
    g = g[g['size'] >= 30]
    w = g['size'] / g['size'].sum()
    between = float(((g['mean'] - (g['mean'] * w).sum()) ** 2 * w).sum())
    print(f'  {lab:<10} 그룹 {len(g):>5}개   집단간 분산 {between:.5f}   '
          f'전체 대비 {between/tot:6.2%}')
print(f'  (전체 분산 {tot:.5f} — 이진 타겟이라 대부분은 환원 불가능한 베르누이 잡음)')

print('\n' + '=' * 78)
print('2) 투수 성적의 시즌 간 지속성 — cond_* 가 기대야 하는 근거')
n = df.groupby(['season', 'pitcher_id']).size().rename('n').reset_index()
r = df.groupby(['season', 'pitcher_id'])['control_success'].mean().rename('r').reset_index()
pr = n.merge(r, on=['season', 'pitcher_id'])
for a, b in zip(S[:-1], S[1:]):
    for mn in (300, 1000):
        A = pr[(pr.season == a) & (pr.n >= mn)].set_index('pitcher_id')['r']
        B = pr[(pr.season == b) & (pr.n >= mn)].set_index('pitcher_id')['r']
        j = pd.concat([A, B], axis=1, join='inner')
        if len(j) >= 30 and mn == 300:
            print(f'  {a}->{b}  {mn}구+  {len(j):>3}명  상관 {j.corr().iloc[0,1]:.3f}', end='')
        elif len(j) >= 30:
            print(f'   | {mn}구+ {len(j):>3}명 상관 {j.corr().iloc[0,1]:.3f}')
print('\n  → 상관이 낮으면 "작년 잘한 투수가 올해도 잘한다" 가 약하다는 뜻이고,')
print('    그건 cond_* 계열의 천장이 낮다는 뜻이다.')

print('\n' + '=' * 78)
print('3) 결측 패턴 — 시즌마다 다른가 (다르면 그 자체가 드리프트)')
miss = df.isna().mean()
mc = miss[miss > 0].sort_values(ascending=False)
if len(mc) == 0:
    print('  원본 49컬럼에 결측 없음')
else:
    print(f'{"컬럼":<38}' + ''.join(f'{s:>9}' for s in S))
    for c in mc.index[:12]:
        g = df.groupby('season')[c].apply(lambda x: x.isna().mean())
        print(f'{c[:36]:<38}' + ''.join(f'{g[s]:>9.3f}' for s in S))

print('\n' + '=' * 78)
print('4) 시즌별 데이터 구성 — test(2025) 가 어떻게 다를지 가늠')
g = df.groupby('season').agg(
    행수=('row_id', 'size'),
    투수수=('pitcher_id', 'nunique'),
    타자수=('batter_id', 'nunique'),
    경기타입=('game_type', 'nunique'))
g['투수당투구'] = (g['행수'] / g['투수수']).round(0)
print(g.to_string())

print('\n  신규 투수 비율 (직전 시즌 대비)')
for a, b in zip(S[:-1], S[1:]):
    pa = set(df[df.season <= a]['pitcher_id'])
    pb = df[df.season == b]
    new = ~pb['pitcher_id'].isin(pa)
    print(f'    {b}: 투수 {new.groupby(pb["pitcher_id"]).first().mean():.1%} 신규 / '
          f'투구 {new.mean():.1%} 가 신규 투수')

print('\n' + '=' * 78)
print('5) 우리가 안 쓰는 구조 — 타석 내 위치 (balls+strikes 는 투구수의 하한)')
df['pa_pitch'] = df['balls_before'] + df['strikes_before']
g = df.groupby(['season', 'pa_pitch'])['control_success'].agg(['mean', 'size']).reset_index()
g = g[g['size'] >= 2000]
piv = g.pivot(index='pa_pitch', columns='season', values='mean').dropna()
lg = pd.Series({s: df[df.season == s]['control_success'].mean() for s in S})
sh = piv.sub(lg[piv.columns], axis=1)
print(f'{"볼+스트라이크":<14}' + ''.join(f'{s:>9}' for s in piv.columns) + f'{"전체비중":>10}')
tot_n = len(df)
for k in piv.index:
    share = (df['pa_pitch'] == k).sum() / tot_n
    print(f'{k:<14}' + ''.join(f'{v:>+9.4f}' for v in sh.loc[k]) + f'{share:>10.1%}')
print(f'  → 리그 제거 후 잔차 변화폭 {(sh.max(axis=1)-sh.min(axis=1)).mean():.4f}')

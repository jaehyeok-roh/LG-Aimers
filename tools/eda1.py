# EDA 1 — 드리프트의 해부.
#
# 열흘 동안 모델만 두들겼고 데이터를 들여다본 적이 없다. 우리 진단
# ("피처-결과 관계가 시즌마다 바뀐다")도 모델 실패에서 **역추론**한 것이지
# 데이터에서 확인한 적이 없다. 여기서 직접 본다.
#
# 묻는 것:
#   1) 하락은 리그 전체의 **수준 이동**인가, 아니면 슬라이스마다 다른가
#   2) 조건부 관계(카운트/이닝/주자/손 매치업)의 **모양**이 바뀌는가
#   3) 바뀐다면 어디가 얼마나
#
# 수준만 바뀌면 재중심화로 끝난 문제다. 모양이 바뀌면 그게 우리가 잃는 64% 다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']   # 'None' 제외 (4-3)
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
S = sorted(df['season'].unique())
print(f'{len(df):,}행 x {df.shape[1]}컬럼 | 시즌 {S}\n')

lg = df.groupby('season')['control_success'].mean()
print('리그 평균 성공률')
print('  ' + '  '.join(f'{s}:{lg[s]:.4f}' for s in S))
print('  전년대비 ' + '  '.join(f'{s}:{lg[s]-lg[s-1]:+.4f}' for s in S[1:]) + '\n')


def slice_drift(col, label=None, min_n=3000, top=None):
    """슬라이스별 성공률의 시즌 궤적. 수준 이동인지 모양 변화인지 본다."""
    g = df.groupby(['season', col])['control_success'].agg(['mean', 'size']).reset_index()
    g = g[g['size'] >= min_n]
    piv = g.pivot(index=col, columns='season', values='mean')
    piv = piv.dropna()
    if top:
        n = df.groupby(col).size()
        piv = piv.loc[[k for k in n.sort_values(ascending=False).index if k in piv.index][:top]]
    if len(piv) < 2:
        return
    print(f'--- {label or col} ---')
    hdr = ''.join(f'{s:>9}' for s in piv.columns)
    print(f'{"":<16}{hdr}{"총하락":>9}{"리그대비":>10}')
    lg_drop = lg[S[-1]] - lg[S[0]]
    for k, row in piv.iterrows():
        d = row[S[-1]] - row[S[0]]
        print(f'{str(k)[:15]:<16}' + ''.join(f'{v:>9.4f}' for v in row)
              + f'{d:>+9.4f}{d - lg_drop:>+10.4f}')
    # 리그 수준을 뺀 잔차의 시즌 간 변동 = '모양' 이 얼마나 바뀌는가
    resid = piv.sub(lg[piv.columns], axis=1)
    print(f'  → 리그 제거 후 잔차 표준편차(시즌평균) {resid.std(axis=0).mean():.4f}'
          f' | 잔차의 시즌간 변화폭 {(resid.max(axis=1)-resid.min(axis=1)).mean():.4f}\n')


df['count_str'] = df['balls_before'].astype(str) + '-' + df['strikes_before'].astype(str)
df['hand'] = df['pitcher_hand'].astype(str) + 'v' + df['batter_hand'].astype(str)
slice_drift('count_str', '볼카운트')
slice_drift('base_state', '주자 상태')
slice_drift('hand', '투타 손')
slice_drift('inning', '이닝', top=9)
slice_drift('outs_before', '아웃카운트')

# 투수 등판량 티어 — 구성 변화 vs 내부 변화
n_by = df.groupby(['season', 'pitcher_id']).size().rename('n').reset_index()
df2 = df.merge(n_by, on=['season', 'pitcher_id'])
df2['tier'] = pd.cut(df2['n'], [0, 200, 800, 2000, 99999],
                     labels=['~200', '200-800', '800-2000', '2000+'])
g = df2.groupby(['season', 'tier'], observed=True)['control_success'].mean().unstack()
print('--- 투수 등판량 티어 ---')
print(f'{"":<12}' + ''.join(f'{s:>9}' for s in g.index) + f'{"총하락":>9}')
for t in g.columns:
    print(f'{str(t):<12}' + ''.join(f'{g.loc[s, t]:>9.4f}' for s in g.index)
          + f'{g.loc[S[-1], t] - g.loc[S[0], t]:>+9.4f}')
print()

# 같은 투수만 (구성 변화 제거) — 인접 시즌 쌍
print('--- 동일 투수 인접 시즌 (양 시즌 300구 이상) ---')
for a, b in zip(S[:-1], S[1:]):
    pa = n_by[(n_by.season == a) & (n_by.n >= 300)]['pitcher_id']
    pb = n_by[(n_by.season == b) & (n_by.n >= 300)]['pitcher_id']
    both = set(pa) & set(pb)
    ra = df[(df.season == a) & df.pitcher_id.isin(both)].groupby('pitcher_id')['control_success'].mean()
    rb = df[(df.season == b) & df.pitcher_id.isin(both)].groupby('pitcher_id')['control_success'].mean()
    d = (rb - ra).dropna()
    print(f'  {a}->{b}  투수 {len(d):>3}명  개인평균 {d.mean():+.4f}  '
          f'리그 {lg[b]-lg[a]:+.4f}  하락한 비율 {(d < 0).mean():.0%}  '
          f'개인차 표준편차 {d.std():.4f}')

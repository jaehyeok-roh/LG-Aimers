# EDA 27 — 정답 매핑이 배포 조건(2024 트랙맨 -> 2025 test)에서 커버리지를 얼마나 올리나.
import numpy as np, pandas as pd
gold = pd.read_csv('cache/raw/gold_pitcher_map.csv')
v2   = pd.read_csv('data/pitcher_id_mapping_v2.csv')
tm   = pd.read_parquet('cache/raw/trackman.parquet',
                       columns=['season','pitcher_trackman_id','pitcher_team'])
tr   = pd.read_parquet('cache/raw/train.parquet', columns=['season','pitcher_id'])

print('매핑 규모:  gold', len(gold), '투수 | v2',
      v2.dropna(subset=['pitcher_id']).pitcher_id.nunique(), '투수')
print('gold 는 시즌 무관 1:1, v2 는 시즌별 행:', v2.shape)

tm24 = set(tm[tm.season == 2024].pitcher_trackman_id.dropna().unique())
for lab, ids in [('gold', gold.set_index('pitcher_id').pitcher_trackman_id),
                 ('v2', v2[v2.season == 2024].dropna(subset=['pitcher_id']).drop_duplicates('pitcher_id')
                        .set_index('pitcher_id').pitcher_trackman_id)]:
    d24 = tr[tr.season == 2024]
    hit = d24.pitcher_id.map(ids)
    ok = hit.notna() & hit.isin(tm24)
    print(f'  {lab:5s}: 2024 train 투구 중 2024 트랙맨에 닿는 비율 {ok.mean():.4%}  '
          f'(투수 {d24[ok].pitcher_id.nunique()}/{d24.pitcher_id.nunique()})')

# 2025 에 등장할 투수 = 2024 에 던진 투수라고 보고, 그중 트랙맨 커버
print('\n2024 에 던진 투수별 투구량 가중 커버리지(=2025 test 근사):')
w = tr[tr.season == 2024].pitcher_id.value_counts(normalize=True)
for lab, ids in [('gold', gold.set_index('pitcher_id').pitcher_trackman_id),
                 ('v2', v2[v2.season == 2024].dropna(subset=['pitcher_id']).drop_duplicates('pitcher_id')
                        .set_index('pitcher_id').pitcher_trackman_id)]:
    m = w.index.map(ids); ok = pd.notna(m) & pd.Series(m).isin(tm24).to_numpy()
    print(f'  {lab:5s}: {w[ok].sum():.4%}')

# 1군/2군 분리 가능 여부
pre = tm['pitcher_team'].astype(str).str.split('_').str[0]
tm['lvl'] = np.where(pre.eq('MIN') | tm['pitcher_team'].isin(['KBO_ARM','KBO_POL']), '2군', '1군')
share = tm[tm.season == 2024].groupby('pitcher_trackman_id')['lvl'].apply(lambda s: (s == '2군').mean())
g = gold.set_index('pitcher_id')['pitcher_trackman_id'].map(share)
print(f'\ngold 매핑된 투수의 2024 트랙맨 2군 비중: 중앙 {g.median():.3f}  '
      f'0% {(g==0).mean():.1%}  100% {(g==1).mean():.1%}')

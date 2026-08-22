# EDA 23 — game_type=F 는 퓨처스(2군)다 (eda18: 매칭된 F 240경기가 전부 트랙맨 2군).
#   그럼 2022 .7087 -> 2023 .4729 의 '체제 변화' 는 무엇인가.
import numpy as np, pandas as pd
pd.set_option('display.width', 220)
C = ['row_id','season','game_month','game_dayofweek','game_type','inning','top_bottom',
     'balls_before','strikes_before','outs_before','run_total_before','pitcher_id',
     'batter_id','pitcher_team_id','batter_team_id','control_success','asof_pitcher_n',
     'asof_pitcher_success_rate','asof_pitcher_middle_rate','asof_pitcher_ball_rate',
     'asof_pitcher_strike_rate','asof_pitcher_reverse_rate','li']
tr = pd.read_parquet('cache/raw/train.parquet', columns=C)
c = ['season','game_month','game_dayofweek','game_type']
tr['gid'] = ((tr[c] != tr[c].shift()).any(axis=1) |
             (tr['inning'].diff() < 0) | (tr['run_total_before'].diff() < 0)).cumsum()

print('=' * 78); print('1) F 의 규모와 팀 구성'); print('=' * 78)
g = tr.groupby(['season','game_type']).agg(경기=('gid','nunique'), 투구=('row_id','size'),
        성공률=('control_success','mean'), 투수=('pitcher_id','nunique'))
print(g.round(4).to_string())
print('\nF 에 나오는 투수팀 id (시즌별):')
for s in sorted(tr.season.unique()):
    f = tr[(tr.season==s)&(tr.game_type=='F')]
    print(f'  {s}: {sorted(f.pitcher_team_id.unique())}')

print('\n' + '=' * 78); print('2) 2023 붕괴는 구성인가 실제인가 — 양 시즌 모두 F 에 나온 투수'); print('=' * 78)
for a, b in [(2021,2022),(2022,2023),(2023,2024)]:
    f = tr[tr.game_type=='F']
    A = f[f.season==a].groupby('pitcher_id')['control_success'].agg(['mean','size'])
    B = f[f.season==b].groupby('pitcher_id')['control_success'].agg(['mean','size'])
    j = A.join(B, lsuffix='_a', rsuffix='_b', how='inner')
    j = j[(j['size_a']>=50)&(j['size_b']>=50)]
    print(f'  {a}->{b}: 동일투수 {len(j):3d}명  평균변화 {(j["mean_b"]-j["mean_a"]).mean():+.4f}  '
          f'전체F변화 {f[f.season==b].control_success.mean()-f[f.season==a].control_success.mean():+.4f}')

print('\n' + '=' * 78); print('3) 같은 투수가 같은 시즌에 R 과 F 양쪽에 던질 때'); print('=' * 78)
for s in sorted(tr.season.unique()):
    d = tr[tr.season==s]
    p = d.groupby(['pitcher_id','game_type'])['control_success'].agg(['mean','size']).unstack()
    p = p[(p[('size','F')]>=30)&(p[('size','R')]>=30)]
    if len(p): print(f'  {s}: {len(p):3d}명  F-R 차 {(p[("mean","F")]-p[("mean","R")]).mean():+.4f}')

print('\n' + '=' * 78); print('4) F 안에서 실패 유형 구성이 바뀌었나 (asof 비율의 시즌 평균)'); print('=' * 78)
cols = ['asof_pitcher_success_rate','asof_pitcher_middle_rate','asof_pitcher_reverse_rate',
        'asof_pitcher_ball_rate','asof_pitcher_strike_rate']
print(tr[tr.game_type=='F'].groupby('season')[cols].mean().round(4).to_string())
print('\n(R 비교)')
print(tr[tr.game_type=='R'].groupby('season')[cols].mean().round(4).to_string())

print('\n' + '=' * 78); print('5) 경기 축의 초과분산 — 투수 구성을 뺀 뒤에도 남는가'); print('=' * 78)
d = tr[(tr.season==2024)&(tr.game_type=='R')].copy()
# 투수-시즌 평균으로 기대값을 만들고 (leave-one-out), 그 잔차를 경기별로 모은다
pm = d.groupby('pitcher_id')['control_success'].agg(['sum','size'])
exp = (pm['sum'].reindex(d.pitcher_id).to_numpy() - d['control_success'].to_numpy()) / \
      (pm['size'].reindex(d.pitcher_id).to_numpy() - 1)
d['resid'] = d['control_success'].to_numpy() - exp
gm = d.groupby('gid')['resid'].agg(['mean','size']); gm = gm[gm['size']>=100]
noise = (0.25/gm['size']).mean()
ex = max(0.0, gm['mean'].var() - noise)
print(f'  경기 {len(gm)}개  잔차평균 sd {gm["mean"].std():.4f}  잡음기대 {np.sqrt(noise):.4f}')
print(f'  투수 구성 제거 후 경기 초과분산 {ex:.6f}  = 전체분산의 {ex/0.2498:.4%}')
print('  (원 초과분산 1.05% 중 이만큼이 투수로 설명 안 되는 경기 고유 효과)')

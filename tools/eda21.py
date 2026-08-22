# EDA 21 — 경기 복원이 열어주는 것들.
#   CLAUDE.md 는 "train/test 에 경기 ID도 날짜도 없어 경기내 피로도는 복원 불가능" 이라 적혀 있다.
#   train 쪽은 복원된다 (eda16). 그럼 실제로 신호가 있는가?
import numpy as np, pandas as pd
pd.set_option('display.width', 220)
C = ['row_id','season','game_month','game_dayofweek','game_type','inning','top_bottom',
     'balls_before','strikes_before','outs_before','run_total_before','pitcher_id',
     'batter_id','pitcher_team_id','control_success','asof_pitcher_n',
     'asof_pitcher_prev1_game_success_rate']
tr = pd.read_parquet('cache/raw/train.parquet', columns=C)
c = ['season','game_month','game_dayofweek','game_type']
tr['gid'] = ((tr[c] != tr[c].shift()).any(axis=1) |
             (tr['inning'].diff() < 0) | (tr['run_total_before'].diff() < 0)).cumsum()

# 등판(appearance) = (경기, 투수)
tr['app'] = tr.groupby(['gid','pitcher_id']).ngroup()
tr['pc_app'] = tr.groupby('app').cumcount()          # 그 등판에서 몇 번째 투구인가
tr['pc_game'] = tr.groupby('gid').cumcount()
app = tr.groupby('app').agg(n=('row_id','size'), gid=('gid','first'), pid=('pitcher_id','first'),
                            se=('season','first'), inn0=('inning','first'))
print(f'등판 {len(app):,}  경기당 {len(app)/tr.gid.nunique():.1f}등판')
print(f'등판 투구수: 중앙 {app.n.median():.0f} q90 {app.n.quantile(.9):.0f} max {app.n.max()}')

print('\n' + '=' * 78)
print('A) 등판 내 투구수(피로) vs 성공률 — 2024')
print('=' * 78)
d = tr[tr.season == 2024]
b = pd.cut(d['pc_app'], [-1,9,19,29,44,59,79,99,10**6],
           labels=['1-10','11-20','21-30','31-45','46-60','61-80','81-100','100+'])
t = d.groupby(b, observed=True)['control_success'].agg(['mean','size'])
t['dev'] = t['mean'] - d['control_success'].mean()
print(t.round(4).to_string())
print('\n선발만 (그 등판이 1회에 시작 & 50구 이상):')
st = app[(app.inn0 == 1) & (app.n >= 50)].index
ds = d[d['app'].isin(st)]
t2 = ds.groupby(pd.cut(ds['pc_app'], [-1,14,29,44,59,74,89,10**6],
        labels=['1-15','16-30','31-45','46-60','61-75','76-90','90+']), observed=True)['control_success'].agg(['mean','size'])
t2['dev'] = t2['mean'] - ds['control_success'].mean()
print(t2.round(4).to_string())

print('\n' + '=' * 78)
print('B) 경기 내 시점(투구 순번) / 이닝 반복 — 같은 타자를 다시 만날 때')
print('=' * 78)
d = d.copy()
d['tto'] = d.groupby(['gid','pitcher_id','batter_id']).cumcount() + 1   # times through
t3 = d.groupby('tto')['control_success'].agg(['mean','size'])
print(t3.head(5).round(4).to_string())

print('\n' + '=' * 78)
print('C) 복원한 경기 경계가 주최측 prev1_game 과 맞는가 (검증)')
print('=' * 78)
# 같은 등판 안에서는 prev1_game 값이 변하면 안 된다
v = tr.groupby('app')['asof_pitcher_prev1_game_success_rate'].nunique(dropna=False)
print(f'  등판 안에서 prev1 값이 1개뿐인 비율: {(v <= 1).mean():.4%}   (경계가 맞으면 100%)')
# 다른 등판 사이에서는 바뀌어야 한다
f = tr.groupby('app').agg(pid=('pitcher_id','first'), se=('season','first'),
                          v=('asof_pitcher_prev1_game_success_rate','first')).reset_index()
f = f.sort_values('app')
ch = f.groupby(['pid','se'])['v'].apply(lambda s: (s.diff().abs() > 1e-12).mean())
print(f'  연속 등판 사이에 prev1 이 바뀌는 비율: {ch.mean():.4%}')

print('\n' + '=' * 78)
print('D) 경기 수준 통계 — 심판/구장 효과의 대리')
print('=' * 78)
g = tr[tr.season==2024].groupby('gid')['control_success'].agg(['mean','size'])
g = g[g['size'] >= 100]
print(f'  경기 {len(g):,}개  성공률 sd {g["mean"].std():.4f}')
exp = np.sqrt((g["mean"].mean()*(1-g["mean"].mean())/g["size"]).mean())
print(f'  이항 잡음만으로 기대되는 sd {exp:.4f}  -> 초과 분산 {max(0,g["mean"].var()-exp**2):.6f}')
print(f'  (초과분산 / 전체분산 = {max(0,g["mean"].var()-exp**2)/0.2498:.4%}  경기 축의 신호 상한)')

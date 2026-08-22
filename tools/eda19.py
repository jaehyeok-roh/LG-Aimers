# EDA 19 — 1:1 정렬된 부분집합을 실제로 만들고, **그 투구의 물리량**이
# control_success 를 얼마나 설명하는지 잰다.
import numpy as np, pandas as pd, hashlib
pd.set_option('display.width', 220)

tr = pd.read_parquet('cache/raw/train.parquet')
tm = pd.read_parquet('cache/raw/trackman.parquet')
tm['dt'] = pd.to_datetime(tm['game_date'].astype(str), format='mixed', errors='coerce')
c = ['season', 'game_month', 'game_dayofweek', 'game_type']
tr['gid'] = ((tr[c] != tr[c].shift()).any(axis=1) |
             (tr['inning'].diff() < 0) | (tr['run_total_before'].diff() < 0)).cumsum()
tr['pidx'] = tr.groupby('gid').cumcount()
tm = tm.sort_values(['trackman_game_id', 'pitch_no'])
tm['pidx'] = tm.groupby('trackman_game_id').cumcount()

mj = pd.read_parquet('cache/raw/game_match.parquet')[['gid', 'trackman_game_id']]
a = tr.merge(mj, on='gid', how='inner')
d = a.merge(tm, on=['trackman_game_id', 'pidx'], how='inner', suffixes=('', '_tm'))
print(f'정렬된 투구 {len(d):,}  (train 의 {len(d)/len(tr):.1%})')

# ---- 정렬 검증: 독립적으로 겹치는 컬럼이 실제로 일치하는가 ----
print('\n정렬 검증 (겹치는 컬럼 일치율):')
for c1, c2 in [('inning','inning_tm'), ('balls_before','balls_before_tm'),
               ('strikes_before','strikes_before_tm'), ('outs_before','outs_before_tm'),
               ('season','season_tm'), ('game_month','game_month_tm'),
               ('game_dayofweek','game_dayofweek_tm')]:
    print(f'  {c1:22s} {(d[c1]==d[c2]).mean():.6f}')
hd = {1:'Left', 2:'Right'}
print(f'  pitcher_hand           {(d["pitcher_hand"].map(hd)==d["pitcher_hand_tm"]).mean():.6f}')
print(f'  batter_hand            {(d["batter_hand"].map(hd)==d["batter_hand_tm"]).mean():.6f}')
print(f'  투수 1:1 (pitcher_id x pitcher_trackman_id 최빈 일치율):')
g = d.groupby('pitcher_id')['pitcher_trackman_id']
top = g.agg(lambda s: s.value_counts(normalize=True).iloc[0])
print(f'    투수 {len(top)}명, 최빈 트랙맨id 점유율 중앙 {top.median():.4f}, 90% 이상 {(top>=.9).mean():.2%}')
gb = d.groupby('batter_id')['batter_trackman_id']
topb = gb.agg(lambda s: s.value_counts(normalize=True).iloc[0])
print(f'    타자 {len(topb)}명, 점유율 중앙 {topb.median():.4f}, 90% 이상 {(topb>=.9).mean():.2%}')

print('\n' + '=' * 78)
print('★ 그 투구의 실제 물리량이 control_success 를 얼마나 맞히나 (teacher 상한)')
print('=' * 78)
meas = ['rel_speed','spin_rate','induced_vert_break','horz_break','extension',
        'rel_height','rel_side','zone_speed']
d24 = d[d.season == 2024].copy()
print(f'2024 정렬 투구 {len(d24):,}  성공률 {d24.control_success.mean():.4f}')

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import HistGradientBoostingClassifier
def bss(X, y, note=''):
    r = y.mean(); U = r*(1-r)
    est = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=0)
    p = cross_val_predict(est, X, y, cv=KFold(4, shuffle=True, random_state=0),
                          method='predict_proba')[:, 1]
    s = (1 - ((np.clip(p,1e-6,1-1e-6)-y)**2).mean()/U) * 100000
    print(f'  {note:52s} 스킬 {s:8,.0f}')
    return s

y = d24['control_success'].to_numpy(float)
pt = pd.get_dummies(d24['pitch_type_group'].astype(str), prefix='pt').astype(float)
PRE = ['balls_before','strikes_before','outs_before','inning','li','num_runners_on',
       'asof_pitcher_n','asof_pitcher_success_rate','asof_pitcher_middle_rate',
       'asof_batter_success_rate','pitcher_hand','batter_hand','score_diff_pitcher_team']
Xpre = d24[PRE].astype(float).to_numpy()
Xm   = d24[meas].astype(float).to_numpy()
bss(Xpre, y, 'A. 사전 정보만 (student 가 쓸 수 있는 것)')
bss(Xm, y, 'B. 트랙맨 측정 8개만')
bss(np.hstack([Xm, pt.to_numpy()]), y, 'C. 측정 8개 + 구종군')
bss(np.hstack([Xpre, Xm, pt.to_numpy()]), y, 'D. 사전 + 측정 + 구종군  (teacher)')
bss(np.hstack([Xpre, pt.to_numpy()]), y, 'E. 사전 + 구종군만')

print('\n구종군별 성공률 (2024 정렬분):')
print(d24.groupby('pitch_type_group')['control_success'].agg(['mean','size']).sort_values('size', ascending=False).to_string())
print('\n측정값 x 성공 여부 평균차 (2024, 표준화):')
z = (d24[meas] - d24[meas].mean()) / d24[meas].std()
print((z[d24.control_success==1].mean() - z[d24.control_success==0].mean()).round(4).to_string())
d.to_parquet('cache/raw/aligned.parquet', index=False)
print(f'\n-> cache/raw/aligned.parquet ({len(d):,}행)')

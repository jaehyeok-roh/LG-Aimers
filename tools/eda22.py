# EDA 22 — 정렬을 최대한 넓히고, 그 결과로 **정답 ID 매핑**을 만든다.
#   전체 지문 일치는 55.5% 였다. 트랙맨이 경기 중간 몇 구를 빠뜨리면 지문이 깨진다.
#   그래서 (시즌,월,요일) + 앞 20구 지문으로 후보를 잡고, 투구수/손 일치로 검증한다.
import numpy as np, pandas as pd, hashlib
pd.set_option('display.width', 220)
TC = ['trackman_game_id','season','game_date','pitch_no','inning','top_bottom',
      'balls_before','strikes_before','outs_before','pitcher_trackman_id',
      'batter_trackman_id','pitcher_hand','batter_hand','pitcher_team']
tm = pd.read_parquet('cache/raw/trackman.parquet', columns=TC)
tm['dt'] = pd.to_datetime(tm['game_date'].astype(str), format='mixed', errors='coerce')
tm = tm.sort_values(['trackman_game_id','pitch_no'])
TR = ['row_id','season','game_month','game_dayofweek','game_type','inning','top_bottom',
      'balls_before','strikes_before','outs_before','run_total_before',
      'pitcher_id','batter_id','pitcher_hand','batter_hand']
tr = pd.read_parquet('cache/raw/train.parquet', columns=TR)
c = ['season','game_month','game_dayofweek','game_type']
tr['gid'] = ((tr[c] != tr[c].shift()).any(axis=1) |
             (tr['inning'].diff() < 0) | (tr['run_total_before'].diff() < 0)).cumsum()

def sig(df, g, K=None):
    s = (df['inning'].astype(str) + df['top_bottom'].astype(str).str[0].str.upper() +
         df['balls_before'].astype(str) + df['strikes_before'].astype(str) + df['outs_before'].astype(str))
    d = df.assign(_s=s)
    if K: d = d.groupby(g).head(K)
    return d.groupby(g)['_s'].apply(lambda x: hashlib.md5('|'.join(x).encode()).hexdigest()[:16])

res = {}
for K in [None, 40, 20, 12]:
    a = sig(tr, 'gid', K).rename('h'); b = sig(tm, 'trackman_game_id', K).rename('h')
    a = a[~a.index.isin(res)]                       # 아직 못 붙인 것만
    b = b[~b.isin([])]
    a = a[~a.duplicated(keep=False)]; b = b[~b.duplicated(keep=False)]
    b = b[~b.index.isin(res.values())]
    m = pd.merge(a.reset_index(), b.reset_index(), on='h')
    for _, r in m.iterrows(): res[r['gid']] = r['trackman_game_id']
    print(f'K={str(K):>4}: 신규 {len(m):5,}  누적 {len(res):5,}  ({len(res)/tr.gid.nunique():.1%})')

mp = pd.DataFrame({'gid': list(res.keys()), 'trackman_game_id': list(res.values())})
tr['pidx'] = tr.groupby('gid').cumcount(); tm['pidx'] = tm.groupby('trackman_game_id').cumcount()
d = tr.merge(mp, on='gid').merge(tm, on=['trackman_game_id','pidx'], suffixes=('','_tm'))
hd = {1:'Left', 2:'Right'}
ok = ((d['inning']==d['inning_tm']) & (d['balls_before']==d['balls_before_tm']) &
      (d['strikes_before']==d['strikes_before_tm']) & (d['outs_before']==d['outs_before_tm']))
print(f'\n확장 정렬 {len(d):,}행 ({len(d)/len(tr):.1%})   상태 일치 {ok.mean():.4%}')
d = d[ok]
print(f'  손 일치: 투수 {(d["pitcher_hand"].map(hd)==d["pitcher_hand_tm"]).mean():.4%} '
      f'타자 {(d["batter_hand"].map(hd)==d["batter_hand_tm"]).mean():.4%}')

print('\n' + '=' * 78); print('정답 ID 매핑 (다수결)'); print('=' * 78)
def gold(d, a, b, lab):
    t = d.groupby([a, b]).size().rename('n').reset_index()
    tot = t.groupby(a)['n'].transform('sum')
    t['share'] = t['n'] / tot
    top = t.sort_values('n').groupby(a).tail(1).sort_values('n', ascending=False)
    print(f'{lab}: {len(top)}명  점유율 중앙 {top.share.median():.4f}  '
          f'>=95% {(top.share>=.95).mean():.2%}  >=80% {(top.share>=.80).mean():.2%}')
    return top
gp = gold(d, 'pitcher_id', 'pitcher_trackman_id', '투수')
gb = gold(d, 'batter_id', 'batter_trackman_id', '타자')
# 역방향 유일성
print(f'  투수: trackman_id 하나가 여러 pitcher_id 의 최빈인 경우 '
      f'{gp.pitcher_trackman_id.duplicated().sum()}')

old = pd.read_csv('data/pitcher_id_mapping_v2.csv')
oldm = old.dropna(subset=['pitcher_id','pitcher_trackman_id']).drop_duplicates(['pitcher_id','pitcher_trackman_id'])[['pitcher_id','pitcher_trackman_id']]
j = gp[['pitcher_id','pitcher_trackman_id','share','n']].merge(
        oldm, on='pitcher_id', how='left', suffixes=('_gold','_v2'))
agree = (j['pitcher_trackman_id_gold'] == j['pitcher_trackman_id_v2'])
print(f'\nv2 매핑과 비교: 투수 {len(j)}명 중 v2 에 있는 {j.pitcher_trackman_id_v2.notna().sum()}명, '
      f'일치 {agree.sum()} ({agree.sum()/max(1,j.pitcher_trackman_id_v2.notna().sum()):.1%})')
gp.to_csv('cache/raw/gold_pitcher_map.csv', index=False)
gb.to_csv('cache/raw/gold_batter_map.csv', index=False)
mp.to_csv('cache/raw/game_map.csv', index=False)
print('-> cache/raw/gold_pitcher_map.csv, gold_batter_map.csv, game_map.csv')

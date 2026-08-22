# EDA 18 — train 과 trackman 을 **경기 단위**로 붙일 수 있는가.
#
# eda16/17 에서 확인된 두 가지가 이걸 가능하게 만든다:
#   - train 은 전역 시간순이고, (시즌,월,요일,게임타입) 변화 + 이닝/득점 리셋으로
#     4,868 경기로 깨끗하게 쪼개진다 (시즌당 정규시즌 720 경기 정확히 일치)
#   - trackman 은 trackman_game_id 가 있고, game_date 는 **두 가지 포맷이 섞여 있었다**
#     (2019~21 MM/DD/YYYY, 2022~24 YYYY-MM-DD). format='mixed' 로 읽으면 결측 0%.
#
# 팀 매핑에 의존하지 않는 방법을 쓴다: 경기 안 투구 상태열
#   (inning, top_bottom, balls, strikes, outs) 의 **지문**으로 맞춘다.
import numpy as np, pandas as pd, hashlib
pd.set_option('display.width', 220)

tr = pd.read_parquet('cache/raw/train.parquet')
tm = pd.read_parquet('cache/raw/trackman.parquet')
tm['dt'] = pd.to_datetime(tm['game_date'].astype(str), format='mixed', errors='coerce')
print(f'트랙맨 날짜 결측 {tm.dt.isna().mean():.4%}  범위 {tm.dt.min().date()}~{tm.dt.max().date()}')

# ---- train 경기 분할 ----
c = ['season', 'game_month', 'game_dayofweek', 'game_type']
tr['gid'] = ((tr[c] != tr[c].shift()).any(axis=1) |
             (tr['inning'].diff() < 0) | (tr['run_total_before'].diff() < 0)).cumsum()
print(f'train 경기 {tr.gid.nunique():,}')

# ---- 요일 규약 확인 ----
tmd = tm.drop_duplicates('trackman_game_id')[['trackman_game_id', 'season', 'dt', 'game_dayofweek', 'game_month']]
print('\n트랙맨 game_dayofweek vs 날짜 파생:')
for off in range(7):
    agree = ((tmd['dt'].dt.dayofweek + off) % 7 == tmd['game_dayofweek']).mean()
    if agree > 0.9: print(f'  dow = (pandas dayofweek + {off}) % 7   일치 {agree:.2%}')
print('  train dow 범위', tr.game_dayofweek.min(), '~', tr.game_dayofweek.max(),
      '| tm dow 범위', tm.game_dayofweek.min(), '~', tm.game_dayofweek.max())
print('  트랙맨 month vs 날짜 month 일치', (tmd['dt'].dt.month == tmd['game_month']).mean())

# ---- 상태열 지문 ----
def fp(df, gcol, tbcol):
    s = (df['inning'].astype(str) + df[tbcol].astype(str).str[0].str.upper() +
         df['balls_before'].astype(str) + df['strikes_before'].astype(str) +
         df['outs_before'].astype(str))
    return df.assign(_s=s).groupby(gcol)['_s'].apply(
        lambda x: hashlib.md5('|'.join(x).encode()).hexdigest()[:16])

tm = tm.sort_values(['trackman_game_id', 'pitch_no'])
f_tr = fp(tr, 'gid', 'top_bottom').rename('h')
f_tm = fp(tm, 'trackman_game_id', 'top_bottom').rename('h')
print(f'\n지문 유일성: train {f_tr.nunique()}/{len(f_tr)}   trackman {f_tm.nunique()}/{len(f_tm)}')
m = pd.merge(f_tr.reset_index(), f_tm.reset_index(), on='h')
print(f'★ 전체 상태열 완전일치 경기: {len(m):,}  (train 의 {len(m)/len(f_tr):.1%})')

# ---- 앞 K 투구만으로 (트랙맨 결측 투구 대비) ----
for K in [20, 50]:
    a = tr.groupby('gid').head(K); b = tm.groupby('trackman_game_id').head(K)
    ha = fp(a, 'gid', 'top_bottom').rename('h'); hb = fp(b, 'trackman_game_id', 'top_bottom').rename('h')
    ha = ha[~ha.duplicated(keep=False)]; hb = hb[~hb.duplicated(keep=False)]
    mm = pd.merge(ha.reset_index(), hb.reset_index(), on='h')
    print(f'  앞 {K}투구 지문 (양쪽 유일한 것만): 일치 {len(mm):,} 경기 ({len(mm)/len(f_tr):.1%})')

# ---- 매칭된 경기의 성질 ----
if len(m):
    info_tr = tr.groupby('gid').agg(se=('season','first'), mo=('game_month','first'),
                                    dow=('game_dayofweek','first'), gt=('game_type','first'),
                                    n=('row_id','size'), teams=('pitcher_team_id', lambda s: tuple(sorted(s.unique()))))
    pre = tm['pitcher_team'].astype(str).str.split('_').str[0]
    tm['lvl'] = np.where(pre.eq('MIN') | tm['pitcher_team'].isin(['KBO_ARM','KBO_POL']), '2군', '1군')
    info_tm = tm.groupby('trackman_game_id').agg(dt=('dt','first'), n=('pitch_no','size'),
                                                 lvl=('lvl','first'),
                                                 tms=('pitcher_team', lambda s: tuple(sorted(s.unique()))))
    mj = m.join(info_tr, on='gid').join(info_tm, on='trackman_game_id', rsuffix='_tm')
    print('\n매칭된 경기 — 시즌 x 게임타입:')
    print(mj.groupby(['se','gt']).size().unstack().to_string())
    print('\n매칭된 경기 — train game_type vs trackman 리그수준:')
    print(pd.crosstab(mj['gt'], mj['lvl']).to_string())
    print('\n투구수 일치:', (mj['n'] == mj['n_tm']).mean())
    print('\n팀코드 <-> 팀id 대응 (매칭 경기에서 투표):')
    from collections import Counter
    cnt = {}
    for _, r in mj.iterrows():
        if len(r['teams']) == 2 and len(r['tms']) == 2:
            cnt.setdefault(r['teams'], Counter())[r['tms']] += 1
    rows = [(k, v.most_common(1)[0][0], v.most_common(1)[0][1], sum(v.values())) for k, v in cnt.items()]
    rows.sort(key=lambda x: -x[3])
    for a, b, c_, t in rows[:12]:
        print(f'  {a} <-> {b}   {c_}/{t}')
    mj.to_parquet('cache/raw/game_match.parquet')
    print('\n-> cache/raw/game_match.parquet 저장')

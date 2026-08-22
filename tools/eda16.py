# EDA 16 — eda15 에서 튀어나온 세 가지를 확인한다.
#   A) 트랙맨 game_date 가 정말 2019~2021 에만 있는가  (있다면 v8 휴식일 피처의 근거가 붕괴)
#   B) trackman_game_id 의 단위 — 시즌마다 재사용되는가, 경기 수는 맞는가
#   C) train 을 경기 단위로 쪼갤 수 있는가
import numpy as np, pandas as pd
pd.set_option('display.width', 220)
tm = pd.read_parquet('cache/raw/trackman.parquet')
tr = pd.read_parquet('cache/raw/train.parquet')

print('=' * 78); print('A) game_date 결측 — 시즌별'); print('=' * 78)
gd = pd.to_datetime(tm['game_date'], errors='coerce')
t = pd.DataFrame({'season': tm['season'], 'na': gd.isna(), 'dt': gd})
print(t.groupby('season')['na'].agg(['mean', 'sum']).rename(columns={'mean': '결측비율', 'sum': '결측행'}).to_string())
print('\n비결측 game_date 의 시즌별 범위:')
print(t.dropna(subset=['dt']).groupby('season')['dt'].agg(['min', 'max', 'count']).to_string())
print('\n원본 문자열 샘플 (결측인 행):')
print(tm.loc[gd.isna(), ['season', 'game_date', 'game_month', 'game_dayofweek']].head(5).to_string())

print('\n' + '=' * 78); print('B) trackman_game_id 의 단위'); print('=' * 78)
print(f'고유 game_id {tm.trackman_game_id.nunique():,}')
gs = tm.groupby('trackman_game_id')['season'].nunique()
print(f'  한 game_id 가 여러 시즌에 걸치는 경우: {(gs > 1).sum():,}개  -> {"재사용됨" if (gs>1).any() else "시즌 고유"}')
per = tm.groupby('season')['trackman_game_id'].nunique()
print('\n시즌별 고유 game_id 수:'); print(per.to_string())
pre = tm['pitcher_team'].astype(str).str.split('_').str[0]
tm['lvl'] = np.where(pre.eq('MIN'), '2군', np.where(pre.eq('KBO'), '올스타',
                     np.where(pre.eq('ACE'), '기타', '1군')))
print('\n시즌 x 리그수준 별 고유 game_id 수 (1군 정규시즌은 720 경기여야 한다):')
print(tm.groupby(['season', 'lvl'])['trackman_game_id'].nunique().unstack().to_string())
# 한 game_id 안에 여러 수준이 섞이는가
mix = tm.groupby('trackman_game_id')['lvl'].nunique()
print(f'\n한 game_id 안에 리그수준이 섞인 경기: {(mix > 1).sum():,}')
g1 = tm[tm.lvl == '1군']
print(f'1군 투구 {len(g1):,} / 1군 경기 {g1.trackman_game_id.nunique():,} '
      f'-> 경기당 {len(g1)/g1.trackman_game_id.nunique():.0f}구')

print('\n' + '=' * 78); print('C) train 을 경기 단위로 쪼갤 수 있는가'); print('=' * 78)
# 파일이 전역 시간순이라면: 새 경기 = run_total_before 가 0 으로 리셋 & inning=1
tr['newg'] = ((tr['inning'] == 1) & (tr['run_total_before'] == 0) &
              (tr['outs_before'] == 0) & (tr['balls_before'] == 0) &
              (tr['strikes_before'] == 0))
print(f'inning=1 & 0점 & 0아웃 & 0-0 인 행: {tr["newg"].sum():,}')
print('  (경기당 1회 = 1회초 첫 투구. KBO 6시즌 1군 정규시즌 ~4,320경기 예상)')
print(tr.groupby('season')['newg'].sum().to_string())
# 그 지표로 경기 id 를 붙이고 검증
tr['gid'] = tr['newg'].cumsum()
gg = tr.groupby('gid')
chk = gg.agg(n=('row_id', 'size'), inn_mono=('inning', lambda s: s.is_monotonic_increasing),
             run_mono=('run_total_before', lambda s: s.is_monotonic_increasing),
             seasons=('season', 'nunique'), months=('game_month', 'nunique'),
             dows=('game_dayofweek', 'nunique'))
print(f'\n복원된 경기 {len(chk):,}개')
print(f'  투구수 중앙 {chk.n.median():.0f}  q10 {chk.n.quantile(.1):.0f}  q90 {chk.n.quantile(.9):.0f}  max {chk.n.max()}')
print(f'  이닝 비감소 {chk.inn_mono.mean():.2%} | 득점 비감소 {chk.run_mono.mean():.2%}')
print(f'  시즌 1개 {(chk.seasons==1).mean():.2%} | 월 1개 {(chk.months==1).mean():.2%} | 요일 1개 {(chk.dows==1).mean():.2%}')

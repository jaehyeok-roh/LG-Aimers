import numpy as np, pandas as pd
pd.set_option('display.width', 220)
tm = pd.read_parquet('cache/raw/trackman.parquet')
tr = pd.read_parquet('cache/raw/train.parquet')

print('=' * 78); print('A) game_date 파싱 실패의 정체'); print('=' * 78)
print('dtype:', tm['game_date'].dtype)
s = tm['game_date'].astype(str)
print('2019 샘플 repr:', [repr(x) for x in s[tm.season == 2019].head(3)])
print('2022 샘플 repr:', [repr(x) for x in s[tm.season == 2022].head(3)])
print('길이 분포:', s.str.len().value_counts().to_dict())
for fmt in ['%Y-%m-%d']:
    p = pd.to_datetime(s, format=fmt, errors='coerce')
    print(f'format={fmt} 결측 {p.isna().mean():.2%}')
p2 = pd.to_datetime(s, errors='coerce', format='mixed')
print(f'format=mixed 결측 {p2.isna().mean():.2%}')
tm['dt'] = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
print('\n시즌별 실제 날짜 범위 / 고유 날짜 수:')
print(tm.groupby('season')['dt'].agg(['min', 'max', 'nunique']).to_string())

print('\n' + '=' * 78); print('B) train 경기 분할 — 제대로'); print('=' * 78)
c = ['season', 'game_month', 'game_dayofweek', 'game_type']
chg = (tr[c] != tr[c].shift()).any(axis=1)
drop = (tr['inning'].diff() < 0) | (tr['run_total_before'].diff() < 0)
tr['gid'] = (chg | drop).cumsum()
gg = tr.groupby('gid')
chk = gg.agg(n=('row_id', 'size'), maxinn=('inning', 'max'),
             np=('pitcher_id', 'nunique'), nteam=('pitcher_team_id', 'nunique'),
             gt=('game_type', 'first'), se=('season', 'first'))
print(f'분할된 덩어리 {len(chk):,}개  (시즌당 {len(chk)/6:.0f})')
print(f'  투구수: 중앙 {chk.n.median():.0f} q10 {chk.n.quantile(.1):.0f} q90 {chk.n.quantile(.9):.0f} max {chk.n.max()}')
print(f'  덩어리당 투수팀 수: {chk.nteam.value_counts().head().to_dict()}')
print(f'  덩어리당 최대이닝 분포: {chk.maxinn.value_counts().sort_index().head(12).to_dict()}')
print('\n같은 (시즌,월,요일) 안에 여러 경기가 있으면 이 방식으론 못 나눈다 -> 팀쌍으로 확인')
print('시즌x게임타입별 덩어리 수:')
print(chk.groupby(['se', 'gt']).size().unstack().to_string())

print('\n' + '=' * 78); print('C) game_type F 의 정체'); print('=' * 78)
print('행 비율:'); print((tr.groupby('season')['game_type'].value_counts(normalize=True).unstack() * 100).round(2).to_string())
for gt in sorted(tr.game_type.unique()):
    d = tr[tr.game_type == gt]
    print(f'\n--- game_type={gt}  {len(d):,}행 ---')
    print(f'  투수팀 id: {sorted(d.pitcher_team_id.unique())}')
    print(f'  투수 수 {d.pitcher_id.nunique()}  타자 수 {d.batter_id.nunique()}')
    print(f'  월 분포: {d.game_month.value_counts().sort_index().to_dict()}')
    print(f'  이닝 최대 {d.inning.max()}  성공률 {d.control_success.mean():.4f}')
sp = set(tr[tr.game_type=='F'].pitcher_id) & set(tr[tr.game_type=='R'].pitcher_id)
print(f'\nR 과 F 양쪽에 나오는 투수: {len(sp)} / F 전체 {tr[tr.game_type=="F"].pitcher_id.nunique()}')
print('F 전용 투수 수:', len(set(tr[tr.game_type=='F'].pitcher_id) - set(tr[tr.game_type=='R'].pitcher_id)))

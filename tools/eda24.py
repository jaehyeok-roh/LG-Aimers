# EDA 24 — F(퓨처스) 라벨 체제 변화가 프로젝트의 '시즌 드리프트' 서사를 얼마나 오염시켰나.
import numpy as np, pandas as pd
pd.set_option('display.width', 220)
tr = pd.read_parquet('cache/raw/train.parquet',
        columns=['season','game_type','control_success','pitcher_id','asof_pitcher_n'])

print('=' * 78); print('1) 리그 평균 — 전체 vs R 전용 vs F 전용'); print('=' * 78)
tot = tr.groupby('season')['control_success'].mean()
byg = tr.groupby(['season','game_type'])['control_success'].mean().unstack()
sh  = tr.groupby('season')['game_type'].value_counts(normalize=True).unstack()['F']
t = pd.DataFrame({'전체': tot, 'R': byg['R'], 'F': byg['F'], 'F비중': sh})
t['전체_하락'] = t['전체'].diff(); t['R_하락'] = t['R'].diff()
print(t.round(4).to_string())
print('\n★ CLAUDE.md 의 "2022->2023 은 6년 중 최대 낙폭 -0.0290" 은 전체 기준이다.')
print(f'   R 전용으로 보면 {t.loc[2023,"R_하락"]:+.4f} 로 사실상 0 이다. 낙폭은 F 재라벨링이었다.')

print('\n' + '=' * 78); print('2) 2025 베이스레이트 재외삽'); print('=' * 78)
def lin(y, x=None, k=3):
    x = np.arange(len(y)) if x is None else x
    c = np.polyfit(x[-k:], y[-k:], 1); return np.polyval(c, x[-1]+1)
R = t['R'].to_numpy(); F = t['F'].to_numpy(); S = t['F비중'].to_numpy()
r25, f25, s25 = lin(R), lin(F[-2:], k=2), S[-3:].mean()
print(f'  R 최근3년 선형 -> 2025 R = {r25:.4f}')
print(f'  F 2023~24 선형 -> 2025 F = {f25:.4f}   (2019~22 는 다른 체제라 못 씀)')
print(f'  F 비중 최근3년 평균 {s25:.4f}')
print(f'  -> 2025 전체 예상 = {r25*(1-s25)+f25*s25:.4f}')
print(f'  (오염된 전체 계열로 외삽하면 {lin(tot.to_numpy()):.4f} — CLAUDE.md 의 0.462~0.475 대역)')
print(f'  같은 방법으로 2024 를 맞혀보면: R {lin(R[:-1]):.4f}(실제 {R[-1]:.4f}) '
      f'전체 {lin(R[:-1])*(1-S[-4:-1].mean())+F[-2]*S[-4:-1].mean():.4f}(실제 {tot.iloc[-1]:.4f})')

print('\n' + '=' * 78); print('3) 2023 홀드아웃이 왜 상수예측보다 못했나'); print('=' * 78)
d23 = tr[tr.season == 2023]
fm = d23.game_type.eq('F')
# 2019~2022 로 학습한 모델은 F 행을 그 시절 수준으로 예측한다
p_F_old = tr[(tr.season < 2023) & (tr.game_type == 'F')]['control_success'].mean()
p_R_old = tr[(tr.season < 2023) & (tr.game_type == 'R')]['control_success'].mean()
act_F, act_R = d23[fm].control_success.mean(), d23[~fm].control_success.mean()
print(f'  2019~22 F 평균 {p_F_old:.4f}  -> 2023 F 실제 {act_F:.4f}   오차 {p_F_old-act_F:+.4f}')
print(f'  2019~22 R 평균 {p_R_old:.4f}  -> 2023 R 실제 {act_R:.4f}   오차 {p_R_old-act_R:+.4f}')
r = d23.control_success.mean(); U = r*(1-r)
pen = fm.mean() * (p_F_old-act_F)**2 + (~fm).mean() * (p_R_old-act_R)**2
print(f'  이 편향만으로 늘어나는 Brier {pen:.5f}  -> 스킬 {-pen/U*100000:,.0f} 점')
print('  (실측된 2023 홀드아웃 Optuna 전 trial 이 -937 ~ -1360 이었다)')

print('\n' + '=' * 78); print('4) F 체제가 asof 커리어 통계를 얼마나 오염시키나 (2024 시점)'); print('=' * 78)
old = tr[(tr.season < 2023)]
fshare = old.groupby('pitcher_id')['game_type'].apply(lambda s: (s=='F').mean()).rename('구체제F비중')
n_old = old.groupby('pitcher_id').size().rename('구체제투구')
d24 = tr[tr.season == 2024]
p24 = d24.groupby('pitcher_id').agg(n24=('control_success','size'), y24=('control_success','mean'))
j = p24.join(fshare).join(n_old).dropna()
j = j[(j.n24 >= 200) & (j.구체제투구 >= 200)]
b = pd.cut(j['구체제F비중'], [-.01,.001,.1,.3,.6,1.01], labels=['0','0-10%','10-30%','30-60%','60%+'])
print(j.groupby(b, observed=True).agg(투수=('y24','size'), y2024=('y24','mean'),
      구체제F비중=('구체제F비중','mean')).round(4).to_string())
print('  -> 구체제 F 비중이 높은 투수일수록 asof 커리어 성공률이 위로 부풀려져 있다')

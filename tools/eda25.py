# EDA 25 — F 체제가 우리가 **실제로 쓰는 피처**를 얼마나 왜곡하는가.
#   cond_* 는 '그 시즌 리그평균 대비 편차' 로 디트렌드한다. 그런데 2019~22 F 의
#   리그평균은 0.68 이고 R 은 0.51 이다. 시즌 하나로 뭉뚱그리면 F 를 많이 던진 투수는
#   편차가 통째로 +0.17 만큼 부풀려진다.
import numpy as np, pandas as pd
pd.set_option('display.width', 220)
tr = pd.read_parquet('cache/raw/train.parquet',
        columns=['season','game_type','control_success','pitcher_id'])

lg_s  = tr.groupby('season')['control_success'].mean()                       # 현행
lg_sg = tr.groupby(['season','game_type'])['control_success'].mean()          # 제안
tr['dev_s']  = tr['control_success'] - tr['season'].map(lg_s)
tr['dev_sg'] = tr['control_success'] - pd.MultiIndex.from_arrays(
                    [tr['season'], tr['game_type']]).map(lg_sg)

SE = 2024
past = tr[tr.season < SE]
C = 200
def cond(col):
    g = past.groupby('pitcher_id')[col].agg(['sum','size'])
    return (g['sum'] / (g['size'] + C)).rename(col)
a, b = cond('dev_s'), cond('dev_sg')
j = pd.concat([a, b], axis=1).join(past.groupby('pitcher_id').size().rename('n'))
j = j[j.n >= 200]
print(f'투수 {len(j)}명')
print(f'  cond_p 두 판의 상관 {j.dev_s.corr(j.dev_sg):.4f}   '
      f'평균절대차 {(j.dev_s-j.dev_sg).abs().mean():.4f}  최대차 {(j.dev_s-j.dev_sg).abs().max():.4f}')
fs = past.groupby('pitcher_id')['game_type'].apply(lambda s: (s=='F').mean())
j['F비중'] = fs
bb = pd.cut(j['F비중'], [-.01,.001,.1,.3,.6,1.01], labels=['0','0-10%','10-30%','30-60%','60%+'])
print('\nF 비중별 두 판의 차이 (현행 - 제안):')
print(j.assign(diff=j.dev_s-j.dev_sg).groupby(bb, observed=True)
        .agg(투수=('n','size'), 현행=('dev_s','mean'), 제안=('dev_sg','mean'),
             차이=('diff','mean')).round(4).to_string())

print('\n' + '=' * 78)
print('어느 판이 2024 성적을 더 잘 맞히나 (2024 투수별 편차와의 상관)')
print('=' * 78)
d24 = tr[tr.season == SE]
y = d24.groupby('pitcher_id').agg(n24=('control_success','size'), y=('dev_sg','mean'))
k = j.join(y, how='inner'); k = k[k.n24 >= 200]
print(f'  대상 투수 {len(k)}명')
for c in ['dev_s','dev_sg']:
    r = np.corrcoef(k[c], k['y'])[0,1]
    w = np.sqrt(k['n24']); rw = np.cov(k[c], k['y'], aweights=w)[0,1]/np.sqrt(
        np.cov(k[c],k[c],aweights=w)[0,1]*np.cov(k['y'],k['y'],aweights=w)[0,1])
    print(f'  {c:8s}  상관 {r:+.4f}   투구수 가중 {rw:+.4f}')

print('\n' + '=' * 78)
print('wseason 디트렌드도 같은 문제 — 당해 시즌 비율을 시즌 리그평균으로 뺀다')
print('=' * 78)
for s in sorted(tr.season.unique()):
    d = tr[tr.season==s]
    print(f'  {s}: 시즌평균 {lg_s[s]:.4f} | R {lg_sg[(s,"R")]:.4f} F {lg_sg[(s,"F")]:.4f} '
          f'| F 행이 받는 오차 {lg_sg[(s,"F")]-lg_s[s]:+.4f}')

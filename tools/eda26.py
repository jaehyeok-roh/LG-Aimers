# EDA 26 — 1:1 정렬이 준 새 라벨: **그 투구의 구종**.
#   eda19 에서 사전정보 613 -> 사전+구종 905 (+291). 구종 자체는 test 에서 못 쓴다.
#   하지만 '그 상황에서 이 투수가 무엇을 던질 확률' 은 train 으로 만드는 룩업이라 합법이다
#   (규칙 4번). 지금 우리는 커리어 구종비율 3개만 갖고 있고 eda10 에서 증분 +0 이었다.
#   문제는 그게 **상황 조건부가 아니라서** 일 수 있다. 여기서 조건부 변동 폭을 잰다.
import numpy as np, pandas as pd
pd.set_option('display.width', 220)
d = pd.read_parquet('cache/raw/aligned.parquet',
      columns=['season','game_type','control_success','pitcher_id','pitch_type_group',
               'balls_before','strikes_before','outs_before','inning','batter_hand',
               'pitcher_hand','num_runners_on','li'])
d = d[d.pitch_type_group.isin(['fastball','breaking','offspeed'])]
print(f'정렬 투구 {len(d):,}  시즌 {sorted(d.season.unique())}')
d['cnt'] = d.balls_before.astype(str) + '-' + d.strikes_before.astype(str)

print('\n' + '=' * 78); print('1) 구종 배합의 카운트 조건부 변동'); print('=' * 78)
t = pd.crosstab(d['cnt'], d['pitch_type_group'], normalize='index')
t['n'] = d.groupby('cnt').size()
t['성공률'] = d.groupby('cnt')['control_success'].mean()
print(t.round(4).to_string())
print(f'\n  fastball 비율 범위 {t.fastball.min():.3f} ~ {t.fastball.max():.3f}  '
      f'(전체 {d.pitch_type_group.eq("fastball").mean():.3f})')

print('\n' + '=' * 78); print('2) 구종 x 카운트 별 성공률 — 구종 효과는 카운트와 독립인가'); print('=' * 78)
p = d.pivot_table(index='cnt', columns='pitch_type_group', values='control_success', aggfunc='mean')
print(p.round(4).to_string())
print('\n  구종 간 격차(fastball - breaking) 카운트별:')
print((p['fastball'] - p['breaking']).round(4).to_string())

print('\n' + '=' * 78); print('3) 구종 예측 가능성 — 조건부 엔트로피 감소'); print('=' * 78)
def H(p):
    p = p[p > 0]; return -(p*np.log2(p)).sum()
base = H(d.pitch_type_group.value_counts(normalize=True))
def cond_H(keys):
    g = d.groupby(keys)['pitch_type_group'].value_counts(normalize=True)
    w = d.groupby(keys).size(); w = w/w.sum()
    h = g.groupby(level=list(range(len(keys)))).apply(lambda s: H(s.to_numpy()))
    return float((h*w).sum())
print(f'  무조건 엔트로피            {base:.4f} bit')
for k, lab in [(['cnt'],'카운트'), (['pitcher_id'],'투수'),
               (['pitcher_id','cnt'],'투수 x 카운트'),
               (['pitcher_id','cnt','batter_hand'],'투수 x 카운트 x 타자손')]:
    h = cond_H(k)
    print(f'  {lab:26s} {h:.4f} bit   감소 {base-h:.4f} ({(base-h)/base:.1%})')
print('  (투수를 알면 대부분 설명된다 = 커리어 구종비율이 이미 담고 있다.')
print('   카운트가 그 위에 얼마나 더하는지가 새 정보의 크기다)')

print('\n' + '=' * 78); print('4) 실패 유형 대리 — 구종군별 시즌 추세'); print('=' * 78)
print(d.pivot_table(index='season', columns='pitch_type_group',
                    values='control_success', aggfunc='mean').round(4).to_string())
print('\n구종 배합의 시즌 추세:')
print(pd.crosstab(d['season'], d['pitch_type_group'], normalize='index').round(4).to_string())

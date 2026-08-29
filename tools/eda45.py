# 큰 피처(wseason5 +61.72 / wsbat +30.11)의 정합성 검산 — 학습 0, 순수 pandas.
#
# 투수측은 복원 success 를 control_success 와 대조해 일치율 1.000000 을 확인한 기록이
# 있는데 **타자측은 그 검산이 문서에 없다.** 배포 점수의 30점이 미검증 상태다.
# 여기에 어긋남이 있으면 '버그 수정' 부류(전달률 0.45)의 후보가 된다.
#
# 보는 것 넷:
#   1) asof_batter_n 이 투구 단위인가 타석 단위인가  (복원 분모가 맞는지)
#   2) 타자 당해시즌 success 복원값 vs 라벨로 직접 계산한 값
#   3) 투수측 같은 검산을 전수로 (기록은 391명 표본이었다)
#   4) 복원값이 범위를 벗어나는 행 (분모<0, 비율 [0,1] 밖) = 룩업이 틀린 신호
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
C = ['row_id', 'season', 'pitcher_id', 'batter_id', 'control_success',
     'asof_pitcher_n', 'asof_pitcher_success_rate',
     'asof_batter_n', 'asof_batter_success_rate']
df = pd.read_csv('data/train.csv', usecols=C, keep_default_na=False, na_values=_NA)
print('%s행\n' % f'{len(df):,}')

# ---------- 1) asof_batter_n 의 단위 ----------
d = df.sort_values(['batter_id', 'asof_batter_n'])
inc = d.groupby('batter_id')['asof_batter_n'].diff()
v = inc.dropna()
print('=== 1) asof_batter_n 증분 분포 ===')
print(v.value_counts().head(5).to_string())
print('  +1 비율 %.6f  (1.0 이면 투구 단위, 0 이 섞이면 타석 단위)' % (v == 1).mean())
dp = df.sort_values(['pitcher_id', 'asof_pitcher_n'])
vp = dp.groupby('pitcher_id')['asof_pitcher_n'].diff().dropna()
print('  (대조) asof_pitcher_n +1 비율 %.6f' % (vp == 1).mean())

# ---------- 2~3) 당해 시즌 복원 vs 라벨 ----------
def check(entity, ncol, rcol):
    g = df.sort_values([entity, ncol])
    # 그 (엔티티, 시즌) 의 첫 행 = 시즌 시작 시점의 커리어 상태
    f = g.groupby([entity, 'season']).first().reset_index()
    f['n0'] = f[ncol]
    f['x0'] = (f[rcol] * f[ncol]).round()
    # 시즌 마지막 행 = 시즌 끝 시점(그 투구 직전까지)
    l = g.groupby([entity, 'season']).last().reset_index()
    l['n1'] = l[ncol]
    l['x1'] = (l[rcol] * l[ncol]).round()
    m = f[[entity, 'season', 'n0', 'x0']].merge(
        l[[entity, 'season', 'n1', 'x1']], on=[entity, 'season'])
    # 복원: 그 시즌 동안의 성공 수 / 투구 수 (마지막 행 자신은 asof 에 안 들어간다)
    m['rec_n'] = m['n1'] - m['n0'] + 1
    m['rec_x'] = m['x1'] - m['x0']
    lab = df.groupby([entity, 'season']).agg(
        true_n=('control_success', 'size'), true_x=('control_success', 'sum')).reset_index()
    m = m.merge(lab, on=[entity, 'season'])
    m = m[m['rec_n'] > 0]
    dn = (m['rec_n'] - m['true_n']).abs()
    # 마지막 투구 자신의 결과는 asof 에 없으므로 x 는 최대 1 어긋난다 (구조적)
    dx = (m['rec_x'] - (m['true_x'] - m['control_last'])).abs() if 'control_last' in m else None
    print('\n=== %s 당해시즌 복원 검산 (%s개 엔티티-시즌) ===' % (entity, f'{len(m):,}'))
    print('  투구수:  일치 %.6f | 최대차 %d | 평균차 %.4f'
          % ((dn == 0).mean(), dn.max(), dn.mean()))
    ax = (m['rec_x'] - m['true_x']).abs()
    print('  성공수:  차이<=1 %.6f | 최대차 %d | 평균차 %.4f'
          % ((ax <= 1).mean(), ax.max(), ax.mean()))
    bad = m[ax > 1]
    if len(bad):
        print('  ⚠️ 어긋난 엔티티-시즌 %s개 (%.2f%%)' % (f'{len(bad):,}', 100 * len(bad) / len(m)))
        print(bad[[entity, 'season', 'rec_n', 'true_n', 'rec_x', 'true_x']].head(8).to_string(index=False))
    return m


mp = check('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_success_rate')
mb = check('batter_id', 'asof_batter_n', 'asof_batter_success_rate')

# ---------- 4) 복원 비율이 범위를 벗어나는가 ----------
for nm, m in (('투수', mp), ('타자', mb)):
    r = m['rec_x'] / m['rec_n'].clip(lower=1)
    out = ((r < 0) | (r > 1)).mean()
    print('\n%s 복원비율: 평균 %.4f | [0,1] 밖 %.4f%% | 분모<=0 %d건'
          % (nm, r.mean(), 100 * out, (m['rec_n'] <= 0).sum()))

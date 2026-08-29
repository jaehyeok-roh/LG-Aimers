# 투수 유리 카운트의 결손은 '못 잡는 신호' 인가 '없는 신호' 인가 — 학습 0.
#
# eda42: 투수 유리 카운트(28.4%)가 비중대로면 253 을 낼 자리에서 158 만 낸다
#        (행당으로 타자 유리 카운트의 45%). 유일하게 남은 편차다.
#
# 두 해석이 있고 처방이 정반대다:
#   (a) 못 잡는 신호   -> 그 구간용 피처를 만들면 먹는다
#   (b) 없는 신호      -> 0-2/1-2 에서는 투수가 일부러 존을 벗어나게 던지므로
#                        투수 실력의 분산 자체가 작다. 어떤 피처도 못 먹는다.
#
# 가르는 법: 반쪽 분할 신뢰도 (eda33 의 방법을 **카운트 구간별로** 쪼갠다).
#   한 투수의 그 구간 행을 무작위 절반으로 갈라 각각 성공률을 잰다.
#   두 측정치는 같은 참값의 독립 추정이므로  Cov(A,B) = Var(참값) 이다.
#   신호 std = sqrt(Cov)  /  신뢰도 = corr(A,B).
#   신호 std 가 작으면 그 구간엔 투수별 차이가 애초에 없다는 뜻이다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
C = ['season', 'game_type', 'pitcher_id', 'batter_hand', 'control_success',
     'balls_before', 'strikes_before', 'asof_pitcher_n']
df = pd.read_csv('data/train.csv', usecols=C, keep_default_na=False, na_values=_NA)
df = df[(df.season == 2024) & (df.game_type == 'R')].copy()

pa = (df.balls_before < df.strikes_before)
ba = (df.balls_before > df.strikes_before)
nt = (df.balls_before == df.strikes_before) & (df.balls_before.isin([1, 2]))
df['count_adv'] = np.where(pa, 'Pitcher', np.where(ba, 'Batter',
                           np.where(nt, 'Neutral', 'None')))
print('2024 R %s행' % f'{len(df):,}')
print(df.count_adv.value_counts(normalize=True).round(3).to_string(), '\n')

rng = np.random.default_rng(42)
df['half'] = rng.integers(0, 2, len(df))


def split_rel(sub, minn):
    g = sub.groupby(['pitcher_id', 'half'])['control_success'].agg(['mean', 'size'])
    w = g.unstack('half')
    if w.shape[1] < 4:
        return None
    a, b = w[('mean', 0)], w[('mean', 1)]
    na, nb = w[('size', 0)], w[('size', 1)]
    ok = (na >= minn) & (nb >= minn) & a.notna() & b.notna()
    a, b = a[ok].to_numpy(), b[ok].to_numpy()
    if len(a) < 30:
        return None
    cov = float(np.cov(a, b)[0, 1])
    return dict(n=len(a), rel=float(np.corrcoef(a, b)[0, 1]),
                sig=float(np.sqrt(max(cov, 0.0))),
                obs=float(np.std(np.concatenate([a, b]))),
                rate=float(sub.control_success.mean()))


print('=== 카운트 구간별 투수 신호 (반쪽 분할, 각 반쪽 최소 40구) ===')
print('%-10s%8s%9s%11s%11s%9s' % ('구간', '투수수', '성공률', '신뢰도', '신호std', '관측std'))
base = None
for v in ['Batter', 'None', 'Neutral', 'Pitcher']:
    r = split_rel(df[df.count_adv == v], 40)
    if r is None:
        print('%-10s  표본 부족' % v); continue
    if v == 'Batter':
        base = r['sig']
    print('%-10s%8d%9.4f%11.3f%11.4f%9.4f'
          % (v, r['n'], r['rate'], r['rel'], r['sig'], r['obs']))

r_all = split_rel(df, 100)
print('\n(대조) 전 구간 합: 투수 %d명 신뢰도 %.3f 신호std %.4f'
      % (r_all['n'], r_all['rel'], r_all['sig']))

print('\n=== 정확한 12칸 카운트별 ===')
print('%-8s%8s%9s%11s%11s' % ('카운트', '투수수', '성공률', '신뢰도', '신호std'))
df['cnt'] = df.balls_before.astype(str) + '-' + df.strikes_before.astype(str)
for v in sorted(df.cnt.unique()):
    r = split_rel(df[df.cnt == v], 15)
    if r is None:
        print('%-8s  표본 부족 (행 %s)' % (v, f'{(df.cnt == v).sum():,}')); continue
    print('%-8s%8d%9.4f%11.3f%11.4f' % (v, r['n'], r['rate'], r['rel'], r['sig']))

# EDA 5 — 신호가 어느 '행' 에 있는가. 그리고 진짜 천장은 얼마인가.
#
# eda3/eda4: 신호는 거의 전부 **투수 정체성**이다 (0.82%, 잡음 보정 후).
#            상황 변수는 전부 합쳐 0.1% 미만.
# eda3: 2024 투구의 **19.9% 가 그 전에 없던 신규 투수**다.
#
# 그러면 당연한 질문: 투수 이력이 없는 행에서 우리는 무엇을 할 수 있나.
# 신호의 대부분이 투수 정체성인데 20% 의 행에 그게 없다면, 거기가 통째로 비어 있다.
#
# 1) 분할 반분 신뢰도로 **진짜 천장**을 잡는다 (표본잡음을 제거한 신호 분산)
# 2) 이력 있는 행 vs 없는 행으로 나눠 각각의 천장을 잰다
# 3) 신규 투수에게 쓸 수 있는 대리 정보가 있는지 본다
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
SE = 2024
prior = set(df[df.season < SE]['pitcher_id'])
d = df[df.season == SE].reset_index(drop=True)
d['seen'] = d['pitcher_id'].isin(prior)
y = d['control_success'].to_numpy(dtype='float64')
r = y.mean()
U = r * (1 - r)

rng = np.random.default_rng(0)


def split_half_var(keys, mask=None, reps=12, min_n=40):
    """분할 반분 신뢰도로 '진짜' 집단간 분산을 추정한다.

    같은 그룹을 무작위 반으로 갈라 두 반쪽 평균의 **공분산**을 본다.
    표본잡음은 두 반쪽에서 독립이므로 공분산에서 사라지고 신호만 남는다.
    """
    m = np.ones(len(y), bool) if mask is None else mask
    k = pd.Series(keys)[m].to_numpy()
    yy = y[m]
    out = []
    for _ in range(reps):
        h = rng.random(len(yy)) < 0.5
        t = pd.DataFrame({'k': k, 'y': yy, 'h': h})
        a = t[t.h].groupby('k')['y'].agg(['mean', 'size'])
        b = t[~t.h].groupby('k')['y'].agg(['mean', 'size'])
        j = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
        j = j[(j['size_a'] >= min_n) & (j['size_b'] >= min_n)]
        if len(j) < 15:
            continue
        w = (j['size_a'] + j['size_b'])
        w = w / w.sum()
        ma = (j['mean_a'] * w).sum()
        mb = (j['mean_b'] * w).sum()
        out.append(float((w * (j['mean_a'] - ma) * (j['mean_b'] - mb)).sum()))
    return (float(np.mean(out)) if out else np.nan), (len(j) if out else 0)


print(f'{SE} 시즌 {len(d):,}행 | 리그평균 {r:.4f}')
print(f'  이력 있는 투수의 투구 {d["seen"].mean():.1%} / 신규 투수 {1-d["seen"].mean():.1%}')
print(f'  투수 수: 이력 있음 {d[d.seen]["pitcher_id"].nunique()} / '
      f'신규 {d[~d.seen]["pitcher_id"].nunique()}\n')

print('=' * 76)
print('1) 분할 반분으로 잰 진짜 신호 분산 (= 도달 가능한 최대 스킬)')
print(f'{"축":<26}{"그룹":>7}{"신호분산":>12}{"최대스킬":>11}')
d['count_str'] = d['balls_before'].astype(str) + '-' + d['strikes_before'].astype(str)
d['ph'] = d['pitcher_id'].astype(str) + '|' + d['batter_hand'].astype(str)
d['pc'] = d['pitcher_id'].astype(str) + '|' + d['count_str']
for lab, k, mn in [('투수', d['pitcher_id'], 40), ('타자', d['batter_id'], 40),
                   ('투수 x 타자손', d['ph'], 40), ('투수 x 카운트', d['pc'], 25),
                   ('볼카운트', d['count_str'], 500), ('투수팀', d['pitcher_team_id'], 500)]:
    v, ng = split_half_var(k.to_numpy(), min_n=mn)
    print(f'{lab:<26}{ng:>7,}{v:>12.6f}{v/U:>10.3%}')

print('\n' + '=' * 76)
print('2) 이력 있는 행 vs 신규 투수 행 — 각각의 투수 신호')
for lab, m in [('이력 있는 투수', d['seen'].to_numpy()), ('신규 투수', (~d['seen']).to_numpy())]:
    v, ng = split_half_var(d['pitcher_id'].to_numpy(), mask=m, min_n=40)
    share = m.mean()
    print(f'  {lab:<16} 행 {m.sum():>7,} ({share:>5.1%})  투수 {ng:>3}명  '
          f'신호분산 {v:.6f}  그 안에서 {v/U:>6.3%}  전체기여 {v/U*share:>6.3%}')

print("""
  → '전체기여' 는 그 조각이 전체 스킬에 낼 수 있는 최대치다.
    두 조각의 합이 우리가 투수 축에서 뽑을 수 있는 전부다.""")

print('\n' + '=' * 76)
print('3) 신규 투수에게 쓸 수 있는 대리 정보가 있는가')
nw = d[~d['seen']]
print(f'  신규 투수 {nw["pitcher_id"].nunique()}명, {len(nw):,}구, 평균 성공률 {nw["control_success"].mean():.4f}')
print(f'  (이력 있는 투수 {d[d.seen]["control_success"].mean():.4f})')
print(f'\n  투수당 투구수 분포: ' + '  '.join(
    f'{q}%={int(nw.groupby("pitcher_id").size().quantile(q/100))}' for q in (25, 50, 75, 90)))
g = nw.groupby('pitcher_id').agg(n=('row_id', 'size'), rate=('control_success', 'mean'))
g = g[g['n'] >= 40]
print(f'  40구 이상 신규 투수 {len(g)}명의 성공률 표준편차 {g["rate"].std():.4f} '
      f'(잡음 기대치 {np.sqrt(0.25/g["n"].mean()):.4f})')

print('\n  팀별 신규 투수 성공률 — 팀이 대리변수가 되는가')
t = nw.groupby('pitcher_team_id')['control_success'].agg(['mean', 'size'])
t = t[t['size'] >= 1500].sort_values('mean')
for k, row in t.iterrows():
    print(f'    팀 {k}: {row["mean"]:.4f}  ({int(row["size"]):,}구)')
v, ng = split_half_var(nw['pitcher_team_id'].to_numpy(),
                       mask=None if len(nw) == len(d) else None, min_n=400)

print('\n  신규 투수 행에서, 시즌 내 누적(asof)이 얼마나 빨리 쓸모 있어지는가')
nw2 = nw.sort_values(['pitcher_id', 'asof_pitcher_n'])
nw2['within'] = nw2.groupby('pitcher_id').cumcount()
for lo, hi in [(0, 50), (50, 150), (150, 400), (400, 1000), (1000, 99999)]:
    seg = nw2[(nw2['within'] >= lo) & (nw2['within'] < hi)]
    if len(seg) < 2000:
        continue
    v, ng = split_half_var(seg['pitcher_id'].to_numpy(),
                           mask=np.ones(len(seg), bool), min_n=25) if False else (np.nan, 0)
    print(f'    시즌 내 {lo}~{hi}번째 투구: {len(seg):>6,}구 '
          f'({len(seg)/len(d):>5.1%} of 2024)  성공률 {seg["control_success"].mean():.4f}')

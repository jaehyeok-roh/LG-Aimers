# EDA 6 — 신규 투수 조각에서 우리가 무엇을 놓치고 있는가.
#
# eda5: 2024 투구의 19.9% 가 신규 투수인데, 그 조각의 투수 신호가 오히려 더 크다
#       (1.152% vs 이력 있는 투수 0.743%). 전체 기여 상한 0.229% = 229점.
#       컷까지 격차 117 의 두 배다.
#
# 그 조각에서 `cond_*` 는 전부 NaN 이다 (직전 시즌 이력으로 만들기 때문).
# 남은 것은 주최측이 주는 `asof_*` 뿐인데, 그건 **시즌 내에서 누적된다** —
# 학습에서도 추론에서도 똑같이 그렇다. 즉 합법이고 분포 불일치도 없다.
#
# 그러면 질문은 하나: **asof_* 만으로 그 조각에서 얼마나 뽑을 수 있나.**
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
SE = 2024
prior = set(df[df.season < SE]['pitcher_id'])
d = df[df.season == SE].reset_index(drop=True)
d['seen'] = d['pitcher_id'].isin(prior)
r_all = d['control_success'].mean()
print(f'{SE} {len(d):,}행 | 리그 {r_all:.4f} | 신규 투수 {(~d["seen"]).mean():.1%}\n')


def skill(p, yv):
    r = float(yv.mean())
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yv) ** 2).mean() / (r * (1 - r))) * 100000


print('=' * 78)
print('1) asof_pitcher_success_rate 를 그대로 예측으로 쓰면 (shrink C 최적화)')
print('   — 학습 없이, 그 행 이전 정보만으로. 조각별 상한의 근사치다.')
print(f'{"조각":<20}{"행수":>9}{"최적C":>7}{"스킬":>9}{"리그예측대비":>12}')
for lab, m in [('전체', np.ones(len(d), bool)),
               ('이력 있는 투수', d['seen'].to_numpy()),
               ('신규 투수', (~d['seen']).to_numpy())]:
    s = d[m]
    yv = s['control_success'].to_numpy(dtype='float64')
    a = s['asof_pitcher_success_rate'].to_numpy(dtype='float64')
    n = s['asof_pitcher_n'].to_numpy(dtype='float64')
    prior_m = float(np.nanmean(a))
    bs, bc = -9e9, None
    for C in (0, 25, 50, 100, 200, 400, 800, 1600):
        p = np.where(np.isnan(a), prior_m, (a * n + prior_m * C) / np.maximum(n + C, 1))
        v = skill(p, yv)
        if v > bs:
            bs, bc = v, C
    print(f'{lab:<20}{m.sum():>9,}{bc:>7}{bs:>9,.0f}{bs:>12,.0f}')

print('\n' + '=' * 78)
print('2) 신규 투수의 시즌 내 위치별 — 누적이 언제부터 쓸모 있어지는가')
nw = d[~d['seen']].copy().sort_values(['pitcher_id', 'asof_pitcher_n'])
nw['within'] = nw.groupby('pitcher_id').cumcount()
prior_m = float(nw['asof_pitcher_success_rate'].mean())
print(f'{"시즌내 위치":<18}{"행수":>9}{"비중":>8}{"asof 예측 스킬":>15}')
for lo, hi in [(0, 30), (30, 100), (100, 300), (300, 800), (800, 99999)]:
    s = nw[(nw['within'] >= lo) & (nw['within'] < hi)]
    if len(s) < 1500:
        continue
    yv = s['control_success'].to_numpy(dtype='float64')
    a = s['asof_pitcher_success_rate'].to_numpy(dtype='float64')
    n = s['asof_pitcher_n'].to_numpy(dtype='float64')
    p = np.where(np.isnan(a), prior_m, (a * n + prior_m * 100) / np.maximum(n + 100, 1))
    print(f'{f"{lo}~{hi}":<18}{len(s):>9,}{len(s)/len(d):>7.1%}{skill(p, yv):>15,.0f}')

print('\n' + '=' * 78)
print('3) 신규 투수는 정말 이력이 없는가 — asof_pitcher_n 분포')
print('   (주최측 asof 는 커리어 누적이다. 신규 투수도 시즌 내에서 쌓인다)')
q = nw['asof_pitcher_n'].quantile([.1, .25, .5, .75, .9]).astype(int)
print(f'   신규 투수 행의 asof_pitcher_n: ' + '  '.join(f'{int(k*100)}%={v}' for k, v in q.items()))
sn = d[d['seen']]['asof_pitcher_n'].quantile([.1, .5, .9]).astype(int)
print(f'   이력 있는 투수 행:            ' + '  '.join(f'{int(k*100)}%={v}' for k, v in sn.items()))
print(f'\n   신규 투수 행 중 asof_pitcher_n < 100 인 비율: '
      f'{(nw["asof_pitcher_n"] < 100).mean():.1%}  '
      f'(= 전체의 {(nw["asof_pitcher_n"] < 100).sum()/len(d):.1%})')

print('\n' + '=' * 78)
print('4) 그 20% 조각에서 상황 축은 쓸모가 있는가 (분할 반분)')
rng = np.random.default_rng(0)


def shv(keys, yy, reps=12, min_n=30):
    out = []
    for _ in range(reps):
        h = rng.random(len(yy)) < 0.5
        t = pd.DataFrame({'k': keys, 'y': yy, 'h': h})
        a = t[t.h].groupby('k')['y'].agg(['mean', 'size'])
        b = t[~t.h].groupby('k')['y'].agg(['mean', 'size'])
        j = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
        j = j[(j['size_a'] >= min_n) & (j['size_b'] >= min_n)]
        if len(j) < 10:
            continue
        w = (j['size_a'] + j['size_b'])
        w = w / w.sum()
        out.append(float((w * (j['mean_a'] - (j['mean_a'] * w).sum())
                          * (j['mean_b'] - (j['mean_b'] * w).sum())).sum()))
    return (float(np.mean(out)) if out else np.nan), (len(j) if out else 0)


yv = nw['control_success'].to_numpy(dtype='float64')
Un = yv.mean() * (1 - yv.mean())
nw['ph'] = nw['pitcher_id'].astype(str) + '|' + nw['batter_hand'].astype(str)
for lab, k, mn in [('투수', nw['pitcher_id'], 40), ('투수 x 타자손', nw['ph'], 30),
                   ('투수팀', nw['pitcher_team_id'], 400)]:
    v, ng = shv(k.astype(str).to_numpy(), yv, min_n=mn)
    print(f'  {lab:<18}그룹 {ng:>4}   신호 {v:.6f}   조각 내 {v/Un:>6.3%}   '
          f'전체기여 {v/Un*len(nw)/len(d):>6.3%}')

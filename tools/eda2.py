# EDA 2 — 피처별 '관계 모양' 이 시즌마다 얼마나 바뀌는가.
#
# eda1 에서 상황 변수(카운트/이닝/주자/아웃)의 조건부 구조는 거의 안 변한다는 것을 봤다
# (잔차 변화폭 0.004~0.02 vs 리그 수준 이동 0.079). 그런데 **투수 등판량 티어는 부호가
# 뒤집혔다** (2019 에는 적게 던지는 투수가 더 좋고, 2024 에는 더 나쁘다).
#
# 그래서 실제 사용 피처 전부에 같은 것을 잰다:
#   전역 분위수로 10구간을 만들고, (시즌 x 구간) 평균 성공률에서 그 시즌 리그평균을 뺀다.
#   남는 것이 '관계의 모양'. 그 모양이 시즌마다 얼마나 움직이는지 본다.
#
# 핵심 지표 두 개:
#   spread  = 모양의 크기 (구간 간 최대-최소, 시즌 평균). 작으면 애초에 신호가 없다.
#   drift   = 모양의 시즌 간 이동 (구간별 시즌간 표준편차의 평균).
#   flip    = 초기 3시즌 모양과 후기 3시즌 모양의 상관. 음수면 관계가 뒤집혔다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
y, ss = df['control_success'].to_numpy(), df['season'].to_numpy()
S = sorted(df['season'].unique())
lg = pd.Series({s: y[ss == s].mean() for s in S})

skip = {'row_id', 'control_success', 'season', 'pitcher_id', 'batter_id'}
num = [c for c in df.columns
       if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
print(f'수치 피처 {len(num)}개 | 시즌 {S}\n')

rows = []
for c in num:
    v = df[c].to_numpy(dtype='float64')
    ok = ~np.isnan(v)
    try:
        q = np.unique(np.nanquantile(v[ok], np.linspace(0, 1, 11)))
    except Exception:
        continue
    if len(q) < 4:
        continue
    b = np.digitize(v, q[1:-1])
    t = pd.DataFrame({'b': b, 's': ss, 'y': y})[ok]
    g = t.groupby(['s', 'b'])['y'].agg(['mean', 'size']).reset_index()
    g = g[g['size'] >= 800]
    piv = g.pivot(index='b', columns='s', values='mean').dropna()
    if piv.shape[0] < 4 or piv.shape[1] < len(S):
        continue
    shape = piv.sub(lg[piv.columns], axis=1)          # 리그 수준 제거 = 관계의 모양
    early = shape[S[:3]].mean(axis=1)
    late = shape[S[-3:]].mean(axis=1)
    rows.append({
        'feature': c,
        'bins': piv.shape[0],
        'spread': float((shape.max(axis=0) - shape.min(axis=0)).mean()),
        'drift': float(shape.std(axis=1).mean()),
        'flip': float(np.corrcoef(early, late)[0, 1]),
        'e_rng': float(early.max() - early.min()),
        'l_rng': float(late.max() - late.min()),
    })

d = pd.DataFrame(rows)
d['ratio'] = d['drift'] / d['spread'].clip(lower=1e-9)
d = d.sort_values('spread', ascending=False)

print('=' * 96)
print('신호가 큰 피처 20개 — 모양의 크기(spread) vs 모양의 이동(drift)')
print(f'{"피처":<40}{"spread":>9}{"drift":>9}{"이동/크기":>10}{"전후상관":>10}')
for _, x in d.head(20).iterrows():
    print(f'{x["feature"][:38]:<40}{x["spread"]:>9.4f}{x["drift"]:>9.4f}'
          f'{x["ratio"]:>10.2f}{x["flip"]:>10.3f}')

print('\n' + '=' * 96)
print('★ 관계가 뒤집히거나 무너지는 피처 — 전후상관이 낮은 순 (spread 0.01 이상만)')
print(f'{"피처":<40}{"spread":>9}{"전후상관":>10}{"전반범위":>10}{"후반범위":>10}')
big = d[d['spread'] >= 0.01].sort_values('flip')
for _, x in big.head(18).iterrows():
    print(f'{x["feature"][:38]:<40}{x["spread"]:>9.4f}{x["flip"]:>10.3f}'
          f'{x["e_rng"]:>10.4f}{x["l_rng"]:>10.4f}')

print('\n' + '=' * 96)
print('★ 모양의 이동이 크기 대비 큰 피처 (이동/크기 비율, spread 0.01 이상만)')
for _, x in big.sort_values('ratio', ascending=False).head(12).iterrows():
    print(f'{x["feature"][:38]:<40}{x["spread"]:>9.4f}{x["drift"]:>9.4f}{x["ratio"]:>10.2f}')

d.to_csv('cache/eda_shape.csv', index=False)
print(f'\n저장: cache/eda_shape.csv  ({len(d)}개 피처)')

# --- 등판량 역전을 직접 확인 ---
print('\n' + '=' * 96)
print('asof_pitcher_n (커리어 누적 투구수) 5분위별 성공률 — 리그 제거 후')
v = df['asof_pitcher_n'].to_numpy(dtype='float64')
b = np.digitize(v, np.nanquantile(v, [.2, .4, .6, .8]))
t = pd.DataFrame({'b': b, 's': ss, 'y': y})
piv = t.groupby(['s', 'b'])['y'].mean().unstack()
sh = piv.sub(lg, axis=0)
print(f'{"시즌":<8}' + ''.join(f'{f"Q{i+1}":>10}' for i in range(sh.shape[1])))
for s in S:
    print(f'{s:<8}' + ''.join(f'{sh.loc[s, k]:>+10.4f}' for k in sh.columns))
print(f'{"원값":<8}' + ''.join(f'{piv.loc[S[-1], k]:>10.4f}' for k in piv.columns) + '  (2024)')

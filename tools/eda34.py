# EDA 34 — 투수 커리어 추정치가 '상대 타자의 질' 로 편향돼 있는가.
#
# eda33 의 재해석: wseason/wsbat 이 이긴 방식은 '최근 정보' 가 아니라
# **커리어 추정치의 편향 제거**였다 (옛 시즌 리그 수준 오염을 씻어냄).
# 그러면 남은 질문은 "커리어 추정치가 또 어디서 편향돼 있나" 다.
#
# 가장 유력한 후보: **상대 타자의 질**.
#   `asof_pitcher_success_rate` 는 누구를 상대했는지 보정하지 않는다.
#   어려운 타자를 많이 상대한 투수(선발·필승조)는 값이 눌려 있고,
#   약한 타자를 상대한 투수(추격조·2군)는 부풀려져 있다.
#   그리고 그 보정은 **다른 행에서 정보를 끌어와야** 한다 — 트리가 만들 수 없다.
#
# 판정 기준은 '설명력' 이 아니라 **다음 시즌 예측력**이다:
#   보정한 추정치가 raw 보다 그 투수의 **다음 시즌** 편차를 잘 맞히면 진짜다.
#   (in-sample 적합도는 상대질을 넣으면 무조건 오른다 — 그건 의미 없다.)
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
tr = pd.read_csv('data/train.csv',
                 usecols=['season', 'game_type', 'pitcher_id', 'batter_id',
                          'control_success'],
                 keep_default_na=False, na_values=_NA)
print(f'{len(tr):,}행\n')

# 시즌 x game_type 리그평균으로 디트렌드 (F 체제 변경 반영)
tr['dev'] = tr['control_success'] - tr.groupby(
    ['season', 'game_type'])['control_success'].transform('mean')

# ---- 타자 질: 그 타자를 상대한 전체 투구의 평균 편차 (커리어) ----
# ⚠️ 자기 투수의 기여를 빼야 순환하지 않는다 (leave-one-pitcher-out).
bt = tr.groupby('batter_id')['dev'].agg(['sum', 'size'])
pb = tr.groupby(['pitcher_id', 'batter_id'])['dev'].agg(['sum', 'size'])
key = pd.MultiIndex.from_arrays([tr['pitcher_id'], tr['batter_id']])
bs = bt['sum'].reindex(tr['batter_id']).to_numpy()
bn = bt['size'].reindex(tr['batter_id']).to_numpy()
ps = pb['sum'].reindex(key).to_numpy()
pn = pb['size'].reindex(key).to_numpy()
C_B = 200.0
tr['bq'] = (bs - ps) / np.maximum(bn - pn, 1) * (
    (bn - pn) / (bn - pn + C_B))          # 표본 적으면 0 으로 shrink
print(f'타자 질(자기 투수 제외): std {tr["bq"].std():.4f}  '
      f'범위 {tr["bq"].quantile(.01):+.4f} ~ {tr["bq"].quantile(.99):+.4f}')

# ---- 투수-시즌 단위 집계 ----
g = tr.groupby(['pitcher_id', 'season']).agg(
    dev=('dev', 'mean'), bq=('bq', 'mean'), n=('dev', 'size')).reset_index()
g = g[g['n'] >= 200]
print(f'투수-시즌 {len(g):,}개 (200구+)')
print(f'  상대 타자 질의 투수-시즌 간 std : {g["bq"].std():.4f}')
print(f'  투수 편차의 std                : {g["dev"].std():.4f}')
print(f'  둘의 상관                      : {np.corrcoef(g["bq"], g["dev"])[0,1]:+.3f}')

# ---- 커리어 추정치: raw vs 상대질 보정 ----
# 각 (투수, 시즌) 에 대해 **그 시즌 이전까지**의 커리어 추정치를 만든다 (leak-free)
g = g.sort_values(['pitcher_id', 'season'])
gg = g.groupby('pitcher_id')
for c in ('dev', 'bq'):
    w = (g[c] * g['n']).groupby(g['pitcher_id']).cumsum() - g[c] * g['n']
    nn = gg['n'].cumsum() - g['n']
    g['past_' + c] = np.where(nn > 0, w / np.maximum(nn, 1), np.nan)
g['past_n'] = gg['n'].cumsum() - g['n']

d = g[(g['past_n'] >= 300) & g['past_dev'].notna()].copy()
# 상대질 보정: 과거 편차에서 그때 상대한 타자 질의 몫을 뺀다
beta = np.polyfit(d['past_bq'], d['past_dev'], 1)[0]
d['adj_dev'] = d['past_dev'] - beta * (d['past_bq'] - d['past_bq'].mean())
print(f'\n보정 계수 beta = {beta:+.3f}  (과거편차 ~ 과거상대질)')

print('\n' + '=' * 66)
print('★ 다음 시즌 편차를 얼마나 맞히나 (이게 유일한 판정 기준)')
print(f'{"추정치":<34}{"상관":>9}{"R2":>9}')
print('-' * 52)
y = d['dev'].to_numpy()
for lab, x in [('과거 커리어 편차 (raw)', d['past_dev'].to_numpy()),
               ('상대질 보정한 과거 편차', d['adj_dev'].to_numpy()),
               ('(참고) 그 시즌 상대질만', d['bq'].to_numpy())]:
    r = float(np.corrcoef(x, y)[0, 1])
    print(f'{lab:<34}{r:>+9.4f}{r*r:>9.4f}')

# 둘을 같이 넣으면?
X = np.column_stack([d['past_dev'], d['past_bq'], np.ones(len(d))])
b, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ b
r2 = float(np.corrcoef(pred, y)[0, 1] ** 2)
print(f'{"과거편차 + 과거상대질 (2변수)":<34}{np.sqrt(r2):>+9.4f}{r2:>9.4f}')
print(f'   계수: past_dev {b[0]:+.3f}  past_bq {b[1]:+.3f}')

print(f'\n표본 {len(d):,} 투수-시즌 (과거 300구+)')
print("""
읽는 법:
  '보정' 이 raw 보다 상관이 **뚜렷이** 높으면 -> 커리어 추정치가 상대질로 편향돼
    있고, 보정 피처를 만들 값어치가 있다 (wseason/wsbat 과 같은 부류).
  차이가 없으면 -> KBO 일정이 균형적이라 상대질이 투수마다 비슷하다는 뜻이고,
    이 축은 닫힌다.
⚠️ 이건 대리지표다. 양수여도 스크리너/리더보드로 판정한다 (오늘 대리지표 3연패).""")

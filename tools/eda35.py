# EDA 35 — 투수 커리어 추정치가 '상대 타자의 질' 로 편향돼 있는가.
#
# eda33 의 재해석: wseason/wsbat 이 이긴 방식은 '최근 정보' 가 아니라
# **커리어 추정치의 편향 제거**였다 (옛 시즌 리그 수준 오염을 씻어냄).
# 그러면 남은 질문은 "커리어 추정치가 또 어디서 편향돼 있나" 다.
#
# eda34 에서 '상대 타자의 질' 은 닫혔다 (투수-시즌 간 std 0.0022 = 투수 편차의 1/21,
# KBO 균형 편성이라 다들 같은 타선을 상대한다). 같은 틀로 다음 후보를 잰다.
#
# 후보: **상황 믹스**.
#   선발은 1회부터 주자 없이 시작하고, 필승조는 주자 있는 접전에 나오고,
#   추격조는 점수차 큰 상황에 나온다. 상황마다 제구 성공률이 다르므로
#   `asof_pitcher_success_rate` 는 **어떤 상황을 던졌느냐로 편향**돼 있을 수 있다.
#   보정하려면 리그 전체의 '상황별 난이도' 를 알아야 한다 = 다른 행에서 끌어온다.
#
# 판정 기준은 '설명력' 이 아니라 **다음 시즌 예측력**이다:
#   보정한 추정치가 raw 보다 그 투수의 **다음 시즌** 편차를 잘 맞히면 진짜다.
#   (in-sample 적합도는 상대질을 넣으면 무조건 오른다 — 그건 의미 없다.)
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
tr = pd.read_csv('data/train.csv',
                 usecols=['season', 'game_type', 'pitcher_id', 'batter_id',
                          'balls_before', 'strikes_before', 'base_state', 'inning',
                          'batter_hand', 'li', 'control_success'],
                 keep_default_na=False, na_values=_NA)
print(f'{len(tr):,}행\n')

# 시즌 x game_type 리그평균으로 디트렌드 (F 체제 변경 반영)
tr['dev'] = tr['control_success'] - tr.groupby(
    ['season', 'game_type'])['control_success'].transform('mean')

# ---- 상황 난이도: 리그 전체의 상황별 평균 편차 ----
# 셀 = 볼카운트 x 주자상태 x 이닝구간 x 타자손. 투수 정체성이 안 섞이도록
# **그 투수를 뺀** 리그 평균으로 만든다 (leave-one-pitcher-out).
tr['cell'] = (tr['balls_before'].astype(str) + '-' + tr['strikes_before'].astype(str)
              + '|' + tr['base_state'].astype(str)
              + '|' + np.clip((tr['inning'] - 1) // 3, 0, 2).astype(str)
              + '|' + tr['batter_hand'].astype(str))
ct = tr.groupby('cell')['dev'].agg(['sum', 'size'])
pc = tr.groupby(['pitcher_id', 'cell'])['dev'].agg(['sum', 'size'])
key = pd.MultiIndex.from_arrays([tr['pitcher_id'], tr['cell']])
cs = ct['sum'].reindex(tr['cell']).to_numpy()
cn = ct['size'].reindex(tr['cell']).to_numpy()
ps = pc['sum'].reindex(key).to_numpy()
pn = pc['size'].reindex(key).to_numpy()
C_B = 200.0
tr['bq'] = (cs - ps) / np.maximum(cn - pn, 1) * (
    (cn - pn) / (cn - pn + C_B))
print(f'상황 셀 {tr["cell"].nunique():,}개 | 난이도 std {tr["bq"].std():.4f}  '
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
               ('상황믹스 보정한 과거 편차', d['adj_dev'].to_numpy()),
               ('(참고) 그 시즌 상황난이도만', d['bq'].to_numpy())]:
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
  '보정' 이 raw 보다 상관이 **뚜렷이** 높으면 -> 커리어 추정치가 상황믹스로 편향돼
    있고, 보정 피처를 만들 값어치가 있다 (wseason/wsbat 과 같은 부류).
  차이가 없으면 -> 보직이 달라도 상황 난이도 평균은 비슷하다는 뜻이고,
    이 축은 닫힌다.
⚠️ 이건 대리지표다. 양수여도 스크리너/리더보드로 판정한다 (오늘 대리지표 3연패).""")

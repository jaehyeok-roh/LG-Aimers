# EDA 12 — 데이터 위생 감사. 불가능한 값 / 중복 / 컬럼 간 관계.
#
# 트리는 피처 이상치에 둔감하다 (분할이 순위 기반). 하지만 둔감하지 않은 것:
#   - 라벨 오류: 트리가 맞추려고 용량을 쓴다
#   - 불가능한 값/데이터 버그: 분할 자체를 왜곡한다
#   - 컬럼 간 결정적 관계: 있으면 피처가 중복이고, 깨져 있으면 버그다
#
# 열흘 동안 아무도 이걸 안 봤다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
n = len(df)
print(f'{n:,}행 x {df.shape[1]}컬럼\n')

print('=' * 76)
print('1) 비율 컬럼이 [0,1] 을 벗어나는가 / 개수가 정수인가')
rates = [c for c in df.columns if c.endswith('_rate')]
bad = 0
for c in rates:
    v = df[c].to_numpy(dtype='float64')
    o = np.nansum((v < -1e-9) | (v > 1 + 1e-9))
    if o:
        print(f'  ⚠️ {c}: 범위 밖 {o:,}행')
        bad += 1
print(f'  범위 밖 컬럼 {bad}개 / {len(rates)}개' if bad else
      f'  ✅ 비율 {len(rates)}개 전부 [0,1] 안')

for rc, nc in [('asof_pitcher_success_rate', 'asof_pitcher_n'),
               ('asof_pitcher_fastball_rate', 'asof_pitcher_pitchmix_n'),
               ('asof_batter_success_rate', 'asof_batter_n')]:
    x = df[rc].fillna(0).to_numpy(dtype='float64') * df[nc].fillna(0).to_numpy(dtype='float64')
    err = np.abs(x - np.round(x))
    print(f'  {rc:<34} x {nc:<26} 정수 오차 최대 {err.max():.6f}')

print('\n' + '=' * 76)
print('2) 컬럼 간 결정적 관계 — 합이 1 인가 (있으면 피처가 중복)')
combos = [
    (['asof_pitcher_success_rate', 'asof_pitcher_middle_rate', 'asof_pitcher_reverse_rate'],
     '성공 + 몰림 + 반대'),
    (['asof_pitcher_ball_rate', 'asof_pitcher_strike_rate'], '볼 + 스트라이크'),
    (['asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate',
      'asof_pitcher_offspeed_rate'], '직구 + 변화 + 오프스피드'),
    (['asof_batter_success_rate', 'asof_batter_middle_rate'], '타자 성공 + 몰림'),
]
for cols, lab in combos:
    if not all(c in df.columns for c in cols):
        continue
    s = df[cols].sum(axis=1)
    q = s.quantile([0, .5, 1]).round(4).tolist()
    print(f'  {lab:<26} 합 최소/중앙/최대 {q}  |  1.0 인 비율 '
          f'{float((s.sub(1).abs() < 1e-6).mean()):.1%}')

print('\n' + '=' * 76)
print('3) 중복 행 / 퇴화 컬럼')
key = ['season', 'pitcher_id', 'batter_id', 'asof_pitcher_n']
d = int(df.duplicated(subset=key).sum())
print(f'  (시즌,투수,타자,asof_n) 중복 {d:,}행 ({d/n:.3%})')
const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print(f'  상수 컬럼: {const if const else "없음"}')
near = [(c, df[c].value_counts(normalize=True, dropna=False).iloc[0])
        for c in df.columns if df[c].nunique(dropna=False) <= 30]
near = [(c, v) for c, v in near if v > 0.97]
print(f'  한 값이 97% 이상인 컬럼: {[(c, round(v,4)) for c, v in near] if near else "없음"}')

print('\n' + '=' * 76)
print('4) asof_pitcher_n 이 투수별로 정확히 +1 씩 증가하는가')
sub = df[['pitcher_id', 'asof_pitcher_n']].copy()
sub = sub.sort_values(['pitcher_id', 'asof_pitcher_n'], kind='stable')
g = sub.groupby('pitcher_id')['asof_pitcher_n']
diff = g.diff()
print(f'  증분 분포: ' + '  '.join(
    f'{int(k)}={int(v):,}' for k, v in diff.value_counts().head(4).items()))
print(f'  시작값이 0 이 아닌 투수: '
      f'{int((g.min() != 0).sum())}명 / {sub.pitcher_id.nunique()}명')

print('\n' + '=' * 76)
print('5) 극단 행 — 표본이 거의 없는 상태의 asof')
for c, nc in [('asof_pitcher_success_rate', 'asof_pitcher_n'),
              ('asof_batter_success_rate', 'asof_batter_n')]:
    v = df[nc].fillna(0).to_numpy(dtype='float64')
    for th in (0, 5, 20, 100):
        m = v <= th
        if m.sum() == 0:
            continue
        rr = df.loc[m, 'control_success'].mean()
        print(f'  {nc:<22} <= {th:>4}: {m.sum():>8,}행 ({m.mean():>5.2%})  '
              f'성공률 {rr:.4f}  {c} 결측 {df.loc[m, c].isna().mean():.1%}')
    print()

print('=' * 76)
print('6) 라벨 쪽 — 완전히 같은 상황인데 결과가 갈리는 비율 (내재 잡음의 하한)')
kk = ['pitcher_id', 'batter_id', 'balls_before', 'strikes_before', 'outs_before',
      'base_state', 'inning', 'season']
g2 = df.groupby(kk)['control_success'].agg(['size', 'mean'])
g2 = g2[g2['size'] >= 2]
mixed = float(((g2['mean'] > 0) & (g2['mean'] < 1)).mean())
print(f'  2회 이상 반복된 (투수x타자x상황) 조합 {len(g2):,}개')
print(f'  그중 결과가 갈리는 조합 {mixed:.1%}')
print(f'  -> 완전히 동일한 조건에서도 결과가 갈린다면 그만큼은 환원 불가능한 잡음이다')

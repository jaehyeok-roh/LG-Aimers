# EDA 15 — 상황별 구종 성향을 제대로 다시 잰다.
#
# eda14 에서 P(구종|투수,카운트,손) 이 -12(=0) 였는데, 그때 base 에 **투수-시즌 평균
# 물리량 8개**가 들어 있었다. 그건 투수의 레퍼토리를 거의 다 담으므로 성향이 중복됐다.
# 불공정한 비교였다.
#
# 논리적으로는 오히려 유망한 이유가 있다:
#   - `cond_pc`(투수x카운트 **성적**)는 표본이 카운트 12개로 쪼개져 잡음투성이다
#   - **구종 선택은 결정론적에 가까워** 훨씬 적은 표본으로 정확히 추정된다
#   - 변화구가 제구하기 어렵다는 것은 +478 로 확인됐다 (실제 구종을 알 때)
#
# 그래서: 실제 모델과 비슷한 base + shrink + **배포 조건**(직전 시즌까지의 트랙맨)
# 으로 다시 잰다. 2024 를 평가 시즌으로 두고 트랙맨은 <=2023 만 쓴다.
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
SE = 2024
tm = pd.read_csv('data/trackman_history.csv', keep_default_na=False, na_values=_NA)
mp = pd.read_csv('data/pitcher_id_mapping_v2.csv', keep_default_na=False, na_values=_NA)
tr = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)

tm = tm.merge(mp[['pitcher_id', 'pitcher_trackman_id', 'season']].dropna(),
              on=['pitcher_trackman_id', 'season'], how='inner')
tm = tm[tm['season'] < SE]                       # ★ 배포 조건: 직전 시즌까지만
tm['ct'] = tm['balls_before'].astype(str) + '-' + tm['strikes_before'].astype(str)
TYPES = sorted(tm['pitch_type_group'].dropna().unique())
print(f'트랙맨(<= {SE-1}) 매핑 후 {len(tm):,}행 | 구종군 {TYPES}')

d = tr[tr['season'] == SE].reset_index(drop=True)
d['ct'] = d['balls_before'].astype(str) + '-' + d['strikes_before'].astype(str)
d['bh'] = d['batter_hand'].map({1: 'Left', 2: 'Right'}).fillna(d['batter_hand'].astype(str))
y = d['control_success'].to_numpy(dtype='float64')
r = y.mean()
U = r * (1 - r)
print(f'{SE} {len(d):,}행 | 리그 {r:.4f}\n')


def sk(X, tag=''):
    p = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
        X, y, cv=KFold(4, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / U) * 100000


def mix(keys, C, name):
    """키별 구종 비율을 투수 전체 믹스로 shrink 한다.
    구종 선택은 결정론적에 가까워 적은 표본으로도 잘 추정된다."""
    g = tm.groupby(keys + ['pitch_type_group']).size().unstack(fill_value=0)
    g = g.reindex(columns=TYPES, fill_value=0)
    p_all = tm.groupby(['pitcher_id', 'pitch_type_group']).size().unstack(fill_value=0)
    p_all = p_all.reindex(columns=TYPES, fill_value=0)
    p_all = p_all.div(p_all.sum(1).clip(lower=1), axis=0)
    if keys == ['pitcher_id']:
        base = p_all
    else:
        base = p_all.reindex(g.index.get_level_values('pitcher_id')).to_numpy()
    n = g.sum(1).to_numpy()[:, None]
    sh = (g.to_numpy() + np.asarray(base) * C) / (n + C)
    out = pd.DataFrame(sh, index=g.index, columns=[f'{name}_{t}' for t in TYPES])
    idx = pd.MultiIndex.from_arrays([d[k if k != 'ct' else 'ct'] if k != 'batter_hand'
                                     else d['bh'] for k in keys])
    v = out.reindex(idx).to_numpy(dtype='float64')
    miss = np.isnan(v[:, 0]).mean()
    v = np.where(np.isnan(v), np.nanmean(v, axis=0), v)
    print(f'  {name:<10} 셀 {len(out):>7,}  중앙 표본 {np.median(n):>5.0f}  결측 {miss:>5.1%}')
    return v


print('구종 성향표 (shrink C=30, 투수 전체 믹스로 당김)')
P_p = mix(['pitcher_id'], 0.0, 'mix_p')                      # 투수 전체 (기준)
P_pc = mix(['pitcher_id', 'ct'], 30.0, 'mix_pc')             # x 카운트
P_pch = mix(['pitcher_id', 'ct', 'batter_hand'], 30.0, 'mix_pch')   # x 카운트 x 손

# 상황 성향의 **투수 전체 대비 편차** = 진짜 증분 정보
D_pc = P_pc - P_p
D_pch = P_pch - P_p
print(f'\n  편차 크기: pc {np.abs(D_pc).mean():.4f}  pch {np.abs(D_pch).mean():.4f}')

# ---- base: 실제 모델이 가진 투수/상황 정보의 근사 ----
cols = ['asof_pitcher_success_rate', 'asof_pitcher_n', 'asof_pitcher_middle_rate',
        'asof_pitcher_reverse_rate', 'asof_pitcher_fastball_rate',
        'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate',
        'balls_before', 'strikes_before', 'outs_before', 'inning',
        'asof_batter_success_rate']
B = d[cols].astype('float64').fillna(d[cols].astype('float64').median()).to_numpy()
B = np.hstack([B, pd.get_dummies(d['bh'], prefix='bh').to_numpy(float),
               pd.get_dummies(d['base_state'].astype(str), prefix='bs').to_numpy(float)])

print('\n' + '=' * 66)
print(f'{"구성":<44}{"스킬":>9}{"증분":>9}')
b0 = sk(B)
print(f'{"base (투수 asof + 커리어 믹스 + 상황)":<44}{b0:>9,.0f}')
for lab, V in [('+ 투수 전체 구종 믹스 (트랙맨)', P_p),
               ('+ 투수x카운트 성향', P_pc),
               ('+ 투수x카운트x손 성향', P_pch),
               ('+ 투수x카운트 성향의 **편차**', D_pc),
               ('+ 편차 (pc + pch)', np.hstack([D_pc, D_pch]))]:
    v = sk(np.hstack([B, V]))
    print(f'{lab:<44}{v:>9,.0f}{v-b0:>+9,.0f}')

print("""
읽는 법: 상황 성향이 base 위에서 +30 이상이면 후보로 만들 값어치가 있다.
  '편차' 항이 원본 성향보다 나으면, 값어치가 투수 수준이 아니라 **상황 조정**에 있다는 뜻.
  ⚠️ 트랙맨 기반이라 2025 에는 1년 묵은 값을 쓴다 (staleness 77% 손실 전례).""")

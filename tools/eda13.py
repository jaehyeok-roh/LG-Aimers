# EDA 13 — 트랙맨 데이터를 처음으로 제대로 본다.
#
# 지금까지 트랙맨은 '피처를 만들어 시도했다 실패했다' 는 기록만 있다
# (매핑 재구축 +3.56 / 릴리스 동역학 -2.48 / 휴식·등판밀도 -6.85 / zone_speed -3.35).
# 그런데 **데이터 자체를 들여다본 적이 없다.**
#
# 오늘 분산 분해로 알게 된 것: 신호는 거의 전부 투수 정체성(0.83%)이다.
# 그러면 트랙맨에 대한 결정적 질문은 하나다:
#
#   **트랙맨 측정값이 투수의 제구 능력을 얼마나 설명하는가?**
#
# 투수-시즌 단위로 트랙맨 집계 -> 그 투수-시즌의 실제 제구 성공률을 얼마나 맞히나.
# 이게 0 이면 트랙맨은 이 타겟에 대해 정보가 없는 것이고 축이 완전히 닫힌다.
# 0 이 아니면 우리가 못 뽑아낸 것이 있다는 뜻이다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
tm = pd.read_csv('data/trackman_history.csv', keep_default_na=False, na_values=_NA)
tr = pd.read_csv('data/train.csv',
                 usecols=['season', 'pitcher_id', 'control_success', 'pitcher_hand'],
                 keep_default_na=False, na_values=_NA)
mp = pd.read_csv('data/pitcher_id_mapping_v2.csv', keep_default_na=False, na_values=_NA)

print(f'트랙맨 {len(tm):,}행 x {tm.shape[1]}컬럼 | train {len(tr):,}행')
print(f'매핑 v2 {len(mp):,}행  컬럼 {list(mp.columns)[:6]}\n')

print('=' * 76)
print('1) 트랙맨 구성 — 시즌 x 리그(팀코드 접두어)')
tm['lg'] = tm['pitcher_team'].astype(str).str.split('_').str[0]
piv = tm.pivot_table(index='lg', columns='season', values='pitch_no', aggfunc='size').fillna(0)
piv = piv.astype(int).sort_values(piv.columns[-1], ascending=False)
print(piv.head(8).to_string())
print(f'\n  전체 대비 비중: ' + '  '.join(
    f'{k}={v/len(tm):.1%}' for k, v in tm['lg'].value_counts().head(5).items()))

print('\n' + '=' * 76)
print('2) 측정 컬럼의 결측 / 분포')
meas = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
        'extension', 'rel_height', 'rel_side', 'zone_speed']
print(f'{"컬럼":<22}{"결측":>8}{"평균":>10}{"표준편차":>10}{"1%":>9}{"99%":>9}')
for c in meas:
    v = tm[c]
    print(f'{c:<22}{v.isna().mean():>8.1%}{v.mean():>10.2f}{v.std():>10.2f}'
          f'{v.quantile(.01):>9.2f}{v.quantile(.99):>9.2f}')

print('\n' + '=' * 76)
print('3) ★ 트랙맨이 투수의 제구 능력을 설명하는가')
print('   (투수-시즌 단위: 트랙맨 집계 -> 그 투수-시즌의 실제 제구 성공률)')

# 매핑
col = [c for c in mp.columns if 'trackman' in c.lower()][0]
pcol = [c for c in mp.columns if c.lower().startswith('pitcher_id')][0]
m = mp[[pcol, col] + (['season'] if 'season' in mp.columns else [])].dropna()
print(f'   매핑 사용: {pcol} <-> {col}' + (' (+season)' if 'season' in m.columns else ''))

agg = {c: ['mean', 'std'] for c in meas}
agg['pitch_no'] = 'size'
g = tm.groupby(['pitcher_trackman_id', 'season']).agg(agg)
g.columns = ['_'.join(x) for x in g.columns]
g = g.reset_index().rename(columns={'pitch_no_size': 'tm_n'})

if 'season' in m.columns:
    g = g.merge(m, left_on=['pitcher_trackman_id', 'season'],
                right_on=[col, 'season'], how='inner')
else:
    g = g.merge(m, left_on='pitcher_trackman_id', right_on=col, how='inner')

lab = tr.groupby(['pitcher_id', 'season'])['control_success'].agg(['mean', 'size'])
lab = lab.reset_index().rename(columns={'mean': 'rate', 'size': 'n_train'})
d = g.merge(lab, left_on=[pcol, 'season'], right_on=['pitcher_id', 'season'], how='inner')
d = d[(d['n_train'] >= 300) & (d['tm_n'] >= 200)]
print(f'   결합 후 투수-시즌 {len(d):,}개 (train 300구+ & 트랙맨 200구+)')

if len(d) < 60:
    print('   표본이 너무 적어 판정 불가')
else:
    lgm = tr.groupby('season')['control_success'].mean()
    d['dev'] = d['rate'] - d['season'].map(lgm)      # 시즌 디트렌드
    feats = [c for c in d.columns if any(c.startswith(x + '_') for x in meas)] + ['tm_n']
    feats = [c for c in feats if d[c].notna().mean() > 0.8]
    print(f'\n   개별 상관 (시즌 디트렌드한 제구 편차와)')
    cors = []
    for c in feats:
        ok = d[c].notna()
        if ok.sum() < 50:
            continue
        r = float(np.corrcoef(d.loc[ok, c], d.loc[ok, 'dev'])[0, 1])
        cors.append((abs(r), c, r))
    cors.sort(reverse=True)
    for a, c, r in cors[:10]:
        print(f'     {c:<30}{r:>+8.3f}')

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    X = d[feats].fillna(d[feats].median()).to_numpy()
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    yv = d['dev'].to_numpy()
    r2 = cross_val_score(Ridge(alpha=10.0), X, yv, cv=5, scoring='r2').mean()
    print(f'\n   ★ 트랙맨 {len(feats)}개로 제구 편차를 예측: 교차검증 R2 = {r2:+.4f}')
    print(f'     (제구 편차의 표준편차 {yv.std():.4f})')
    # 비교: 그 투수의 직전 시즌 제구율은 얼마나 설명하나
    prv = lab.copy()
    prv['season'] = prv['season'] + 1
    prv = prv.rename(columns={'rate': 'prev_rate'})[['pitcher_id', 'season', 'prev_rate']]
    d2 = d.merge(prv, on=['pitcher_id', 'season'], how='inner')
    if len(d2) > 50:
        pv = d2['prev_rate'] - d2['season'].map(lgm).add(0)
        pv = (d2['prev_rate'].to_numpy()
              - d2['season'].sub(1).map(lgm).to_numpy())
        r2p = cross_val_score(Ridge(alpha=1.0), pv.reshape(-1, 1),
                              d2['dev'].to_numpy(), cv=5, scoring='r2').mean()
        print(f'     비교) 그 투수의 **직전 시즌 제구 편차** 하나로: R2 = {r2p:+.4f} '
              f'({len(d2):,}개)')
    print("""
   읽는 법: R2 가 0 근처면 트랙맨은 이 타겟에 대해 정보가 없다 -> 축 완전 종료.
     0.1 이상이면 우리가 못 뽑아낸 것이 있다는 뜻이다.""")

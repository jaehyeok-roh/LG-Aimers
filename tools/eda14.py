# EDA 14 — train 과 trackman 을 **투구 1:1** 로 정렬할 수 있는가.
#
# 주최측 Q&A (2026-08-17 참가자 질의 / 08-20 참가자 질의) 답변: **"모두 가능합니다."**
#   - train x trackman 매칭 부분집합에서 teacher(현재 투구의 구종·트랙맨 포함) 학습
#   - student 는 pre-pitch 만 입력받아 teacher 예측을 증류
#   - 제출은 student 만  -> LUPI / distillation 허용
#
# 그리고 한 참가자(08-14) 질문에 "경기·투구 단위로 정렬하면 상당수 투구에 대해
# 해당 투구의 Trackman 측정값을 확인할 수 있습니다" 라고 적혀 있다.
# **우리 매핑은 투수 단위다. 투구 1:1 정렬은 한 번도 안 해봤다.**
#
# 왜 큰가: 라벨 잡음이 엄청나다 (동일 조건 54% 가 결과가 갈림). 증류는 정확히 그
# 잡음을 줄인다. teacher 가 그 투구의 실제 물리량을 보면 훨씬 잘 맞히고, student 는
# 그 **확률**을 배운다 (0/1 라벨보다 정보가 많다).
#
# 여기서는 딱 두 가지만 본다:
#   1) 1:1 정렬이 실제로 되는가, 커버리지는 얼마인가
#   2) 정렬된 부분집합에서 **그 투구의 트랙맨 측정값이 control_success 를 얼마나 맞히는가**
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
tm = pd.read_csv('data/trackman_history.csv', keep_default_na=False, na_values=_NA)
tr = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
mp = pd.read_csv('data/pitcher_id_mapping_v2.csv', keep_default_na=False, na_values=_NA)
print(f'트랙맨 {len(tm):,} | train {len(tr):,}\n')

# ---- 정렬 키 만들기 ----
tm['game_date'] = pd.to_datetime(tm['game_date'], errors='coerce')
tm['game_month'] = tm['game_date'].dt.month
tm['game_dayofweek'] = tm['game_date'].dt.dayofweek
tm['tb'] = tm['top_bottom'].astype(str).str[0].str.upper()
tr['tb'] = tr['top_bottom'].astype(str).str[0].str.upper()
print('top_bottom 값:  train', sorted(tr['tb'].unique()), ' / 트랙맨', sorted(tm['tb'].unique()))
print('game_dayofweek 범위: train', tr['game_dayofweek'].min(), '~', tr['game_dayofweek'].max(),
      ' / 트랙맨(월=0)', tm['game_dayofweek'].min(), '~', tm['game_dayofweek'].max())

# 투수 매핑 붙이기
tm = tm.merge(mp[['pitcher_id', 'pitcher_trackman_id', 'season']].dropna(),
              on=['pitcher_trackman_id', 'season'], how='inner')
print(f'\n투수 매핑 후 트랙맨 {len(tm):,}행 ({len(tm)/1793078:.1%})')

KEY = ['season', 'game_month', 'pitcher_id', 'inning', 'tb',
       'balls_before', 'strikes_before', 'outs_before']
for k in ['pitch_of_pa']:
    if k in tm.columns and k in tr.columns:
        KEY.append(k)

print(f'\n정렬 키: {KEY}')
a = tr.groupby(KEY).size().rename('n_tr')
b = tm.groupby(KEY).size().rename('n_tm')
j = pd.concat([a, b], axis=1).dropna()
uniq = j[(j['n_tr'] == 1) & (j['n_tm'] == 1)]
print(f'  공통 키 조합 {len(j):,}개')
print(f'  양쪽 다 유일한 조합 {len(uniq):,}개  -> train 의 {len(uniq)/len(tr):.1%} 를 1:1 정렬 가능')

# 더 좁은 키로 (요일 추가)
KEY2 = KEY + ['game_dayofweek']
a2 = tr.groupby(KEY2).size().rename('n_tr')
b2 = tm.groupby(KEY2).size().rename('n_tm')
j2 = pd.concat([a2, b2], axis=1).dropna()
u2 = j2[(j2['n_tr'] == 1) & (j2['n_tm'] == 1)]
print(f'  + 요일까지: 유일 조합 {len(u2):,}개  -> train 의 {len(u2)/len(tr):.1%}')

BEST = KEY2 if len(u2) > len(uniq) else KEY
ub = u2 if len(u2) > len(uniq) else uniq
print(f'\n채택 키: {BEST}')

# ---- 정렬해서 라벨과 물리량을 붙인다 ----
idx = ub.index
trm = tr.set_index(BEST).loc[idx, ['control_success']].reset_index()
meas = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
        'extension', 'rel_height', 'rel_side', 'zone_speed']
tmm = tm.set_index(BEST).loc[idx, meas + ['pitch_type_group']].reset_index()
d = trm.join(tmm[meas + ['pitch_type_group']])
d = d.dropna(subset=['control_success'])
print(f'\n정렬된 투구 {len(d):,}개 ({len(d)/len(tr):.1%} of train)')
print(f'  성공률 {d["control_success"].mean():.4f} (전체 train {tr["control_success"].mean():.4f})')

if len(d) < 20000:
    print('\n표본이 적어 teacher 검증은 생략')
else:
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = d[meas].to_numpy(dtype='float64')
    pt = pd.get_dummies(d['pitch_type_group'].astype(str), prefix='pt').to_numpy(dtype=float)
    X = np.hstack([X, pt])
    y = d['control_success'].to_numpy(dtype='float64')
    r = y.mean()
    U = r * (1 - r)

    def bss(est, X, y):
        from sklearn.model_selection import cross_val_predict
        p = cross_val_predict(est, X, y, cv=KFold(4, shuffle=True, random_state=0),
                              method='predict_proba')[:, 1]
        return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / U) * 100000

    est = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    print(f'\n★ 그 투구의 트랙맨 측정값 {X.shape[1]}개만으로 control_success 예측')
    print(f'   교차검증 스킬 = {bss(est, X, y):,.0f}')
    print(f'   (참고: 우리 모델의 실전 스킬 ~1,000, OOF ~2,200)')
    print("""
   읽는 법: 이 값이 2,000 을 크게 넘으면 teacher 가 student 보다 훨씬 잘 안다는 뜻이고
     증류가 의미 있다. 1,000 근처면 그 투구의 물리량으로도 제구 성공을 잘 못 맞히는
     것이고 (위치 정보가 없으니 당연할 수 있다) 증류의 여지가 작다.""")

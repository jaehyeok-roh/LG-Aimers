# EDA 20 — 그 투구의 트랙맨 측정값이 **우리 96피처 위에** 얼마를 더하는가.
# eda19 는 사전정보를 13개로만 근사했다 (614). 우리 실제 모델은 그보다 훨씬 강하다.
# 강한 모델 위에서의 증분이어야 실제 기대와 맞는다.
import numpy as np, pandas as pd, pickle, gc
pd.set_option('display.width', 220)

rid = np.load('cache/row_id.npy', allow_pickle=True)
y_all = np.load('cache/y.npy'); se_all = np.load('cache/season.npy')
al = pd.read_parquet('cache/raw/aligned.parquet',
        columns=['row_id','season','control_success','pitcher_id','pitch_type_group',
                 'rel_speed','spin_rate','induced_vert_break','horz_break',
                 'extension','rel_height','rel_side','zone_speed'])
al24 = al[al.season == 2024].reset_index(drop=True)
print(f'2024 정렬 {len(al24):,}')

pos = pd.Series(np.arange(len(rid)), index=pd.Index(rid))
idx = pos.reindex(al24['row_id']).to_numpy()
assert not np.isnan(idx).any()
idx = idx.astype(int)
assert (y_all[idx] == al24['control_success'].to_numpy()).all(), 'y 불일치'
assert (se_all[idx] == 2024).all()
print('row_id 정렬 확인 OK')

X = pd.read_pickle('cache/X.pkl')
Xs = X.iloc[idx].reset_index(drop=True)
del X; gc.collect()
print(f'우리 피처 {Xs.shape}')

meas = ['rel_speed','spin_rate','induced_vert_break','horz_break','extension',
        'rel_height','rel_side','zone_speed']
cat = [c for c in Xs.columns if str(Xs[c].dtype) == 'category']
y = al24['control_success'].to_numpy(float)

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
r = y.mean(); U = r*(1-r)

def run(Xd, note):
    p = np.zeros(len(y))
    for trn, val in StratifiedKFold(4, shuffle=True, random_state=0).split(Xd, y):
        m = CatBoostClassifier(iterations=600, depth=8, learning_rate=0.05,
                               l2_leaf_reg=8.5, verbose=0, thread_count=14,
                               cat_features=[c for c in cat if c in Xd.columns])
        m.fit(Xd.iloc[trn], y[trn])
        p[val] = m.predict_proba(Xd.iloc[val])[:, 1]
    s = (1 - ((np.clip(p,1e-6,1-1e-6)-y)**2).mean()/U) * 100000
    print(f'  {note:46s} 스킬 {s:8,.0f}')
    return s, p

M = al24[meas].astype(float).reset_index(drop=True)
PT = al24['pitch_type_group'].astype(str).astype('category').rename('pt').reset_index(drop=True)

s0, p0 = run(Xs, 'A. 우리 96피처 (student 가 쓸 수 있는 것)')
s1, _  = run(pd.concat([Xs, M], axis=1), 'B. + 그 투구 측정 8개')
cat.append('pt')
s2, p2 = run(pd.concat([Xs, M, PT], axis=1), 'C. + 측정 8개 + 구종군  (teacher)')
s3, _  = run(pd.concat([Xs, PT], axis=1), 'D. + 구종군만')
print(f'\n증분:  측정 {s1-s0:+,.0f}   측정+구종 {s2-s0:+,.0f}   구종만 {s3-s0:+,.0f}')
print(f'teacher/student 예측 상관 {np.corrcoef(p0, p2)[0,1]:.4f}')

print('\n' + '=' * 78)
print('측정값의 within-pitcher 변동만으로도 신호가 있는가 (투수 평균을 뺀 값)')
print('=' * 78)
Mw = M - M.groupby(al24['pitcher_id'].to_numpy()).transform('mean')
s4, _ = run(pd.concat([Xs, Mw.add_suffix('_w')], axis=1), 'E. + 투수평균 제거한 측정 8개')
print(f'  증분 {s4-s0:+,.0f}   (양수면 투수 정체성으로 설명 안 되는 새 정보)')

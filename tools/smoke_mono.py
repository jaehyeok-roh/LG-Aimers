# mono 후보 연막시험. 캐글에 올리기 전에 CatBoost 가 실제로 받아주는지만 본다.
# 로컬은 본 학습이 16코어를 쓰고 있으므로 thread_count 를 2 로 묶는다.
import json, sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, 'tools')
import screen

X = pd.read_pickle('cache/X.pkl').head(4000)
y = np.load('cache/y.npy')[:4000]
meta = json.load(open('cache/meta.json', encoding='utf-8'))
cat = meta['cat_features']
for c in cat:
    X[c] = X[c].astype(str).astype('category')

for name, agg in (('mono', False), ('mono2', True)):
    mm = screen._mono_map(list(X.columns), agg)
    assert all(c in X.columns for c in mm), '없는 컬럼에 제약을 걸었다'
    assert not (set(mm) & set(cat)), '범주형에 단조 제약은 불가'
    p = dict(meta['best_params'])
    p.update(iterations=20, cat_features=cat, task_type='CPU', thread_count=2,
             verbose=0, monotone_constraints=mm, boosting_type='Plain')
    p.pop('bagging_temperature', None)
    try:
        m = CatBoostClassifier(**p).fit(X, y)
        pr = m.predict_proba(X)[:, 1]
        print(f'{name:<6} OK  제약 {len(mm):>2}개  '
              f'(+{sum(v>0 for v in mm.values())} / -{sum(v<0 for v in mm.values())})  '
              f'예측 {pr.min():.3f}~{pr.max():.3f}')
    except Exception as e:
        print(f'{name:<6} 실패: {type(e).__name__}: {str(e)[:300]}')

# 제약이 실제로 걸렸는지 확인: 제약 피처를 단조 증가시키며 예측이 안 내려가는지 본다
mm = screen._mono_map(list(X.columns), False)
c = 'asof_pitcher_success_rate'
if c in mm:
    g = X.head(1).copy()
    rows = pd.concat([g] * 40, ignore_index=True)
    for cc in cat:
        rows[cc] = rows[cc].astype(str).astype('category')
    rows[c] = np.linspace(float(X[c].min()), float(X[c].max()), 40)
    pr = m.predict_proba(rows)[:, 1]
    d = np.diff(pr)
    print(f'\n{c} 를 최소->최대로 훑을 때 예측 변화: '
          f'하락 구간 {int((d < -1e-9).sum())}개 / 39  (0 이어야 제약이 걸린 것)')

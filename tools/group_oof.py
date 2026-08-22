# 'in-distribution 2.21%' 가 진짜인지 검증한다.
#
# 우리는 열흘 동안 "학습 분포 2.21% vs 미래 시즌 0.80% = 64% 를 전이에서 잃는다" 를
# 전제로 움직였고, 세 자릿수가 남은 유일한 곳이라고 적어뒀다. 그런데 EDA 결과
# **주요 피처의 관계 모양이 6시즌 동안 전후상관 0.97~0.98 로 거의 안 변한다.**
# 관계가 안 변하는데 64% 를 잃는다는 것은 앞뒤가 안 맞는다.
#
# 의심: 2.21% 는 **무작위 K-fold OOF** 다. 투수 한 명이 시즌당 ~1,600구를 던지고
# 10-fold 면 그중 ~1,440구가 학습에 들어간다. 즉 검증행마다 모델은 **같은 투수의
# 같은 시즌**을 이미 1,440번 봤다. 2025 test 에는 그런 것이 없다.
#
# 그래서 세 가지를 같은 조건(같은 학습량, 같은 파라미터)에서 잰다:
#   A. 무작위 fold        — 지금 쓰는 OOF. 같은 투수·시즌이 양쪽에 있다
#   B. 투수x시즌 그룹 fold — 그 투수의 그 시즌을 통째로 뺀다
#   C. 투수 그룹 fold      — 그 투수를 커리어 통째로 뺀다
#
# A >> B 면 2.21% 는 누수로 부푼 숫자이고, '전이 손실 64%' 의 상당 부분은 허수다.
# 그러면 남은 여지도 훨씬 작다는 뜻이라 로드맵을 다시 써야 한다.
import json, os, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold

CACHE = os.environ.get('SCREEN_CACHE', 'cache')
ITERS = int(os.environ.get('GO_ITERS', '1000'))
FOLDS = 3

X = pd.read_pickle(f'{CACHE}/X.pkl')
for c in X.columns:
    if X[c].dtype == np.float64:
        X[c] = X[c].astype(np.float32)
y = np.load(f'{CACHE}/y.npy')
season = np.load(f'{CACHE}/season.npy')
rid = np.load(f'{CACHE}/row_id.npy', allow_pickle=True)
meta = json.load(open(f'{CACHE}/meta.json', encoding='utf-8'))
cat = meta['cat_features']

# pitcher_id 는 피처에서 빠져 있으므로 train.csv 에서 row_id 로 붙인다
_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
import glob
_c = ([f for f in ('data/train.csv',) if os.path.exists(f)]
      + glob.glob('/kaggle/input/**/train.csv', recursive=True))
if not _c:
    raise SystemExit('train.csv 를 못 찾음')
print('train.csv:', _c[0], flush=True)
tr = pd.read_csv(_c[0], usecols=['row_id', 'pitcher_id'],
                 keep_default_na=False, na_values=_NA)
pid = pd.Series(tr.set_index('row_id')['pitcher_id']).reindex(pd.Index(rid)).to_numpy()
assert not pd.isna(pid).any(), 'pitcher_id 매칭 실패'
print(f'{len(X):,}행 | 투수 {len(np.unique(pid)):,}명 | 반복 {ITERS} | fold {FOLDS}', flush=True)

p = dict(meta['best_params'])
p.update(iterations=ITERS, cat_features=cat, task_type='CPU',
         thread_count=-1, verbose=0, random_seed=42)
p.pop('bagging_temperature', None)
p.pop('early_stopping_rounds', None)


def skill(pr, yv):
    r = float(yv.mean())
    return (1 - ((np.clip(pr, 1e-6, 1 - 1e-6) - yv) ** 2).mean() / (r * (1 - r))) * 100000


ps = pd.Series(pid).astype(str) + '_' + pd.Series(season).astype(str)
schemes = {
    'A. 무작위 fold': list(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(X, y)),
    'B. 투수x시즌 그룹': list(GroupKFold(FOLDS).split(X, y, groups=ps.to_numpy())),
    'C. 투수 그룹': list(GroupKFold(FOLDS).split(X, y, groups=pid)),
}

t0 = time.time()
res = {}
for name, splits in schemes.items():
    oof = np.zeros(len(y))
    for k, (ti, vi) in enumerate(splits):
        m = CatBoostClassifier(**p).fit(X.iloc[ti], y[ti])
        oof[vi] = m.predict_proba(X.iloc[vi])[:, 1]
        print(f'  [{name}] fold {k+1}/{FOLDS}  ({(time.time()-t0)/60:.0f}분)', flush=True)
    res[name] = skill(oof, y)
    # 최신 시즌만 따로 — 2024 홀드아웃(0.79%)과 가장 비교 가능한 조각
    m24 = season == 2024
    res[name + ' (2024행만)'] = skill(oof[m24], y[m24])
    print(f'  [{name}] 전체 {res[name]:,.0f} / 2024행 {res[name+" (2024행만)"]:,.0f}\n', flush=True)

print('=' * 72)
print(f'{"검증 방식":<26}{"스킬":>10}{"A 대비":>10}')
a = res['A. 무작위 fold']
for k, v in res.items():
    print(f'{k:<26}{v:>10,.0f}{v - a:>+10,.0f}')
print('=' * 72)
print("""
읽는 법: 우리 2024 시즌 홀드아웃(~2023 학습 -> 2024 예측)이 약 790 이다.
  B 나 C 가 790 근처면 -> 2.21% 는 누수로 부푼 값이고 '전이 손실' 은 거의 없다.
                          남은 여지도 작다는 뜻이므로 로드맵을 다시 써야 한다.
  B, C 가 여전히 1,500+ 면 -> 전이 손실은 실재한다. 기존 진단 유지.
""")
json.dump({k: round(float(v), 1) for k, v in res.items()},
          open('cache/group_oof.json', 'w'), indent=2, ensure_ascii=False)
print(f'저장: cache/group_oof.json   총 {(time.time()-t0)/60:.0f}분')

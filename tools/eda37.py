# EDA 37 — 새 피처가 들어간 뒤 보정(calibration)이 나빠졌는가.
#
# 2026-08-21 에 보정 축은 닫혔다:
#   Murphy 분해로 천장 51점, 실행 가능한 접근은 -48 ~ -566.
#   원인은 구조적이다 — 과대예측 폭 b 가 시즌마다 달라(2023 +0.0192 / 2024 +0.0072)
#   한 시즌에서 잰 보정을 다음 시즌에 옮기면 과보정한다.
#
# 그런데 **오늘 CPU 이득이 +7.88 -> -1.8 로 뒤집혔다** (피처가 좋아지자 추정효율
# 기법이 무의미해짐). 그래서 옛 측정을 그냥 인용하지 않고 다시 잰다.
#
# 여기서 찾는 것은 '기회' 가 아니라 **버그**다:
#   wseason5+wsbat 이 들어가면서 예측 분포가 바뀌었는데, fold 별 isotonic 이
#   여전히 잘 옮겨가는가? 신뢰도(reliability)가 커졌다면 그건 고칠 손상이다.
import json
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, 'tools')
import importlib.util
spec = importlib.util.spec_from_file_location('sc', 'tools/screen.py')
sc = importlib.util.module_from_spec(spec)
sys.modules['sc'] = sc
spec.loader.exec_module(sc)

ITERS = 1000
CACHE = 'cache'
X = pd.read_pickle(f'{CACHE}/X.pkl')
y = np.load(f'{CACHE}/y.npy')
season = np.load(f'{CACHE}/season.npy')
meta = json.load(open(f'{CACHE}/meta.json', encoding='utf-8'))
for c in X.columns:
    if X[c].dtype == np.float64:
        X[c] = X[c].astype(np.float32)

mh, mv = season <= 2023, season == 2024
params = dict(meta['best_params'])
params.update(iterations=ITERS, cat_features=meta['cat_features'], task_type='CPU',
              thread_count=-1, verbose=0, allow_writing_files=False)

W = sc._wseason5_cols()
B = sc._wsbat_cols()
Xh, Xv = X[mh].reset_index(drop=True), X[mv].reset_index(drop=True)
for D in (W, B):
    for k, v in D.items():
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
yh, yv = y[mh], y[mv]
print(f'\n학습 {len(Xh):,} -> 검증 {len(Xv):,} | 피처 {Xh.shape[1]}\n', flush=True)

raw = np.zeros(len(Xv))
cal = np.zeros(len(Xv))
for tr_i, va_i in StratifiedKFold(3, shuffle=True, random_state=42).split(Xh, yh):
    m = CatBoostClassifier(**params)
    m.fit(Xh.iloc[tr_i], yh[tr_i])
    p_va = m.predict_proba(Xh.iloc[va_i])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(p_va, yh[va_i])
    p_te = m.predict_proba(Xv)[:, 1]
    raw += p_te / 3
    cal += iso.transform(p_te) / 3

r = float(yv.mean())
U = r * (1 - r)


def skill(p):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yv) ** 2).mean() / U) * 100000


def murphy(p, nb=50):
    """Brier = 신뢰도 - 분해능 + 불확실성 (분위 구간)."""
    q = pd.qcut(pd.Series(p).rank(method='first'), nb, labels=False)
    d = pd.DataFrame({'p': p, 'y': yv, 'q': q}).groupby('q').agg(
        pm=('p', 'mean'), ym=('y', 'mean'), n=('y', 'size'))
    rel = float((d['n'] * (d['pm'] - d['ym']) ** 2).sum() / len(p))
    res = float((d['n'] * (d['ym'] - r) ** 2).sum() / len(p))
    return rel, res


print(f'{"판":<30}{"스킬":>9}{"평균예측":>10}{"신뢰도pt":>10}{"분해능pt":>10}')
print('-' * 70)
for lab, p in [('raw (isotonic 없음)', raw), ('fold별 isotonic (현행)', cal),
               ('현행 + 재중심화', sc.recenter(cal, r))]:
    rel, res = murphy(p)
    print(f'{lab:<30}{skill(p):>9,.0f}{p.mean():>10.4f}'
          f'{rel/U*100000:>10,.0f}{res/U*100000:>10,.0f}')

# 오라클 상한 (2024 정답을 봤을 때. 실제로는 못 쓴다)
base = sc.recenter(cal, r)
orc = IsotonicRegression(out_of_bounds='clip').fit_transform(base, yv)
print(f'{"[오라클] 완전 isotonic":<30}{skill(orc):>9,.0f}{orc.mean():>10.4f}')

print('\n' + '=' * 70)
rel, res = murphy(base)
print(f'현행의 신뢰도 손실 = **{rel/U*100000:,.0f}점**  (완벽 보정 시 회수 가능한 최대치)')
print(f'2026-08-21 측정치  = 51점')
print("""
읽는 법:
  신뢰도 손실이 51 근처거나 더 작다 -> 보정은 여전히 닫힌 축이다. 손대지 말 것.
  훨씬 커졌다 -> 새 피처가 보정을 망가뜨린 것이고, 그건 기회가 아니라 **버그**다.
⚠️ 천장이 커져도 '실행 가능한 접근이 전부 음수' 라는 결론은 별개다
   (b 가 시즌마다 달라 한 시즌에서 잰 보정이 안 옮겨간다).""")

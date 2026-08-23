# EDA 30 — 방금 +61.72 를 준 '당해 시즌 폼' 축에 얼마나 남았는가.
#
# group_oof 가 이렇게 말했다 (2024 행 기준):
#   A. 무작위 fold (그 투수의 당해 시즌을 봄)   1092
#   B. 투수x시즌 그룹 (못 봄)                    791
#   차이 301 = '당해 시즌 폼을 아는 것' 의 값어치
#
# wseason5 는 그 301 중 얼마를 가져왔나? 남은 게 크면 이 축을 더 파야 하고,
# 다 가져왔으면 다른 데를 봐야 한다. 목표가 이분법(컷 아니면 0)이므로
# **어느 광맥에 남았는지**를 아는 것이 무엇보다 값어치 있다.
#
# 오라클 = 그 투수의 2024 전체 성공률 (자기 행 제외). 추론엔 못 쓰지만 **천장**이다.
import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, 'tools')
import importlib.util
_sp = next(q for q in ('tools/screen.py', 'screen_lib.py') if os.path.exists(q))
spec = importlib.util.spec_from_file_location('sc', _sp)
sc = importlib.util.module_from_spec(spec)
sys.modules['sc'] = sc
spec.loader.exec_module(sc)

_a = [x for x in sys.argv[1:] if x.isdigit()]
ITERS = int(_a[0]) if _a else 1000
CACHE = os.environ.get('SCREEN_CACHE', 'cache')
X = pd.read_pickle(f'{CACHE}/X.pkl')
y = np.load(f'{CACHE}/y.npy')
season = np.load(f'{CACHE}/season.npy')
rid = np.load(f'{CACHE}/row_id.npy', allow_pickle=True)
meta = __import__('json').load(open(f'{CACHE}/meta.json', encoding='utf-8'))
for c in X.columns:
    if X[c].dtype == np.float64:
        X[c] = X[c].astype(np.float32)

mh, mv = season <= 2023, season == 2024
cat = meta['cat_features']
params = dict(meta['best_params'])
params.update(iterations=ITERS, cat_features=cat, task_type='CPU',
              thread_count=-1, verbose=0, allow_writing_files=False)

# ---- 오라클: 그 투수의 당해 시즌 성공률
# ⚠️ leave-one-out 으로 만들면 **역산 누수**가 난다 (eda29 1차 실행에서 -28,492).
#    loo = (S - y)/(N-1) 이므로 모델이 투수별 S, N 을 외우면 y 를 정확히 복원한다.
#    학습 시즌에서 그 역산을 배우고, 검증 시즌에서는 S/N 이 달라 확신에 찬 오답을 낸다.
#    그래서 **반쪽 분할**을 쓴다 — 각 투수-시즌을 무작위로 반 갈라 반대쪽 평균을 준다.
#    역산이 불가능하고(자기 행이 안 들어감) 표본은 절반이라 천장을 약간 보수적으로 잡는다.
tr = sc._read_tr(['season', 'pitcher_id'])
g = pd.DataFrame({'p': tr['pitcher_id'], 's': tr['season'], 'y': y})
half = np.random.default_rng(0).integers(0, 2, len(g))
g['h'] = half
agg2 = g.groupby(['p', 's', 'h'])['y'].agg(['sum', 'size'])
other = pd.MultiIndex.from_arrays([g['p'], g['s'], 1 - g['h']])
S = agg2['sum'].reindex(other).to_numpy()
N = agg2['size'].reindex(other).to_numpy()
loo = S / np.maximum(N, 1)                    # 반대쪽 절반의 평균 (자기 행 없음)
loo = np.where(N > 0, loo, np.nan)
lg = pd.Series(y).groupby(g['s'].to_numpy()).transform('mean').to_numpy()
ORACLE = {'orc_rate': loo - lg, 'orc_n': N.astype('float64')}
print(f'오라클: 투수-시즌 {agg2.index.droplevel(2).nunique():,}개 | 반쪽 표본 중앙 {np.nanmedian(N):.0f} | '
      f'편차 std {np.nanstd(ORACLE["orc_rate"]):.4f}\n', flush=True)

W = sc._wseason5_cols()
print(flush=True)


def run(extra, tag):
    Xh, Xv = X[mh].reset_index(drop=True), X[mv].reset_index(drop=True)
    for D in extra:
        for k, v in D.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    yh, yv = y[mh], y[mv]
    p = np.zeros(len(Xv))
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    for tr_i, va_i in skf.split(Xh, yh):
        m = CatBoostClassifier(**params)
        m.fit(Xh.iloc[tr_i], yh[tr_i])
        iso = IsotonicRegression(out_of_bounds='clip').fit(
            m.predict_proba(Xh.iloc[va_i])[:, 1], yh[va_i])
        p += iso.transform(m.predict_proba(Xv)[:, 1]) / 3
    p = sc.recenter(p, float(yv.mean()))
    s = sc.skill(p, yv)
    print(f'{tag:<40}{s:>9,.0f}', flush=True)
    return s


print(f'{"구성":<40}{"스킬":>9}')
print('-' * 50, flush=True)
b = run([], 'A. base (96피처)')
w = run([W], 'B. + wseason5 (당해시즌 복원)')
o = run([ORACLE], 'C. + 오라클 당해시즌 폼 (추론 불가)')
wo = run([W, ORACLE], 'D. + 둘 다')

print('-' * 50)
print(f'{"wseason5 가 가져온 몫":<40}{w-b:>+9,.0f}')
print(f'{"오라클 천장":<40}{o-b:>+9,.0f}')
print(f'{"★ wseason5 위에 오라클이 더 얹는 것":<40}{wo-w:>+9,.0f}')
if o - b > 0:
    print(f'{"회수율":<40}{(w-b)/(o-b):>9.0%}')
print("""
읽는 법:
  D-B 가 크다  -> 당해 시즌 축에 아직 많이 남았다. 더 정교한 복원을 파야 한다.
  D-B 가 작다  -> wseason5 가 이 광맥을 거의 다 캤다. 다른 데를 봐야 한다.""")

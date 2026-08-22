# 피처별 **전이 안정성** 감사.
#
# 진단: 모델은 멀쩡하다 (in-sample 스킬 2.232% ~= OOF 2.21% -> 안 외운다,
# 단조 제약 +0 -> 이동 구간에서 이상한 모양을 안 만든다, 규제 전부 음수).
# 그렇다면 미래 시즌에서 잃는 64% 는 **피처와 결과의 관계가 시즌마다 바뀌기** 때문이다.
#
# 그러면 물어볼 것은 하나다: **어느 피처의 관계가 바뀌는가.**
# 월/요일 제거(+4.52, 리더보드)가 정확히 이 부류였고 **우연히** 찾았다. 체계화한다.
#
# 방법: ~2022 로 학습하고 두 곳에서 순열 중요도를 잰다.
#   (a) 2023 홀드아웃  = 학습 직후 시즌  -> "가까운 미래" 기여도
#   (b) 2024 홀드아웃  = 두 시즌 뒤      -> "먼 미래" 기여도
# 둘 다 학습에 안 들어갔으므로 순수하게 **시간 거리**만 다르다.
# (a) 는 크고 (b) 는 작은 피처가 전이 불안정 피처다.
#
# ⚠️ 왜 in-sample 을 (a) 로 쓰지 않는가: 학습에 쓴 행의 중요도는 암기 성분이 섞여
#    무엇을 재는지 모호해진다. 두 미래 시즌을 비교하면 축이 '시간 거리' 하나로 깨끗하다.
#
# 사용: python tools/transfer_audit.py [--iters=1000] [--rep=2]
import json, os, sys, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

CACHE = os.environ.get('SCREEN_CACHE', 'cache')
OUT = os.environ.get('AUDIT_OUT', 'cache/transfer_audit.csv')
ITERS, REP, SEED = 1000, 2, 42
for a in sys.argv[1:]:
    if a.startswith('--iters='):
        ITERS = int(a.split('=')[1])
    elif a.startswith('--rep='):
        REP = int(a.split('=')[1])


def skill(p, yv, r):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yv) ** 2).mean() / (r * (1 - r))) * 100000


def recenter(p, target):
    """평균 예측을 target 에 맞추는 로짓 시프트. 순열이 평균을 흔드는 것을 상쇄한다."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(q / (1 - q))
    off = 0.0
    for _ in range(200):
        cur = 1.0 / (1.0 + np.exp(-(lo + off)))
        e = cur.mean() - target
        if abs(e) < 1e-9:
            break
        off -= e * 4.0
    return 1.0 / (1.0 + np.exp(-(lo + off)))


X = pd.read_pickle(f'{CACHE}/X.pkl')
for c in X.columns:
    if X[c].dtype == np.float64:
        X[c] = X[c].astype(np.float32)
y = np.load(f'{CACHE}/y.npy')
season = np.load(f'{CACHE}/season.npy')
meta = json.load(open(f'{CACHE}/meta.json', encoding='utf-8'))
cat = meta['cat_features']

mt, m23, m24 = season <= 2022, season == 2023, season == 2024
Xt, yt = X[mt].reset_index(drop=True), y[mt]
sets = {'2023': (X[m23].reset_index(drop=True), y[m23]),
        '2024': (X[m24].reset_index(drop=True), y[m24])}
print(f'학습 {mt.sum():,}행(~2022) | 평가 2023 {m23.sum():,} / 2024 {m24.sum():,}', flush=True)
print(f'피처 {X.shape[1]}개 (범주형 {len(cat)}) | 반복 {ITERS} | 순열 {REP}회\n', flush=True)

p = dict(meta['best_params'])
p.update(iterations=ITERS, cat_features=cat, task_type='CPU',
         thread_count=-1, verbose=0, random_seed=SEED)
p.pop('bagging_temperature', None)
p.pop('early_stopping_rounds', None)

t0 = time.time()
m = CatBoostClassifier(**p).fit(Xt, yt)
print(f'학습 완료 {(time.time()-t0)/60:.0f}분\n', flush=True)

base = {}
for k, (Xv, yv) in sets.items():
    r = float(yv.mean())
    s = skill(recenter(m.predict_proba(Xv)[:, 1], r), yv, r)
    base[k] = s
    print(f'  기준선 {k}: {s:,.0f}  (리그평균 {r:.4f})', flush=True)
print(flush=True)

rng = np.random.default_rng(SEED)
rows = []
feats = list(X.columns)
for i, f in enumerate(feats):
    rec = {'feature': f, 'is_cat': f in cat}
    for k, (Xv, yv) in sets.items():
        r = float(yv.mean())
        orig = Xv[f].to_numpy(copy=True)
        drops = []
        for _ in range(REP):
            Xv[f] = orig[rng.permutation(len(orig))]
            drops.append(base[k] - skill(recenter(m.predict_proba(Xv)[:, 1], r), yv, r))
        Xv[f] = orig
        rec[f'imp{k}'] = float(np.mean(drops))
    # 전이 유지율: 먼 미래에 얼마나 남는가
    a, b = rec['imp2023'], rec['imp2024']
    rec['delta'] = b - a
    rec['keep'] = (b / a) if a > 1e-9 else np.nan
    rows.append(rec)
    if (i + 1) % 12 == 0 or i + 1 == len(feats):
        print(f'  {i+1}/{len(feats)}  ({(time.time()-t0)/60:.0f}분)', flush=True)

d = pd.DataFrame(rows).sort_values('imp2023', ascending=False)
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
d.to_csv(OUT, index=False)

print('\n' + '=' * 78)
print('기여도가 큰 피처 25개 — 2023(가까운 미래) vs 2024(먼 미래)')
print(f'{"피처":<40}{"2023":>9}{"2024":>9}{"차이":>9}{"유지율":>9}')
for _, x in d.head(25).iterrows():
    kp = '   —' if pd.isna(x['keep']) else f'{x["keep"]:8.2f}'
    print(f'{x["feature"][:38]:<40}{x["imp2023"]:>9.1f}{x["imp2024"]:>9.1f}'
          f'{x["delta"]:>+9.1f}{kp}')

print('\n' + '=' * 78)
print('★ 2024 에서 기여가 음수인 피처 = 먼 미래에 오히려 해가 된다 (제거 후보)')
bad = d[d['imp2024'] < 0].sort_values('imp2024')
if len(bad) == 0:
    print('  없음')
for _, x in bad.iterrows():
    print(f'{x["feature"][:38]:<40}{x["imp2023"]:>9.1f}{x["imp2024"]:>9.1f}{x["delta"]:>+9.1f}')

print('\n' + '=' * 78)
print('★ 기여도가 가장 많이 무너지는 피처 15개 (2023 에서 5 이상 기여한 것 중)')
big = d[d['imp2023'] > 5].sort_values('keep')
for _, x in big.head(15).iterrows():
    print(f'{x["feature"][:38]:<40}{x["imp2023"]:>9.1f}{x["imp2024"]:>9.1f}{x["keep"]:>9.2f}')
print(f'\n저장: {OUT}   총 {(time.time()-t0)/60:.0f}분')

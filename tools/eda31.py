# EDA 31 — 배포 모델이 `season` 을 실제로 어떻게 쓰는가.
#
# 배경: `season` 제거는 -424 로 즉사한다. 그래서 '필수' 로 적어뒀는데,
# **모델이 그걸로 무엇을 하는지는 한 번도 안 봤다.**
#
# 사실 하나가 이미 분명하다: `season` 은 cat_features 에 없고 int64 다.
# 트리는 `season <= X` 로만 쪼개므로 X <= 2024 인 모든 분기에서
# **2025 는 2024 와 같은 쪽으로 간다.** 즉 원리적으로 구분이 불가능하다.
#
# 여기서 실제 배포 모델(submit_cpu30 의 30개)로 확인한다:
#   1) season 을 2025 로 바꾸면 예측이 정말 안 변하는가 (= 외삽 불가 확정)
#   2) season 을 2019~2024 로 바꾸면 얼마나 변하는가 (= 시즌 학습의 크기)
#   3) 그 변화가 '수준 이동' 인가 '순위 재배열' 인가
#      -> 수준뿐이면 재중심화가 이미 다 하고 있다는 뜻이고,
#         재배열이 크면 '시즌 x 피처' 상호작용이 실재한다는 뜻이다.
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ZIP = sys.argv[1] if len(sys.argv) > 1 else 'out/submit_cpu30.zip'
NMOD = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NROW = int(sys.argv[3]) if len(sys.argv) > 3 else 60000

work = tempfile.mkdtemp(prefix='eda31_')
with zipfile.ZipFile(ZIP) as z:
    names = [n for n in z.namelist() if n.endswith('.cbm')][:NMOD]
    for n in names:
        z.extract(n, work)
print(f'모델 {len(names)}개 로드 ({os.path.basename(ZIP)})')

X = pd.read_pickle('cache/X.pkl')
y = np.load('cache/y.npy')
season = np.load('cache/season.npy')
m24 = season == 2024
idx = np.where(m24)[0][:NROW]
Xs = X.iloc[idx].reset_index(drop=True)
ys = y[idx]
print(f'2024 행 {len(Xs):,}개 | 실제 성공률 {ys.mean():.4f}\n')

CATS = [c for c in json.load(open('cache/meta.json', encoding='utf-8'))['cat_features']
        if c in Xs.columns]
models = []
for n in names:
    m = CatBoostClassifier()
    m.load_model(os.path.join(work, n))
    models.append(m)


def pred(s_val):
    d = Xs.copy()
    d['season'] = s_val
    pool = Pool(d, cat_features=CATS)
    p = np.zeros(len(d))
    for m in models:
        p += m.predict_proba(pool)[:, 1] / len(models)
    return p


base = pred(2024)
print(f'{"season":<10}{"평균예측":>10}{"2024 대비":>11}{"행별 상관":>11}{"행별 최대차":>13}')
print('-' * 55)
rows = {}
for s in [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2030]:
    p = pred(s)
    r = float(np.corrcoef(p, base)[0, 1])
    rows[s] = p
    print(f'{s:<10}{p.mean():>10.5f}{p.mean()-base.mean():>+11.5f}'
          f'{r:>11.6f}{np.abs(p-base).max():>13.2e}')

print('\n' + '=' * 55)
same = np.abs(rows[2025] - base).max()
if same < 1e-12:
    print(f'✅ season=2025 예측이 2024 와 **완전히 동일** (최대차 {same:.1e})')
    print('   트리는 2025 를 2024 로 취급한다. 외삽이 원리적으로 불가능하다.')
else:
    print(f'⚠️ season=2025 가 2024 와 다르다 (최대차 {same:.2e}) — 예상 밖. 조사할 것.')

# 3) 시즌 이동이 '수준' 인가 '재배열' 인가
print('\n시즌을 바꿨을 때 변화의 성질 (2024 -> 2019)')
p19 = rows[2019]
shift = p19.mean() - base.mean()
resid = (p19 - base) - shift
print(f'  평균 이동          {shift:+.5f}')
print(f'  이동을 뺀 잔차 std {resid.std():.5f}')
print(f'  잔차/이동 비율     {resid.std()/abs(shift):.2f}')
print("""
읽는 법:
  비율이 0 에 가까우면 season 은 **수준 이동만** 한다 -> 재중심화가 이미 다 하고 있고
    '시즌 x 피처' 를 손보는 후보(마스킹/재학습 등)는 기대값 0 이다.
  비율이 1 을 넘으면 시즌마다 **순위가 재배열**된다 -> 그 상호작용이 실재하고,
    2025 에 2024 의 재배열을 그대로 적용하는 것이 진짜 위험이다.""")

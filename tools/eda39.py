# EDA 39 — 어떤 피처의 '의미' 가 시즌을 건너면서 뒤집히는가.
#
# 중요도가 낮다는 것은 제거 근거가 못 된다 (제거 20전 1승, 유일한 성공이 월/요일 +4.52).
# 실제로 해로운 피처는 **많이 쓰이는데 의미가 변하는** 것이다 — game_type 이 그랬다
# (F 가 R 대비 +0.20 -> -0.03 으로 부호 반전, 순열 중요도 -554 로 96개 중 최악).
#
# 학습 없이 잰다: 피처를 구간으로 나누고, 각 구간의 '리그평균 대비 편차' 프로파일을
#   옛 시즌(2019~2021)  vs  최근 시즌(2023~2024)
# 두 번 만들어 **가중 상관**을 본다.
#   상관 ~1.0  = 의미가 그대로다 (전이 안전)
#   상관 낮음/음수 = 그 피처가 말하는 바가 달라졌다 (전이 위험)
#
# ⚠️ 대리지표다 (오늘까지 대리지표 3연패). **후보 생성용이지 판정용이 아니다.**
#    game_type 이 이 표에서 최악으로 나와야 방법론이 검증된 것이고,
#    그런데도 game_type 제거는 -46.7 이다 — 낮은 상관은 '제거하라' 가 아니라
#    '왜 변했는지 데이터에서 확인하라' 는 뜻이다.
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'tools')
import importlib.util
spec = importlib.util.spec_from_file_location('sc', 'tools/screen.py')
sc = importlib.util.module_from_spec(spec)
sys.modules['sc'] = sc
spec.loader.exec_module(sc)

CACHE = 'cache'
X = pd.read_pickle(f'{CACHE}/X.pkl')
y = np.load(f'{CACHE}/y.npy').astype('float64')
season = np.load(f'{CACHE}/season.npy')
meta = json.load(open(f'{CACHE}/meta.json', encoding='utf-8'))
CATS = set(meta['cat_features'])

for D in (sc._wseason5_cols(), sc._wsbat_cols()):
    for k, v in D.items():
        X[k] = np.asarray(v, dtype=np.float32)
print(f'피처 {X.shape[1]} | 행 {len(X):,}\n', flush=True)

EARLY = np.isin(season, [2019, 2020, 2021])
LATE = np.isin(season, [2023, 2024])
# 각 시기의 리그평균을 빼서 드리프트 자체는 제거한다 (우리가 보려는 건 '모양' 이다)
dev = y - np.where(EARLY, y[EARLY].mean(), np.where(LATE, y[LATE].mean(), y.mean()))

rows = []
for c in X.columns:
    s = X[c]
    if c in CATS or str(s.dtype) in ('object', 'category'):
        v = s.astype(str)
        keep = v.value_counts().head(24).index
        b = np.where(v.isin(keep), v, '__other__')
    else:
        x = pd.to_numeric(s, errors='coerce')
        nu = x.nunique(dropna=True)
        if nu < 2:
            continue
        if nu <= 12:
            # ⚠️ 저카디널리티는 값 그대로 쓴다. qcut(rank) 을 쓰면 같은 값 안을
            #    **행 순서로** 쪼개는데, 행 순서가 시즌과 상관돼 인공물이 나온다
            #    (strikes_before 가 -0.993 으로 나왔던 원인).
            b = x.fillna(-999.0).to_numpy(dtype='float64')
        else:
            try:
                b = pd.qcut(x.rank(method='first'), 10, labels=False).astype('float')
            except Exception:
                continue
            b = np.where(np.isnan(x), -1.0, b)      # 결측을 독립 구간으로
    d = pd.DataFrame({'b': b, 'dev': dev, 'e': EARLY, 'l': LATE})
    ge = d[d.e].groupby('b')['dev'].agg(['mean', 'size'])
    gl = d[d.l].groupby('b')['dev'].agg(['mean', 'size'])
    j = ge.join(gl, lsuffix='_e', rsuffix='_l', how='inner')
    j = j[(j['size_e'] >= 500) & (j['size_l'] >= 500)]
    if len(j) < 3:
        continue
    w = j['size_l'].to_numpy(dtype='float64')
    a, bb = j['mean_e'].to_numpy(), j['mean_l'].to_numpy()
    am, bm = np.average(a, weights=w), np.average(bb, weights=w)
    ca = np.average((a - am) * (bb - bm), weights=w)
    va = np.average((a - am) ** 2, weights=w)
    vb = np.average((bb - bm) ** 2, weights=w)
    r = float(ca / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan
    rows.append(dict(f=c, r=r, spread_e=float(np.sqrt(va)),
                     spread_l=float(np.sqrt(vb)), bins=len(j)))

d = pd.DataFrame(rows).dropna(subset=['r'])
# 배포 모델의 중요도를 붙인다 (있으면)
try:
    imp = json.load(open('out/imp_v10wp.json', encoding='utf-8'))
    d['imp'] = d['f'].map(imp).fillna(0.0)
except Exception:
    d['imp'] = np.nan

d['risk'] = d['imp'] * (1 - d['r'])          # 많이 쓰이는데 의미가 변한 것
d = d.sort_values('r')
print('=' * 88)
print('의미가 가장 많이 변한 피처 (옛 2019~21  vs  최근 2023~24 프로파일 상관)')
print(f'{"피처":<42}{"상관":>8}{"퍼짐(옛)":>10}{"퍼짐(최근)":>11}{"중요도":>8}{"위험":>8}')
print('-' * 88)
for _, r in d.head(24).iterrows():
    print(f'{r.f:<42}{r.r:>+8.3f}{r.spread_e:>10.4f}{r.spread_l:>11.4f}'
          f'{r.imp:>8.2f}{r.risk:>8.2f}')

print('\n' + '=' * 88)
print('★ 위험 = 중요도 x (1 - 상관)  — 많이 쓰이면서 의미가 변한 것')
print(f'{"피처":<42}{"상관":>8}{"중요도":>8}{"위험":>8}')
print('-' * 88)
for _, r in d.sort_values('risk', ascending=False).head(16).iterrows():
    print(f'{r.f:<42}{r.r:>+8.3f}{r.imp:>8.2f}{r.risk:>8.2f}')

print('\n' + '=' * 88)
print('가장 안정적인 피처 (참고 — 상관 1.0 근처면 전이 안전)')
for _, r in d.sort_values('r', ascending=False).head(8).iterrows():
    print(f'  {r.f:<42}{r.r:>+8.3f}{r.imp:>8.2f}')
d.to_csv('out/eda39_table.csv', index=False, encoding='utf-8')
print('\n-> out/eda39_table.csv')
print("""
읽는 법:
  game_type 이 하위권에 나와야 방법론이 맞는 것이다 (실제로 부호가 뒤집혔다).
  '위험' 상위 = 후보. 단 **제거가 답이라는 뜻이 아니다** — game_type 은 이 표에서
  최악인데 제거하면 -46.7 이다 (재학습하면 season 과 결합해 보완한다).
  낮은 상관은 '데이터에서 왜 변했는지 확인하라' 는 신호다.""")

# EDA 28 — teacher +797 이 진짜인가, 경기 지문 누수인가.
#
# 주장: 우리 96피처(983) 위에 그 투구의 트랙맨 측정 8개를 얹으면 1,780 (+797).
# 의심 근거 셋:
#   1) 트랙맨에는 **위치 정보가 없다**. control_success 는 위치/의도 판정이다.
#   2) eda14 는 "구종 위에 물리량 편차" 를 +57 로 쟀다. 새 숫자는 같은 것을 +369 로 본다.
#   3) 보고서 자신이 **경기축 초과분산 0.76~1.05%** 를 쟀다. 연속 측정 8개는
#      (투수 x 경기) 를 사실상 유일하게 지문화하므로, 무작위 fold 에서는
#      **같은 경기의 다른 투구**를 통해 경기 효과를 target-encoding 할 수 있다.
#
# 가설이 갈리는 지점:
#   A) 진짜 신호 (릴리스가 흔들린 공이 실제로 벗어난다) -> 경기로 그룹핑해도 증분이 남는다
#   B) 경기 지문 누수                                  -> GroupKFold(gid) 에서 증분이 무너진다
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, GroupKFold

MEAS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
        'extension', 'rel_height', 'rel_side', 'zone_speed']
SE = 2024
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 400

a = pd.read_parquet('cache/raw/aligned.parquet')
a = a[a['season'] == SE].copy()
X = pd.read_pickle('cache/X.pkl')
rid = np.load('cache/row_id.npy', allow_pickle=True)
X.index = pd.Index(rid, name='row_id')
y_all = pd.Series(np.load('cache/y.npy'), index=X.index)

a = a[a['row_id'].isin(X.index)]
B = X.loc[a['row_id']].reset_index(drop=True)
y = y_all.loc[a['row_id']].to_numpy(dtype='float64')
gid = a['gid'].to_numpy()
M = a[MEAS].to_numpy(dtype='float64')
pt = pd.get_dummies(a['pitch_type_group'].astype(str), prefix='pt').astype('float32')

cat = [c for c in B.columns if str(B[c].dtype) == 'category']
r = y.mean()
U = r * (1 - r)
print(f'{SE} 정렬분 {len(B):,}행 | 피처 {B.shape[1]} | 경기 {len(np.unique(gid)):,} | '
      f'리그 {r:.4f} | 반복 {ITERS}\n', flush=True)


def run(Xd, groups, tag):
    """groups=None 이면 무작위 fold, 아니면 경기 그룹 fold."""
    cv = (KFold(4, shuffle=True, random_state=0).split(Xd)
          if groups is None else GroupKFold(4).split(Xd, y, groups))
    p = np.zeros(len(Xd))
    for tr_i, va_i in cv:
        m = CatBoostClassifier(iterations=ITERS, depth=8, learning_rate=0.05,
                               cat_features=cat, verbose=0, thread_count=-1,
                               allow_writing_files=False)
        m.fit(Xd.iloc[tr_i], y[tr_i])
        p[va_i] = m.predict_proba(Xd.iloc[va_i])[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (1 - ((p - y) ** 2).mean() / U) * 100000


SETS = [('A. 우리 96피처', B),
        ('B. + 그 투구 측정 8개', pd.concat([B, a[MEAS].reset_index(drop=True)], axis=1)),
        ('D. + 구종군만', pd.concat([B, pt.reset_index(drop=True)], axis=1))]
# E: 투수 평균을 뺀 편차
dev = a[MEAS].to_numpy('float64') - a.groupby('pitcher_id')[MEAS].transform('mean').to_numpy('float64')
SETS.append(('E. + 투수평균 뺀 편차 8개',
             pd.concat([B, pd.DataFrame(dev, columns=[c + '_dev' for c in MEAS])
                        .reset_index(drop=True)], axis=1)))

print(f'{"구성":<28}{"무작위 fold":>14}{"경기 그룹 fold":>16}')
print('-' * 58, flush=True)
res = {}
for tag, Xd in SETS:
    s_rand = run(Xd, None, tag)
    s_grp = run(Xd, gid, tag)
    res[tag] = (s_rand, s_grp)
    print(f'{tag:<28}{s_rand:>14,.0f}{s_grp:>16,.0f}', flush=True)

b = res['A. 우리 96피처']
print('-' * 58)
print(f'{"증분":<28}{"무작위":>14}{"경기그룹":>16}')
for tag, (sr, sg) in res.items():
    if tag.startswith('A'):
        continue
    print(f'{tag:<28}{sr-b[0]:>+14,.0f}{sg-b[1]:>+16,.0f}')

print("""
읽는 법:
  경기그룹 증분이 무작위 증분과 비슷하다  -> 진짜 신호다 (릴리스 흔들림 -> 미스)
  경기그룹에서 증분이 크게 줄거나 사라진다 -> 경기 지문 누수였다. teacher 상한은 허수다.""")

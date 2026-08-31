# 투수 유리 카운트가 왜 약한가 — 라벨 구성으로 진단한다. 학습 0.
#
#   python tools/eda64.py
#
# eda62 가 가장 약한 구간으로 카운트 0-2(자체스킬 784) / 1-2(809)를 지목했다
# (평균 1221, 둘 합쳐 16%). claude.md 의 기존 진단은
#   "0-2/1-2 에서는 투수가 일부러 존을 벗어나게 던지므로 의도적 볼과 제구 실패가
#    라벨상 구분되지 않는다"
# 인데, **라벨 정의와 안 맞을 수 있다.** 라벨은 '존' 이 아니라 '포수가 요구한 곳' 이다.
# 일부러 바깥으로 뺐는데 포수가 거기 앉아 있었으면 그건 **성공**이어야 한다.
#
# 가른다:
#   (a) 0-2 의 실패 구성이 '크게 벗어남' 쪽으로 쏠린다 -> 진짜 제구 난이도. 처방 없음
#   (b) '반대방향' 쪽으로 쏠린다 -> 포수 요구 지점 추정이 흔들린다 = 라벨 잡음
#   (c) 구성이 다른 카운트와 비슷한데 성공률만 낮다 -> 단순히 어려운 타겟
#
# 그리고 투수별 신뢰도를 카운트별로 재서, **카운트 안에서 투수를 구분할 수 있는지**를
# 본다. 반쪽 분할 신뢰도(eda33 방식)가 낮으면 그 구간은 원리적으로 못 짜낸다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
y = df.control_success.to_numpy(dtype='float64')

n = df.asof_pitcher_n.to_numpy(dtype='float64')
g = df.pitcher_id.to_numpy()
o = np.lexsort((n, g))
ns, gs = n[o], g[o]
ok = np.r_[(gs[:-1] == gs[1:]) & (ns[1:] - ns[:-1] == 1), False]
lab = {}
for kk in ['success', 'middle', 'reverse', 'ball', 'strike']:
    cum = np.round(df['asof_pitcher_%s_rate' % kk].fillna(0)
                   .to_numpy(dtype='float64')[o] * ns)
    d = np.r_[cum[1:] - cum[:-1], np.nan]
    v = np.where(ok & np.isin(d, [0.0, 1.0]), d, np.nan)
    b = np.full(len(df), np.nan)
    b[o] = v
    lab[kk] = b
m = np.isfinite(lab['success'])
print('라벨 복원 검산: success 일치율 %.6f' % (lab['success'][m] == y[m]).mean())

good = np.isfinite(lab['middle']) & np.isfinite(lab['reverse']) & np.isfinite(lab['ball'])
cnt = (df.balls_before.astype(str) + '-' + df.strikes_before.astype(str)).to_numpy()
R = (df.game_type == 'R').to_numpy()
mi, re = lab['middle'] == 1, lab['reverse'] == 1
ball = lab['ball'] == 1

print('\n=== 카운트별 성공률과 실패 구성 (R, 라벨 복원 행) ===')
print('%-7s%9s%9s%10s%9s%9s%9s%9s'
      % ('카운트', '비중', '성공률', '볼비율', '몰림만', '반대만', '둘다', '크게벗어남'))
order = ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2',
         '3-0', '3-1', '3-2']
rows = {}
for c in order:
    k = R & good & (cnt == c)
    if k.sum() < 2000:
        continue
    f = k & (lab['success'] == 0)
    comp = [float((mi & ~re)[f].mean()), float((re & ~mi)[f].mean()),
            float((mi & re)[f].mean()), float((~mi & ~re)[f].mean())]
    rows[c] = (float(k.mean()), float(y[k].mean()), float(ball[k].mean()), comp)
    print('%-7s%8.1f%%%9.4f%10.4f%9.4f%9.4f%9.4f%9.4f'
          % (c, 100 * k.mean(), y[k].mean(), ball[k].mean(), *comp))

print('''
  '크게벗어남' 이 0-2/1-2 에서 유독 높으면 (a) 진짜 난이도,
  '반대만' 이 높으면 (b) 포수 요구 지점 추정 잡음, 둘 다 아니면 (c) 그냥 낮은 성공률이다.''')

# ── 카운트 안에서 투수를 구분할 수 있는가 (반쪽 분할 신뢰도, eda33 방식) ──
print('\n=== 카운트별 투수 반쪽 분할 신뢰도 (2024, R) ===')
print('  그 카운트 안에서 투수를 얼마나 구분할 수 있나. 낮으면 원리적으로 못 짜낸다.')
s24 = R & good & (df.season == 2024).to_numpy()
pid = df.pitcher_id.to_numpy()
rng = np.random.default_rng(0)
half = rng.random(len(df)) < 0.5
print('  %-7s%10s%12s%12s%10s' % ('카운트', '투수수', '전반 편차std', '신뢰도', '신호std'))
for c in order:
    k = s24 & (cnt == c)
    if k.sum() < 5000:
        continue
    a, b = k & half, k & ~half
    da = pd.Series(y[a]).groupby(pid[a]).agg(['mean', 'size'])
    db = pd.Series(y[b]).groupby(pid[b]).agg(['mean', 'size'])
    j = da.join(db, lsuffix='_a', rsuffix='_b', how='inner')
    j = j[(j['size_a'] >= 25) & (j['size_b'] >= 25)]
    if len(j) < 30:
        print('  %-7s%10s  표본 부족' % (c, len(j)))
        continue
    va = j['mean_a'].to_numpy() - y[k].mean()
    vb = j['mean_b'].to_numpy() - y[k].mean()
    rel = float(np.corrcoef(va, vb)[0, 1])
    sig = float(np.sqrt(max(rel, 0)) * np.std(va))
    print('  %-7s%10d%12.4f%12.3f%10.4f' % (c, len(j), np.std(va), rel, sig))

print('''
  신뢰도(반쪽 상관)가 0 근처면 그 카운트에서 투수 간 차이는 전부 표본 잡음이다
  -> 어떤 피처를 넣어도 못 짜낸다. 다른 카운트와 비슷하면 아직 여지가 있다.''')

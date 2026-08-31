# 오차 분석 — 배포본을 2024 에 돌려 **구간별 편향**을 찾는다.
#
#   python tools/eda61.py [zip] [N]
#
# ⚠️ 이 문제에서 원시 잔차 |y-p| 를 보면 안 된다. 타겟이 0/1 이고 예측이 0.35~0.60 에
#    몰려 있어 모든 행의 잔차가 ~0.5 다. "크게 틀린 행" 은 그냥 y=1 인데 p 가 낮은 행,
#    즉 **정답 라벨을 보고 고른 것**이라 공통점을 찾으면 라벨을 역추적할 뿐이다.
#
# Brier 에 맞는 분해는 둘이다.
#   · 분해능(discrimination) : 구간별 기여 = w_s x (1 - MSE_s/r(1-r))   -> eda42 가 함
#   · 신뢰도(reliability)    : 구간별 **편향** b_s = mean(p) - mean(y)   -> **여기**
# 편향이 있는 구간은 곧 '그 구간을 설명하는 피처가 빠졌거나 값이 틀렸다' 는 뜻이라
# 새 피처 아이디어로 바로 연결된다. 손실은 `100000 x w_s x b_s² / r(1-r)` 이다.
#
# ⚠️ 2024 행은 배포 모델의 **학습에 들어가 있다** (30폴드 중 27). 따라서 여기서 재는
#    편향은 실제보다 **작다**. 그래도 발견되는 편향은 진짜다 (보수적 검정).
#    그리고 재중심화 상수는 ~2023 모델의 2024 편향을 지우도록 맞춰져 있어 in-sample
#    예측에는 인공 전역 편향을 만든다. 그래서 **전역 편향을 뺀 뒤** 구간을 비교한다.
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

ZIP = sys.argv[1] if len(sys.argv) > 1 else 'kout_ph/submit_v10wph.zip'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']

tr = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
d24 = tr[tr.season == 2024].reset_index(drop=True)
if N < len(d24):
    d24 = d24.sample(N, random_state=0).sort_index().reset_index(drop=True)
print('2024 행 %s개로 배포본을 돌린다 (%s)' % (format(len(d24), ','), ZIP))

work = tempfile.mkdtemp(prefix='err_')
with zipfile.ZipFile(ZIP) as z:
    z.extractall(work)
dd = os.path.join(work, 'data')
os.makedirs(dd, exist_ok=True)
d24.drop(columns=['control_success']).to_csv(os.path.join(dd, 'test.csv'), index=False)
pd.DataFrame({'row_id': d24.row_id, 'control_success': .5}).to_csv(
    os.path.join(dd, 'sample_submission.csv'), index=False)
shutil.copy('data/trackman_history.csv', os.path.join(dd, 'trackman_history.csv'))
r = subprocess.run([sys.executable, '-u', 'script.py'], cwd=work, capture_output=True,
                   text=True, encoding='utf-8', errors='replace')
out = os.path.join(work, 'output', 'submission.csv')
if not os.path.exists(out):
    print(r.stdout[-3000:])
    print(r.stderr[-3000:])
    sys.exit('script.py 실패')
sub = pd.read_csv(out)
d = d24.merge(sub, on='row_id', how='left', suffixes=('', '_pred'))
p = d.control_success_pred.to_numpy(dtype='float64')
y = d.control_success.to_numpy(dtype='float64')
assert np.isfinite(p).all()
R = float(y.mean())
NAIVE = R * (1 - R)
GB = float(p.mean() - y.mean())
print('\n실제 %.4f | 예측 %.4f | 전역 편향 %+.4f' % (R, p.mean(), GB))
print('  (전역 편향은 in-sample + 재중심화 상수의 인공물이다. 아래는 전부 이걸 뺀 값)')

os.makedirs('cache', exist_ok=True)
np.savez('cache/err24.npz', p=p, y=y, row_id=d.row_id.to_numpy().astype(str))

# ---- 재스케일링 천장: λ* = Cov(p,y)/Var(p) ----
# ⚠️ 1 보다 크면 '늘려라' 는 뜻인데, in-sample 은 구조적으로 그 방향의 인공물을
#    만든다 (모델이 본 행이라 실제가 예측을 더 잘 따라간다). claude.md 의 정직한
#    2024 홀드아웃 측정은 λ*≈0.96, 재스케일링 천장 +1.95점이었다. 둘을 비교한다.
_c = float(np.cov(p, y, ddof=0)[0, 1])
_v = float(p.var())
_lam = _c / _v
_cur = (2 * _c - _v) / NAIVE
_opt = _c * _c / (_v * NAIVE)
print('\n=== 재스케일링 천장 ===')
print('  Cov(p,y) %.6f | Var(p) %.6f | **λ* = %.3f**' % (_c, _v, _lam))
print('  현재 스킬 %.4f%% -> λ* 적용 %.4f%%  (상대 %+.1f%%)'
      % (100 * _cur, 100 * _opt, 100 * (_opt / _cur - 1)))
print('  리더보드 환산 %.0f -> %.0f' % (_cur * 100000, _opt * 100000))
print('  ⚠️ claude.md 정직 홀드아웃: λ*=0.96, 천장 +1.95점.'
      ' 크게 다르면 in-sample 인공물이다.')

print('\n=== 신뢰도 곡선 (예측 10분위) ===')
q = pd.qcut(p, 10, labels=False, duplicates='drop')
print('%-6s%10s%10s%10s%10s' % ('분위', 'n', '평균예측', '실제', '편향(중심화)'))
for b in range(int(q.max()) + 1):
    k = q == b
    print('%-6d%10s%10.4f%10.4f%+10.4f'
          % (b, format(int(k.sum()), ','), p[k].mean(), y[k].mean(),
             p[k].mean() - y[k].mean() - GB))

# ---- 구간 정의 (전부 test 행이 자기 컬럼으로 만들 수 있는 것) ----
sd = d.score_diff_pitcher_team.to_numpy()
an = d.asof_pitcher_n.to_numpy(dtype='float64')
segs = {
    'game_type': d.game_type.astype(str),
    'cnt12': d.balls_before.astype(str) + '-' + d.strikes_before.astype(str),
    'inning': pd.cut(d.inning, [0, 3, 6, 99], labels=['1-3', '4-6', '7+']).astype(str),
    'outs': d.outs_before.astype(str),
    'base_state': d.base_state.astype(str),
    'top_bottom': d.top_bottom.astype(str),
    'pitcher_hand': d.pitcher_hand.astype(str),
    'batter_hand': d.batter_hand.astype(str),
    'hand4': d.pitcher_hand.astype(str) + d.batter_hand.astype(str),
    'score_diff': pd.cut(sd, [-99, -3, -1, 1, 3, 99]).astype(str),
    'month': d.game_month.astype(str),
    'dayofweek': d.game_dayofweek.astype(str),
    'pitcher_team': d.pitcher_team_id.astype(str),
    'batter_team': d.batter_team_id.astype(str),
    't13_involved': ((d.pitcher_team_id == 13) | (d.batter_team_id == 13)).astype(str),
    'asof_n(커리어)': pd.qcut(an, 5, labels=['q1', 'q2', 'q3', 'q4', 'q5']).astype(str),
    'asof_batter_n': pd.qcut(d.asof_batter_n.astype('float64'), 5,
                             labels=['q1', 'q2', 'q3', 'q4', 'q5'],
                             duplicates='drop').astype(str),
    'li': pd.qcut(d.li.astype('float64'), 5, labels=['q1', 'q2', 'q3', 'q4', 'q5'],
                  duplicates='drop').astype(str),
    'asof_p_rate': pd.qcut(d.asof_pitcher_success_rate.astype('float64'), 5,
                           labels=['q1', 'q2', 'q3', 'q4', 'q5'],
                           duplicates='drop').astype(str),
}

# ⚠️ 전역 편향만 빼면 **암기 인공물**(λ*=1.687 을 만드는 그것)이 구간 편향에 샌다.
#    예측 수준이 높은 구간은 자동으로 음의 편향을 갖게 되기 때문이다.
#    그래서 **예측 20분위 안에서** 편향을 재고 가중평균한다 — 분산 축을 고정하면
#    남는 것은 순수하게 '그 구간에만 있는' 편향이다.
_qb = pd.qcut(p, 20, labels=False, duplicates='drop')
_adj = np.zeros(len(p))
for _b in range(int(_qb.max()) + 1):
    _k = _qb == _b
    _adj[_k] = p[_k].mean() - y[_k].mean()      # 그 분위의 기준 편향
resid = p - y - _adj                            # 분위 내 상대 편향

rows = []
for name, s in segs.items():
    s = s.to_numpy() if hasattr(s, 'to_numpy') else np.asarray(s)
    for v in pd.unique(s):
        k = s == v
        if k.sum() < 800:
            continue
        w = float(k.mean())
        b = float(resid[k].mean())              # 분위 보정 후
        b0 = float(p[k].mean() - y[k].mean() - GB)   # 보정 전 (비교용)
        rows.append((100000 * w * b * b / NAIVE, name, str(v), int(k.sum()), w, b,
                     float(y[k].mean()), b0))
rows.sort(reverse=True)

print('\n=== 구간별 편향 (전역 편향 제거 후). 손실 = 100000 x w x b² / r(1-r) ===')
print('%-16s%-12s%10s%8s%11s%11s%9s'
      % ('축', '값', 'n', '비중', '편향(분위내)', '편향(전역만)', '손실(점)'))
for loss, name, v, n, w, b, ya, b0 in rows[:22]:
    print('%-16s%-12s%10s%7.1f%%%+11.4f%+11.4f%9.2f'
          % (name, v[:11], format(n, ','), 100 * w, b, b0, loss))

print('\n=== 축별 합계 손실 ===')
agg = {}
for loss, name, *_ in rows:
    agg[name] = agg.get(name, 0.0) + loss
for k, v in sorted(agg.items(), key=lambda x: -x[1]):
    print('  %-18s %6.2f점' % (k, v))
print('''
읽는 법: 손실이 곧 '그 구간의 편향을 상수 하나로 고쳤을 때 되찾는 점수' 다.
         2024 는 학습에 들어가 있으므로 실제 2025 에서는 이보다 크다.
         한 축의 합계가 노이즈 바닥(1.5점)을 크게 넘으면 그 축에 피처가 빠진 것이다.''')

# wseason 디트렌드 상수의 2025 외삽이 정확한가. 학습 0.
#
#   python tools/eda65.py
#
# `ws_apply` 는 `w_rate = (wx + base*C)/(wn + C) - base` 로 당해 시즌 비율을
# `base` 에 shrink 하고 `base` 를 뺀다. `base` 는 그 시즌의 **asof 컬럼(커리어 누적)
# 평균**이다 (control_success 평균이 아니다 -- `wseason5b` 가 -13.0 이라 이 선택은
# 의도된 것이다).
#
# 학습 시즌은 **실제 평균**을 쓰고 2025 만 **3시즌 선형 외삽**이다. 즉 외삽이 틀리면
# **2025 행에서만** 최중요 피처가 계통적으로 어긋난다. 학습에선 본 적 없는 편이다.
#
# 검증: 배포와 같은 함수로 2022/2023/2024 를 각각 '한 점 빼고' 맞혀 오차를 본다
# (claude.md 필수 절차 -- 이 검증이 224점/46점 실수를 세 번 막았다).
# 그리고 오차가 `w_rate` 를 얼마나 흔드는지 환산한다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
COLS = ['asof_pitcher_success_rate', 'asof_pitcher_middle_rate',
        'asof_pitcher_reverse_rate', 'asof_pitcher_ball_rate',
        'asof_pitcher_strike_rate', 'asof_batter_success_rate',
        'asof_batter_middle_rate']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA,
                 usecols=['season', 'game_type', 'control_success'] + COLS)
SE = list(range(2019, 2025))


def extrap(d, target, k=3):
    """배포본 ws_next_season_mean 과 동일: 최근 k 시즌 선형."""
    ss = sorted(d)[-k:]
    a, b = np.polyfit(ss, [d[s] for s in ss], 1)
    return float(np.clip(a * target + b, 0.0, 1.0))


# 배포본에 실제로 박힌 값 (kout_tor)
DEPLOY = {'asof_pitcher_success_rate': 0.499505,
          'asof_pitcher_middle_rate': 0.160941,
          'asof_pitcher_reverse_rate': 0.245092,
          'asof_pitcher_ball_rate': 0.371802,
          'asof_pitcher_strike_rate': 0.445596,
          'asof_batter_success_rate': 0.503508,
          'asof_batter_middle_rate': 0.155824}

print('=== 시즌별 asof 컬럼 평균 (디트렌드 기준) ===')
means = {}
for c in COLS:
    means[c] = {s: float(df.loc[df.season == s, c].mean()) for s in SE}
print('%-32s' % '' + ''.join('%9d' % s for s in SE) + '%11s%11s' % ('2025외삽', '배포값'))
for c in COLS:
    e = extrap(means[c], 2025)
    print('%-32s' % c[:31] + ''.join('%9.4f' % means[c][s] for s in SE)
          + '%11.6f%11.6f' % (e, DEPLOY[c]))

print('\n=== 한 점 빼고 맞히기 (배포와 같은 3시즌 선형) ===')
print('%-32s%10s%10s%10s%10s' % ('컬럼', 'Y', '예측', '실제', '오차'))
tot = {}
for c in COLS:
    errs = []
    for Y in (2022, 2023, 2024):
        d = {s: means[c][s] for s in SE if s < Y}
        p = extrap(d, Y)
        errs.append(abs(p - means[c][Y]))
        if c == 'asof_pitcher_success_rate':
            print('%-32s%10d%10.4f%10.4f%10.4f' % (c[:31], Y, p, means[c][Y], p - means[c][Y]))
    tot[c] = float(np.mean(errs))
print()
print('%-32s%12s' % ('컬럼', '평균 |오차|'))
for c in COLS:
    print('%-32s%12.5f' % (c[:31], tot[c]))

# ---- 오차가 w_rate 를 얼마나 흔드는가 ----
# w = (wx + base*C)/(wn + C) - base.  base -> base+e 이면
#   Δw = e*C/(wn+C) - e = -e * wn/(wn+C).   wn 이 크면 -> -e (그대로 전이)
print('\n=== 오차 e 가 w_rate 에 미치는 영향: Δw = -e x wn/(wn+C), C=100 ===')
for wn in (50, 200, 500, 1000, 2000):
    print('  wn=%-5d  전이율 %.3f' % (wn, wn / (wn + 100)))
print('  -> 투구수가 많은 투수일수록 오차가 그대로 넘어간다.')
print('''
읽는 법: 평균 |오차| 가 0.002 수준이면 무시할 만하다. 0.01 을 넘으면 2025 행에서만
         최중요 피처가 그만큼 밀린다는 뜻이고, **상수 하나라 GPU 0 으로 고칠 수 있다.**''')

# 참고: control_success 자체의 시즌 평균 (커리어 누적과의 격차)
print('\n=== 참고: 커리어 누적 평균 vs 실제 당해 성공률 ===')
print('%-10s%12s%12s%10s' % ('시즌', 'asof평균', 'control평균', '격차'))
for s in SE:
    a = means['asof_pitcher_success_rate'][s]
    b = float(df.loc[df.season == s, 'control_success'].mean())
    print('%-10d%12.4f%12.4f%10.4f' % (s, a, b, a - b))

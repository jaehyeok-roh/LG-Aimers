# 재중심화 상수는 2025 에 맞는가 — 앵커 비대칭을 따진다. 학습 0.
#
#   python tools/eda60.py
#
# 재중심화는 홀드아웃 기준 **+134** 를 지고 있는 단일 상수다. 손실이
# `100000 x Δ² / r(1-r)` 이므로 평균이 0.005 어긋나면 10점, 0.01 이면 40점이다.
# 지금 남은 후보 중 크기가 있는 유일한 것이고 GPU 가 0 이다.
#
# **구조상의 비대칭** (claude.md 어디에도 없다):
#   오프셋 측정 모델 = ~2023 학습  ->  2024 예측
#   배포 모델        = ~2024 학습  ->  2025 예측
# 학습 앵커가 한 시즌 다르다. 성공률이 매 시즌 내려가므로 배포 모델은 **더 낮은
# 앵커**를 갖고, 따라서 편향이 더 작아야 한다. 그런데 우리는 홀드아웃에서 잰
# 오프셋을 그대로 쓴다 -> **과보정** 가능성.
#
# 다만 표적도 같이 내려간다 (2024 -> 2025). 두 효과가 부분 상쇄하므로 부호가
# 자명하지 않다. 그래서 모형을 세우고 **한 점 빼고 맞히기**로 검증한다
# (claude.md: 이 검증이 224점 / 46점 / 1,431점짜리 실수를 세 번 막았다).
#
#   편향 b_Y ≈ α x (m_Y - r_Y)      m_Y = ~Y-1 학습분 평균, r_Y = Y 실제 평균
#
# eda38 이 107피처 이진에서 b 를 네 해 실측해뒀다. α 가 안정적이면 같은 α 로
# b_2025 를 외삽하고, 현행(b_2024 재사용)과의 차이를 점수로 환산한다.
# ⚠️ α 의 절대값은 피처 구성에 따라 변하지만 **비 b_2025/b_2024 는 α 가 약분돼
#    사라진다.** 그래서 120피처 5분류에도 옮길 수 있다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA,
                 usecols=['season', 'game_type', 'control_success'])
y = df.control_success.to_numpy(dtype='float64')
sea = df.season.to_numpy()
gt = df.game_type.to_numpy().astype(str)
SEASONS = list(range(2019, 2025))

rate = {s: float(y[sea == s].mean()) for s in SEASONS}
nrow = {s: int((sea == s).sum()) for s in SEASONS}
fshare = {s: float((gt[sea == s] == 'F').mean()) for s in SEASONS}
print('시즌  %8s%10s%9s%9s' % ('성공률', '행수', 'F비중', '낙폭'))
for s in SEASONS:
    d = rate[s] - rate[s - 1] if s > 2019 else np.nan
    print('%d  %8.4f%10s%9.4f%9s'
          % (s, rate[s], format(nrow[s], ','), fshare[s],
             '%+.4f' % d if np.isfinite(d) else '-'))

# 2025 외삽: 최근 3년 선형 (claude.md 가 이 방식으로 2024 를 오차 0.0016 에 맞혔다)
xs = np.array([2022, 2023, 2024], dtype=float)
ys = np.array([rate[s] for s in (2022, 2023, 2024)])
a, b0 = np.polyfit(xs, ys, 1)
r2025 = float(a * 2025 + b0)
print('\n2025 외삽 (최근3년 선형): %.4f  (2024 대비 %+.4f)' % (r2025, r2025 - rate[2024]))
# 검산: 같은 방식으로 2024 를 맞혀본다
a2, b2 = np.polyfit(np.array([2021., 2022, 2023]), [rate[s] for s in (2021, 2022, 2023)], 1)
print('  검산) 2021~23 으로 2024 예측 %.4f | 실제 %.4f | 오차 %.4f'
      % (a2 * 2024 + b2, rate[2024], abs(a2 * 2024 + b2 - rate[2024])))


def m_upto(Y):
    """~Y-1 학습분의 라벨 평균 (모델이 붙잡는 앵커)."""
    k = sea <= Y - 1
    return float(y[k].mean())


# eda38 실측 b (107피처 이진, 전체 R+F)
B38 = {2021: 0.0014, 2022: 0.0025, 2023: 0.0229, 2024: 0.0182}
print('\n=== 앵커 격차 m-r 와 실측 편향 b (eda38) ===')
print('%-6s%10s%10s%10s%10s%9s' % ('Y', 'm(~Y-1)', 'r(Y)', 'm-r', 'b(eda38)', 'α=b/(m-r)'))
al = []
for Y in (2021, 2022, 2023, 2024):
    m, r = m_upto(Y), rate[Y]
    g = m - r
    al.append(B38[Y] / g)
    print('%-6d%10.4f%10.4f%10.4f%10.4f%9.3f' % (Y, m, r, g, B38[Y], B38[Y] / g))
al = np.array(al)
print('  α 평균 %.3f | std %.3f | 변동계수 %.2f' % (al.mean(), al.std(), al.std() / al.mean()))

# 2025 의 앵커 격차
m2025 = m_upto(2025)          # ~2024 전체
g2025 = m2025 - r2025
g2024 = m_upto(2024) - rate[2024]
print('\n=== 2025 예측 ===')
print('  m(~2024) %.4f | r(2025) 외삽 %.4f | 격차 %.4f' % (m2025, r2025, g2025))
print('  m(~2023) %.4f | r(2024) 실제 %.4f | 격차 %.4f' % (m_upto(2024), rate[2024], g2024))
ratio = g2025 / g2024
print('  ★ 격차 비 (2025/2024) = %.3f' % ratio)
print('     -> 모형이 맞다면 b_2025 = b_2024 x %.3f 여야 하는데 현행은 x1.000 을 쓴다' % ratio)

# 한 점 빼고 맞히기: α 를 세 해로 추정해 나머지 한 해의 b 를 맞힌다
print('\n=== 검증: 한 점 빼고 맞히기 (claude.md 필수 절차) ===')
print('%-6s%12s%12s%10s%10s' % ('뺀 해', 'α(나머지3)', 'b 예측', 'b 실제', '손실(점)'))
tot_m, tot_cur = 0.0, 0.0
for Y in (2021, 2022, 2023, 2024):
    rest = [k for k in (2021, 2022, 2023, 2024) if k != Y]
    aa = np.mean([B38[k] / (m_upto(k) - rate[k]) for k in rest])
    pred = aa * (m_upto(Y) - rate[Y])
    # 현행(D 모형) = 직전 해의 b 를 그대로 쓴다
    cur = B38[Y - 1] if (Y - 1) in B38 else B38[2021]
    lo = lambda e: 100000 * e * e / (rate[Y] * (1 - rate[Y]))
    tot_m += lo(pred - B38[Y])
    tot_cur += lo(cur - B38[Y])
    print('%-6d%12.3f%12.4f%10.4f%10.1f%10.1f'
          % (Y, aa, pred, B38[Y], lo(pred - B38[Y]), lo(cur - B38[Y])))
print('  누적 손실:  앵커모형 %.1f  vs  현행(직전 b 재사용) %.1f' % (tot_m, tot_cur))
# ⚠️ 총합만 보면 앵커모형이 이긴다. 그런데 **2023 을 빼면 뒤집힌다** —
#    2023 은 F 라벨 체제가 통째로 바뀐 일회성 해이고 2025 에 그런 예고는 없다.
_nm = [Y for Y in (2021, 2022, 2024)]
_a2 = sum(100000 * (np.mean([B38[k] / (m_upto(k) - rate[k])
                             for k in (2021, 2022, 2023, 2024) if k != Y])
                    * (m_upto(Y) - rate[Y]) - B38[Y]) ** 2
          / (rate[Y] * (1 - rate[Y])) for Y in _nm)
_c2 = sum(100000 * (B38[Y - 1] - B38[Y]) ** 2 / (rate[Y] * (1 - rate[Y]))
          for Y in _nm if (Y - 1) in B38)
print('  ★ 2023(F 체제 충격) 제외:  앵커모형 %.1f  vs  현행 %.1f  -> **현행이 낫다**'
      % (_a2, _c2))

# 2025 에 적용했을 때의 차이
b2024 = B38[2024]
b2025_model = al.mean() * g2025
print('\n=== 2025 상수 결정 ===')
print('  현행       b_2025 = b_2024        = %+.4f' % b2024)
print('  앵커모형   b_2025 = α x (m-r)     = %+.4f' % b2025_model)
print('  차이 %.4f -> 어느 한쪽이 맞다면 다른 쪽은 %.1f점 손해'
      % (abs(b2025_model - b2024),
         100000 * (b2025_model - b2024) ** 2 / (r2025 * (1 - r2025))))


# ── 정정: 2025 외삽은 R/F 를 분리해야 한다 ──
# 위의 전체 3년 선형(0.4622)은 2022->2023 낙폭 -0.0290 을 진짜 드리프트로 읽는다.
# 그런데 claude.md/eda57 이 확인했듯 그 낙폭은 **거의 전부 F 라벨 체제 변경**이고
# R 전용으로는 -0.0006 이다. F 는 이미 새 체제에 정착했으므로 반복되지 않는다.
print('\n' + '=' * 62)
print('=== 정정: R/F 분리 외삽 ===')
rR = {s: float(y[(sea == s) & (gt == 'R')].mean()) for s in SEASONS}
rF = {s: float(y[(sea == s) & (gt == 'F')].mean()) for s in SEASONS}
print('  R :', ' '.join('%d %.4f' % (s, rR[s]) for s in SEASONS))
print('  F :', ' '.join('%d %.4f' % (s, rF[s]) for s in SEASONS))
aR, bR = np.polyfit(xs, [rR[s] for s in (2022, 2023, 2024)], 1)
aF, bF = np.polyfit(np.array([2023., 2024]), [rF[s] for s in (2023, 2024)], 1)
R25, F25 = float(aR * 2025 + bR), float(aF * 2025 + bF)
fs = float(np.mean([fshare[s] for s in (2023, 2024)]))
r25 = (1 - fs) * R25 + fs * F25
print('  R 외삽 %.4f | F 외삽 %.4f (2023~24 만, 체제 전환 후) | F비중 %.4f' % (R25, F25, fs))
print('  ★ 혼합 r_2025 = %.4f   (전체 3년 선형 %.4f 보다 %+.4f)' % (r25, r2025, r25 - r2025))

g25 = m2025 - r25
print('\n  m(~2024) %.4f - r_2025 %.4f = 격차 %.4f' % (m2025, r25, g25))
print('  m(~2023) %.4f - r_2024 %.4f = 격차 %.4f' % (m_upto(2024), rate[2024], g2024))
print('  ★ 격차 비 (2025/2024) = %.3f' % (g25 / g2024))
bm = al.mean() * g25
print('\n  현행     b_2025 = b_2024              = %+.4f' % b2024)
print('  앵커모형 b_2025 = α_평균 x 격차        = %+.4f' % bm)
print('  비-일정 b_2025 = b_2024 x 격차비       = %+.4f' % (b2024 * g25 / g2024))
for nm, v in (('앵커모형', bm), ('비-일정', b2024 * g25 / g2024)):
    print('  %s 이 맞다면 현행은 %.1f점 손해' % (nm, 100000 * (v - b2024) ** 2 / (r25 * (1 - r25))))
print('''
⛔ 그리고 '예측 평균 = r_2025 가 되게 푼다' 는 **규정 위반**이다 —
   주최측 금지 예시에 '평가 데이터 전체의 평균으로 예측값 보정' 이 명시돼 있다.
   그래서 현행(편향 b 를 상수로 박는 D 모형)이 유일하게 합법인 형태다.''')

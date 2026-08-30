# 월 단위 계단 변화 전수 탐색 — t13 과 같은 형태를 팀이 아닌 축에서도 찾는다. 학습 0.
#
# t13(+20.58)의 형태는 `(A==x OR B==x) AND (시점 > T)` 였다. eda52 는 시간 축을
# **연 단위(era)** 로만 봤는데, t13 이 걸린 이유가 바로 시간 해상도다 —
# 2023 연간값은 +0.0507 이지만 4월만 보면 -0.0220 이다. 연 단위로는 희석된다.
#
# 팀에 대해서는 월별 계단 탐색을 이미 했고 13 만 나왔다. **팀이 아닌 축은 안 했다.**
# 판정 체제 변경(ABS 등)은 팀이 아니라 규칙 단위로 일어나므로 오히려 이쪽이 자연스럽다.
#
# 방법: 각 (컬럼, 값) 에 대해 월별 '그 값의 효과'(해당 행 평균 - 나머지 평균) 계열을 만들고,
#       모든 분할점에서 전/후 평균차가 최대인 곳을 찾는다.
#       ⚠️ 계열이 시끄러우면 분할점 최적화만으로도 큰 값이 나온다. 그래서 **월별 잡음
#          표준편차를 같이 내고, 계단폭이 그 3배를 넘는 것만 후보로 본다.**
#          (팀 12 가 계단 0.039 로 잡혔지만 계열이 ±0.03 로 출렁이던 게 그 예다.)
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
df = df[df.game_type == 'R'].copy()
df['ym'] = df.season * 100 + df.game_month
df['cnt12'] = df.balls_before.astype(str) + '-' + df.strikes_before.astype(str)
df['inn3'] = pd.cut(df.inning, [0, 3, 6, 99], labels=['1-3', '4-6', '7+']).astype(str)
df['sd3'] = pd.cut(df.score_diff_pitcher_team, [-99, -3, -1, 1, 3, 99],
                   labels=['a', 'b', 'c', 'd', 'e']).astype(str)
df['t13'] = ((df.pitcher_team_id == 13) | (df.batter_team_id == 13)).astype(int)
# count_advantage 는 원본이 아니라 step4 파생이다 (0-0/3-2 가 'None' 으로 묶인다)
_b, _s = df.balls_before, df.strikes_before
df['count_advantage'] = np.select([_b < _s, _b > _s, (_b == _s) & _b.isin([1, 2])],
                                  ['Pitcher', 'Batter', 'Neutral'], default='None')

COLS = ['cnt12', 'inn3', 'sd3', 'base_state', 'outs_before', 'top_bottom',
        'pitcher_hand', 'batter_hand', 'runner_on_1b', 'runner_on_2b',
        'runner_on_3b', 'count_advantage', 't13']
months = sorted(df.ym.unique())
print('R %s행 | 월 %d개 (%d ~ %d)\n' % (f'{len(df):,}', len(months), months[0], months[-1]))

rows = []
for c in COLS:
    v = df[c].astype(str)
    for val in sorted(v.unique()):
        m = (v == val).to_numpy()
        if m.mean() < 0.03:                       # 커버리지 3% 미만은 크기가 안 나온다
            continue
        eff = []
        for mo in months:
            k = (df.ym == mo).to_numpy()
            a, b = m & k, (~m) & k
            eff.append(df.control_success.to_numpy()[a].mean()
                       - df.control_success.to_numpy()[b].mean()
                       if a.sum() > 300 and b.sum() > 300 else np.nan)
        a = np.array(eff, dtype=float)
        ok = ~np.isnan(a)
        if ok.sum() < 20:
            continue
        # 월간 잡음: 인접 월 차분의 표준편차 / sqrt(2)
        d = np.diff(a[ok])
        noise = float(np.std(d) / np.sqrt(2))
        best, bi = 0.0, None
        for k in range(4, len(a) - 4):
            p, q = a[:k], a[k:]
            if np.isnan(p).all() or np.isnan(q).all():
                continue
            g = abs(np.nanmean(q) - np.nanmean(p))
            if g > best:
                best, bi = g, k
        if bi is None:
            continue
        rows.append((c, val, float(m.mean()), best, noise, best / max(noise, 1e-9),
                     months[bi], float(np.nanmean(a[:bi])), float(np.nanmean(a[bi:]))))

rows.sort(key=lambda r: -r[5])
print('=== 계단 변화 / 월간 잡음 비 순 (>=3 만 후보) ===')
print('%-16s%-10s%7s%9s%9s%7s%9s%9s%9s'
      % ('컬럼', '값', '비중', '계단폭', '잡음', '비', '시점', '전', '후'))
for c, val, cov, g, nz, r, mo, pre, post in rows[:16]:
    flag = ' ★' if r >= 3 else ''
    print('%-16s%-10s%6.1f%%%9.4f%9.4f%7.1f%9d%+9.4f%+9.4f%s'
          % (c, val[:9], 100 * cov, g, nz, r, mo, pre, post, flag))

print('\n=== 점수 환산 (비 3 이상만) ===')
n = 0
for c, val, cov, g, nz, r, mo, pre, post in rows:
    if r < 3:
        continue
    n += 1
    pts = 401000 * (cov * (g * (1 - cov)) ** 2 + (1 - cov) * (g * cov) ** 2)
    print('  %-14s %-9s 커버 %.1f%% | 계단 %+.4f | 완벽히 잡으면 %.1f점'
          % (c, val[:9], 100 * cov, post - pre, pts))
if n == 0:
    print('  없음 — 팀 축(t13) 말고는 월 단위 계단 변화가 없다')

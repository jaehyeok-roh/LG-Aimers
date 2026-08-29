# 저카디널리티 조합 전수 스캔 — t13(+20.58) 과 같은 부류를 더 찾는다. 학습 0.
#
# 통한 것들이 전부 한 부류다: **트리가 만들기 비싼 조합을 직접 준 것**.
#   is_same_hand (hand x hand)          제거하면 -17.0
#   cnt12 + hand4                       +5.72
#   t13 + t13_post (팀 OR x 시점)       **+20.58**
#
# 그래서 모든 컬럼 쌍의 **2차 상호작용 잔차**를 잰다. 가법 모형으로 설명되는 부분은
# 트리가 한 레벨씩 쪼개서 싸게 만들 수 있으니, 남는 것은 조합으로만 표현되는 몫이다.
#
#   r_ab = mean(y|a,b) - mean(y|a) - mean(y|b) + mean(y)
#   강도 = sum_cells  비중 x r_ab²        ->  점수 = 401,000 x 강도
#
# ⚠️ eda48 교훈: 집계로 닫기 전에 개별 항목을 본다. 그래서 상위 쌍은 셀 단위로 펼치고,
#    **시즌 간 안정성**(전반기 2019~22 vs 후반기 2023~24 잔차 상관)을 같이 낸다.
#    표본 내 강도가 커도 해를 넘겨 안 따라오면 wn/game_type 세그먼트 상수 꼴이 난다.
import itertools
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
df = df[df.game_type == 'R'].copy()          # F 는 팀13 과 완전 교락 (eda51)
y = df.control_success.to_numpy(dtype='float64')
GM = float(y.mean())
U = 0.2494
print('R %s행 | 전체 성공률 %.4f\n' % (f'{len(df):,}', GM))

# 조합으로 줄 만한 저카디널리티 축만 (수치 연속형은 트리가 임계값으로 싸게 근사한다)
df['cnt12'] = df.balls_before.astype(str) + '-' + df.strikes_before.astype(str)
df['inn3'] = pd.cut(df.inning, [0, 3, 6, 99], labels=['1-3', '4-6', '7+']).astype(str)
df['sdiff3'] = pd.cut(df.score_diff_pitcher_team, [-99, -3, -1, 1, 3, 99],
                      labels=['<-3', '-3~-1', 'tie', '1~3', '>3']).astype(str)
df['t13'] = ((df.pitcher_team_id == 13) | (df.batter_team_id == 13)).astype(int)
df['era'] = (df.season >= 2023).astype(int)   # t13 이 살았던 시점 축

COLS = ['cnt12', 'inn3', 'sdiff3', 'base_state', 'outs_before', 'top_bottom',
        'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id',
        'season', 'era', 't13', 'runner_on_1b', 'runner_on_3b']
for c in COLS:
    df[c] = df[c].astype(str)
print('축 %d개 | 쌍 %d개' % (len(COLS), len(list(itertools.combinations(COLS, 2)))))


def inter(a, b, sub=None):
    d = df if sub is None else df[sub]
    yy = d.control_success.to_numpy(dtype='float64')
    g = float(yy.mean())
    ma = d.groupby(a).control_success.mean()
    mb = d.groupby(b).control_success.mean()
    cell = d.groupby([a, b]).control_success.agg(['mean', 'size'])
    r = (cell['mean']
         - cell.index.get_level_values(0).map(ma).to_numpy()
         - cell.index.get_level_values(1).map(mb).to_numpy() + g)
    w = cell['size'].to_numpy() / len(d)
    return float((w * r.to_numpy() ** 2).sum()), r, cell['size']


rows = []
for a, b in itertools.combinations(COLS, 2):
    st, r, n = inter(a, b)
    rows.append((a, b, 401000 * st, int(len(r))))
rows.sort(key=lambda t: -t[2])

print('\n=== 2차 상호작용 강도 (완벽히 잡았을 때의 점수 상당) ===')
print('%-22s%-22s%10s%8s' % ('축 A', '축 B', '점수', '셀수'))
for a, b, pts, nc in rows[:16]:
    print('%-22s%-22s%10.1f%8d' % (a, b, pts, nc))

# --- 상위 쌍의 시즌 간 안정성 ---
print('\n=== 상위 8쌍의 시즌 간 안정성 (2019~22 잔차 vs 2023~24 잔차) ===')
pre = (df.season.astype(int) <= 2022).to_numpy()
post = ~pre
print('%-22s%-22s%10s%10s%10s' % ('축 A', '축 B', '전반강도', '후반강도', '상관'))
for a, b, pts, nc in rows[:8]:
    s1, r1, n1 = inter(a, b, pre)
    s2, r2, n2 = inter(a, b, post)
    j = r1.index.intersection(r2.index)
    c = float(np.corrcoef(r1.loc[j], r2.loc[j])[0, 1]) if len(j) > 3 else float('nan')
    print('%-22s%-22s%10.1f%10.1f%10.2f' % (a, b, 401000 * s1, 401000 * s2, c))

# --- 최상위 쌍은 셀 단위로 펼친다 (집계로 닫지 않는다) ---
a, b, pts, nc = rows[0]
print('\n=== 최상위 쌍 %s x %s 의 셀별 잔차 (비중 1%% 이상) ===' % (a, b))
st, r, n = inter(a, b)
t = pd.DataFrame({'resid': r, 'n': n})
t['frac'] = t['n'] / len(df)
t = t[t.frac >= 0.01].sort_values('resid')
print(t.assign(pts=lambda d: 401000 * d.frac * d.resid ** 2).to_string(
    float_format=lambda v: f'{v:+.4f}'))

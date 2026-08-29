# eda52 후속 — t13 을 빼고도 남는 조합이 있는가, 그리고 2023 vs 2024 로 안정적인가.
#
# eda52 상위:
#   season x t13        202.0   <- 이미 먹었다 (v10wt13 +20.58)
#   pitcher_team x era   95.7   <- t13 을 일반화한 형태. era 는 2025 로 확장된다
#   batter_team  x era   64.1
#   hand x pitcher_team  38.1   <- 모델에 없다
#   pitcher_team x batter_team 27.8
#
# ⚠️ 'x season' 형태는 2025 가 학습에 없는 값이라 학습 시대에 얼어붙는다
#    (auxrev 전달률 0.10 의 기전). era(2023 이후 0/1)만 쓴다.
#
# 두 가지를 본다:
#   1) t13 관여 행을 **제외**하고도 team x era 강도가 남는가 (남으면 새 후보)
#   2) 2023 잔차 vs 2024 잔차 상관 — 2025 는 후반 체제이므로 이게 옳은 안정성 검사다
#      (eda48 에서 시즌 간 상관 0.21 짜리를 채택했다면 wn/game_type 꼴이 났다)
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
df = df[df.game_type == 'R'].copy()
df['t13'] = ((df.pitcher_team_id == 13) | (df.batter_team_id == 13)).astype(int)
df['era'] = (df.season >= 2023).astype(int)
for c in ('pitcher_team_id', 'batter_team_id', 'era', 'pitcher_hand', 'batter_hand'):
    df[c] = df[c].astype(str)


def strength(d, a, b):
    yy = d.control_success
    g = float(yy.mean())
    ma, mb = d.groupby(a).control_success.mean(), d.groupby(b).control_success.mean()
    cell = d.groupby([a, b]).control_success.agg(['mean', 'size'])
    r = (cell['mean']
         - cell.index.get_level_values(0).map(ma).to_numpy()
         - cell.index.get_level_values(1).map(mb).to_numpy() + g)
    w = cell['size'].to_numpy() / len(d)
    return 401000 * float((w * r.to_numpy() ** 2).sum()), r


print('=== 1) t13 관여 행을 빼면 team x era 가 남는가 ===')
print('%-34s%12s%12s' % ('조합', '전체', 't13 제외'))
no13 = df[df.t13 == 0]
for a, b in [('pitcher_team_id', 'era'), ('batter_team_id', 'era'),
             ('pitcher_hand', 'pitcher_team_id'), ('pitcher_team_id', 'batter_team_id'),
             ('pitcher_hand', 'batter_hand')]:
    s_all, _ = strength(df, a, b)
    s_no, _ = strength(no13, a, b)
    print('%-34s%12.1f%12.1f' % (f'{a} x {b}', s_all, s_no))

print('\n=== 2) 2023 잔차 vs 2024 잔차 (2025 는 후반 체제) ===')
d23, d24 = df[df.season == 2023], df[df.season == 2024]
print('%-34s%10s%10s%10s%8s' % ('조합', '2023강도', '2024강도', '상관', '셀수'))
for a, b in [('pitcher_team_id', 'batter_team_id'), ('pitcher_hand', 'pitcher_team_id'),
             ('pitcher_hand', 'batter_team_id'), ('batter_hand', 'pitcher_team_id'),
             ('pitcher_hand', 'batter_hand')]:
    s3, r3 = strength(d23, a, b)
    s4, r4 = strength(d24, a, b)
    j = r3.index.intersection(r4.index)
    c = float(np.corrcoef(r3.loc[j], r4.loc[j])[0, 1]) if len(j) > 3 else float('nan')
    print('%-34s%10.1f%10.1f%10.2f%8d' % (f'{a} x {b}', s3, s4, c, len(j)))

print('\n=== 3) 팀별 era 전후 격차 (t13 제외, R 전용) — 두 번째 t13 이 있나 ===')
post = df[df.era == '1']
pre = df[df.era == '0']
print('%-8s%12s%12s%10s%10s' % ('팀', '전(19~22)', '후(23~24)', '변화', '후반비중'))
rows = []
for t in sorted(df.pitcher_team_id.unique()):
    fa = (pre.pitcher_team_id == t) | (pre.batter_team_id == t)
    fb = (post.pitcher_team_id == t) | (post.batter_team_id == t)
    da = pre[fa].control_success.mean() - pre[~fa].control_success.mean()
    db = post[fb].control_success.mean() - post[~fb].control_success.mean()
    rows.append((t, da, db, db - da, fb.mean()))
for t, da, db, ch, cov in sorted(rows, key=lambda r: -abs(r[3])):
    print('%-8s%+12.4f%+12.4f%+10.4f%9.1f%%' % (t, da, db, ch, 100 * cov))

print('\n=== 4) 2023 vs 2024 만으로 본 팀 격차 안정성 (t13 다음 후보용) ===')
print('%-8s%12s%12s%10s' % ('팀', '2023', '2024', '평균'))
for t in sorted(df.pitcher_team_id.unique()):
    v = []
    for d in (d23, d24):
        f = (d.pitcher_team_id == t) | (d.batter_team_id == t)
        v.append(d[f].control_success.mean() - d[~f].control_success.mean())
    print('%-8s%+12.4f%+12.4f%+10.4f' % (t, v[0], v[1], np.mean(v)))

# w5_* (당해 시즌 복원, 최대 피처군) 를 구간화해 다른 축과 교차 스캔 — 학습 0.
#
# eda52 는 **원본 범주형끼리**만 봤다 (105쌍). 거기서 t13(+20.58)과 phteam(+7.79)이
# 나왔고 나머지는 닫혔다. 아직 안 본 대상이 하나 있다 — 배포 점수의 92점을 지고 있는
# w5_*/wb_* 를 구간화해서 다른 축과 교차한 조합이다.
#
# 방법은 같다: 2차 상호작용 잔차의 강도 x 2023<->2024 안정성.
#   r_ab = mean(y|a,b) - mean(y|a) - mean(y|b) + mean(y)
#   강도 = sum 비중 x r²  ->  401,000 x 강도 = 완벽히 잡았을 때의 점수
#
# 교정점 두 개를 같이 찍는다 (방법이 살아 있는지 확인):
#   pitcher_hand x batter_hand   강도 49.5 / 상관 1.00  -> 이미 모델에 있음
#   pitcher_hand x pitcher_team  강도 38.1 / 상관 0.40  -> 어제 +7.79
#
# ⚠️ 수치형은 트리가 임계값으로 싸게 쪼개므로 범주형 OR 구조만큼 비싸지 않다.
#    즉 같은 강도라도 이득은 더 작을 수 있다. 그래도 강도 x 상관이 phteam 급이면 후보다.
import itertools
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
df = df[df.game_type == 'R'].copy()

# ---- w5_* 복원 (wseason5 와 같은 방식: 시즌 시작 시점 커리어 상태를 뺀다) ----
g = df.sort_values(['pitcher_id', 'asof_pitcher_n'])
f = g.groupby(['pitcher_id', 'season']).first()
df = g.copy()
key = pd.MultiIndex.from_arrays([df.pitcher_id, df.season])
n0 = f['asof_pitcher_n'].reindex(key).to_numpy()
x0 = (f['asof_pitcher_success_rate'] * f['asof_pitcher_n']).round().reindex(key).to_numpy()
df['w_n'] = df.asof_pitcher_n.to_numpy() - n0
df['w_x'] = (df.asof_pitcher_success_rate * df.asof_pitcher_n).round().to_numpy() - x0
df['w_rate'] = (df.w_x + 0.5 * 100) / (df.w_n + 100)      # 리그평균으로 shrink

gb = df.sort_values(['batter_id', 'asof_batter_n'])
fb = gb.groupby(['batter_id', 'season']).first()
kb = pd.MultiIndex.from_arrays([df.batter_id, df.season])
bn0 = fb['asof_batter_n'].reindex(kb).to_numpy()
bx0 = (fb['asof_batter_success_rate'] * fb['asof_batter_n']).round().reindex(kb).to_numpy()
df['wb_n'] = df.asof_batter_n.to_numpy() - bn0
df['wb_rate'] = ((df.asof_batter_success_rate * df.asof_batter_n).round().to_numpy() - bx0
                 + 0.5 * 100) / (df.wb_n + 100)
print('R %s행 | w_rate 중앙 %.4f | wb_rate 중앙 %.4f'
      % (f'{len(df):,}', df.w_rate.median(), df.wb_rate.median()))

# ---- 구간화 (5분위) + 기존 축 ----
for c, s in [('w5q', 'w_rate'), ('wnq', 'w_n'), ('wb5q', 'wb_rate'), ('wbnq', 'wb_n')]:
    df[c] = pd.qcut(df[s], 5, labels=[f'q{i}' for i in range(5)],
                    duplicates='drop').astype(str)
df['cnt12'] = df.balls_before.astype(str) + '-' + df.strikes_before.astype(str)
df['inn3'] = pd.cut(df.inning, [0, 3, 6, 99], labels=['1-3', '4-6', '7+']).astype(str)
df['t13'] = ((df.pitcher_team_id == 13) | (df.batter_team_id == 13)).astype(int)

NEW = ['w5q', 'wnq', 'wb5q', 'wbnq']
OLD = ['cnt12', 'inn3', 'base_state', 'outs_before', 'pitcher_hand', 'batter_hand',
       'pitcher_team_id', 'batter_team_id', 't13', 'top_bottom']
for c in NEW + OLD:
    df[c] = df[c].astype(str)

d23, d24 = df[df.season == 2023], df[df.season == 2024]


def st(d, a, b):
    gm = float(d.control_success.mean())
    ma, mb = d.groupby(a).control_success.mean(), d.groupby(b).control_success.mean()
    c = d.groupby([a, b]).control_success.agg(['mean', 'size'])
    r = (c['mean'] - c.index.get_level_values(0).map(ma).to_numpy()
         - c.index.get_level_values(1).map(mb).to_numpy() + gm)
    return 401000 * float((c['size'].to_numpy() / len(d) * r.to_numpy() ** 2).sum()), r


pairs = [(a, b) for a in NEW for b in NEW + OLD if a != b]
pairs = list({tuple(sorted(p)) for p in pairs})
rows = []
for a, b in pairs:
    s_all, _ = st(df, a, b)
    s3, r3 = st(d23, a, b)
    s4, r4 = st(d24, a, b)
    j = r3.index.intersection(r4.index)
    c = float(np.corrcoef(r3.loc[j], r4.loc[j])[0, 1]) if len(j) > 3 else float('nan')
    rows.append((a, b, s_all, s3, s4, c, len(j)))
rows.sort(key=lambda t: -(t[2] * max(t[5], 0)))

print('\n=== w5_* 교차 조합 (강도 x 안정성 순) ===')
print('%-14s%-18s%9s%8s%8s%8s%6s' % ('축A', '축B', '전체', '2023', '2024', '상관', '셀'))
for a, b, sa, s3, s4, c, n in rows[:14]:
    print('%-14s%-18s%9.1f%8.1f%8.1f%8.2f%6d' % (a, b, sa, s3, s4, c, n))

print('\n=== 교정점 (방법이 살아 있는지) ===')
for a, b in [('pitcher_hand', 'batter_hand'), ('pitcher_hand', 'pitcher_team_id')]:
    sa, _ = st(df, a, b)
    s3, r3 = st(d23, a, b)
    s4, r4 = st(d24, a, b)
    j = r3.index.intersection(r4.index)
    print('%-14s%-18s%9.1f%8.1f%8.1f%8.2f'
          % (a, b, sa, s3, s4, float(np.corrcoef(r3.loc[j], r4.loc[j])[0, 1])))

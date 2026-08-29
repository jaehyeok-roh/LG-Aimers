# eda50 후속 — team13 효과에서 F 교락을 걷어내고 R 안에서만 본다.
#
# eda50 이 두 가지가 겹쳐 있음을 드러냈다:
#   (a) F(퓨처스) 행은 **전부** team13 관여다 -> 시즌별 표가 F 의 .7087 에 오염돼 있었다
#   (b) 그런데 R 만 봐도 2022 -0.041 -> 2023 +0.051 -> 2024 +0.049 로 부호가 뒤집힌다
#
# (b) 가 claude.md 에 없는 새 현상이다. 여기서 확인할 것:
#   1) F 에서 team13 이 왜 100% 인가 (팀 ID 코딩 문제인가)
#   2) R 안에서 시즌 x 월 전환점
#   3) 투수측/타자측 중 어디가 원인인가
#   4) 모델이 이미 잡고 있는가 — pitcher_team_id/batter_team_id 를 갖고 있으니
#      '팀13 여부' 단독이 아니라 **OR 조합 x 시점** 이 새 정보인지가 관건
#   5) 점수 환산
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
C = ['season', 'game_month', 'game_type', 'pitcher_team_id', 'batter_team_id',
     'pitcher_id', 'control_success']
df = pd.read_csv('data/train.csv', usecols=C, keep_default_na=False, na_values=_NA)

print('=== 1) game_type 별 팀 ID 분포 ===')
for gt in ('R', 'F'):
    d = df[df.game_type == gt]
    pt = sorted(d.pitcher_team_id.unique())
    bt = sorted(d.batter_team_id.unique())
    print('  %s  %s행 | pitcher_team %s' % (gt, f'{len(d):,}', pt))
    print('      batter_team  %s' % bt)
    print('      team13 관여 %.4f' % ((d.pitcher_team_id == 13) | (d.batter_team_id == 13)).mean())

R = df[df.game_type == 'R'].copy()
R['t13'] = ((R.pitcher_team_id == 13) | (R.batter_team_id == 13)).astype(int)
R['p13'] = (R.pitcher_team_id == 13).astype(int)
R['b13'] = (R.batter_team_id == 13).astype(int)
print('\nR %s행 | team13 관여 %.2f%% (투수측 %.2f%% / 타자측 %.2f%%)'
      % (f'{len(R):,}', 100 * R.t13.mean(), 100 * R.p13.mean(), 100 * R.b13.mean()))

print('\n=== 2) R 전용 시즌별 ===')
print('%-8s%10s%10s%10s%12s%12s' % ('시즌', '관여', '비관여', '차이', '투수13차이', '타자13차이'))
for s in sorted(R.season.unique()):
    d = R[R.season == s]
    base = d[d.t13 == 0].control_success.mean()
    a = d[d.t13 == 1].control_success.mean()
    pp = d[d.p13 == 1].control_success.mean() - base
    bb = d[d.b13 == 1].control_success.mean() - base
    print('%-8d%10.4f%10.4f%+10.4f%+12.4f%+12.4f' % (s, a, base, a - base, pp, bb))

print('\n=== 3) R 전용 월별 (2022~2024) ===')
print('%-12s%10s%10s%10s%9s' % ('시즌-월', '관여', '비관여', '차이', '관여행수'))
for s in (2022, 2023, 2024):
    for m in sorted(R[R.season == s].game_month.unique()):
        d = R[(R.season == s) & (R.game_month == m)]
        a = d[d.t13 == 1]
        if len(a) < 500:
            continue
        base = d[d.t13 == 0].control_success.mean()
        print('%-12s%10.4f%10.4f%+10.4f%9s'
              % (f'{s}-{m:02d}', a.control_success.mean(), base,
                 a.control_success.mean() - base, f'{len(a):,}'))

print('\n=== 4) 투수 구성 때문인가 — 투수-시즌 평균 제거 후 (R) ===')
R['pdev'] = R.control_success - R.groupby(['pitcher_id', 'season']).control_success.transform('mean')
print('%-8s%14s%14s' % ('시즌', '원시차이', '투수효과제거후'))
for s in sorted(R.season.unique()):
    d = R[R.season == s]
    raw = d[d.t13 == 1].control_success.mean() - d[d.t13 == 0].control_success.mean()
    adj = d[d.t13 == 1].pdev.mean() - d[d.t13 == 0].pdev.mean()
    print('%-8d%+14.4f%+14.4f' % (s, raw, adj))

print('\n=== 5) 점수 환산 (2024 R, 전환 후 체제) ===')
d = R[R.season == 2024]
cov = d.t13.mean()
gap = d[d.t13 == 1].control_success.mean() - d[d.t13 == 0].control_success.mean()
adj = d[d.t13 == 1].pdev.mean() - d[d.t13 == 0].pdev.mean()
for nm, g in (('원시', gap), ('투수효과 제거', adj)):
    v = 401000 * (cov * (g * (1 - cov)) ** 2 + (1 - cov) * (g * cov) ** 2)
    print('  %-14s 격차 %+.4f | 커버 %.1f%% | 완벽히 맞히면 %.1f점' % (nm, g, 100 * cov, v))
print('  ⚠️ 모델은 두 팀 ID 를 이미 갖고 있다. 새 정보는 (OR 조합) x (시점) 뿐이고,')
print('     그건 트리가 3레벨을 써야 만든다 -- nosh(-17.0)/cnt12h(+5.72) 와 같은 자리.')

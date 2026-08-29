# 팀 13 관여 x 2023-05 체제 전환 — 2위가 주최측에 문의한 파생변수를 직접 검증한다.
#
# 배경: 2026-08-29 Q&A 에서 리더보드 2위가 두 파생변수의 합법성을 문의했다.
#   1) team13 = (pitcher_team_id==13) or (batter_team_id==13)          0/1
#   2) switch = team13 x [season>2023 or (season==2023 and (game_month>=5 or game_type=='F'))]
#   둘 다 행 자기 컬럼만 쓰고 전환점은 train 라벨 통계에서 상수로 고정했다고 한다.
#
# 우리 eda48 이 같은 것을 손에 쥐고 놓쳤다. 구장(홈팀)별 편차에서 13 만 t=-8.95 로
# 튀었고 시즌 계열이 -0.0433 -0.0320 -0.0070 -0.0259 **+0.0161 +0.0153** 으로
# 2022->2023 에 부호가 뒤집혔는데, 10개 구장 평균(std 0.0052, 시즌상관 0.21)에
# 묻어서 '불안정하니 ~2점' 으로 닫아버렸다.
#
# ⚠️ 그리고 우리 모델은 이걸 표현할 수 없다: DROP_CAL 로 game_month 를 버렸으므로
#    '2023년 5월 전후' 라는 경계가 아예 없다.
#
# 재는 것:
#   1) 커버리지 (홈경기만이 아니라 관여 전체)
#   2) 시즌 x 월 격자에서 전환점이 정말 2023-05 인가
#   3) game_type 별로 갈리는가 (2위는 2023 F 를 전환 후로 분류했다)
#   4) 다른 팀에도 같은 게 있는가 (13 만 특별한가)
#   5) 점수 환산
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
C = ['season', 'game_month', 'game_type', 'pitcher_team_id', 'batter_team_id',
     'control_success']
df = pd.read_csv('data/train.csv', usecols=C, keep_default_na=False, na_values=_NA)
df['t13'] = ((df.pitcher_team_id == 13) | (df.batter_team_id == 13)).astype(int)
print('%s행 | team13 관여 %.2f%%\n' % (f'{len(df):,}', 100 * df.t13.mean()))

print('=== 1) 시즌별 성공률: team13 관여 vs 비관여 ===')
print('%-8s%10s%10s%10s%10s' % ('시즌', '관여', '비관여', '차이', '관여행수'))
for s in sorted(df.season.unique()):
    d = df[df.season == s]
    a, b = d[d.t13 == 1], d[d.t13 == 0]
    print('%-8d%10.4f%10.4f%+10.4f%10s'
          % (s, a.control_success.mean(), b.control_success.mean(),
             a.control_success.mean() - b.control_success.mean(), f'{len(a):,}'))

print('\n=== 2) 2022~2023 월별 (전환점 확인) ===')
print('%-14s%10s%10s%10s%9s' % ('시즌-월', '관여', '비관여', '차이', '관여행수'))
for s in (2022, 2023):
    for m in sorted(df[df.season == s].game_month.unique()):
        d = df[(df.season == s) & (df.game_month == m)]
        a, b = d[d.t13 == 1], d[d.t13 == 0]
        if len(a) < 500:
            continue
        print('%-14s%10.4f%10.4f%+10.4f%9s'
              % (f'{s}-{m:02d}', a.control_success.mean(), b.control_success.mean(),
                 a.control_success.mean() - b.control_success.mean(), f'{len(a):,}'))

print('\n=== 3) game_type 별 (2위는 2023 F 를 전환 후로 뒀다) ===')
print('%-16s%10s%10s%10s%9s' % ('시즌/유형', '관여', '비관여', '차이', '관여행수'))
for s in (2022, 2023, 2024):
    for gt in ('R', 'F'):
        d = df[(df.season == s) & (df.game_type == gt)]
        a, b = d[d.t13 == 1], d[d.t13 == 0]
        if len(a) < 300:
            continue
        print('%-16s%10.4f%10.4f%+10.4f%9s'
              % (f'{s} {gt}', a.control_success.mean(), b.control_success.mean(),
                 a.control_success.mean() - b.control_success.mean(), f'{len(a):,}'))

print('\n=== 4) 13 만 특별한가 — 팀별 (전환 전 2019~22 vs 후 2023~24) ===')
pre, post = df[df.season <= 2022], df[df.season >= 2023]
print('%-8s%12s%12s%10s' % ('팀', '전(19~22)', '후(23~24)', '변화'))
rows = []
for t in sorted(set(df.pitcher_team_id.unique())):
    fa = ((pre.pitcher_team_id == t) | (pre.batter_team_id == t))
    fb = ((post.pitcher_team_id == t) | (post.batter_team_id == t))
    da = pre[fa].control_success.mean() - pre[~fa].control_success.mean()
    db = post[fb].control_success.mean() - post[~fb].control_success.mean()
    rows.append((t, da, db, db - da))
for t, da, db, d in sorted(rows, key=lambda r: -abs(r[3])):
    print('%-8d%+12.4f%+12.4f%+10.4f' % (t, da, db, d))

print('\n=== 5) 점수 환산 (2024 = 배포에 가장 가까운 해) ===')
d24 = df[df.season == 2024]
cov = d24.t13.mean()
gap = d24[d24.t13 == 1].control_success.mean() - d24[d24.t13 == 0].control_success.mean()
# 전역 평균 보존 시프트: 관여쪽 +gap*(1-cov), 비관여쪽 -gap*cov
print('  커버리지 %.1f%% | 2024 격차 %+.4f' % (100 * cov, gap))
print('  완벽히 맞힌다고 가정: 401,000 x [%.3f x %.5f² + %.3f x %.5f²] = %.1f점'
      % (cov, gap * (1 - cov), 1 - cov, gap * cov,
         401000 * (cov * (gap * (1 - cov)) ** 2 + (1 - cov) * (gap * cov) ** 2)))
print('  ⚠️ 모델은 pitcher_team_id / batter_team_id 를 이미 갖고 있다.')
print('     따라서 이 값은 상한이고, 실제 이득은 "트리가 못 만드는 부분" 뿐이다.')

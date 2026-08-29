# 이적(팀 변경)이 제구에 영향을 주는가 — 학습 0, 순수 pandas.
#
# 왜 이 각도인가:
#   test 행이 담고 있는 2025 정보를 다시 세어보면 asof 19컬럼 말고 하나가 더 있다 —
#   pitcher_team_id / batter_team_id 는 **그 선수의 2025 소속**이다.
#   그리고 타겟 정의가 "포수가 요구한 곳에 던졌는가" 이므로 팀이 바뀌면
#   **요구를 내는 주체(포수진)가 통째로 바뀐다.**
#
#   모델은 이걸 볼 수 없다. pitcher_id 와 pitcher_team_id 를 둘 다 갖고 있지만,
#   2025 에 이적한 투수는 학습에 없던 (투수, 팀) 조합으로 나타날 뿐이고
#   트리는 '바뀌었다' 를 표현할 수단이 없다. 즉 이적 플래그는 다른 행(train)에서
#   끌어오는 **새 정보** 부류다 (wseason5 +61.72 / wsbat +30.11 과 같은 통로).
#
# 재는 것:
#   1) 이적이 얼마나 흔한가 (커버리지). 드물면 크기가 안 나온다.
#   2) 이적 시즌의 제구 변화량 vs 잔류 투수의 변화량 (같은 시즌쌍 안에서 비교)
#   3) 시즌 초반에 더 큰가 (새 포수 적응 가설이 맞다면 그래야 한다)
#   4) 타자측도 같이
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
C = ['season', 'game_type', 'pitcher_id', 'batter_id', 'pitcher_team_id',
     'batter_team_id', 'control_success', 'asof_pitcher_n']
df = pd.read_csv('data/train.csv', usecols=C, keep_default_na=False, na_values=_NA)
df = df[df.game_type == 'R'].copy()
df['w_n'] = df.asof_pitcher_n - df.groupby(['pitcher_id', 'season']).asof_pitcher_n.transform('min')
print('R %s행\n' % f'{len(df):,}')


def analyze(ent, team, label, minn=200):
    # (선수, 시즌) 별 주 소속팀 + 성적
    g = (df.groupby([ent, 'season'])
           .agg(rate=('control_success', 'mean'), n=('control_success', 'size'),
                team=(team, lambda s: s.mode().iat[0]),
                nteam=(team, 'nunique')).reset_index())
    g = g[g.n >= minn]
    prev = g.copy()
    prev['season'] = prev['season'] + 1
    m = g.merge(prev[[ent, 'season', 'rate', 'team', 'n']],
                on=[ent, 'season'], suffixes=('', '_p'))
    m['moved'] = (m['team'] != m['team_p']).astype(int)
    m['d'] = m['rate'] - m['rate_p']
    # 그 시즌의 리그 변화량을 빼서 드리프트 제거
    lg = df.groupby('season').control_success.mean()
    m['d_adj'] = m['d'] - (m['season'].map(lg) - (m['season'] - 1).map(lg))

    print('=== %s ===' % label)
    print('  비교 가능한 (선수,시즌쌍) %s개 | 이적 %d개 (%.1f%%)'
          % (f'{len(m):,}', m.moved.sum(), 100 * m.moved.mean()))
    a = m[m.moved == 1]['d_adj']
    b = m[m.moved == 0]['d_adj']
    if len(a) < 10:
        print('  이적 표본 부족\n'); return m
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print('  드리프트 보정 변화량:  이적 %+.5f (n=%d)  잔류 %+.5f (n=%d)'
          % (a.mean(), len(a), b.mean(), len(b)))
    print('  차이 %+.5f  (표준오차 %.5f, t=%.2f)' % (a.mean() - b.mean(), se,
                                                 (a.mean() - b.mean()) / se))
    print('  시즌 중 다팀(트레이드) %d개 (%.1f%%)\n'
          % ((m.nteam > 1).sum(), 100 * (m.nteam > 1).mean()))
    return m


mp = analyze('pitcher_id', 'pitcher_team_id', '투수 이적')
mb = analyze('batter_id', 'batter_team_id', '타자 이적', minn=200)

# ---- 3) 시즌 초반에 더 큰가 (새 포수 적응 가설) ----
mv = mp[mp.moved == 1][['pitcher_id', 'season']].assign(moved=1)
d = df.merge(mv, on=['pitcher_id', 'season'], how='left')
d['moved'] = d['moved'].fillna(0)
# 직전 시즌 성적이 있는 투수만 (이적/잔류 비교가 성립하는 모집단)
have = mp[['pitcher_id', 'season']].assign(cmp=1)
d = d.merge(have, on=['pitcher_id', 'season'], how='inner')
d['bucket'] = pd.cut(d.w_n, [-1, 150, 400, 900, 1e9],
                     labels=['0-150', '150-400', '400-900', '900+'])
print('=== 이적 투수의 시즌 진행별 성공률 (2020~2024, 직전시즌 있는 투수만) ===')
print('%-10s%12s%12s%10s%10s' % ('당해투구수', '이적', '잔류', '차이', '이적행수'))
for bk in ['0-150', '150-400', '400-900', '900+']:
    s = d[d.bucket == bk]
    a, b = s[s.moved == 1], s[s.moved == 0]
    if len(a) < 500:
        print('%-10s  표본 부족 (%d)' % (bk, len(a))); continue
    print('%-10s%12.4f%12.4f%+10.4f%10s'
          % (bk, a.control_success.mean(), b.control_success.mean(),
             a.control_success.mean() - b.control_success.mean(), f'{len(a):,}'))

print('\n=== 2024 커버리지 (2025 에서 쓸 수 있는 크기의 대리) ===')
m24 = mp[mp.season == 2024]
mv24 = set(m24[m24.moved == 1].pitcher_id)
d24 = df[df.season == 2024]
print('  이적 투수 %d명 / 비교가능 %d명' % (len(mv24), len(m24)))
print('  그들이 던진 비중 %.2f%%' % (100 * d24.pitcher_id.isin(mv24).mean()))

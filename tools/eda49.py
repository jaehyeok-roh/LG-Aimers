# 남은 격차 중 '원리적으로 도달 불가능한 몫' 을 직접 잰다 — 학습 0.
#
# group_oof: 무작위 fold 로 2024 행을 예측하면 1,092, 투수x시즌 그룹으로 빼면 791.
# 현재 홀드아웃은 914 이므로 **아직 178 이 남아 있다.**
#
# 무작위 fold 가 갖고 우리가 못 갖는 것은 하나다 — 그 투수의 **당해 시즌 나머지 투구**,
# 즉 이 투구 이후의 기록. test 에서는 asof 가 '직전까지' 라 원리적으로 없다.
#
# 그 몫이 얼마인지 재면 남은 178 을 쫓을 가치가 있는지 판정된다:
#   (a) w5_success  = 시즌 진행분까지의 성공률   <- 우리가 가진 것
#   (b) 시즌 전체 성공률 (그 투구 제외)          <- 무작위 fold 가 사실상 가진 것
#   둘의 단독 예측력 차이 = 도달 불가능한 몫
#
# 둘 다 같은 shrink(C=100, 리그평균으로)를 적용해 공정하게 비교한다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
C = ['season', 'game_type', 'pitcher_id', 'control_success',
     'asof_pitcher_n', 'asof_pitcher_success_rate']
df = pd.read_csv('data/train.csv', usecols=C, keep_default_na=False, na_values=_NA)
df = df[df.season == 2024].copy()
r = float(df.control_success.mean())
U = r * (1 - r)
print('2024 %s행 | r=%.4f\n' % (f'{len(df):,}', r))

g = df.sort_values(['pitcher_id', 'asof_pitcher_n'])
# 시즌 시작 시점 커리어 상태 -> 당해 시즌 진행분 복원 (wseason5 와 같은 방식)
first = g.groupby('pitcher_id').first()
n0 = first['asof_pitcher_n']
x0 = (first['asof_pitcher_success_rate'] * first['asof_pitcher_n']).round()
g['w_n'] = g['asof_pitcher_n'] - g['pitcher_id'].map(n0)
g['w_x'] = (g['asof_pitcher_success_rate'] * g['asof_pitcher_n']).round() - g['pitcher_id'].map(x0)

# (b) 시즌 전체 (leave-one-out: 자기 투구는 뺀다)
tot_n = g.groupby('pitcher_id')['control_success'].transform('size')
tot_x = g.groupby('pitcher_id')['control_success'].transform('sum')
g['f_n'] = tot_n - 1
g['f_x'] = tot_x - g['control_success']

C_SHRINK = 100.0


def skill(p):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - g.control_success) ** 2).mean() / U) * 100000


def shrunk(x, n):
    return (x + r * C_SHRINK) / (n + C_SHRINK)


print('%-34s%12s%12s' % ('단독 예측자', '스킬', '가중평균 n'))
print('%-34s%12.0f' % ('상수 r', skill(np.full(len(g), r))))
p_a = shrunk(g.w_x, g.w_n)
p_b = shrunk(g.f_x, g.f_n)
p_c = shrunk((g.asof_pitcher_success_rate * g.asof_pitcher_n).round(), g.asof_pitcher_n)
print('%-34s%12.0f%12.0f' % ('(c) 커리어 누적 (운영측 원본)', skill(p_c), g.asof_pitcher_n.mean()))
print('%-34s%12.0f%12.0f' % ('(a) 당해 진행분  = 우리가 가진 것', skill(p_a), g.w_n.mean()))
print('%-34s%12.0f%12.0f' % ('(b) 당해 시즌 전체 = 무작위fold', skill(p_b), g.f_n.mean()))
print('\n  (b) - (a) = %+.0f  <- 이 투구 이후를 아는 것의 값어치 (도달 불가능)'
      % (skill(p_b) - skill(p_a)))
print('  (a) - (c) = %+.0f  <- wseason5 이 실제로 회수한 몫' % (skill(p_a) - skill(p_c)))

# 진행도별로 쪼개보면 도달 불가능 몫이 어디에 몰려 있는지 보인다
print('\n=== 시즌 진행도별 (a) vs (b) ===')
print('%-12s%9s%11s%11s%9s' % ('당해투구수', '비중', '(a)진행분', '(b)전체', '차이'))
g['bk'] = pd.cut(g.w_n, [-1, 100, 400, 900, 1e9], labels=['0-100', '100-400', '400-900', '900+'])
for bk in ['0-100', '100-400', '400-900', '900+']:
    m = (g.bk == bk).to_numpy()
    if m.sum() < 1000:
        continue
    sa = (1 - ((p_a[m] - g.control_success[m]) ** 2).mean() / U) * 100000
    sb = (1 - ((p_b[m] - g.control_success[m]) ** 2).mean() / U) * 100000
    print('%-12s%8.1f%%%11.0f%11.0f%+9.0f' % (bk, 100 * m.mean(), sa, sb, sb - sa))

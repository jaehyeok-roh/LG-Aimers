# li / win_expectancy 가 상황의 **순수 함수**인가, 아니면 바깥 정보를 담는가. 학습 0.
#
#   python tools/eda66.py
#
# 이 세 컬럼은 원본이 아니라 **계산된 값**이다.
#   li, home_win_expectancy, away_win_expectancy
# 보통 (이닝, 공수, 아웃, 주자, 점수차)의 함수로 만든다. 그렇다면 트리가 스스로
# 만들 수 있으므로 이미 죽은 정보다 (claude.md: 한 행 안 파생은 3번 다 0).
#
# ⭐ 그런데 그 계산에 **팀 전력이나 그 시즌 득점환경**이 들어가면 이야기가 다르다.
#    2025 test 행의 그 값은 **2025 팀 전력**을 담게 되고, 그건 우리가 다른 어떤
#    경로로도 못 얻는 정보다 (train 은 2019~24 뿐). `wseason5` 와 같은 부류 --
#    '평가 행이 몰래 갖고 있는 당해 시즌 정보' 다.
#
# 검사: 상황 조합의 **각 칸 안에서** 값이 유일한가. 유일하면 순수 함수 = 죽은 정보.
# 흩어져 있으면 그 분산이 어디서 오는지(팀/시즌/월) 본다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
USE = ['season', 'game_month', 'inning', 'top_bottom', 'outs_before', 'base_state',
       'score_diff_pitcher_team', 'li', 'home_win_expectancy', 'away_win_expectancy',
       'pitcher_team_id', 'batter_team_id', 'balls_before', 'strikes_before',
       'game_type', 'control_success']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA, usecols=USE)
print('%s행' % format(len(df), ','))
print('win_exp 합 = 1 인가: 최대 |h+a-1| = %.6f'
      % float((df.home_win_expectancy + df.away_win_expectancy - 1).abs().max()))

# 상황 키: 게임 상태를 결정하는 컬럼 전부
KEY = ['inning', 'top_bottom', 'outs_before', 'base_state', 'score_diff_pitcher_team']
KEY2 = KEY + ['balls_before', 'strikes_before']

for name, keys in (('상황 (이닝/공수/아웃/주자/점수차)', KEY),
                   ('상황 + 볼카운트', KEY2)):
    print('\n=== %s 안에서의 유일성 ===' % name)
    g = df.groupby(keys, observed=True)
    for c in ('li', 'home_win_expectancy'):
        st = g[c].agg(['nunique', 'std', 'size'])
        st = st[st['size'] >= 50]
        w = st['size'] / st['size'].sum()
        print('  %-22s 칸 %s개 | 칸당 고유값 중앙 %.0f | 가중평균 std %.5f'
              % (c, format(len(st), ','), st['nunique'].median(),
                 float((st['std'].fillna(0) * w).sum())))

print('''
  칸당 고유값이 1 이고 std 가 0 이면 **순수 함수**다 -> 트리가 만들 수 있고 죽은 정보.
  흩어져 있으면 바깥 정보가 섞여 있다는 뜻이다.''')

# ---- 흩어져 있다면: 그 분산이 무엇으로 설명되는가 ----
print('\n=== 잔차가 무엇으로 설명되는가 (상황 칸 평균을 뺀 뒤) ===')
key = KEY
for c in ('li', 'home_win_expectancy'):
    v = df[c].to_numpy(dtype='float64')
    m = df.groupby(key, observed=True)[c].transform('mean').to_numpy()
    r = v - m
    if np.nanstd(r) < 1e-9:
        print('  %-22s 잔차 없음 (순수 함수)' % c)
        continue
    print('  %-22s 잔차 std %.5f (원본 std %.5f)' % (c, np.nanstd(r), np.nanstd(v)))
    for nm, col in (('season', df.season), ('season x 홈팀', None),
                    ('game_month', df.game_month), ('game_type', df.game_type)):
        if col is None:
            home = np.where(df.top_bottom.astype(str).str[0] == 'T',
                            df.pitcher_team_id, df.batter_team_id)
            col = pd.Series(df.season.astype(str) + '_' + pd.Series(home).astype(str))
        gm = pd.Series(r).groupby(col.to_numpy(), observed=True).transform('mean').to_numpy()
        print('      %-16s 설명 분산 %.4f' % (nm, 1 - np.nanvar(r - gm) / np.nanvar(r)))

print('''
  'season x 홈팀' 이 크게 설명하면 그 컬럼은 **팀-시즌 전력**을 담고 있다는 뜻이고,
  2025 test 행은 우리가 다른 경로로 못 얻는 **2025 팀 전력**을 갖고 있는 셈이다.''')

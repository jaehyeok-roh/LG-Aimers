# EDA 32 — 투수 x 상황 축 중에 아직 안 판 것이 있는가 (학습 없이 판정).
#
# 주최측 문제 소개 자료가 '코치의 네 가지 질문' 을 든다:
#   주자 상황 / 볼카운트 / 경기 후반 / 최근 흐름
# 우리가 조건부 통계(cond_*)로 만든 것은 **카운트와 좌우뿐**이다.
# 주자·이닝 축은 한 번도 안 만들었다 (claude.md 에 '미검증' 으로 남아 있다).
#
# 새 피처를 만들기 전에 **그 축에 진짜 신호가 있는지**부터 잰다.
# 방법: 반쪽 분할 신뢰도(split-half reliability).
#   각 투수의 투구를 무작위로 반 갈라, 양쪽에서 '상황별 편차' 를 따로 계산하고 상관을 본다.
#   상관이 0 이면 그 편차는 전부 표본 잡음이다 -> 피처로 만들 값어치가 없다.
#   상관이 크면 투수마다 실재하는 성향이다 -> cond_* 슬라이스로 만들 값어치가 있다.
#
# 교정점으로 **투수 x 카운트**(cond_pc, 실측 +27)를 같이 잰다.
# 그것보다 신뢰도가 크게 낮으면 그 축은 열지 않는다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
COLS = ['season', 'game_type', 'pitcher_id', 'control_success', 'balls_before',
        'strikes_before', 'inning', 'base_state', 'runner_on_1b', 'runner_on_2b',
        'runner_on_3b', 'num_runners_on', 'li', 'score_diff_pitcher_team',
        'outs_before', 'batter_hand']
tr = pd.read_csv('data/train.csv', usecols=COLS, keep_default_na=False, na_values=_NA)
print(f'{len(tr):,}행\n')

# 시즌 x game_type 리그평균으로 디트렌드 (F 체제 변경 반영)
tr['dev'] = tr['control_success'] - tr.groupby(
    ['season', 'game_type'])['control_success'].transform('mean')

b, s = tr['balls_before'], tr['strikes_before']
pa = ((b == 0) & (s == 1)) | ((b == 0) & (s == 2)) | ((b == 1) & (s == 2))
ba = (((b == 1) & (s == 0)) | ((b == 2) & (s == 0)) | ((b == 3) & (s == 0))
      | ((b == 2) & (s == 1)) | ((b == 3) & (s == 1)))
nu = ((b == 1) & (s == 1)) | ((b == 2) & (s == 2))
tr['count_adv'] = np.select([pa, ba, nu], ['P', 'B', 'N'], default='X')

AXES = {
    'count_adv (교정점, cond_pc = +27)': tr['count_adv'],
    'batter_hand (교정점, cond_ph)': tr['batter_hand'].astype(str),
    '주자 유무': np.where(tr['num_runners_on'] > 0, 'on', 'empty'),
    '주자 상태 (base_state)': tr['base_state'].astype(str),
    'RISP': np.where((tr['runner_on_2b'] > 0) | (tr['runner_on_3b'] > 0), 'risp', 'no'),
    '이닝 구간': pd.cut(tr['inning'], [0, 3, 6, 99], labels=['1-3', '4-6', '7+']).astype(str),
    '아웃카운트': tr['outs_before'].astype(str),
    '압박(li) 3분위': pd.qcut(tr['li'].rank(method='first'), 3,
                             labels=['low', 'mid', 'high']).astype(str),
    '점수차 구간': pd.cut(tr['score_diff_pitcher_team'], [-99, -4, -1, 1, 4, 99],
                        labels=['크게뒤', '뒤', '접전', '앞', '크게앞']).astype(str),
}

rng = np.random.default_rng(0)
tr['half'] = rng.integers(0, 2, len(tr))
MIN_N = 40          # 셀당 최소 투구수 (반쪽 기준)

print(f'{"축":<34}{"셀":>6}{"반쪽신뢰도":>11}{"편차 std":>11}{"신호 std":>10}')
print('-' * 74)
out = []
for name, key in AXES.items():
    d = tr.assign(k=np.asarray(key))
    g = d.groupby(['pitcher_id', 'k', 'half'])['dev'].agg(['mean', 'size'])
    g = g[g['size'] >= MIN_N]['mean'].unstack('half')
    g = g.dropna()
    if len(g) < 200:
        print(f'{name:<34}{len(g):>6}    표본 부족')
        continue
    # 투수 주효과를 빼야 '상황' 축만 남는다
    a = g[0] - g[0].groupby(level=0).transform('mean')
    c = g[1] - g[1].groupby(level=0).transform('mean')
    ok = a.notna() & c.notna()
    r = float(np.corrcoef(a[ok], c[ok])[0, 1])
    obs = float(np.std(np.concatenate([a[ok], c[ok]])))
    sig = obs * np.sqrt(max(r, 0.0))       # 신뢰도로 잡음을 걷어낸 진짜 신호 크기
    out.append((name, r, sig))
    print(f'{name:<34}{ok.sum():>6}{r:>11.3f}{obs:>11.4f}{sig:>10.4f}')

print('\n' + '=' * 74)
if out:
    cal = next((x for x in out if x[0].startswith('count_adv')), None)
    if cal:
        print(f'교정점 count_adv: 신뢰도 {cal[1]:.3f} / 신호 std {cal[2]:.4f} '
              f'(이게 리더보드 +27 짜리다)')
        print()
        for name, r, sig in sorted(out, key=lambda z: -z[2]):
            if name.startswith(('count_adv', 'batter_hand')):
                continue
            v = ('★ 파볼 만함' if sig > cal[2] * 0.7 else
                 '△ 절반 수준' if sig > cal[2] * 0.35 else '✕ 무시')
            print(f'  {name:<32}신호 {sig:.4f}  = 교정점의 {sig/cal[2]:.0%}   {v}')
print("""
읽는 법: '신호 std' 는 반쪽 신뢰도로 잡음을 걷어낸 **진짜 투수x상황 편차**의 크기다.
  cond_pc(카운트)가 리더보드 +27 을 냈으므로, 그 대비 비율이 그대로 기대치의 눈금이 된다.
  ⚠️ 다만 축들끼리 겹칠 수 있다 (주자 유무 ~ li ~ 점수차). 겹침은 여기서 안 잰다.""")

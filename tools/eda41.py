# EDA 41 — 실패 유형을 '보조 타겟' 으로 쓸 근거가 있는가.
#
# 복원은 eda40 에서 확정됐다 (success 검산 일치율 1.000000, 147만 행).
# 여기서 묻는 것은 **왜 유형별로 나누면 좋아지는가** 의 기전이다.
#
# 기본 우려: middle/reverse 는 같은 X 의 함수 + 잡음이라 E[y|X] 를 바꾸지 않는다.
#   이득 경로가 '정보 추가' 가 아니라 '추정 효율' 이고, 우리는 거기 안 묶여 있다
#   (in-sample 2.232% ~= OOF 2.21%). 증류가 죽은 이유와 같다.
#
# 그 벽을 우회하는 판 두 가지를 잰다:
#   (A) 유형마다 **드리프트가 다른가**. 다르면 유형별 모델링은 추정 효율이 아니라
#       **구조** 문제가 된다 — 재중심화가 134점을 지고 있는 그 축이다.
#   (B) 유형마다 **투수 수준 신뢰도가 다른가**. middle 이 success 보다 신뢰도가
#       높으면 그쪽 모델이 더 날카롭다 = 합쳐서 이득이 날 여지.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
R = ['success', 'middle', 'reverse', 'ball', 'strike']
t = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA,
                usecols=['season', 'game_type', 'pitcher_id', 'control_success',
                         'asof_pitcher_n'] + [f'asof_pitcher_{k}_rate' for k in R])
o = np.lexsort((t['asof_pitcher_n'].to_numpy(), t['pitcher_id'].to_numpy()))
t = t.iloc[o].reset_index(drop=True)
n = t['asof_pitcher_n'].to_numpy(dtype='float64')
g = t['pitcher_id'].to_numpy()
ok = np.r_[(g[:-1] == g[1:]) & (n[1:] - n[:-1] == 1), False]
for k in R:
    cum = np.round(t[f'asof_pitcher_{k}_rate'].fillna(0).to_numpy(dtype='float64') * n)
    d = np.r_[cum[1:] - cum[:-1], np.nan]
    t[k] = np.where(ok & np.isin(d, [0.0, 1.0]), d, np.nan)
assert (t.loc[ok, 'success'] == t.loc[ok, 'control_success']).all(), '복원 검산 실패'
# ⚠️ middle 과 reverse 는 **배타적이 아니다** (둘 다 1인 행이 실패의 7.2%).
# 포수가 바깥에 앉았는데 가운데-몸쪽으로 몰리면 '몰림' 이면서 '반대방향' 이다.
# 따라서 wild = 실패 & !middle & !reverse 로 정의해야 한다 (1-s-m-r 은 틀렸다).
t['both'] = np.where(t['success'].notna(), t['middle'] * t['reverse'], np.nan)
t['wild'] = np.where(t['success'].notna(),
                     (1 - t['success']) * (1 - t['middle']) * (1 - t['reverse']), np.nan)
t['inplay'] = np.where(t['ball'].notna(), 1 - t['ball'] - t['strike'], np.nan)
d = t[ok].copy()
print(f'복원 {len(d):,}행 | wild 음수 {int((d["wild"]<0).sum())} | inplay 음수 {int((d["inplay"]<0).sum())}')

TY = ['middle', 'reverse', 'both', 'wild']
print('\n' + '=' * 78)
print('(A) 유형별 드리프트 — R(1군) 전용, 이게 이 아이디어의 핵심 기전이다')
print(f'{"시즌":<8}{"실패율":>9}' + ''.join(f'{k:>10}' for k in TY)
      + ''.join(f'{k:>9}' for k in ['ball', 'strike']))
print('-' * 78)
r = d[d.game_type == 'R']
tab = r.groupby('season')[['success'] + TY + ['ball', 'strike']].mean()
for s, row in tab.iterrows():
    print(f'{s:<8}{1-row["success"]:>9.4f}'
          + ''.join(f'{row[k]:>10.4f}' for k in TY)
          + ''.join(f'{row[k]:>9.4f}' for k in ['ball', 'strike']))
f0, f1 = tab.iloc[0], tab.iloc[-1]
print('-' * 78)
tot = (1 - f1['success']) - (1 - f0['success'])
print(f'2019 -> 2024 실패율 변화 {tot:+.4f}  = 유형별 분해:')
for k in TY:
    dd = f1[k] - f0[k]
    print(f'    {k:<10}{dd:+.4f}   (전체 변화의 {100*dd/tot:>5.1f}%)')
print(f'    ball {f1["ball"]-f0["ball"]:+.4f} | strike {f1["strike"]-f0["strike"]:+.4f}')

print('\n' + '=' * 78)
print('(B) 유형별 투수 수준 신뢰도 — 반쪽 분할 (eda33 과 같은 방법)')
print(f'{"타겟":<12}{"신뢰도":>9}{"신호 std":>11}{"평균":>9}   해석')
print('-' * 78)
rng = np.random.default_rng(0)
half = rng.random(len(d)) < 0.5
for k in ['success'] + TY + ['ball', 'strike']:
    v = d[k].to_numpy()
    m = ~np.isnan(v)
    a = pd.DataFrame({'p': d['pitcher_id'].to_numpy()[m], 'v': v[m], 'h': half[m]})
    g1 = a[a.h].groupby('p')['v'].agg(['mean', 'size'])
    g2 = a[~a.h].groupby('p')['v'].agg(['mean', 'size'])
    j = g1.join(g2, lsuffix='1', rsuffix='2', how='inner')
    j = j[(j['size1'] >= 100) & (j['size2'] >= 100)]
    w = (j['size1'] + j['size2']).to_numpy(dtype='float64')
    x, y = j['mean1'].to_numpy(), j['mean2'].to_numpy()
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    cv = np.average((x - xm) * (y - ym), weights=w)
    vx = np.average((x - xm) ** 2, weights=w)
    vy = np.average((y - ym) ** 2, weights=w)
    rel = float(cv / np.sqrt(vx * vy))
    sig = float(np.sqrt(max(cv, 0.0)))          # 잡음 제거한 진짜 신호 std
    note = ''
    if k == 'success':
        note = '<- 교정점'
        base = sig
    else:
        note = f'신호가 success 의 {100*sig/base:.0f}%'
    print(f'{k:<12}{rel:>9.3f}{sig:>11.4f}{np.nanmean(v):>9.4f}   {note}')
print(f'\n투수 {len(j):,}명 (양쪽 100구+)')
print("""
읽는 법:
  (A) 유형별 변화율이 **크게 다르면** -> 유형별 모델링은 구조 문제다. 해볼 값어치.
      다 같은 비율로 움직이면 -> 이진 타겟이 이미 다 담고 있다.
  (B) 어떤 유형의 신호가 success 보다 **크면** 그 유형 모델이 더 날카롭다.
⚠️ 대리지표다 (오늘까지 4연패). 판정은 스크리너/리더보드로만.""")

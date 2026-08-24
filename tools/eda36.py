# EDA 36 — 혹사와 경험: 궤적(변화의 방향)에 신호가 있는가.
#
# eda34(상대 타자 질) · eda35(상황 믹스)는 둘 다 **수준의 편향**을 찾다가 닫혔다.
# 이번은 성격이 다르다 — **변화의 방향**이다:
#   가설 A (혹사)   이번 시즌 많이 던진 투수는 다음 시즌 나빠진다
#   가설 B (경험)   신인은 커리어가 쌓일수록 좋아진다
#
# ★ 모델이 이걸 못 만든다는 것이 핵심이다.
#   2025 test 행이 가진 것: 그 투수의 **올해** 투구수(w_n), 커리어 총량(asof_pitcher_n).
#   없는 것: **작년에 얼마나 던졌는지**. 그건 (2024말 커리어) - (2023말 커리어) 라
#   룩업이 두 개 필요하다 — wseason 과 같은 부류의 '다른 행에서 끌어온 정보' 다.
#
# 판정은 eda34/35 와 같은 기준: **다음 시즌 편차를 더 잘 맞히는가.**
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
tr = pd.read_csv('data/train.csv',
                 usecols=['season', 'game_type', 'pitcher_id', 'control_success'],
                 keep_default_na=False, na_values=_NA)
tr['dev'] = tr['control_success'] - tr.groupby(
    ['season', 'game_type'])['control_success'].transform('mean')

g = tr.groupby(['pitcher_id', 'season']).agg(
    dev=('dev', 'mean'), n=('dev', 'size')).reset_index()
g = g.sort_values(['pitcher_id', 'season'])
gg = g.groupby('pitcher_id')
g['sidx'] = gg.cumcount() + 1                       # 그 투수의 몇 번째 시즌인가
g['career_n'] = gg['n'].cumsum() - g['n']           # 그 시즌 **이전까지**의 커리어 투구수
_w = (g['dev'] * g['n']).groupby(g['pitcher_id']).cumsum() - g['dev'] * g['n']
_nn = gg['n'].cumsum() - g['n']
g['past_dev'] = np.where(_nn > 0, _w / _nn.replace(0, np.nan), np.nan)
# 다음 시즌 값
g['next_dev'] = gg['dev'].shift(-1)
g['next_season'] = gg['season'].shift(-1)
g['gap'] = g['next_season'] - g['season']

d = g[(g['gap'] == 1) & (g['n'] >= 200) & g['next_dev'].notna()].copy()
print(f'연속 두 시즌 쌍 {len(d):,}개 (그 시즌 200구+)')
print(f'  이번 시즌 투구수  중앙 {d["n"].median():.0f}  범위 '
      f'{d["n"].quantile(.05):.0f}~{d["n"].quantile(.95):.0f}')
print(f'  편차 std {d["dev"].std():.4f} | 다음 시즌 편차 std {d["next_dev"].std():.4f}\n')


def partial(y, x, ctrl):
    """ctrl 을 뺀 뒤 x 와 y 의 부분상관."""
    A = np.column_stack([np.asarray(c, dtype='float64') for c in ctrl]
                        + [np.ones(len(y))])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


print('=' * 68)
print('가설 A — 이번 시즌 혹사가 다음 시즌 하락을 부르는가')
print('-' * 68)
y = d['next_dev'].to_numpy()
dev = d['dev'].to_numpy()
logn = np.log(d['n'].to_numpy())
print(f'  투구수 ~ 이번 시즌 편차   상관 {np.corrcoef(logn, dev)[0,1]:+.3f}  '
      '(좋은 투수가 많이 던진다)')
print(f'  ★ 투구수 -> 다음 시즌 편차 (이번 시즌 편차 통제)  '
      f'부분상관 {partial(y, logn, [dev]):+.4f}')
print(f'    + 커리어 투구수까지 통제                      '
      f'부분상관 {partial(y, logn, [dev, np.log1p(d["career_n"])]):+.4f}')

print()
print('=' * 68)
print('가설 B — 경험이 쌓이면 좋아지는가 (투수 고정효과 안에서)')
print('-' * 68)
e = g[g['n'] >= 200].copy()
e['dev_c'] = e['dev'] - e.groupby('pitcher_id')['dev'].transform('mean')  # 투수 평균 제거
for lab, col in [('시즌 순번 (1,2,3...)', 'sidx'),
                 ('커리어 투구수 log', None)]:
    x = (np.log1p(e['career_n'].to_numpy()) if col is None
         else e[col].to_numpy(dtype='float64'))
    ok = np.isfinite(x)
    r = float(np.corrcoef(x[ok], e['dev_c'].to_numpy()[ok])[0, 1])
    print(f'  {lab:<24} 투수내 상관 {r:+.4f}')
print(f'  (참고) 1시즌차 평균편차 {e[e.sidx==1]["dev"].mean():+.4f} | '
      f'2시즌차 {e[e.sidx==2]["dev"].mean():+.4f} | '
      f'3+시즌차 {e[e.sidx>=3]["dev"].mean():+.4f}')

print()
print('=' * 68)
print('★ 판정 — 다음 시즌 편차 예측이 좋아지는가 (eda34/35 와 같은 기준)')
print('-' * 68)
dd = d[d['past_dev'].notna() & (d['career_n'] >= 300)].copy()
yy = dd['next_dev'].to_numpy()
base = [dd['past_dev'].to_numpy(), dd['dev'].to_numpy()]
sets = [('과거편차 + 이번시즌편차 (기준)', base),
        ('+ 이번시즌 투구수', base + [np.log(dd['n'].to_numpy())]),
        ('+ 커리어 투구수', base + [np.log1p(dd['career_n'].to_numpy())]),
        ('+ 시즌 순번', base + [dd['sidx'].to_numpy(dtype='float64')]),
        ('+ 셋 다', base + [np.log(dd['n'].to_numpy()),
                            np.log1p(dd['career_n'].to_numpy()),
                            dd['sidx'].to_numpy(dtype='float64')])]
print(f'{"구성":<34}{"R2":>9}{"기준 대비":>11}')
b0 = None
for lab, cols in sets:
    A = np.column_stack([np.asarray(c, dtype='float64') for c in cols]
                        + [np.ones(len(yy))])
    p = A @ np.linalg.lstsq(A, yy, rcond=None)[0]
    r2 = float(np.corrcoef(p, yy)[0, 1] ** 2)
    if b0 is None:
        b0 = r2
    print(f'{lab:<34}{r2:>9.4f}{r2-b0:>+11.4f}')
print(f'\n표본 {len(dd):,} 쌍')
print("""
읽는 법: R2 증분이 +0.01 을 넘으면 궤적 신호가 실재한다 -> 피처로 만들 값어치.
  0 근처면 '혹사/경험' 은 이미 커리어 편차 안에 녹아 있거나 존재하지 않는다.
⚠️ 대리지표다. 양수여도 스크리너/리더보드로 판정한다.""")

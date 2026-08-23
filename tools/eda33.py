# EDA 33 — 정보원 천장 지도. 아직 안 판 축이 있는가 (학습 없이).
#
# 지금까지 이득은 전부 **'다른 행에서 정보를 끌어온' 축**에서 나왔다:
#   wseason5(투수 당해시즌) +61.72 / wsbat(타자 당해시즌) +30.11 / cond_p 가족 +27
# 그리고 그 주력 광맥(asof 커리어 누적 분해)은 이제 말랐다 — 분해 가능한 10개 중
# 투수 5 · 타자 2 를 다 썼고 구종 3 은 0 이었다 (eda10).
#
# 그래서 **남은 축이 있는지**를 한 눈금 위에 올려 본다.
# 방법: 반쪽 분할 신뢰도. 각 그룹의 편차를 무작위 반쪽 두 개에서 따로 재고 상관을 본다.
#   상관이 0 이면 그 편차는 전부 표본 잡음 -> 피처로 만들 값어치가 없다.
#   신호 std = 관측 std x sqrt(신뢰도)  <- 잡음을 걷어낸 진짜 크기
#
# ★ 핵심: **부모 효과를 먼저 뺀다.** 그래야 '증분' 을 잰다.
#   예) pitcher_team 은 투수 효과를 빼야 '팀(포수·전력분석) 고유' 가 남는다.
#
# 교정점 두 개를 같이 재서 눈금을 만든다:
#   pitcher x season (= wseason5 가 캐는 축, 리더보드 +61.72)
#   batter  x season (= wsbat  이 캐는 축, 리더보드 +30.11)
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
COLS = ['season', 'game_month', 'game_type', 'pitcher_id', 'batter_id',
        'pitcher_team_id', 'batter_team_id', 'pitcher_hand', 'batter_hand',
        'inning', 'control_success']
tr = pd.read_csv('data/train.csv', usecols=COLS, keep_default_na=False, na_values=_NA)
print(f'{len(tr):,}행 | 투수 {tr.pitcher_id.nunique():,} 타자 {tr.batter_id.nunique():,}\n')

tr['dev'] = tr['control_success'] - tr.groupby(
    ['season', 'game_type'])['control_success'].transform('mean')
rng = np.random.default_rng(0)
tr['h'] = rng.integers(0, 2, len(tr))
tr['half_season'] = np.where(tr['game_month'] <= 6, 'H1', 'H2')


def reliab(keys, parent=None, min_n=40, label=''):
    """keys 그룹의 편차를 반쪽 분할로 잰다. parent 가 있으면 그 효과를 먼저 뺀다."""
    d = tr
    v = d['dev'].to_numpy(dtype='float64')
    if parent:
        v = v - d.groupby(parent)['dev'].transform('mean').to_numpy(dtype='float64')
    g = pd.DataFrame({'v': v, 'h': d['h']})
    for k in keys:
        g[k] = d[k].to_numpy()
    a = g.groupby(keys + ['h'])['v'].agg(['mean', 'size'])
    ok = a[a['size'] >= min_n]
    cov = float(ok['size'].sum()) / len(d)          # ★ 통과한 칸이 덮는 행 비율
    a = ok['mean'].unstack('h').dropna()
    if len(a) < 150:
        return None, len(a), 0.0, 0.0, cov
    r = float(np.corrcoef(a[0], a[1])[0, 1])
    obs = float(np.std(np.concatenate([a[0].to_numpy(), a[1].to_numpy()])))
    return r, len(a), obs, obs * np.sqrt(max(r, 0.0)), cov


AXES = [
    # (라벨, 그룹키, 먼저 뺄 부모효과)
    ('★ pitcher (커리어)',            ['pitcher_id'], None),
    ('★ pitcher x season',            ['pitcher_id', 'season'], ['pitcher_id']),
    ('★ batter (커리어)',             ['batter_id'], None),
    ('★ batter x season',             ['batter_id', 'season'], ['batter_id']),
    ('pitcher x season x 전후반',      ['pitcher_id', 'season', 'half_season'],
     ['pitcher_id', 'season']),
    ('pitcher x season x 월',          ['pitcher_id', 'season', 'game_month'],
     ['pitcher_id', 'season']),
    ('pitcher_team (포수 대리)',       ['pitcher_team_id'], ['pitcher_id']),
    ('pitcher_team x season',          ['pitcher_team_id', 'season'],
     ['pitcher_id', 'season']),
    ('batter_team',                    ['batter_team_id'], ['batter_id']),
    ('pitcher x batter (맞대결)',      ['pitcher_id', 'batter_id'],
     ['pitcher_id']),
    ('pitcher x 타자손',               ['pitcher_id', 'batter_hand'], ['pitcher_id']),
    ('batter x 투수손',                ['batter_id', 'pitcher_hand'], ['batter_id']),
    ('pitcher x 이닝',                 ['pitcher_id', 'inning'], ['pitcher_id']),
]

print(f'{"축":<32}{"셀":>8}{"커버리지":>9}{"신뢰도":>8}{"신호std":>9}{"x커버":>9}')
print('-' * 76)
res = {}
for lab, keys, par in AXES:
    r, n, obs, sig, cov = reliab(keys, par, label=lab)
    if r is None:
        print(f'{lab:<32}{n:>8}   커버 {cov:>5.1%}  표본 부족')
        continue
    eff = sig * cov                       # 커버리지를 반영한 실효 신호
    res[lab] = eff
    print(f'{lab:<32}{n:>8,}{cov:>9.1%}{r:>8.3f}{sig:>9.4f}{eff:>9.4f}')

print('\n' + '=' * 70)
ps = res.get('★ pitcher x season')
bs = res.get('★ batter x season')
if ps and bs:
    print(f'교정점(커버리지 반영):  pitcher x season {ps:.4f} = LB +61.72')
    print(f'                        batter  x season {bs:.4f} = LB +30.11')
    k = (61.72 + 30.11) / (ps + bs)          # 신호 1 단위당 리더보드 점수 (거친 눈금)
    print(f'         -> 거친 환산 계수 약 {k:.0f} 점 / 신호단위\n')
    for lab, sig in sorted(res.items(), key=lambda z: -z[1]):
        if lab.startswith('★'):
            continue
        v = ('★ 파볼 만함' if sig >= bs * 0.8 else
             '△ 재볼 만함' if sig >= bs * 0.4 else '✕ 무시')
        print(f'  {lab:<30}실효 {sig:.4f}  환산 ~{sig*k:>5.0f}점   {v}')

print("""
⚠️ 환산 계수는 **거친 눈금**이다. 두 점으로 직선을 그은 것이고, 축들끼리 겹치는 것도
   반영 안 했다. 순서를 보는 용도이지 점수를 예측하는 용도가 아니다.
   오늘 대리지표에 한 번 당했다 (ws5gt: 상관 +0.260->+0.477, 점수 -17.8).
   판정은 스크리너 -> 리더보드로만 한다.""")

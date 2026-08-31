# 2023-05 에 팀 13 의 **투구 물리량**도 바뀌었나. 학습 0.
#
#   python tools/eda59.py
#
# eda57 은 라벨(성공률 + 실패 구성비)이 2023-05 에 계단을 밟는다는 것까지 봤다.
# 그런데 그게 **판정 방식이 바뀐 것**인지 **LG 가 실제로 뭔가 다르게 던지기 시작한
# 것**인지는 라벨만으로 못 가른다. 트랙맨은 심판·포수가 아니라 **공 자체**를 재므로
# 이 둘을 정확히 가른다.
#
#   물리량 그대로 + 라벨만 계단  ->  판정 방식 변경 (진단 확정)
#   물리량도 같이 계단           ->  진짜 야구 변화 (진단 오류, 함의가 완전히 다르다)
#
# ⚠️ 리그 전체도 해마다 변하므로 **DiD** 로 본다 (LG 변화 - 나머지 변화).
# ⚠️ game_date 는 포맷이 두 가지다 (claude.md 버그 항목): 2019~21 은 %m/%d/%Y,
#    2022~24 는 %Y-%m-%d. 여기서는 season/game_month 컬럼을 쓰므로 무관하다.
import numpy as np
import pandas as pd

tm = pd.read_csv('data/trackman_history.csv')
tm['ym'] = tm.season * 100 + tm.game_month
codes = sorted(set(tm.pitcher_team.astype(str)) | set(tm.batter_team.astype(str)))
print('팀 코드 %d개' % len(codes))
print(' ', [c for c in codes if not c.startswith('MIN_')][:20])

LG = [c for c in codes if c.startswith('LG')]
print('\nLG 코드:', LG, '| 2군 코드:', [c for c in codes if c.startswith('MIN_LG')])
if not LG:
    raise SystemExit('LG 코드를 못 찾았다 -- 코드 목록을 보고 지정할 것')

pt = tm.pitcher_team.astype(str).to_numpy()
bt = tm.batter_team.astype(str).to_numpy()
inv = np.isin(pt, LG) | np.isin(bt, LG)
# 1군만 본다 (MIN_* 는 2군이고 claude.md 기준 32%). F 는 라벨 쪽에서 따로 봤다.
one = ~(pd.Series(pt).str.startswith('MIN_').to_numpy()
        | pd.Series(bt).str.startswith('MIN_').to_numpy())
ym = tm.ym.to_numpy()
print('1군 %s행 | LG 관여 %.4f' % (format(int(one.sum()), ','), inv[one].mean()))

COLS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
        'extension', 'rel_height', 'rel_side', 'zone_speed']
pre = one & (ym < 202305) & (ym >= 202200)     # 직전 1.5시즌만 (드리프트 최소화)
post = one & (ym >= 202305) & (ym < 202400)
print('\n비교 구간: 2022-01~2023-04 (%s행) vs 2023-05~2023-10 (%s행)'
      % (format(int(pre.sum()), ','), format(int(post.sum()), ',')))

print('\n%-20s%10s%10s%10s%10s' % ('물리량', 'LG 변화', '나머지', 'DiD', 'DiD/std'))
for c in COLS:
    v = tm[c].to_numpy(dtype='float64')
    ok = np.isfinite(v)
    sd = np.nanstd(v[ok & one])
    a = v[pre & inv & ok].mean() - v[post & inv & ok].mean()
    b = v[pre & (~inv) & ok].mean() - v[post & (~inv) & ok].mean()
    print('%-20s%10.4f%10.4f%10.4f%10.4f' % (c, -a, -b, -(a - b), -(a - b) / sd))

# 구종 배합도 본다 (레퍼토리를 통째로 바꿨으면 여기 나온다)
print('\n%-20s%10s%10s%10s' % ('구종군 비율', 'LG 변화', '나머지', 'DiD'))
grp = tm.pitch_type_group.astype(str).to_numpy()
for g in sorted(set(grp[one])):
    if (grp[one] == g).mean() < 0.02:
        continue
    a = (grp[post & inv] == g).mean() - (grp[pre & inv] == g).mean()
    b = (grp[post & (~inv)] == g).mean() - (grp[pre & (~inv)] == g).mean()
    print('%-20s%10.4f%10.4f%10.4f' % (g, a, b, a - b))

print('''
읽는 법: DiD/std 가 전부 0.05 아래면 물리량은 그대로다 = **판정 방식이 바뀐 것**이다.
         라벨 쪽 DiD 는 실패율 -0.113 / 반대만 +0.065 로 컸다 (eda57).''')


# ── 결정적 검정: 2023-04 vs 2023-05 (같은 로스터, 한 달 차이) ──
# 위 비교창은 16개월 대 6개월이라 로스터 교체가 섞인다. rel_side / horz_break 는
# 팔각도·좌우 구성에 직접 붙는 값이라 어느 팀이든 시즌을 건너뛰면 0.1 std 는 움직인다.
# 라벨은 202304 -0.0220 -> 202305 +0.0469 로 **한 달 안에** 뒤집혔으므로,
# 그 한 달만 잘라서 보면 교락이 사라진다.
print('\n=== 결정적 검정: 2023-04 vs 2023-05 (같은 로스터) ===')
p4 = one & (ym == 202304)
p5 = one & (ym == 202305)
print('  2023-04 %s행 (LG 관여 %s) | 2023-05 %s행 (LG 관여 %s)'
      % (format(int(p4.sum()), ','), format(int((p4 & inv).sum()), ','),
         format(int(p5.sum()), ','), format(int((p5 & inv).sum()), ',')))
print('  %-20s%10s%10s%10s%10s' % ('물리량', 'LG 변화', '나머지', 'DiD', 'DiD/std'))
worst = 0.0
for c in COLS:
    v = tm[c].to_numpy(dtype='float64')
    ok = np.isfinite(v)
    sd = np.nanstd(v[ok & one])
    a = v[p5 & inv & ok].mean() - v[p4 & inv & ok].mean()
    b = v[p5 & (~inv) & ok].mean() - v[p4 & (~inv) & ok].mean()
    worst = max(worst, abs((a - b) / sd))
    print('  %-20s%10.4f%10.4f%10.4f%10.4f' % (c, a, b, a - b, (a - b) / sd))
print('\n  물리량 DiD 최대 %.3f std' % worst)
print('  라벨 DiD (eda57, 실패율) -0.1130 = %.3f std  (베르누이 sd 0.5)' % (0.1130 / 0.5))
print('  -> 배율 %.1f배' % ((0.1130 / 0.5) / max(worst, 1e-9)))


# ── 구장별: 측정계 교체인가, LG 자료 처리 변경인가 ──
# trackman_game_id 는 '20190329-Gocheok-1' 꼴이라 구장이 들어 있다.
# 라벨 효과는 투수측/타자측 대칭이었으므로 LG 를 **따라다닌다**. 물리량도 그러면
# 구장 하드웨어가 아니라 LG 경기 자료 자체가 바뀐 것이다.
print('\n=== 구장별 분해 (extension, 2023-04 -> 2023-05) ===')
ven = tm.trackman_game_id.astype(str).str.split('-').str[1].to_numpy()
v = tm['extension'].to_numpy(dtype='float64')
ok = np.isfinite(v)
sd = np.nanstd(v[ok & one])
print('  %-12s%8s%10s%10s%10s' % ('구장', 'n(5월)', 'LG관여', 'LG없음', '차이/std'))
for vv in sorted(set(ven[one & (ym == 202305)])):
    k4 = p4 & (ven == vv) & ok
    k5 = p5 & (ven == vv) & ok
    if k5.sum() < 1500 or k4.sum() < 1500:
        continue
    a = (v[k5 & inv].mean() - v[k4 & inv].mean()) if (k5 & inv).sum() > 400 and (k4 & inv).sum() > 400 else np.nan
    b = (v[k5 & ~inv].mean() - v[k4 & ~inv].mean()) if (k5 & ~inv).sum() > 400 and (k4 & ~inv).sum() > 400 else np.nan
    print('  %-12s%8s%10s%10s%10s'
          % (vv, format(int(k5.sum()), ','),
             '%+.4f' % a if np.isfinite(a) else '-',
             '%+.4f' % b if np.isfinite(b) else '-',
             '%+.3f' % ((a - b) / sd) if np.isfinite(a) and np.isfinite(b) else '-'))
print('''
  잠실에 LG 경기와 두산 경기가 둘 다 있다. 잠실 안에서 LG 관여만 움직이면
  하드웨어가 아니라 **LG 경기 자료**가 바뀐 것이다 (라벨 효과의 대칭성과 일관).''')

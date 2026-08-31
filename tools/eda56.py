# 팀 '관여' OR 플래그 전수 스캔 — t13 이 유일한 사례였는지 직접 확인한다. 학습 0.
#
#   python tools/eda56.py
#
# t13 이 리더보드 +20.58 을 낸 형태는 `(pitcher_team==k OR batter_team==k)` 이고,
# eda50 이 그 효과가 **투수측 +0.0513 / 타자측 +0.0467 로 대칭**임을 보였다.
# 즉 투수 실력이 아니라 **경기 단위** 효과다 (구장·심판조·포수 배정 같은).
# 그리고 두 컬럼의 OR 은 대칭트리에 레벨 두 개를 요구하므로 표현 비용이 최대다
# (claude.md 표현 비용 순서에서 1위, nosh -17.0 이 같은 자리).
#
# 그런데 우리가 실제로 준 것은 **팀 13 하나**뿐이다. 나머지 12개 팀의 OR 플래그는
# 한 번도 모델에 준 적이 없고, 스캔한 적도 없다 —
#   eda52 는 **컬럼 쌍**의 2차 상호작용을 봤지 파생 OR 플래그를 안 봤고,
#   eda55 의 COLS 에는 pitcher_team_id / batter_team_id 가 아예 없다.
#
# 여기서 두 가지를 한 번에 잰다.
#   (A) 정적 효과 : 팀 k 관여 플래그의 시즌별 효과와 2023<->2024 안정성
#   (B) 계단 변화 : 그 효과 계열의 월 단위 계단 (t13 = 2023-05 를 재발견해야 한다)
#
# ⚠️ 판정 기준은 **강도가 아니라 안정성**이다 (claude.md: pitcher_team x batter_team
#    이 강도 120~186 인데 상관 -0.14 라 죽었다). 그리고 계단은 **계단폭/잡음 비**로
#    본다 (eda55: 분할점 최적화만으로도 시끄러운 계열은 큰 계단이 나온다).
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
y = df.control_success.to_numpy(dtype='float64')
pt = df.pitcher_team_id.to_numpy()
bt = df.batter_team_id.to_numpy()
sea = df.season.to_numpy()
ym = (df.season * 100 + df.game_month).to_numpy()
isR = (df.game_type == 'R').to_numpy()
teams = sorted(set(pt) | set(bt))
print('%s행 | 팀 %d개 %s' % (format(len(df), ','), len(teams), teams))

# 팀 주효과를 뺀 잔차로 본다. OR 플래그가 단순히 '그 팀 투수/타자가 약하다' 를
# 반복하는 것이면 CTR 이 이미 갖고 있으므로 새 정보가 아니다.
# 시즌 x 팀 단위 주효과를 뺀다 (드리프트와 팀 수준을 동시에 제거).
res = y.copy()
for s in np.unique(sea):
    m = sea == s
    res[m] -= y[m].mean()
for col in (pt, bt):
    for s in np.unique(sea):
        for k in teams:
            m = (sea == s) & (col == k)
            if m.sum() > 200:
                res[m] -= res[m].mean() * 0.5      # 두 축이 겹치므로 절반씩
print('잔차 std %.5f (원본 %.5f)' % (res.std(), y.std()))


def eff(mask, sub):
    """sub 구간에서 mask 인 행의 잔차 평균 - 아닌 행의 잔차 평균."""
    a, b = mask & sub, (~mask) & sub
    if a.sum() < 300 or b.sum() < 300:
        return np.nan
    return res[a].mean() - res[b].mean()


print('\n=== (A) 팀 관여 OR 플래그: 시즌별 효과 (R 전용) ===')
print('%-6s%7s' % ('팀', '커버') + ''.join('%9d' % s for s in sorted(set(sea)))
      + '%10s%9s' % ('|23-24|', '천장'))
rows = []
for k in teams:
    inv = (pt == k) | (bt == k)
    cov = float((inv & isR).sum() / isR.sum())
    if cov < 0.03:
        continue
    e = [eff(inv, isR & (sea == s)) for s in sorted(set(sea))]
    d = abs(e[-1] - e[-2]) if np.isfinite(e[-1]) and np.isfinite(e[-2]) else np.nan
    # 천장: 2024 효과를 완벽히 잡았을 때 (eda52 와 같은 환산)
    g = e[-1] if np.isfinite(e[-1]) else 0.0
    pts = 401000 * (cov * (g * (1 - cov)) ** 2 + (1 - cov) * (g * cov) ** 2)
    rows.append((k, cov, e, d, pts))
    print('%-6d%6.1f%%' % (k, 100 * cov)
          + ''.join('%+9.4f' % v if np.isfinite(v) else '%9s' % '-' for v in e)
          + '%10.4f%9.1f' % (d, pts))

print('\n  |23-24| 가 작을수록 안정적이다. t13 은 +0.0507/+0.0490 로 0.002 여야 한다.')

print('\n=== (B) 월 단위 계단 변화 (t13 = 2023-05 를 재발견해야 방법이 유효하다) ===')
months = sorted(set(ym[isR]))
out = []
for k in teams:
    inv = (pt == k) | (bt == k)
    cov = float((inv & isR).sum() / isR.sum())
    if cov < 0.03:
        continue
    a = np.array([eff(inv, isR & (ym == mo)) for mo in months], dtype=float)
    ok = np.isfinite(a)
    if ok.sum() < 20:
        continue
    noise = float(np.std(np.diff(a[ok])) / np.sqrt(2))
    best, bi = 0.0, None
    for i in range(4, len(a) - 4):
        if np.isnan(a[:i]).all() or np.isnan(a[i:]).all():
            continue
        g = abs(np.nanmean(a[i:]) - np.nanmean(a[:i]))
        if g > best:
            best, bi = g, i
    if bi is None:
        continue
    out.append((best / max(noise, 1e-9), k, cov, best, noise, months[bi],
                float(np.nanmean(a[:bi])), float(np.nanmean(a[bi:]))))

out.sort(reverse=True)
print('%-6s%7s%9s%9s%7s%9s%9s%9s' % ('팀', '커버', '계단폭', '잡음', '비', '시점', '전', '후'))
for r, k, cov, g, nz, mo, pre, post in out:
    print('%-6d%6.1f%%%9.4f%9.4f%7.1f%9d%+9.4f%+9.4f%s'
          % (k, 100 * cov, g, nz, r, mo, pre, post, ' *' if r >= 3 else ''))

print('\n=== 판정 ===')
hit = [o for o in out if o[0] >= 3 and o[5] >= 202001]
if not hit:
    print('  2020 이후 계단은 없다 -> t13 이 유일한 사례. 이 축 종료.')
else:
    for r, k, cov, g, nz, mo, pre, post in hit:
        pts = 401000 * (cov * (post * (1 - cov)) ** 2 + (1 - cov) * (post * cov) ** 2)
        print('  ★ 팀 %d | %d 전환 | 비 %.1f | 전 %+.4f -> 후 %+.4f | 천장 %.1f점'
              % (k, mo, r, pre, post, pts))

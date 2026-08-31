# 리그 전체의 **실패 구성비**에 시즌 중간 계단이 있는가. 학습 0.
#
#   python tools/eda58.py
#
# eda55 는 **성공률**의 월 단위 계단을 훑어 t13 하나만 찾았다. 그런데 eda57 에서
# 13 이 아닌 나머지 리그도 구성비가 크게 움직인다는 게 보였다 (몰림만 +6.2pp,
# 크게벗어남 -5.5pp). 성공률이 매끄럽게 움직이면서 구성비만 계단을 밟을 수 있다 —
# 그건 라벨 산출 방식이 바뀌었다는 서명이고, eda55 의 스캔은 그걸 못 본다.
#
# 왜 이게 t13 급일 수 있나: `season` 은 cat_features 에 없는 int64 라 분기 경계가
# 학습에서 본 값 위로 안 생기고(eda31), 게다가 **연 단위**다. 전환점이 시즌 중간이면
# season 으로는 표현이 안 되고 game_month 는 DROP_CAL 로 버렸다. t13 이 통한
# 이유 ②가 정확히 그것이었다.
#
# ⚠️ 판정은 계단폭/잡음 비로 한다 (eda55). 그리고 **2020 이후 전환만** 후보다 —
#    2019 경계는 2025 가 어차피 '후' 이고 season 이 이미 가른다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
y = df.control_success.to_numpy(dtype='float64')
ym = (df.season * 100 + df.game_month).to_numpy()
gt = df.game_type.to_numpy().astype(str)

n = df['asof_pitcher_n'].to_numpy(dtype='float64')
g = df['pitcher_id'].to_numpy()
o = np.lexsort((n, g))
ns, gs = n[o], g[o]
ok = np.r_[(gs[:-1] == gs[1:]) & (ns[1:] - ns[:-1] == 1), False]
lab = {}
for kk in ['success', 'middle', 'reverse', 'ball', 'strike']:
    cum = np.round(df['asof_pitcher_%s_rate' % kk].fillna(0)
                   .to_numpy(dtype='float64')[o] * ns)
    d = np.r_[cum[1:] - cum[:-1], np.nan]
    v = np.where(ok & np.isin(d, [0.0, 1.0]), d, np.nan)
    b = np.full(len(df), np.nan)
    b[o] = v
    lab[kk] = b
m = np.isfinite(lab['success'])
print('복원 검산: success 일치율 %.6f' % (lab['success'][m] == y[m]).mean())

good = np.isfinite(lab['middle']) & np.isfinite(lab['reverse']) & \
    np.isfinite(lab['ball']) & np.isfinite(lab['strike'])
mi, re = lab['middle'] == 1, lab['reverse'] == 1
fail = good & (lab['success'] == 0)

# 잴 계열들: 실패 안에서의 구성비 + 판정 비율. 전부 '전체 대비 비율' 이라
# 실패율 드리프트 자체와는 독립이다.
SERIES = {
    '실패율': (good, ~np.isfinite(lab['success']) | (lab['success'] == 0)),
    '몰림만/실패': (fail, mi & ~re),
    '반대만/실패': (fail, re & ~mi),
    '둘다/실패': (fail, mi & re),
    '크게벗어남/실패': (fail, ~mi & ~re),
    '볼/전체': (good, lab['ball'] == 1),
    '스트라이크/전체': (good, lab['strike'] == 1),
}
SERIES['실패율'] = (good, lab['success'] == 0)


def scan(name, base, num, sub, tag):
    mos = sorted(set(ym[sub]))
    a = []
    for mo in mos:
        k = sub & base & (ym == mo)
        a.append(num[k].mean() if k.sum() > 500 else np.nan)
    a = np.array(a, dtype=float)
    good_m = np.isfinite(a)
    if good_m.sum() < 20:
        return None
    noise = float(np.std(np.diff(a[good_m])) / np.sqrt(2))
    best, bi = 0.0, None
    for i in range(4, len(a) - 4):
        if np.isnan(a[:i]).all() or np.isnan(a[i:]).all():
            continue
        gap = abs(np.nanmean(a[i:]) - np.nanmean(a[:i]))
        if gap > best:
            best, bi = gap, i
    if bi is None:
        return None
    # ⚠️ 전/후 평균차 / 잡음 은 **완만한 추세에서 폭발한다** — 인접 월 차분이
    #    작아 잡음 추정이 0 에 가까워지기 때문이다 (여기서 비 15.0 짜리 가짜를
    #    하나 잡았다). 진짜 체제 변화는 **한 달 안에** 다 일어난다. 그래서
    #    분할점에서의 한 달 점프가 전체 계단의 몇 %인지를 같이 낸다.
    j = np.nan
    for back in range(1, 4):
        if bi - back >= 0 and np.isfinite(a[bi - back]) and np.isfinite(a[bi]):
            j = abs(a[bi] - a[bi - back])
            break
    frac = float(j / best) if np.isfinite(j) and best > 0 else np.nan
    return (best / max(noise, 1e-9), name, tag, best, noise, mos[bi],
            float(np.nanmean(a[:bi])), float(np.nanmean(a[bi:])), a, mos, frac)


rows = []
for tag, sub in (('R', gt == 'R'), ('F', gt == 'F')):
    for name, (base, num) in SERIES.items():
        r = scan(name, base, num, sub, tag)
        if r:
            rows.append(r)
rows.sort(reverse=True)

print('\n=== 리그 전체 구성비의 월 단위 계단 ===')
print('%-18s%4s%9s%7s%9s%9s%9s%8s' %
      ('계열', '구분', '계단폭', '비', '시점', '전', '후', '1달몫'))
for r, name, tag, gap, nz, mo, pre, post, _, _, frac in rows:
    star = ' *' if (r >= 3 and mo >= 202001 and frac >= 0.5) else ''
    print('%-18s%4s%9.4f%7.1f%9d%9.4f%9.4f%7.0f%%%s'
          % (name, tag, gap, r, mo, pre, post, 100 * frac, star))

# 후보 조건 넷. 앞의 둘만 걸면 완만한 추세가 통과한다 (비 15.0 짜리 가짜).
#  · 2020 이후 전환   · 비 3 이상
#  · 1달몫 50% 이상   = 계단의 절반 이상이 한 달 안에 일어난다
#  · 시즌 경계가 아님 = 3월/4월 전환은 season 이 이미 가른다
print('\n=== 후보: 2020 이후 + 비 3 + 1달몫 50% + 시즌 경계 아님 ===')
hit = [x for x in rows if x[0] >= 3 and x[5] >= 202001
       and np.isfinite(x[10]) and x[10] >= 0.5 and x[5] % 100 > 4]
if not hit:
    print('  없음 -> 구성비 변화는 계단이 아니라 **매끄러운 드리프트**다.')
    print('     season 이 연 단위로 이미 가르고 있고, 시즌 중간 전환은 없다.')
else:
    for r, name, tag, gap, nz, mo, pre, post, a, mos, frac in hit:
        print('  ★ %s (%s) | %d 전환 | 비 %.1f | 1달몫 %.0f%% | %.4f -> %.4f' %
              (name, tag, mo, r, frac * 100, pre, post))
        print('     계열:', ' '.join('%d:%.3f' % (m, v) for m, v in zip(mos, a)
                                     if np.isfinite(v) and m >= mo - 300)[:200])

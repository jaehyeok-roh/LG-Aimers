# t13 은 왜 혼자 튀는가 — 라벨 체제 변화인지 야구인지 가른다. 학습 0.
#
#   python tools/eda57.py
#
# 지금까지 t13 을 **쓰기만** 했다 (LB +20.58). 기전은 "2023-05 에 뭔가 바뀌었다"
# 로만 적혀 있고, 그게 무엇인지는 본 적이 없다. 네 가지를 직접 확인한다.
#
#  ① 구장 가설 배제: 12(두산)와 13(LG)은 **같은 잠실**이다. 12 가 안 튀면 구장이 아니다.
#  ② 데이터셋 구성: claude.md 는 "F 행은 100% 가 team13 관여" 라고 적어뒀다.
#     사실이면 이건 KBO 사실이 아니라 **데이터셋을 어떻게 모았는가**의 사실이다.
#  ③ 같은 사건인가: F 의 붕괴(.7087 -> .4729)와 t13(R) 의 반전이 같은 달인가.
#  ④ ★ 라벨 정의가 바뀐 것인가: 실패 **구성비**(몰림/반대/크게벗어남)가 같이
#     계단을 밟으면 라벨 산출 방식이 바뀐 것이고, 성공률만 움직이고 구성비가
#     그대로면 난이도가 바뀐 것이다. 이 둘은 처방이 다르다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
y = df.control_success.to_numpy(dtype='float64')
pt, bt = df.pitcher_team_id.to_numpy(), df.batter_team_id.to_numpy()
ym = (df.season * 100 + df.game_month).to_numpy()
gt = df.game_type.to_numpy().astype(str)
inv13 = (pt == 13) | (bt == 13)
inv12 = (pt == 12) | (bt == 12)

print('=== ① 구장 가설 (12 두산 / 13 LG 는 같은 잠실) ===')
for name, m in (('13 관여', inv13), ('12 관여', inv12)):
    v = [y[m & (gt == 'R') & (df.season == s).to_numpy()].mean()
         - y[(~m) & (gt == 'R') & (df.season == s).to_numpy()].mean()
         for s in range(2019, 2025)]
    print('  %-8s' % name + ''.join('%+9.4f' % x for x in v))
print('  -> 같은 구장인데 하나만 튀면 구장이 아니라 **팀**이다.')

print('\n=== ② 데이터셋 구성: F(퓨처스) 행은 누구의 경기인가 ===')
F = gt == 'F'
print('  F 행 %s개 (전체의 %.1f%%)' % (format(F.sum(), ','), 100 * F.mean()))
print('  그중 13 관여 %.4f' % inv13[F].mean())
vc = pd.Series(np.where(pt[F] == 13, bt[F], pt[F])).value_counts()
print('  F 의 상대 팀 분포:', dict(vc.head(8)))
print('  R 행 중 13 관여 %.4f (10팀이면 이론값 0.2)' % inv13[~F].mean())

print('\n=== ③ F 붕괴와 t13(R) 반전은 같은 달인가 ===')
mos = sorted(set(ym))
print('  %-8s%10s%10s%10s' % ('월', 'F 성공률', 't13 R 효과', 'F 행수'))
for mo in mos:
    if not (202208 <= mo <= 202307):
        continue
    k = ym == mo
    f = k & F
    r = k & (~F)
    a, b = r & inv13, r & (~inv13)
    print('  %-8d%10s%10s%10s' % (
        mo,
        '%.4f' % y[f].mean() if f.sum() > 200 else '-',
        '%+.4f' % (y[a].mean() - y[b].mean()) if a.sum() > 300 and b.sum() > 300 else '-',
        format(int(f.sum()), ',')))

print('\n=== ④ ★ 라벨 정의가 바뀐 것인가 — 실패 구성비 (R, 13 관여) ===')
# 실패유형은 asof 인접 행 차분으로 복원한다 (claude.md eda40, success 검산 일치율 1.0)
n = df['asof_pitcher_n'].to_numpy(dtype='float64')
g = df['pitcher_id'].to_numpy()
o = np.lexsort((n, g))
ns, gs = n[o], g[o]
ok = np.r_[(gs[:-1] == gs[1:]) & (ns[1:] - ns[:-1] == 1), False]
lab = {}
for kk in ['success', 'middle', 'reverse']:
    cum = np.round(df['asof_pitcher_%s_rate' % kk].fillna(0)
                   .to_numpy(dtype='float64')[o] * ns)
    d = np.r_[cum[1:] - cum[:-1], np.nan]
    v = np.where(ok & np.isin(d, [0.0, 1.0]), d, np.nan)
    b = np.full(len(df), np.nan)
    b[o] = v
    lab[kk] = b
m = np.isfinite(lab['success'])
print('  복원 검산: success 일치율 %.6f' % (lab['success'][m] == y[m]).mean())

R = (gt == 'R') & np.isfinite(lab['middle']) & np.isfinite(lab['reverse'])
pre = R & inv13 & (ym < 202305)
post = R & inv13 & (ym >= 202305)
ctl_pre = R & (~inv13) & (ym < 202305)
ctl_post = R & (~inv13) & (ym >= 202305)


def compo(mask):
    f = mask & (lab['success'] == 0)
    if f.sum() < 500:
        return None
    mi, re = lab['middle'][f] == 1, lab['reverse'][f] == 1
    return (float(1 - y[mask].mean()),
            float((mi & ~re).mean()), float((re & ~mi).mean()),
            float((mi & re).mean()), float((~mi & ~re).mean()))


print('  %-14s%9s%9s%9s%9s%9s' % ('', '실패율', '몰림만', '반대만', '둘다', '크게벗어남'))
rows = [('13 관여 전', pre), ('13 관여 후', post),
        ('나머지 전', ctl_pre), ('나머지 후', ctl_post)]
vals = {}
for nm, mk in rows:
    c = compo(mk)
    vals[nm] = c
    print('  %-14s' % nm + ''.join('%9.4f' % x for x in c))

a, b = vals['13 관여 전'], vals['13 관여 후']
c, d = vals['나머지 전'], vals['나머지 후']
print('\n  변화량 (후 - 전), 실패 **안에서의 구성비**:')
print('  %-14s%9s%9s%9s%9s%9s' % ('', '실패율', '몰림만', '반대만', '둘다', '크게벗어남'))
print('  %-14s' % '13 관여' + ''.join('%+9.4f' % (b[i] - a[i]) for i in range(5)))
print('  %-14s' % '나머지 ' + ''.join('%+9.4f' % (d[i] - c[i]) for i in range(5)))
print('  %-14s' % '차이(DiD)' + ''.join(
    '%+9.4f' % ((b[i] - a[i]) - (d[i] - c[i])) for i in range(5)))
print('''
  읽는 법: 실패율만 움직이고 구성비 DiD 가 0 이면 **난이도**가 바뀐 것이다.
           구성비까지 크게 움직이면 **라벨 산출 방식**이 바뀐 것이고, 그러면
           t13_post 안에서 피처-라벨 관계 자체가 다르다는 뜻이다.''')

# 5분류 타겟의 재료를 **독립 경로로 교차 검산**한다. 학습 0.
#
#   python tools/eda63.py
#
# eda40 이 asof 인접 행 차분으로 투구 단위 라벨을 복원했고, `success` 는 정답 컬럼이
# 있어 검산됐다 (147만 행 일치율 1.000000). 그런데 **`middle`/`reverse` 는 독립
# 검산이 없다** — 그게 5분류 타겟(LB +12.85)과 aux_rev(LB +13.50)의 재료인데도.
#
# 타자측에 `asof_batter_middle_rate` 가 있다. 라벨은 투구의 속성이므로 투수로
# 집계하든 타자로 집계하든 **같은 투구의 같은 라벨**이 나와야 한다. 즉:
#   투수 경로: (pitcher_id, asof_pitcher_n) 정렬 -> 누적개수 차분
#   타자 경로: (batter_id,  asof_batter_n)  정렬 -> 누적개수 차분
# 두 복원이 일치하면 middle 복원이 옳다는 강한 증거다. 어긋나면 +12.85 짜리
# 피처의 재료에 버그가 있다는 뜻이고, 그건 마감 전에 알아야 한다.
#
# ⚠️ reverse 는 타자측 컬럼이 없어 이 방법으로 검산 불가다. 다만 middle 이
#    통과하면 같은 코드 경로를 쓰는 reverse 도 신뢰할 근거가 된다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
y = df.control_success.to_numpy(dtype='float64')
print('%s행' % format(len(df), ','))


def recover(id_col, n_col, rate_cols):
    """(entity, n) 정렬 후 인접 행 차분으로 투구 단위 라벨을 복원한다."""
    n = df[n_col].to_numpy(dtype='float64')
    g = df[id_col].to_numpy()
    o = np.lexsort((n, g))
    ns, gs = n[o], g[o]
    step1 = (gs[:-1] == gs[1:]) & (ns[1:] - ns[:-1] == 1)
    print('  %-18s 인접 증분 +1 비율 %.6f' % (id_col, step1.mean()))
    ok = np.r_[step1, False]
    out = {}
    for c, name in rate_cols:
        cum = np.round(df[c].fillna(0).to_numpy(dtype='float64')[o] * ns)
        d = np.r_[cum[1:] - cum[:-1], np.nan]
        v = np.where(ok & np.isin(d, [0.0, 1.0]), d, np.nan)
        b = np.full(len(df), np.nan)
        b[o] = v
        out[name] = b
    return out


print('\n=== 투수 경로 ===')
P = recover('pitcher_id', 'asof_pitcher_n',
            [('asof_pitcher_success_rate', 'success'),
             ('asof_pitcher_middle_rate', 'middle'),
             ('asof_pitcher_reverse_rate', 'reverse')])
print('\n=== 타자 경로 ===')
B = recover('batter_id', 'asof_batter_n',
            [('asof_batter_success_rate', 'success'),
             ('asof_batter_middle_rate', 'middle')])

print('\n=== ① 각 경로의 success 자체 검산 (정답 대조) ===')
for tag, R in (('투수', P), ('타자', B)):
    m = np.isfinite(R['success'])
    acc = float((R['success'][m] == y[m]).mean())
    print('  %s 경로: 복원 %s행 (%.1f%%) | success 일치율 **%.6f**'
          % (tag, format(int(m.sum()), ','), 100 * m.mean(), acc))

print('\n=== ② ★ middle 교차 검산 (두 경로가 서로를 검증한다) ===')
m = np.isfinite(P['middle']) & np.isfinite(B['middle'])
agree = float((P['middle'][m] == B['middle'][m]).mean())
print('  둘 다 복원된 행 %s개 (%.1f%%)' % (format(int(m.sum()), ','), 100 * m.mean()))
print('  **일치율 %.6f**' % agree)
print('  투수 경로 middle 율 %.4f | 타자 경로 %.4f'
      % (np.nanmean(P['middle'][m]), np.nanmean(B['middle'][m])))
if agree < 0.999:
    dis = m & (P['middle'] != B['middle'])
    print('\n  ⚠️ 불일치 %s행. 성질을 본다:' % format(int(dis.sum()), ','))
    print('     불일치 행의 success 평균 %.4f (전체 %.4f)' % (y[dis].mean(), y.mean()))
    print('     시즌별 불일치율:',
          {int(s): round(float(dis[(df.season == s).to_numpy()].mean()), 4)
           for s in sorted(df.season.unique())})
else:
    print('  ✅ 두 독립 경로가 일치한다 -> middle 복원은 옳다.')
    print('     reverse 는 타자측 컬럼이 없어 직접 검산 불가지만 **같은 코드 경로**이므로')
    print('     이로써 5분류 타겟(LB +12.85)과 aux_rev(LB +13.50)의 재료가 검증됐다.')

print('\n=== ③ 커버리지: 어느 경로가 더 많이 복원하나 ===')
for tag, R in (('투수', P), ('타자', B)):
    print('  %s 경로 middle 복원 %.4f' % (tag, np.isfinite(R['middle']).mean()))
u = np.isfinite(P['middle']) | np.isfinite(B['middle'])
print('  합집합 %.4f  (투수 단독 대비 %+.4f)'
      % (u.mean(), u.mean() - np.isfinite(P['middle']).mean()))
print('''
  합집합이 유의미하게 크면 타자 경로로 **결측을 메울 수 있다** —
  현재 5분류 타겟은 투수 경로만 쓰고 결측 행을 학습에서 뺀다.''')

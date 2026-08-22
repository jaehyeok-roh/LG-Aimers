# EDA 7 — '당해 시즌 성적' 복원 피처의 값어치.
#
# eda6 에서 발견: `asof_pitcher_success_rate` 는 **커리어 누적**이라 투수마다 다른 것을
# 의미한다. 이력 있는 투수에게는 2019(리그 .5647)부터 섞인 낡은 값이고(단독 예측
# 스킬 **-146**), 신규 투수에게는 순수한 당해 시즌 값이다(**+904**).
#
# 복원 방법 (검증 완료):
#   커리어 성공수 = asof_pitcher_success_rate x asof_pitcher_n     (그 행이 갖고 있다)
#   직전 시즌까지 = train 으로 만든 투수별 룩업                      (test 행 통계 아님 = 합법)
#   차이 -> 당해 시즌 투구수 / 성공수 -> 당해 시즌 성공률
#
# train 2024 에서 검증: 성공수 정수 오차 0.008, 음수 0%, 범위 밖 0.0000%.
# test 5행에서도 asof_n >= train 총투구가 전부 성립.
#
# 여기서는 그 값이 실제로 예측력이 있는지, 어느 조각에서 얼마나인지 잰다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
SE = 2024
prev = df[df.season < SE].groupby('pitcher_id')['control_success'].agg(
    pn='size', ps='sum')
d = df[df.season == SE].join(prev, on='pitcher_id').reset_index(drop=True)
d[['pn', 'ps']] = d[['pn', 'ps']].fillna(0)
d['seen'] = d['pn'] > 0

n_all = d['asof_pitcher_n'].to_numpy(dtype='float64')
s_all = (d['asof_pitcher_success_rate'].fillna(0).to_numpy(dtype='float64') * n_all).round()
d['wn'] = n_all - d['pn'].to_numpy()                    # 당해 시즌 투구수
d['ws'] = s_all - d['ps'].to_numpy()                    # 당해 시즌 성공수
y = d['control_success'].to_numpy(dtype='float64')
r = y.mean()


def skill(p, yv=None):
    yy = y if yv is None else yv
    rr = float(yy.mean())
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yy) ** 2).mean() / (rr * (1 - rr))) * 100000


def shrunk(s, n, C, prior):
    return (s + prior * C) / np.maximum(n + C, 1e-9)


print(f'{SE} {len(d):,}행 | 리그 {r:.4f} | 신규 투수 행 {(~d["seen"]).mean():.1%}')
print(f'당해 시즌 투구수 분포: ' + '  '.join(
    f'{int(q*100)}%={int(d["wn"].quantile(q))}' for q in (.1, .25, .5, .75, .9)) + '\n')

print('=' * 80)
print('단독 예측 스킬 — 어느 값을 쓰느냐만 다르다 (shrink C 는 각각 최적화)')
print(f'{"예측에 쓴 값":<30}{"전체":>10}{"이력 있음":>12}{"신규":>10}')
cands = {
    '커리어 성공률 (현행 asof)': (s_all, n_all),
    '당해 시즌 성공률 (복원)': (d['ws'].to_numpy(), d['wn'].to_numpy()),
    '직전 시즌까지만': (d['ps'].to_numpy(), d['pn'].to_numpy()),
}
seg = {'전체': np.ones(len(d), bool), '이력 있음': d['seen'].to_numpy(),
       '신규': (~d['seen']).to_numpy()}
for lab, (s, n) in cands.items():
    out = []
    for sl, m in seg.items():
        bs = -9e9
        for C in (25, 50, 100, 200, 400, 800, 1600):
            p = shrunk(s[m], n[m], C, r)
            bs = max(bs, skill(p, y[m]))
        out.append(bs)
    print(f'{lab:<30}' + ''.join(f'{v:>11,.0f}' for v in out))

print('\n' + '=' * 80)
print('둘을 같이 쓰면 — 로짓 가법 (당해 시즌 + 직전 시즌까지)')


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


best = (-9e9, None)
for Cw in (50, 100, 200, 400):
    for Cp in (100, 200, 400, 800):
        for w in (0.3, 0.5, 0.7, 1.0):
            pw = shrunk(d['ws'].to_numpy(), d['wn'].to_numpy(), Cw, r)
            pp = shrunk(d['ps'].to_numpy(), d['pn'].to_numpy(), Cp, r)
            z = lg(r) + w * (lg(pw) - lg(r)) + (1 - w) * (lg(pp) - lg(r))
            s = skill(1 / (1 + np.exp(-z)))
            if s > best[0]:
                best = (s, (Cw, Cp, w))
print(f'  최적 {best[0]:,.0f}   (당해 C={best[1][0]}, 과거 C={best[1][1]}, 당해 가중 {best[1][2]})')

print('\n' + '=' * 80)
print('당해 시즌 표본이 쌓일수록 — 구간별 (당해 시즌 성공률 단독)')
print(f'{"당해 시즌 투구수":<20}{"행수":>10}{"비중":>8}{"스킬":>10}')
for lo, hi in [(0, 50), (50, 150), (150, 400), (400, 900), (900, 99999)]:
    m = (d['wn'] >= lo) & (d['wn'] < hi)
    if m.sum() < 3000:
        continue
    p = shrunk(d['ws'].to_numpy()[m], d['wn'].to_numpy()[m], 100, r)
    print(f'{f"{lo}~{hi}":<20}{m.sum():>10,}{m.mean():>7.1%}{skill(p, y[m]):>10,.0f}')

print('\n' + '=' * 80)
print('⚠️ 이 숫자는 단독 예측이다. 모델에는 이미 cond_p / smoothed 등이 있으므로')
print('   증분은 이보다 훨씬 작다. 다만 **현행 asof 가 이력 있는 투수에서 -146 이라는 것**은')
print('   그 80% 구간에 제대로 된 당해 시즌 신호가 없다는 뜻이고, 그게 이 피처의 근거다.')

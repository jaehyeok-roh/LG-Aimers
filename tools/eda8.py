# EDA 8 — '당해 시즌 복원' 의 **증분**. 단독이 아니라 기존 위에 얹었을 때.
#
# eda7: 당해 시즌 성공률 단독 600 vs 커리어 219. 하지만 모델에는 이미 cond_p 가 있다
# (과거 시즌을 **시즌 디트렌드**해서 0 으로 shrink 한 것 = 과거를 제대로 쓴 버전).
# eda7 의 '직전 시즌까지만 -2' 는 디트렌드를 안 한 날것이라 수준 오염으로 죽은 값이고,
# cond_p 의 대역이 아니다.
#
# 그래서 여기서는 축을 순서대로 쌓으며 각 단계의 증분을 본다:
#   1. 리그평균만                     (스킬 0)
#   2. + 과거 시즌 디트렌드 (= cond_p 대역)
#   3. + 카운트 / 타자 / 손
#   4. + **당해 시즌 복원**            <- 우리가 재려는 것
# 4단계의 증분이 이 피처가 낼 수 있는 최대치의 근사다 (모델은 이보다 덜 뽑는다).
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
SE = 2024
lg = df.groupby('season')['control_success'].mean()

# --- 과거 시즌 디트렌드 누적 (cond_p 와 같은 방식) ---
past = df[df.season < SE].copy()
past['dev'] = past['control_success'] - past['season'].map(lg)
pg = past.groupby('pitcher_id')['dev'].agg(psum='sum', pn='size')

d = df[df.season == SE].join(pg, on='pitcher_id').reset_index(drop=True)
d[['psum', 'pn']] = d[['psum', 'pn']].fillna(0)
y = d['control_success'].to_numpy(dtype='float64')
r = y.mean()
U = r * (1 - r)


def skill(p):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / U) * 100000


def lgt(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# --- 당해 시즌 복원 ---
n = d['asof_pitcher_n'].to_numpy(dtype='float64')
s = (d['asof_pitcher_success_rate'].fillna(0).to_numpy(dtype='float64') * n).round()
prevn = df[df.season < SE].groupby('pitcher_id').size().reindex(d['pitcher_id']).fillna(0).to_numpy()
prevs = df[df.season < SE].groupby('pitcher_id')['control_success'].sum().reindex(
    d['pitcher_id']).fillna(0).to_numpy()
wn = np.maximum(n - prevn, 0.0)
ws = np.clip(s - prevs, 0.0, wn)

# --- 축들 (전부 리그평균 기준 로짓 편차로) ---
scale = 1.0 / max(r * (1 - r), 1e-9)


def dev_term(sumdev, cnt, C):
    return (sumdev / (cnt + C)) * scale


def loo_dev(keys, C):
    """그 행 자신을 뺀 그룹 편차 (2024 안에서 만드는 축은 자기누수를 뺀다)."""
    e = y - r
    g = pd.DataFrame({'k': keys, 'e': e})
    sm = g.groupby('k')['e'].transform('sum').to_numpy()
    ct = g.groupby('k')['e'].transform('size').to_numpy()
    return ((sm - e) / ((ct - 1) + C)) * scale


d['cs'] = d['balls_before'].astype(str) + '-' + d['strikes_before'].astype(str)
d['hd'] = d['pitcher_hand'].astype(str) + 'v' + d['batter_hand'].astype(str)

past_t = dev_term(d['psum'].to_numpy(), d['pn'].to_numpy(), 200.0)
w_t = ((ws - r * wn) / (wn + 100.0)) * scale
cnt_t = loo_dev(d['cs'].to_numpy(), 500.0)
bat_t = loo_dev(d['batter_id'].astype(str).to_numpy(), 400.0)
hnd_t = loo_dev(d['hd'].to_numpy(), 500.0)

print(f'{SE} {len(d):,}행 | 리그 {r:.4f}')
print(f'당해 시즌 투구수 중앙값 {np.median(wn):.0f} | 과거 시즌 누적 중앙값 {np.median(d["pn"]):.0f}\n')
print('=' * 74)
print('축을 순서대로 쌓았을 때 (로짓 가법)')
print(f'{"구성":<44}{"스킬":>10}{"증분":>10}')
steps = [
    ('1. 리그평균만', []),
    ('2. + 과거 시즌 디트렌드 (cond_p 대역)', [past_t]),
    ('3. + 카운트 + 타자 + 손', [past_t, cnt_t, bat_t, hnd_t]),
    ('4. + 당해 시즌 복원  ★', [past_t, cnt_t, bat_t, hnd_t, w_t]),
]
prev_s = None
for lab, terms in steps:
    z = lgt(r) + (np.sum(terms, axis=0) if terms else 0.0)
    v = skill(1 / (1 + np.exp(-z)))
    inc = '' if prev_s is None else f'{v - prev_s:>+10,.0f}'
    print(f'{lab:<44}{v:>10,.0f}{inc}')
    prev_s = v

print('\n' + '=' * 74)
print('순서를 바꿔서 — 당해 시즌을 먼저 넣고 과거를 나중에')
z = lgt(r) + w_t + cnt_t + bat_t + hnd_t
a = skill(1 / (1 + np.exp(-z)))
b = skill(1 / (1 + np.exp(-(z + past_t))))
print(f'  당해 + 카운트/타자/손           {a:>10,.0f}')
print(f'  + 과거 시즌                    {b:>10,.0f}   증분 {b-a:>+,.0f}')
print(f'\n  → 두 순서의 증분을 비교하면 겹치는 정보가 얼마인지 보인다.')

print('\n' + '=' * 74)
print('조각별 증분 (3단계 -> 4단계)')
z3 = lgt(r) + past_t + cnt_t + bat_t + hnd_t
z4 = z3 + w_t
seen = (d['pn'] > 0).to_numpy()
for lab, m in [('전체', np.ones(len(d), bool)), ('이력 있는 투수', seen), ('신규 투수', ~seen)]:
    p3 = 1 / (1 + np.exp(-z3[m]))
    p4 = 1 / (1 + np.exp(-z4[m]))
    yy = y[m]
    rr = yy.mean()
    uu = rr * (1 - rr)

    def sk(p):
        return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yy) ** 2).mean() / uu) * 100000
    print(f'  {lab:<16}행 {m.sum():>7,} ({m.mean():>5.1%})   '
          f'{sk(p3):>7,.0f} -> {sk(p4):>7,.0f}   증분 {sk(p4)-sk(p3):>+7,.0f}'
          f'   전체환산 {(sk(p4)-sk(p3))*m.mean():>+7,.0f}')

print("""
⚠️ 이 증분은 **상한**이다. 실제 모델에는 smoothed / prev1,3,5게임 / 트랙맨 등
   부분적으로 겹치는 피처가 더 있고, 트리는 이 가법 조합만큼 깔끔하게 뽑지 못한다.
   그리고 홀드아웃 -> 리더보드 전달률이 0.3 이다.""")

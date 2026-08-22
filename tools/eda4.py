# EDA 4 — 오라클 천장. 이 문제에서 도달 가능한 최대 스킬은 얼마인가.
#
# eda3 의 분산 분해가 충격적이었다: 2024 에서 투수 정체성의 집단간 분산이 전체의
# 0.97% 인데, **우리 리더보드 점수가 스킬 1.00%** 다. 1위도 1.243% 다.
# Brier Skill Score = 설명한 분산의 비율이므로, 이 숫자들은 직접 비교된다.
#
# 그래서 오라클을 만든다: **정답을 보고** 각 축의 효과를 추정해 예측을 만들면
# 스킬이 얼마인가. 이게 그 축에서 뽑을 수 있는 최대치다.
#
# ⚠️ 자기 자신 누수를 반드시 뺀다 (leave-one-out). 안 빼면 표본잡음이 통째로
#    '설명된 분산' 으로 잡혀 천장이 크게 부풀려진다.
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
df = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
df['count_str'] = df['balls_before'].astype(str) + '-' + df['strikes_before'].astype(str)
df['hand'] = df['pitcher_hand'].astype(str) + 'v' + df['batter_hand'].astype(str)

SE = 2024
d = df[df.season == SE].reset_index(drop=True)
y = d['control_success'].to_numpy(dtype='float64')
r = y.mean()
U = r * (1 - r)
print(f'{SE} 시즌 {len(d):,}행 | 리그평균 {r:.4f} | 불확실성 {U:.5f}\n')


def skill(p):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / U) * 100000


def loo_rate(keys, C):
    """자기 자신을 뺀 그룹 평균, C 로 리그평균에 shrink.
    C 가 클수록 표본잡음을 덜 믿는다. C=0 이면 순수 LOO 평균."""
    g = pd.DataFrame({'k': keys, 'y': y})
    s = g.groupby('k')['y'].transform('sum').to_numpy()
    n = g.groupby('k')['y'].transform('size').to_numpy()
    return ((s - y) + C * r) / ((n - 1) + C)


print('=' * 74)
print('축별 오라클 — 정답을 보고 그 축의 효과만으로 예측 (LOO, shrink C 최적화)')
print(f'{"축":<22}{"그룹수":>8}{"최적C":>8}{"스킬":>10}{"변동설명":>10}')
axes = {
    '투수': d['pitcher_id'],
    '타자': d['batter_id'],
    '볼카운트': d['count_str'],
    '투타 손': d['hand'],
    '주자상태': d['base_state'],
    '이닝': d['inning'],
    '투수팀': d['pitcher_team_id'],
    '투수 x 카운트': d['pitcher_id'].astype(str) + '|' + d['count_str'],
    '투수 x 타자손': d['pitcher_id'].astype(str) + '|' + d['batter_hand'].astype(str),
    '투수 x 타자': d['pitcher_id'].astype(str) + '|' + d['batter_id'].astype(str),
}
best = {}
for lab, k in axes.items():
    k = k.astype(str).to_numpy()
    bs, bc = -9e9, None
    for C in (0, 10, 30, 100, 200, 400, 800, 1600, 3200):
        s = skill(loo_rate(k, C))
        if s > bs:
            bs, bc = s, C
    best[lab] = bs
    print(f'{lab:<22}{len(np.unique(k)):>8,}{bc:>8}{bs:>10,.0f}{bs/100000:>9.3%}')

print('\n' + '=' * 74)
print('축을 합치면 — 로짓 가법 결합 (각 축의 최적 C 사용, 잔차에 순차 적합)')


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


combos = [
    (['투수'], '투수만'),
    (['투수', '타자'], '투수 + 타자'),
    (['투수', '타자', '볼카운트'], '투수 + 타자 + 카운트'),
    (['투수', '타자', '볼카운트', '투타 손', '주자상태', '이닝'], '전부'),
]
for names, lab in combos:
    z = np.full(len(y), lg(r))
    for nm in names:
        k = axes[nm].astype(str).to_numpy()
        p = 1 / (1 + np.exp(-z))
        resid = y - p                      # 현재 예측의 잔차를 그 축으로 설명
        g = pd.DataFrame({'k': k, 'e': resid})
        s = g.groupby('k')['e'].transform('sum').to_numpy()
        n = g.groupby('k')['e'].transform('size').to_numpy()
        C = 200 if nm in ('투수', '타자') else 50
        adj = ((s - resid)) / ((n - 1) + C)       # LOO + shrink
        z = z + adj / max(np.mean(p * (1 - p)), 1e-6) * 0.9
    print(f'  {lab:<26}{skill(1/(1+np.exp(-z))):>10,.0f}')

print('\n' + '=' * 74)
print('실측 리더보드 (스킬 = 점수/100000)')
for lab, v in [('공식 베이스라인', 549.51), ('우리 v5', 990.95),
               ('우리 현행 local_cpu', 1002.17), ('100인 컷', 1119.20), ('1위', 1243.0)]:
    print(f'  {lab:<22}{v:>10,.2f}{v/100000:>9.3%}')
print("""
읽는 법: 오라클은 **2024 정답을 보고** 만든 값이라 실전에서 도달할 수 없다.
  그런데도 오라클이 리더보드 근처면 -> 이 대회는 정보 상한 근처에서 싸우는 중이고
  남은 것은 '투수 효과를 얼마나 잘 추정하느냐' 뿐이다.
  오라클이 훨씬 높으면 -> 우리가 못 뽑아낸 것이 아직 많다.""")

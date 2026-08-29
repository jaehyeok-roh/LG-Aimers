# 구장(홈팀) 효과 — 학습 0, 순수 pandas.
#
# claude.md 에 구장/파크팩터 얘기가 한 번도 없다. 그런데 행 안에서 복원된다:
#   top_bottom = T(초)  -> 수비측이 홈  -> pitcher_team_id 가 홈팀 = 구장
#   top_bottom = B(말)  -> batter_team_id 가 홈팀
#
# 이건 행 A 자기 컬럼만 쓰므로 규정상 자명하게 안전하다. 모델은 세 컬럼을 다 갖고
# 있지만 '홈팀' 은 두 레벨 분기를 써야 만들 수 있다 — nosh(-17.0)/cnt12h(+5.72)가
# 증명한 대로 대칭트리는 **만들 수 있지만 비싸다.**
#
# 왜 제구에 영향을 줄 수 있나: 마운드 상태·조명·배경·기후가 구장마다 다르고,
# 홈 포수진이 그 구장에서 계속 앉는다. 타겟이 "포수가 요구한 곳에 던졌는가" 다.
#
# 재는 것:
#   1) 구장별 원시 성공률 (투수 구성 차이가 섞여 있다)
#   2) 투수 주효과를 뺀 뒤의 구장 편차  <- 이게 진짜 파크 효과
#   3) 반쪽 분할 신뢰도 + 시즌 간 안정성 (eda33 방법)
#   4) 홈/원정 투수 차이
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
C = ['season', 'game_type', 'top_bottom', 'pitcher_id', 'pitcher_team_id',
     'batter_team_id', 'control_success']
df = pd.read_csv('data/train.csv', usecols=C, keep_default_na=False, na_values=_NA)
df = df[df.game_type == 'R'].copy()
print('R %s행 | top_bottom 값 %s' % (f'{len(df):,}', sorted(df.top_bottom.unique())))

top = df.top_bottom.astype(str).str.upper().str.startswith('T')
df['park'] = np.where(top, df.pitcher_team_id, df.batter_team_id)
df['is_home_pitcher'] = top.astype(int)
# 검산: 홈팀은 투수팀 아니면 타자팀이어야 하고, 두 팀은 달라야 한다
assert (df.pitcher_team_id != df.batter_team_id).mean() > 0.99
print('구장 %d개 | 홈투수 비율 %.3f\n' % (df.park.nunique(), df.is_home_pitcher.mean()))

print('=== 1) 구장별 원시 성공률 (투수 구성 섞임) ===')
raw = df.groupby('park').control_success.agg(['mean', 'size'])
raw['dev'] = raw['mean'] - df.control_success.mean()
print(raw.sort_values('dev').to_string(float_format=lambda v: f'{v:.4f}'))

# --- 2) 투수 주효과 제거: 각 투수-시즌 평균을 뺀 잔차의 구장별 평균 ---
df['pdev'] = df.control_success - df.groupby(['pitcher_id', 'season']).control_success.transform('mean')
print('\n=== 2) 투수-시즌 평균을 뺀 뒤의 구장 편차 ===')
adj = df.groupby('park').pdev.agg(['mean', 'size'])
adj['se'] = df.groupby('park').pdev.std() / np.sqrt(adj['size'])
adj['t'] = adj['mean'] / adj['se']
print(adj.sort_values('mean').to_string(float_format=lambda v: f'{v:.5f}'))
print('  구장 편차 std %.5f  (참고: 투수 편차 std 는 0.047 수준)' % adj['mean'].std())

# --- 3) 시즌 간 안정성 ---
print('\n=== 3) 구장 효과의 시즌 간 안정성 ===')
pt = df.groupby(['park', 'season']).pdev.mean().unstack('season')
print(pt.to_string(float_format=lambda v: f'{v:+.4f}'))
cs = [pt[a].corr(pt[b]) for a, b in zip(sorted(pt.columns)[:-1], sorted(pt.columns)[1:])]
print('  인접 시즌 상관: %s | 평균 %.3f'
      % (' '.join(f'{c:+.2f}' for c in cs), float(np.nanmean(cs))))

# --- 4) 홈/원정 ---
print('\n=== 4) 홈 투수 vs 원정 투수 (투수-시즌 평균 제거 후) ===')
h = df.groupby('is_home_pitcher').pdev.agg(['mean', 'size', 'std'])
h['se'] = h['std'] / np.sqrt(h['size'])
print(h.to_string(float_format=lambda v: f'{v:.5f}'))
_d = h['mean'].iloc[1] - h['mean'].iloc[0]
_se = np.sqrt((h['se'] ** 2).sum())
print('  홈-원정 차이 %+.5f (se %.5f, t=%.2f)' % (_d, _se, _d / _se))

# --- 5) 점수 환산 ---
print('\n=== 5) 크기 환산 ===')
U = 0.2494
sig = float(adj['mean'].std())
print('  구장 편차를 완벽히 맞힌다고 가정: 401,000 x %.5f² = %.1f점' % (sig, 401000 * sig ** 2))
print('  홈/원정 차이 기준:                401,000 x %.5f² = %.1f점' % (abs(_d), 401000 * _d ** 2))

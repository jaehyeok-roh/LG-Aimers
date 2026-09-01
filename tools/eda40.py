# EDA 40 — 투구 단위 실패유형 라벨이 asof 차분으로 복원되는가 (팀원 제안 검증).
#
# claude.md 는 이걸 '실행 불가능' 으로 적어뒀지만 그 항목은 wseason5 이전 것이다.
# 검증 로그에 '투수 내 asof_pitcher_n 증분 +1 비율 1.000000' 이 있으므로
# 같은 투수의 인접 행은 정확히 한 투구 차이다. 따라서:
#   누적개수(i+1) - 누적개수(i)  ∈ {0,1}  = 투구 i 의 라벨
# asof 는 '직전까지' 이므로 행 i 와 i+1 로 **투구 i** 의 라벨이 나온다.
#
# ⚠️ 규정: 이 차분은 **train 에서만** 한다. test 에서 하는 것은 주최측이
#    명시적으로 '규칙 위반' 이라고 답한 사안이다 (다른 참가자 08-17).
#    train 유래 값에는 제약이 없다 (DACON.GM 08-19).
import numpy as np
import pandas as pd

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
R = ['success', 'middle', 'reverse', 'ball', 'strike']
cols = (['row_id', 'season', 'pitcher_id', 'control_success', 'asof_pitcher_n',
         'balls_before', 'strikes_before', 'batter_hand']
        + [f'asof_pitcher_{k}_rate' for k in R])
t = pd.read_csv('data/train.csv', usecols=cols, keep_default_na=False, na_values=_NA)
print(f'{len(t):,}행')

t = t.sort_values(['pitcher_id', 'asof_pitcher_n'], kind='mergesort').reset_index(drop=True)
n = t['asof_pitcher_n'].to_numpy(dtype='float64')
g = t['pitcher_id'].to_numpy()
same = np.r_[g[:-1] == g[1:], False]
step = np.r_[n[1:] - n[:-1], np.nan]
ok = same & (step == 1)
print(f'투수 내 인접 행 중 n 증분이 정확히 +1 : {ok.sum():,} / {same.sum():,} '
      f'= {ok.sum()/max(same.sum(),1):.6f}')

print('\n' + '=' * 72)
print('라벨 복원 — 누적개수 차분이 {0,1} 로 떨어지는가')
print(f'{"라벨":<10}{"차분 0/1 비율":>15}{"복원 평균":>11}{"asof 최종 비율":>15}')
print('-' * 72)
lab = {}
for k in R:
    cum = np.round(t[f'asof_pitcher_{k}_rate'].fillna(0).to_numpy(dtype='float64') * n)
    d = np.r_[cum[1:] - cum[:-1], np.nan]
    v = np.where(ok, d, np.nan)
    clean = np.isin(v, [0.0, 1.0]) | np.isnan(v)
    frac = float(np.mean(np.isin(v[ok], [0.0, 1.0])))
    lab[k] = v
    # 대조: 그 투수의 마지막 행 asof 비율 (= 커리어 전체 비율)
    last = t.groupby('pitcher_id').tail(1)
    print(f'{k:<10}{frac:>15.6f}{np.nanmean(v):>11.4f}'
          f'{float(last[f"asof_pitcher_{k}_rate"].mean()):>15.4f}')

print('\n' + '=' * 72)
print('★ 정답 대조 — success 라벨은 control_success 와 일치해야 한다')
y = t['control_success'].to_numpy(dtype='float64')
m = ok & np.isin(lab['success'], [0.0, 1.0])
print(f'  복원 success  vs  control_success   일치율 {float((lab["success"][m]==y[m]).mean()):.6f}'
      f'   (표본 {m.sum():,})')
print('  ↑ 1.0 이면 차분 방식이 정확하다는 증거다 (success 는 정답을 아니까 검산이 된다)')

print('\n' + '=' * 72)
print('실패유형의 구조 (복원 라벨끼리)')
mm = ok & np.all([np.isin(lab[k], [0.0, 1.0]) for k in R], axis=0)
D = pd.DataFrame({k: lab[k][mm] for k in R})
D['y'] = y[mm]
print(f'  표본 {len(D):,}')
print(f'  실패(y=0) 중  middle {D[D.y==0]["middle"].mean():.3f}  '
      f'reverse {D[D.y==0]["reverse"].mean():.3f}')
print(f'  성공(y=1) 중  middle {D[D.y==1]["middle"].mean():.3f}  '
      f'reverse {D[D.y==1]["reverse"].mean():.3f}')
print(f'  ball {D["ball"].mean():.3f} | strike {D["strike"].mean():.3f} | '
      f'합 {(D["ball"]+D["strike"]).mean():.3f}')
print(f'  ball  vs y 상관 {np.corrcoef(D["ball"], D["y"])[0,1]:+.3f}')
print(f'  middle+reverse 합 {(D["middle"]+D["reverse"]).mean():.3f} | '
      f'1-y 평균 {(1-D["y"]).mean():.3f}')

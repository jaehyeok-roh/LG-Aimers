# 팀 13 관여 x 2023-05 체제 전환 지시자 — v10wzc(1,121.89) 위.
#
# 발견 경위 (2026-08-29): 다른 참가자가 주최측에 합법성을 문의한 파생변수를 보고
# 직접 검증했다 (tools/eda50.py, eda51.py). 우리 eda48 이 같은 것을 쥐고 놓쳤다 —
# 구장(홈팀) 편차에서 13 만 t=-8.95 로 튀었는데 10개 구장 평균에 묻어서 닫아버렸다.
#
# R(1군) 전용, team13 관여 (커버리지 19.9%):
#   2019 -0.0969  2020 -0.0739  2021 -0.0187  2022 -0.0407  2023 **+0.0507**  2024 **+0.0490**
#   전환점: 2023-04 -0.0220 -> 2023-05 **+0.0469**
#   투수측 +0.0513 / 타자측 +0.0467  <- 대칭이므로 경기 단위 효과 (투수 실력이 아니다)
#   투수-시즌 평균 제거 후에도 +0.0265 생존
#   팀별 전후변화: 13 이 +0.1074, 2위 팀이 -0.0249 로 4.3배 차이 — 13 만 특별하다
#
# 왜 모델이 못 잡을 수 있나:
#   (1) 'team13 관여' 는 두 컬럼의 **OR** 이라 대칭트리가 레벨 두 개를 써야 만든다.
#       is_same_hand 를 빼면 -17.0 이었던 그 자리다 ("못 만드는 게 아니라 비싸다").
#   (2) 전환점이 2023년 5월인데 DROP_CAL 로 game_month 를 버렸다. season 경계로
#       근사하려면 레벨이 하나 더 든다.
#   (3) 6시즌을 뭉치면 팀 13 효과가 상쇄된다 (4시즌 음수 / 2시즌 양수).
#       2025 에 필요한 것은 **양수 체제 하나**뿐이다.
#
# ⚠️ auxrev(전달률 0.10) 함정은 여기 없다 — 2025 행은 전부 전환 후라 지시자가
#    상수 1 이고, season 을 재료로 만든 값이 학습 시대에 얼어붙는 구조가 아니다.
#
# 규정: 행 A 자기 컬럼(team_id / season / game_month / game_type) + train 라벨 통계로
#       고정한 상수. 주최측 허용 범위 1·2·4번이다. CatBoost 는 이미 pitcher_team_id 에
#       CTR(타겟 인코딩)을 돌리고 있으므로 부류상 새로운 것이 아니다.
#       ⚠️ 해당 참가자가 문의 중이고 답변은 아직 없다.
#
# 수치형으로 준다 (범주형 아님). 범주형 조합은 정확히 2개에서 포화하고 4개면 -4.61 인데
# (CTR 경쟁), 0/1 수치 플래그는 CTR 을 만들지 않으므로 그 경쟁에 끼지 않는다.
import ast
import json
import os
import sys

BASE = os.environ.get('T13_BASE', 'aimers_v10wzc.ipynb')
OUT = os.environ.get('T13_OUT', 'experiments/v10w/aimers_v10wt13.ipynb')
ZIP = os.environ.get('T13_ZIP', 'submit_v10wt13.zip')
OLDZIP = os.environ.get('T13_OLDZIP', 'submit_v10wzc.zip')

nb = json.load(open(BASE, encoding='utf-8'))
cells = nb['cells']
code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']


def src(i):
    return ''.join(cells[i]['source'])


def sub(i, old, new, n=1):
    s = src(i)
    if s.count(old) != n:
        sys.exit('셀 %d: 앵커 %d개 기대, %d개\n  %s' % (i, n, s.count(old), old[:150]))
    cells[i]['source'] = s.replace(old, new).splitlines(keepends=True)


def find(anchor):
    hit = [i for i in code if anchor in src(i)]
    if len(hit) != 1:
        sys.exit('앵커 %d곳: %s' % (len(hit), anchor[:110]))
    return hit[0]


# ---------------------------------------------- 1) 지시자 두 개 (STEPS_SRC 안이라 script.py 도 따라온다)
A1 = "    df_proc['cnt12'] = (b.astype(int).astype(str) + '-' + s.astype(int).astype(str))"
N1 = A1 + """
    # 팀13 관여 x 2023-05 체제 전환 (tools/eda51.py). 두 컬럼의 OR 이라 대칭트리는
    # 레벨 두 개를 써야 만든다 — nosh(-17.0)가 증명한 자리. 수치형으로 준다(CTR 경쟁 회피).
    _t13 = ((df_proc['pitcher_team_id'].astype('int64') == 13)
            | (df_proc['batter_team_id'].astype('int64') == 13))
    df_proc['t13'] = _t13.astype('float64')
    # 전환 시점은 train 라벨 통계에서 고정한 상수다 (2023-04 -0.0220 -> 2023-05 +0.0469).
    # 2025 행은 season>2023 이라 전부 1 -> 학습 시대에 얼어붙는 성분이 없다.
    _sea = df_proc['season'].astype('int64')
    _mon = df_proc['game_month'].astype('int64')
    _post = ((_sea > 2023)
             | ((_sea == 2023) & ((_mon >= 5) | (df_proc['game_type'].astype(str) == 'F'))))
    df_proc['t13_post'] = (_t13 & _post).astype('float64')"""
sub(find(A1), A1, N1)
print('step4 에 t13 / t13_post 삽입 (STEPS_SRC 공유 -> script.py 자동 반영)')

# ---------------------------------------------- 2) 학습 전 자체 검증 (4-12)
GUARD = '''
# ---- t13 검증 (학습 전에 터뜨린다, 4-12) ----
for _c in ("t13", "t13_post"):
    assert _c in X_full.columns, f"{_c} 가 X_full 에 없다"
    assert _c not in cat_features, f"{_c} 는 수치형이어야 한다 (CTR 경쟁 회피)"
    assert set(np.unique(X_full[_c].to_numpy())) <= {0.0, 1.0}, _c
_gt = df_processed["game_type"].astype(str).to_numpy()
_t = X_full["t13"].to_numpy()
_p = X_full["t13_post"].to_numpy()
_sea = df_processed["season"].astype("int64").to_numpy()
print(f"  t13 커버리지 전체 {_t.mean():.4f} | R {_t[_gt=='R'].mean():.4f} | F {_t[_gt=='F'].mean():.4f}")
assert 0.18 < _t[_gt == "R"].mean() < 0.22, _t[_gt == "R"].mean()   # eda51: 19.86%
assert _t[_gt == "F"].mean() > 0.999, _t[_gt == "F"].mean()          # F 는 전부 관여
assert (_p <= _t).all(), "t13_post 가 t13 을 넘는다"
# 2024 행은 전부 전환 후여야 한다 (2025 도 마찬가지가 된다)
_m24 = _sea == 2024
assert (_p[_m24] == _t[_m24]).all(), "2024 에서 t13_post != t13"
# 2019~2022 는 전환 전이어야 한다
_mpre = _sea <= 2022
assert _p[_mpre].sum() == 0, "2022 이하에 전환 후가 있다"
print(f"t13 검증 OK | 피처 {X_full.shape[1]}개")

'''
ANC = 'print("최종 파라미터:", BEST_PARAMS)'
sub(find(ANC), ANC, GUARD + ANC)
print('자체 검증 셀 삽입 (학습 전)')

# ---------------------------------------------- 3) 산출물 이름
sub(find(OLDZIP), OLDZIP, ZIP)
print('zip -> %s' % ZIP)

for i in code:
    ast.parse(src(i))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

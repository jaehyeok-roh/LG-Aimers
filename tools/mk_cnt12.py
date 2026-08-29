# 정확한 12칸 카운트를 범주형 하나로 추가한다 (v10wp 위에).
#
# 지금 모델이 카운트에 대해 가진 것:
#   balls_before / strikes_before   **수치형** — 임계값 분할만 된다
#   count_advantage 4칸             ⚠️ 'None' 이 **0-0 과 3-2 를 같이 담는다** (4-3)
#   is_full_count / is_first_pitch  그 두 칸만 따로 빼낸 땜질
# 셋을 합쳐 중요도 3.07% 를 쓰면서도 "0-2 와 2-0 은 다르다" 를 직접 말하지 못한다.
#
# 스크리너 실측 (wsboth 기준, 두 시즌 양수):
#   2024  888.6 -> 896.9  **+8.3**
#   2023  628.5 -> 631.9  **+3.4**
# 그리고 `nosh`(is_same_hand 하나 제거)가 **-17.0 / -13.0** 으로 두 시즌 합의 음수다
# = 손수 만든 저카디널리티 범주형 조합 축이 실재한다는 직접 증거.
#
# ★ 구현이 안전한 이유: cnt12 는 step4 안에서 balls/strikes 만으로 만들어진다.
#   STEPS_SRC 한 곳만 고치면 학습 노트북과 script.py 가 **자동으로 같이 간다**
#   (2장 설계 원칙). 룩업도, 상수도, 행 순서 의존도 없다 — 지금까지 사고가 났던
#   세 경로(트랙맨 exact/asof, ws_* 소실, pf_* 행순서)를 전부 비껴간다.
import ast
import json
import os
import sys

BASE = os.environ.get('C12_BASE', 'experiments/v10w/aimers_v10wp.ipynb')
OUT = os.environ.get('C12_OUT', 'experiments/v10w/aimers_v10wc.ipynb')

nb = json.load(open(BASE, encoding='utf-8'))
cells = nb['cells']
code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']


def src(i):
    return ''.join(cells[i]['source'])


def setsrc(i, s):
    cells[i]['source'] = s.splitlines(keepends=True)


def sub(i, old, new, n=1):
    s = src(i)
    if s.count(old) != n:
        sys.exit(f'셀 {i}: 앵커 {n}개 기대, {s.count(old)}개\n  {old[:100]}')
    setsrc(i, s.replace(old, new))


# ------------------------------------------------- 1) step4 에서 12칸 카운트 생성
ANCHOR = """    df_proc['count_advantage'] = np.select(
        [pitcher_ahead, batter_ahead, neutral], ['Pitcher', 'Batter', 'Neutral'], default='None')"""
NEWSRC = ANCHOR + """
    # 정확한 12칸 카운트. count_advantage 의 'None' 이 0-0 과 3-2 를 같이 담고
    # is_first_pitch / is_full_count 가 그걸 되돌리려는 땜질이라, 12칸을 직접 준다.
    # 대칭트리는 balls/strikes 두 레벨로 이걸 만들 수 **있지만** 비싸다 —
    # 범주형으로 주면 CTR 하나로 끝난다 (nosh -17.0/-13.0 이 이 축을 증명했다).
    df_proc['cnt12'] = (b.astype(int).astype(str) + '-' + s.astype(int).astype(str))
    # hand4 = pitcher_hand x batter_hand (4칸). is_same_hand 는 좌투vs우타와
    # 우투vs좌타를 한 칸에 뭉갠다. 단독으로는 뒤집히지만(2024 +6.5 / 2023 -11.4)
    # cnt12 와 함께면 두 시즌 다 양수다 (cnt12h +19.0 / +4.3, 조합 중 최대).
    df_proc['hand4'] = (df_proc['pitcher_hand'].astype(str)
                        + df_proc['batter_hand'].astype(str))"""

n = 0
for i in code:
    if ANCHOR in src(i):
        sub(i, ANCHOR, NEWSRC)
        n += 1
if n == 0:
    sys.exit('step4 앵커를 못 찾았다')
print(f'step4 수정: {n}곳 (STEPS_SRC 와 실행본 양쪽)')

# ------------------------------------------------- 2) 범주형 목록에 등록
CAT = """                        'is_heating_up', 'is_cooling_down']"""
CATNEW = """                        'is_heating_up', 'is_cooling_down', 'cnt12', 'hand4']"""
m = 0
for i in code:
    if CAT in src(i):
        sub(i, CAT, CATNEW, n=src(i).count(CAT))
        m += src(i).count(CATNEW)
if m == 0:
    sys.exit('created_cat_cols 앵커를 못 찾았다')
print(f'범주형 등록: {m}곳')

# ------------------------------------------------- 3) 산출물 이름
for i in code:
    if 'ZIP_PATH = "submit_v10wp.zip"' in src(i):
        sub(i, 'ZIP_PATH = "submit_v10wp.zip"', 'ZIP_PATH = "submit_v10wc.zip"')
        break

# ------------------------------------------------- 4) 자체 검증 셀
# ⚠️ claude.md 4-12: 학습(63분) **뒤에** 오는 셀의 NameError 로 GPU 를 날린 적이 있다.
#    반드시 cat_features / X_full 이 이미 존재하고 학습은 아직 안 한 지점에 넣는다.
GUARD = """
# ---- cnt12 자체 검증 (학습 전에 터뜨린다) ----
assert "cnt12" in X_full.columns, "cnt12 가 X_full 에 없다"
assert "cnt12" in cat_features and "hand4" in cat_features, "범주형 등록 누락"
assert sorted(X_full["hand4"].astype(str).unique()) == ["11","12","21","22"], "hand4 가 4칸이 아니다"
_u = sorted(X_full["cnt12"].astype(str).unique())
assert len(_u) == 12, f"cnt12 가 12칸이 아니라 {len(_u)}칸이다: {_u}"
print("cnt12 검증 OK:", _u)

"""
ANC2 = 'print("최종 파라미터:", BEST_PARAMS)'
done = False
for i in code:
    if ANC2 in src(i):
        sub(i, ANC2, GUARD + ANC2)
        done = True
        break
if not done:
    sys.exit('가드 앵커를 못 찾았다')
print('자체 검증 셀 삽입 완료 (학습 전)')

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'{OUT} 생성 — {len(code)}셀 문법 OK')

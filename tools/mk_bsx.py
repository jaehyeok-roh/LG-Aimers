# base_state 를 재료로 한 범주형 조합 둘을 추가한다 (aimers_v10wzc.ipynb 위에).
#
#   bs_out (8x3=24)  base_state x outs_before      — 야구의 표준 상태변수(득점 기대값 격자)
#   bs_cnt (8x4=32)  base_state x count_advantage  — 주자 있을 때 불리한 카운트의 의미가 다르다
#
# 왜 지금인가: `cnt12h`(cnt12 + hand4)가 리더보드 **+5.72** 로 나오면서
# 범주형 조합 축이 **5분류 base 위에서 살아 있다**는 것이 실측됐다 (2026-08-27).
#
# 스크리너 실측 (wsboth = 이진 base 기준, 두 시즌 양수):
#   bsx     2024 +3.8 / 2023 +16.2
#   cnt12h  2024 +19.0 / 2023 +4.3   -> 실제 배포(5분류 base)에서 LB +5.72
#
# ⚠️ claude.md: "두 개를 넘기면 상쇄된다" (`bsx` +16.2 + `cnt12` +3.4 = **-1.2**).
#    이 판은 조합이 넷이 되므로 그 경고의 정면이다. 다만 그 측정은 **이진 base** 위였고,
#    같은 표에서 `cnt12h` 는 2023 +4.3 이었는데 실제로는 +5.72 였다 —
#    이진 base 의 비가법성이 5분류 base 로 옮겨간다는 보장이 없다. 리더보드로 판정한다.
#
# ⚠️ 2024 홀드아웃은 판정에 쓰지 않는다. 거짓 음성을 **세 번** 냈다:
#    wsbat +3.0->LB +30.11 / mcaux +2.8->+12.85 / cnt12h **-6 -> +5.72**.
#    마지막 것은 명백한 음수였는데 양수였다 — 어느 방향으로도 거를 수 없다는 뜻이다.
#
# ★ 구현이 안전한 이유는 mk_cnt12.py 와 같다: base_state / outs_before /
#   count_advantage 는 step4 시점에 이미 있는 컬럼이라 STEPS_SRC 한 곳만 고치면
#   학습 노트북과 script.py 가 자동으로 같이 간다. 룩업도 상수도 행 순서 의존도 없다.
import ast
import json
import os
import sys

BASE = os.environ.get('BSX_BASE', 'aimers_v10wzc.ipynb')
OUT = os.environ.get('BSX_OUT', 'aimers_v10wzb.ipynb')
ZIP = os.environ.get('BSX_ZIP', 'submit_v10wzb.zip')

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
        sys.exit('셀 %d: 앵커 %d개 기대, %d개\n  %s' % (i, n, s.count(old), old[:120]))
    setsrc(i, s.replace(old, new))


# ------------------------------------------------- 1) step4 에서 조합 생성
# mk_cnt12.py 가 넣어둔 hand4 줄 바로 뒤에 붙인다 (그 지점엔 df_proc 이 살아 있다).
ANCHOR = """    df_proc['hand4'] = (df_proc['pitcher_hand'].astype(str)
                        + df_proc['batter_hand'].astype(str))"""
NEWSRC = ANCHOR + """
    # base_state x outs = 야구의 표준 상태변수(득점 기대값 격자). 대칭트리는 두 레벨로
    # 만들 수 있지만 비싸다 -- 범주형으로 주면 CTR 하나로 끝난다 (nosh -17.0/-13.0).
    df_proc['bs_out'] = (df_proc['base_state'].astype(str) + '|'
                         + df_proc['outs_before'].astype(int).astype(str))
    # base_state x count_advantage. 주자가 있으면 불리한 카운트의 의미가 달라진다
    # (거를 여유가 없다 vs 있다).
    df_proc['bs_cnt'] = (df_proc['base_state'].astype(str) + '|'
                         + df_proc['count_advantage'].astype(str))"""

n = 0
for i in code:
    if ANCHOR in src(i):
        sub(i, ANCHOR, NEWSRC, n=src(i).count(ANCHOR))
        n += 1
if n == 0:
    sys.exit('step4(hand4) 앵커를 못 찾았다 -- base 가 v10wzc 가 맞는지 확인할 것')
print('step4 수정: %d곳 (STEPS_SRC 와 실행본 양쪽)' % n)

# ------------------------------------------------- 2) 범주형 목록에 등록
CAT = """'cnt12', 'hand4']"""
CATNEW = """'cnt12', 'hand4', 'bs_out', 'bs_cnt']"""
m = 0
for i in code:
    c = src(i).count(CAT)
    if c:
        sub(i, CAT, CATNEW, n=c)
        m += c
if m == 0:
    sys.exit('created_cat_cols 앵커를 못 찾았다')
print('범주형 등록: %d곳' % m)

# ------------------------------------------------- 3) 자체 검증 (학습 전에 터뜨린다)
# ⚠️ 4-12: 학습(63분) **뒤에** 오는 셀의 NameError 로 GPU 를 날린 적이 있다.
GUARD = """
# ---- bsx 자체 검증 (학습 전에 터뜨린다) ----
for _c in ("bs_out", "bs_cnt"):
    assert _c in X_full.columns, f"{_c} 가 X_full 에 없다"
    assert _c in cat_features, f"{_c} 범주형 등록 누락"
_no = X_full["bs_out"].astype(str).nunique()
_nc = X_full["bs_cnt"].astype(str).nunique()
assert 16 <= _no <= 24, f"bs_out 칸수가 {_no} 개다 (8x3=24 기대)"
assert 20 <= _nc <= 32, f"bs_cnt 칸수가 {_nc} 개다 (8x4=32 기대)"
# 'None' 이 살아 있는지 (4-3: count_advantage 의 'None' 은 0-0/3-2 를 뜻하는 실제 값)
assert any("None" in _v for _v in X_full["bs_cnt"].astype(str).unique()), \\
    "bs_cnt 에 'None' 이 없다 -- count_advantage 가 NaN 으로 읽혔을 수 있다 (4-3)"
print(f"bsx 검증 OK: bs_out {_no}칸 / bs_cnt {_nc}칸")

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

# ------------------------------------------------- 4) 산출물 이름
for i in code:
    s = src(i)
    if 'ZIP_PATH' in s and 'submit_v10wzc.zip' in s:
        sub(i, 'submit_v10wzc.zip', ZIP)
        print('zip -> %s' % ZIP)
        break

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

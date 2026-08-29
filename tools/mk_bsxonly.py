# bsx **단독** — v10wz(5분류) 위에 bs_out + bs_cnt 만 얹는다 (cnt12/hand4 없이).
#
# 근거 (2026-08-27 저녁): 범주형 조합 축의 리더보드 실측 세 개가 2023 스크리너
# 순위와 정확히 일치했다.
#
#   2023 스크리너:  bsx +16.2  >  cnt12h +4.3  >  cnt12+bsx -1.2
#   리더보드:                      cnt12h +5.72  >  v10wzb -4.61
#
# 2023 이 이 축에서 가장 크다고 말한 구성(bsx 단독)은 아직 리더보드에 없다.
# 0.3 규칙으로 +4.9, cnt12h 의 실현 전달률(4.3 -> 5.72)로는 그 이상.
# cnt12h 와는 **배타적**이다 — 결합은 -4.61 로 실측 기각됐다. 이기면 교체한다.
#
#   bs_out (8x3=24)  base_state x outs_before      득점 기대값 격자
#   bs_cnt (8x4=32)  base_state x count_advantage  주자 있을 때 불리한 카운트의 의미가 다르다
import ast
import json
import os
import sys

BASE = os.environ.get('BSO_BASE', 'experiments/v10w/aimers_v10wz.ipynb')
OUT = os.environ.get('BSO_OUT', 'experiments/v10w/aimers_v10wbs.ipynb')
ZIP = os.environ.get('BSO_ZIP', 'submit_v10wbs.zip')

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
ANCHOR = """    df_proc['count_advantage'] = np.select(
        [pitcher_ahead, batter_ahead, neutral], ['Pitcher', 'Batter', 'Neutral'], default='None')"""
NEWSRC = ANCHOR + """
    # base_state x outs = 야구의 표준 상태변수(득점 기대값 격자). 대칭트리는 두 레벨로
    # 만들 수 있지만 비싸다 -- 범주형으로 주면 CTR 하나로 끝난다 (nosh -17.0/-13.0).
    df_proc['bs_out'] = (df_proc['base_state'].astype(str) + '|'
                         + df_proc['outs_before'].astype(int).astype(str))
    # base_state x count_advantage. 주자가 있으면 불리한 카운트의 의미가 달라진다.
    df_proc['bs_cnt'] = (df_proc['base_state'].astype(str) + '|'
                         + df_proc['count_advantage'].astype(str))"""
n = 0
for i in code:
    if ANCHOR in src(i):
        sub(i, ANCHOR, NEWSRC)
        n += 1
if n == 0:
    sys.exit('step4 앵커를 못 찾았다')
print('step4 수정: %d곳 (STEPS_SRC 와 실행본 양쪽)' % n)

# ------------------------------------------------- 2) 범주형 목록에 등록
CAT = """'is_heating_up', 'is_cooling_down']"""
CATNEW = """'is_heating_up', 'is_cooling_down', 'bs_out', 'bs_cnt']"""
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
GUARD = """
# ---- bsx 자체 검증 (학습 전에 터뜨린다, 4-12) ----
for _c in ("bs_out", "bs_cnt"):
    assert _c in X_full.columns, f"{_c} 가 X_full 에 없다"
    assert _c in cat_features, f"{_c} 범주형 등록 누락"
_no = X_full["bs_out"].astype(str).nunique()
_nc = X_full["bs_cnt"].astype(str).nunique()
assert 16 <= _no <= 24, f"bs_out 칸수 {_no} (24 기대)"
assert 20 <= _nc <= 32, f"bs_cnt 칸수 {_nc} (32 기대)"
# 4-3: count_advantage 의 'None' 은 0-0/3-2 를 뜻하는 실제 값이다
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
    if 'ZIP_PATH' in s and 'submit_v10wz.zip' in s:
        sub(i, 'submit_v10wz.zip', ZIP)
        print('zip -> %s' % ZIP)
        break

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

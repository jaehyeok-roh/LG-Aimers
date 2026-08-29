# 조건부 투수통계의 카운트 키를 4단계 -> 정확한 12칸으로 — v10wzc(1,121.89) 위.
#
# 근거 (2026-08-29, tools/eda46.py):
#   eda42 에서 투수 유리 카운트(28.4%)가 행당으로 타자 유리 카운트의 45% 밖에 못 번다.
#   그게 '없는 신호' 인지 '못 잡는 신호' 인지 반쪽 분할로 갈랐다:
#
#     구간       신뢰도   신호std   (신호std)²/U = 행당 가용
#     Batter     0.526   0.0470    886   <- 기준
#     Pitcher    0.396   0.0377    570   = 기준의 64%
#
#   가용 64% 인데 실측 45% 다. 일부는 비가역이지만(신뢰도가 낮다) 전부는 아니다.
#
# 그리고 12칸으로 쪼개면 카운트마다 신호 크기가 완전히 다르다:
#     1-0 0.0527 / 2-1 0.0604 / 1-2 0.0508   vs   1-1 0.0228 / 2-0 0.0346 / 0-2 0.0340
#
# 그런데 cond_pc / cond_phc 는 아직 **4단계 count_advantage** 로 조건을 건다.
# 그 4단계의 'None' 은 claude.md 가 직접 지적한 대로 **0-0 과 3-2 를 같은 칸**에 담는다.
# cnt12 는 8/27 에 범주형으로만 들어갔고(+5.72), max_ctr_complexity=1 이라
# CatBoost 가 pitcher x cnt12 CTR 을 자동 생성하지도 않는다.
# 즉 '투수 x 정확한 카운트' 는 모델에 없다. 그리고 이것은 다른 행에서 정보를
# 끌어오는 부류다 — 이득이 났던 통로(cond_* +27, 트랙맨 +9, wseason +61)와 같다.
#
# C 는 칸이 3배 잘게 쪼개지므로 1/3 로 낮춘다 (shrink 비율 n/(n+C) 를 유지).
# claude.md 는 shrink C 축이 넓다고 기록했으므로(10/30/100 이 ±7 안) 정확한 값은
# 민감하지 않다.
#
# ⚠️ **교체**이지 추가가 아니다. 같은 축에 두 벌을 얹으면 상쇄된다 —
#    범주형 조합이 정확히 2개에서 포화하고(4개 -4.61), 보조 타겟이 1개에서 포화한
#    것과 같은 비가법성이다 (claude.md 4번 확인).
import ast
import json
import os
import sys

BASE = os.environ.get('C12_BASE', 'aimers_v10wzc.ipynb')
OUT = os.environ.get('C12_OUT', 'experiments/v10w/aimers_v10wc12.ipynb')
ZIP = os.environ.get('C12_ZIP', 'submit_v10wc12.zip')
OLDZIP = os.environ.get('C12_OLDZIP', 'submit_v10wzc.zip')
C_PC = int(os.environ.get('C12_CPC', '35'))     # 100 / 3
C_PHC = int(os.environ.get('C12_CPHC', '17'))   # 50 / 3

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


# ---------------------------------------------- 1) 학습측 COND_SPECS
A1 = """    (['pitcher_id', 'count_advantage'],                 100, 'cond_pc'),
    (['pitcher_id', 'batter_hand'],                     100, 'cond_ph'),
    (['pitcher_id', 'batter_hand', 'count_advantage'],   50, 'cond_phc'),"""
N1 = """    (['pitcher_id', 'cnt12'],                           %3d, 'cond_pc'),
    (['pitcher_id', 'batter_hand'],                     100, 'cond_ph'),
    (['pitcher_id', 'batter_hand', 'cnt12'],            %3d, 'cond_phc'),""" % (C_PC, C_PHC)
sub(find(A1), A1, N1)
print('COND_SPECS: count_advantage -> cnt12 (C %d / %d)' % (C_PC, C_PHC))

# ---------------------------------------------- 2) 추론측(script.py) 키 목록
A2 = """                       ("cond_pc",  ["pitcher_id", "count_advantage"]),
                       ("cond_ph",  ["pitcher_id", "batter_hand"]),
                       ("cond_phc", ["pitcher_id", "batter_hand", "count_advantage"]),"""
N2 = """                       ("cond_pc",  ["pitcher_id", "cnt12"]),
                       ("cond_ph",  ["pitcher_id", "batter_hand"]),
                       ("cond_phc", ["pitcher_id", "batter_hand", "cnt12"]),"""
sub(find(A2), A2, N2)
print('script.py 키 목록도 교체 (학습/추론 규칙 일치)')

# ---------------------------------------------- 3) 룩업 CSV 라운드트립 검증 교체
# 4-3 검증은 count_advantage 의 'None'(0-0/3-2) 이 read_csv 에서 NaN 이 되는 사고를
# 막으려는 것이다. 키가 cnt12 면 그 값들('0-0'~'3-2')은 NaN 으로 읽힐 여지가 없으므로
# 검증 대상이 바뀐다 — 다만 검증을 **지우지는 않는다**. 칸 수와 결측을 대신 본다.
A3 = '''    assert (_c['count_advantage'].astype(str) == 'None').sum() > 0, "'None' 유실"'''
N3 = '''    # 키가 cnt12 로 바뀌었다. '0-0'~'3-2' 는 read_csv 가 NaN 으로 바꿀 값이 아니므로
    # 4-3 의 'None' 검증은 대상이 없다. 대신 칸 수와 결측을 본다.
    assert 'cnt12' in _c.columns, list(_c.columns)
    assert _c['cnt12'].isna().sum() == 0, "cnt12 에 NaN"
    assert _c['cnt12'].astype(str).nunique() == 12, _c['cnt12'].astype(str).nunique()'''
for _i in code:
    if A3 in src(_i):
        sub(_i, A3, N3)
        print('룩업 CSV 검증을 cnt12 용으로 교체')
        break
else:
    sys.exit('라운드트립 검증 앵커를 못 찾았다')

# ---------------------------------------------- 4) 학습 전 자체 검증 (4-12)
GUARD = '''
# ---- cond_pc12 검증 (학습 전에 터뜨린다, 4-12) ----
# 트랙맨 exact/asof 36점 사고가 '학습과 추론이 다른 키를 쓴' 유형이었다.
_spec = {n: k for k, _, n in COND_SPECS}
assert _spec["cond_pc"] == ["pitcher_id", "cnt12"], _spec["cond_pc"]
assert _spec["cond_phc"] == ["pitcher_id", "batter_hand", "cnt12"], _spec["cond_phc"]
assert "cnt12" in df_processed.columns, "cnt12 가 cond 부착 시점에 없다 (step4 순서 확인)"
_nc = df_processed["cnt12"].astype(str).nunique()
assert _nc == 12, f"cnt12 칸수 {_nc} (12 기대)"
for _c in ("cond_pc", "cond_phc"):
    assert _c in X_full.columns, f"{_c} 누락"
    _m = X_full[_c].isna().mean()
    assert _m < 0.45, f"{_c} 결측 {_m:.1%} -- 키 불일치로 병합이 깨졌을 수 있다"
    print(f"  {_c}: 결측 {_m:.1%}")
print(f"cond_pc12 검증 OK | 피처 {X_full.shape[1]}개")

'''
ANC = 'print("최종 파라미터:", BEST_PARAMS)'
sub(find(ANC), ANC, GUARD + ANC)
print('자체 검증 셀 삽입 (학습 전)')

# ---------------------------------------------- 5) 산출물 이름
sub(find(OLDZIP), OLDZIP, ZIP)
print('zip -> %s' % ZIP)

for i in code:
    ast.parse(src(i))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

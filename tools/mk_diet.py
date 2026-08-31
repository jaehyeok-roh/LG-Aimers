# 트랙맨 유래 피처 20개를 뺀다 — 가장 낡고 가장 전이가 불안정한 블록이다.
#
#   python tools/mk_diet.py
#
# 대상: `past_*` 19개 + `expected_control_difficulty` (둘 다 트랙맨 집계에서 나온다).
#
# 근거 셋.
#  ① **가장 낡다.** 트랙맨에 2025 가 없으므로 배포 시 2024 값이 그대로 간다.
#     eda(2026-08-22) 실측: 같은 시즌 트랙맨의 증분 R2 는 +0.047 인데 **직전 시즌
#     기준으로는 +0.011** 이다. 77% 가 staleness 로 사라진다.
#  ② **전이가 가장 불안정하다.** eda39 의 위험 상위 16 중 **12개가 past_*** 이고
#     프로파일 상관이 -0.20 ~ -0.76 이다.
#  ③ **통째로 빼도 0 이다.** `notm` 스크리너 +2.6(2023) / -2.2(2024).
#     그런데 **리더보드에는 한 번도 안 올렸다** -- 죽은 하네스에서만 잰 값이다.
#
# ⚠️ 기대값은 0 이다. 그런데 지금 필요한 건 기대값이 아니라 **분산**이다:
#    격차 17.5(마감 시점 ~19.5)인데 살아 있는 최선이 +8 이라 기대값 최대화로는
#    산술이 안 맞는다. 최고점 채점이라 하방이 0 이므로 **넓은 분포**를 사는 게 맞다.
#    피처 20개를 빼면 CTR·분할 경쟁이 통째로 바뀌므로 결과가 어디로든 갈 수 있다.
#
# 구현: `DEAD_FEATURES` 에 더하기만 한다 (drop_cols 에 들어간다). 트랙맨 merge 자체는
# 그대로 두므로 script.py 경로는 안 건드린다 -- 학습/추론 불일치 위험이 0 이다.
import ast
import json
import os
import sys

BASE = os.environ.get('FD_BASE', 'experiments/v10w/aimers_v10wtor.ipynb')
OUT = os.environ.get('FD_OUT', 'experiments/v10w/aimers_v10wfd.ipynb')
ZIP = os.environ.get('FD_ZIP', 'submit_v10wfd.zip')
OLDZIP = os.environ.get('FD_OLDZIP', 'submit_v10wtor.zip')

nb = json.load(open(BASE, encoding='utf-8'))
cells = nb['cells']
code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']
src = lambda i: ''.join(cells[i]['source'])


def find(a):
    h = [i for i in code if a in src(i)]
    if len(h) != 1:
        sys.exit('앵커 %d곳: %s' % (len(h), a[:90]))
    return h[0]


def sub(i, old, new, n=1):
    s = src(i)
    if s.count(old) != n:
        sys.exit('셀 %d: 앵커 %d개 기대, %d개' % (i, n, s.count(old)))
    cells[i]['source'] = s.replace(old, new).splitlines(keepends=True)


A = """DEAD_FEATURES = ['is_long_relief', 'is_short_relief',
                 'is_strict_inherited_runner', 'pitches_per_inning']"""
N = A + """
# mk_diet: 트랙맨 유래 20개를 뺀다. 이름은 런타임에 df_processed 에서 뽑는다
# (하드코딩하면 파이프라인이 바뀔 때 조용히 어긋난다).
DIET_TRACKMAN = True"""
sub(find(A), A, N)
print('DIET_TRACKMAN 플래그 삽입')

# feature_cols 를 만드는 자리에서 실제로 뺀다
A2 = "feature_cols = [c for c in df_processed.columns if c not in drop_cols]"
N2 = """_diet = ([c for c in df_processed.columns if c.startswith('past_')]
         + ['expected_control_difficulty']) if DIET_TRACKMAN else []
drop_cols = list(drop_cols) + [c for c in _diet if c in df_processed.columns]
feature_cols = [c for c in df_processed.columns if c not in drop_cols]
print("  다이어트: 트랙맨 유래 %d개 제거" % len(_diet))"""
sub(find(A2), A2, N2)
print('feature_cols 에서 제거')

GUARD = '''
# ---- diet 검증 (학습 전에 터뜨린다, 4-12) ----
_bad = [c for c in X_full.columns if c.startswith("past_")
        or c == "expected_control_difficulty"]
assert not _bad, "트랙맨 피처가 남아 있다: %s" % _bad[:5]
assert 105 <= X_full.shape[1] <= 115, X_full.shape[1]   # v10wtor 130 - 20 = 110
# 나머지 축은 그대로여야 한다 (기준선이 v10wtor 다)
for _c in ("t13", "t13_post", "tor16", "tor21", "w_success", "wb_success"):
    assert _c in X_full.columns, _c
assert "phteam" in cat_features and "cnt12" in cat_features, "범주형 조합 누락"
print("diet 검증 OK | 피처 %d개 (v10wtor 130 -> 110 이어야 한다)" % X_full.shape[1])

'''
ANC = 'print("최종 파라미터:", BEST_PARAMS)'
sub(find(ANC), ANC, GUARD + ANC)
print('자체 검증 셀 삽입 (학습 전)')

sub(find(OLDZIP), OLDZIP, ZIP)
print('zip -> %s' % ZIP)

for i in code:
    ast.parse(src(i))
_chk = 0
for i in code:
    for node in ast.walk(ast.parse(src(i))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and 'def main(' in node.value and 'df_proc' in node.value:
            ast.parse(node.value)
            _chk += 1
if _chk == 0:
    sys.exit('script.py 소스를 못 찾았다')
print('생성될 script.py 문법 OK')

json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

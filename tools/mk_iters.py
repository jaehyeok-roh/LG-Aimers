# 반복수만 바꾼다 — v10wzc(1,121.89) 에서 iterations 한 줄.
#
#   ITERS=1400 python tools/mk_iters.py
#
# 왜 (2026-08-29):
#   iterations=1000 / depth 8 은 **이진 타겟 시절 v5(990점, 96피처)** 에서
#   리더보드로 포화를 확인한 값이다 (500 -23.07 / 700 -4.74 / 1000 기준).
#   v10wz 부터 타겟이 5분류라 잎이 5차원이고 피처는 115개다.
#   claude.md 'optbest 절개': 다른 파라미터 영역에서 잰 효과는 부호까지 뒤집혔다.
#
#   방향이 위쪽인 근거 — 세 축이 전부 1~2개에서 포화하고 더 얹으면 음수다:
#     보조 타겟  reverse 1개 +13.5   / +ball  +0.002
#     범주형 조합 2개        +5.72   / 4개    -4.61
#     다중분류   5분류      +12.85   / 7분류  -6.69
#   세 축이 독립적으로 같은 모양이면 개별 설명(CTR 경쟁)보다 공통 제약,
#   즉 **용량 경쟁**을 의심하는 게 맞다. 부족하면 답은 더 크게다.
#
#   ⚠️ ntree_end(mk_ntree.py)로는 1000 이하만 잴 수 있어 이 방향을 원리적으로
#      못 시험한다. n700 실측 std 0.03424 (1000 은 0.03655) = λ 0.937 이고
#      수축 모형(λ*≈0.96)으로 환산하면 -1 근처 = 노이즈. 그래서 위로 간다.
#
# 본 학습 셀은 early_stopping_rounds 를 pop 하고 eval_set 없이 fit 하므로
# 조기종료에 먹히지 않고 지정한 그루 수를 전부 학습한다 (확인함).
# 보조모델(AUX_ITERS=300)은 건드리지 않는다 — 한 번에 하나만 바꾼다 (4-14).
import ast
import json
import os
import sys

ITERS = int(os.environ.get('ITERS', '1400'))
TAG = 'i%d' % ITERS
BASE = os.environ.get('IT_BASE', 'aimers_v10wzc.ipynb')
OUT = os.environ.get('IT_OUT', 'aimers_v10w%s.ipynb' % TAG)
ZIP = os.environ.get('IT_ZIP', 'submit_v10w%s.zip' % TAG)
OLDZIP = os.environ.get('IT_OLDZIP', 'submit_v10wzc.zip')

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


# ---------------------------------------------- 1) 반복수 한 줄
OLD = 'BEST_PARAMS["iterations"] = 1000'
NEW = 'BEST_PARAMS["iterations"] = %d   # 1000 -> %d (mk_iters.py)' % (ITERS, ITERS)
hit = 0
for i in code:
    if OLD in src(i):
        sub(i, OLD, NEW)
        hit += 1
if hit != 1:
    sys.exit('iterations 앵커 %d개 -- 수동 확인' % hit)
print('iterations -> %d' % ITERS)

# ---------------------------------------------- 2) 자체 검증 (학습 전에 터뜨린다, 4-12)
GUARD = '''
# ---- 반복수 검증 (학습 전에 터뜨린다, 4-12) ----
assert BEST_PARAMS["iterations"] == %d, BEST_PARAMS["iterations"]
assert AUX_ITERS == 300, "보조모델은 건드리지 않는다 (한 번에 하나만, 4-14)"
print(f"iterations = {BEST_PARAMS['iterations']} / AUX_ITERS = {AUX_ITERS}")

''' % ITERS
ANC = 'print("최종 파라미터:", BEST_PARAMS)'
for i in code:
    if ANC in src(i):
        sub(i, ANC, GUARD + ANC)
        break
else:
    sys.exit('가드 앵커를 못 찾았다')
print('자체 검증 셀 삽입 완료 (학습 전)')

# ---------------------------------------------- 3) 산출물 이름
for i in code:
    if 'ZIP_PATH' in src(i) and OLDZIP in src(i):
        sub(i, OLDZIP, ZIP)
        print('zip -> %s' % ZIP)
        break
else:
    sys.exit('ZIP_PATH 앵커를 못 찾았다')

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

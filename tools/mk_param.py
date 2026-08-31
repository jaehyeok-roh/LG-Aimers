# 파라미터 한 줄만 바꾼 판 — depth / seed / iterations.
#
#   PAR_WHAT=depth9  python tools/mk_param.py
#   PAR_WHAT=seed10  python tools/mk_param.py
#   PAR_WHAT=it1400  python tools/mk_param.py
#
# 셋 다 **이진 타겟 96피처 시절**에 리더보드로 재고 닫은 축이다
# (depth9 -0.57 / seed3->10 -1.26 / it700 -4.74). 지금은 5분류 120피처라
# `optbest` 교훈("다른 파라미터 영역에서 잰 효과를 더하지 말 것", 그때는 부호까지
# 뒤집혔다)이 적용되고, 5분류 영역에서는 **반복수만** 재봤다 (n700 -1.47, 잔차 -0.17).
#
# 기대는 전부 0~3 이다. 그래도 던지는 이유는 하나뿐이다 —
# **최고점 채점이라 하방이 0 이고 제출이 남아돈다.** 기대값이 아니라 상방만 산다.
import ast
import json
import os
import sys

WHAT = os.environ.get('PAR_WHAT', 'depth9')
SPEC = {
    'depth9': ('BEST_PARAMS["depth"] = 9   # 8 -> 9 (mk_param)',
               'depth 8 -> 9', 'v10wd9'),
    'seed10': ('SEEDS = [42, 202, 2024, 7, 77, 777, 1234, 31337, 555, 9]   # 3 -> 10 (mk_param)',
               'seed 3 -> 10 (모델 30 -> 100)', 'v10ws10'),
    'it1400': ('BEST_PARAMS["iterations"] = 1400   # 1000 -> 1400 (mk_param)',
               'iterations 1000 -> 1400', 'v10wi14'),
    # fold 수는 이 프로젝트에서 **한 번도 안 잰 축**이다. 재본 것은 반복수·깊이·seed 수뿐.
    # 30 fold x 1 seed 는 모델 수가 30 으로 같아 GPU 시간이 같은데, 각 모델의 학습
    # 데이터가 90.0% -> 96.7% 로 **7.4% 늘어난다.** 다양성을 데이터로 바꾸는 것이고,
    # 근거는 seed 3->10 (-1.26) 이 이미 '앙상블 다양성은 포화' 를 보였다는 것이다.
    # 그리고 '데이터 양이 최신성을 이긴다' 는 이 프로젝트에서 한 번도 안 진 방향이다.
    'f30': (None, 'fold 10x3seed -> 30x1seed (모델 30개 유지, 학습분 90.0% -> 96.7%)',
            'v10wf30'),
    # score_function 은 **분할을 고르는 기준** 자체다 (샘플링도 트리 모양도 아니다).
    # claude.md 에 한 번도 안 나오고 Optuna 탐색 공간에도 없었다. 남은 GPU 손잡이 중
    # 유일하게 두 논거가 **덮지 못하는** 축이다 —
    #   '추정 효율은 148만 행에서 죽는다' (MVS +5.17 -> -2.95)  -> 샘플링 계열 전부 사망
    #   '구조 변형은 더 나쁘다' (Lossguide 731 vs 대칭 803)      -> grow_policy 사망
    # 깊이는 v4 Optuna 가 **이진 96피처**에서 고른 값이고, 5분류로는 어느 base 에서도
    # 리더보드로 잰 적이 없다 (반복수만 n700 -1.47 로 쟀다). 5분류는 잎마다 5차원 값을
    # 내므로 잎당 정보량이 이진과 달라 최적 깊이가 다를 이유가 있다. 그리고 CPU Optuna 는
    # depth 6 을 골랐었다 -- 얕은 쪽 사전값이 있는데 한 번도 안 재봤다.
    # `optbest` 교훈: 다른 파라미터 영역의 효과는 안 옮겨지고 그때는 부호까지 뒤집혔다.
    'depth7': ('BEST_PARAMS["depth"] = 7   # 8 -> 7 (mk_param)',
               'depth 8 -> 7', 'v10wd7'),
    # 보조 모델은 300트리인데 본 모델은 1000이다. 이 값은 aux_rev 를 도입한
    # **이진 타겟 시절**(v10wa)에 정해진 뒤 한 번도 안 건드렸다.
    # `na` 의 -3.53 이 'aux_rev 는 더 잘 추정된 P(reverse) 라서 값어치가 있다' 를
    # 증명했으므로, 그 추정을 더 잘 하면 더 나올 여지가 있다. 학습 행은 OOF,
    # 추론 행은 폴드 평균이라 용량을 키워도 누수 구조는 그대로다.
    'auxit': ('AUX_ITERS = 1000', 'AUX_ITERS 300 -> 1000 (보조모델 용량)', 'v10wai'),
    'newton': ('BEST_PARAMS["score_function"] = "L2"   # Cosine -> L2 (mk_param)',
               'score_function Cosine -> L2', 'v10wnt'),
}
if WHAT not in SPEC:
    sys.exit('PAR_WHAT 은 %s 중 하나' % list(SPEC))
LINE, DESC, TAG = SPEC[WHAT]

BASE = os.environ.get('PAR_BASE', 'experiments/v10w/aimers_v10wph.ipynb')
OUT = os.environ.get('PAR_OUT', 'experiments/v10w/aimers_%s.ipynb' % TAG)
ZIP = os.environ.get('PAR_ZIP', 'submit_%s.zip' % TAG)
OLDZIP = os.environ.get('PAR_OLDZIP', 'submit_v10wph.zip')

nb = json.load(open(BASE, encoding='utf-8'))
cells = nb['cells']
code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']
src = lambda i: ''.join(cells[i]['source'])


def sub(i, old, new, n=1):
    s = src(i)
    if s.count(old) != n:
        sys.exit('셀 %d: 앵커 %d개 기대, %d개' % (i, n, s.count(old)))
    cells[i]['source'] = s.replace(old, new).splitlines(keepends=True)


def find(a):
    h = [i for i in code if a in src(i)]
    if len(h) != 1:
        sys.exit('앵커 %d곳: %s' % (len(h), a[:80]))
    return h[0]


if WHAT == 'f30':
    A, B = 'N_SPLITS = 10', 'SEEDS = [42, 202, 2024]'
    i = find(A)
    sub(i, A, 'N_SPLITS = 30')
    sub(i, B, 'SEEDS = [42]')
elif WHAT == 'seed10':
    A = 'SEEDS = [42, 202, 2024]'
    sub(find(A), A, LINE.split('   #')[0])
elif WHAT == 'auxit':
    A = 'AUX_ITERS = 300'
    sub(find(A), A, LINE)
elif WHAT == 'newton':
    A = 'BEST_PARAMS["cat_features"] = cat_features'
    sub(find(A), A, LINE + '\n' + A)
elif WHAT in ('depth9', 'depth7'):
    A = 'BEST_PARAMS["cat_features"] = cat_features'
    sub(find(A), A, LINE + '\n' + A)
else:
    A = 'BEST_PARAMS["iterations"] = 1000'
    sub(find(A), A, LINE)
print('%s 적용' % DESC)

# 학습 전 가드 — 의도한 값이 실제로 반영됐는지 (조용히 무시되는 사고 방지, 4-6)
CHK = {'depth9': 'assert BEST_PARAMS["depth"] == 9, BEST_PARAMS["depth"]',
       'depth7': 'assert BEST_PARAMS["depth"] == 7, BEST_PARAMS["depth"]',
       'seed10': 'assert len(SEEDS) == 10 and len(set(SEEDS)) == 10, SEEDS',
       'it1400': 'assert BEST_PARAMS["iterations"] == 1400, BEST_PARAMS["iterations"]',
       'newton': 'assert BEST_PARAMS["score_function"] == "L2", BEST_PARAMS',
       'auxit': 'assert AUX_ITERS == 1000, AUX_ITERS',
       'f30': ('assert N_SPLITS == 30 and SEEDS == [42], (N_SPLITS, SEEDS)\n'
               'assert N_SPLITS * len(SEEDS) == 30, "모델 30개 유지가 전제다"')
       }[WHAT]
GUARD = '''
# ---- %s 검증 (학습 전에 터뜨린다, 4-12) ----
%s
print(f"파라미터 확인: depth={BEST_PARAMS['depth']} / iters={BEST_PARAMS['iterations']} / seeds={len(SEEDS)}")

''' % (DESC, CHK)
# ⚠️ auxit 의 가드는 AUX_ITERS 를 참조하는데 그 정의는 '최종 파라미터' 줄보다
#    **아래**에 있다 (aux 블록). 앵커를 나눠야 한다 -- 오늘 d9/t16/na 를 죽인 유형이고
#    nbcheck 가 잡은 7번째 사례다.
if WHAT == 'auxit':
    ANC = 'print("  검산 OK: 클래스0 개수 == success 개수")'
    sub(find(ANC), ANC, ANC + '\n' + GUARD)
else:
    ANC = 'print("최종 파라미터:", BEST_PARAMS)'
    sub(find(ANC), ANC, GUARD + ANC)
sub(find(OLDZIP), OLDZIP, ZIP)

for i in code:
    ast.parse(src(i))
_chk = 0
for i in code:
    for node in ast.walk(ast.parse(src(i))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and 'def main(' in node.value and 'df_proc' in node.value:
            ast.parse(node.value)
            _chk += 1
assert _chk, 'script.py 소스를 못 찾았다'
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('%s 생성 (zip %s) -- %d셀 + script.py 문법 OK' % (OUT, ZIP, len(code)))

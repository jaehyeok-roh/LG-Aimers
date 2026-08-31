# aux_rev 를 뺀다 — 5분류 타겟이 그 정보를 직접 갖고 있으므로 잉여일 수 있다.
#
#   python tools/mk_noaux.py
#
# `aux_rev`(교차적합한 P(reverse|X) 를 피처로)는 LB **+13.50** 으로 채택됐다.
# 그때 claude.md 가 적은 근거는 이것이다:
#   "**본 모델은 이진 y 만 보므로 P(reverse|X) 를 배울 수 없다** (그 라벨을 못 보니까)"
#
# ⛔ **그건 타겟이 이진이던 시절 이야기다.** v10wz 부터 타겟이 5분류이고
#    클래스 2 = '반대만', 클래스 3 = '둘다' 이므로 `P(reverse) = P(2) + P(3)` 을
#    **본 모델이 직접 출력한다.** 그리고 그 이후로 aux_rev 를 다시 잰 적이 없다.
#
# 게다가 claude.md 는 aux_rev 를 **'새 정보가 아니라 기존 신호의 압축'**
# (전달률 0.10~0.60)으로 분류해뒀다. 압축인데 원본을 이제 직접 갖고 있다면
# 용량만 먹는 잉여다. 보조 모델은 폴드 3개 x 300트리를 따로 쓰고, 배포 zip 에
# .cbm 3개를 더 싣고, 오프셋 셀에서도 별도 판을 한 번 더 적합한다.
#
# ⭐ 오늘 세 번 확인된 규칙이 정확히 이 형태다 — **과거에 실측된 이득도 피처가
#    바뀌면 다시 재야 한다**: CPU 전환 +26 -> -2.95, DROP_CAL +4.52 -> +0.08,
#    mc4 +1.88 -> -9.41. 셋 다 '조건이 바뀐 뒤 재측정 안 한 옛 이득' 이었다.
#
# 방향은 양쪽 다 열려 있다. 잉여면 제거가 +, 여전히 더 잘 추정된 판이면 -.
# **기대 -5 ~ +8.** 근거가 노이즈 밖(LB 실측 두 건의 조건 변화)이라는 점이 중요하다.
import ast
import json
import os
import sys

# NA_TARGETS 로 보조 타겟을 지정한다.
#   NA_TARGETS=      -> aux 제거          : 5분류가 reverse 를 담으므로 잉여인가?
#   NA_TARGETS=ball  -> aux 를 ball 로 교체 : 5분류가 **안 담는** 유일한 라벨이다
# claude.md 의 `auxnsb`(rev+ball, +0.002)는 **이진 base** 에서 reverse 위에 얹은
# 증분이라, 5분류 base 의 'ball 단독' 과는 성질이 다르다.
# eda41: ball 의 투수 수준 신호는 success 대비 61% (reverse 는 102%).
TARGETS = [t for t in os.environ.get('NA_TARGETS', '').split(',') if t]
BASE = os.environ.get('NA_BASE', 'experiments/v10w/aimers_v10wtor.ipynb')
OUT = os.environ.get('NA_OUT', 'experiments/v10w/aimers_v10wna.ipynb')
ZIP = os.environ.get('NA_ZIP', 'submit_v10wna.zip')
OLDZIP = os.environ.get('NA_OLDZIP', 'submit_v10wtor.zip')

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


A = "AUX_TARGETS = ['reverse']"
N = ("AUX_TARGETS = %r   # mk_noaux\n"
     "# 원래 근거 '본 모델은 이진 y 만 보므로 reverse 를 못 배운다' 는\n"
     "# v10wz(5분류, class2=반대만 / class3=둘다) 이후로 성립하지 않는다.\n"
     "# 5분류가 담지 않는 라벨은 ball/strike 뿐이다." % (TARGETS,))
sub(find(A), A, N)
print('AUX_TARGETS = %r 적용' % (TARGETS,))

GUARD = '''
# ---- noaux 검증 (학습 전에 터뜨린다, 4-12) ----
assert AUX_TARGETS == {TGT!r}, AUX_TARGETS
_ax = sorted(c for c in X_full.columns if c.startswith("aux_"))
assert _ax == sorted("aux_" + t for t in AUX_TARGETS), _ax   # 양쪽 다 정렬할 것
_sf = json.load(open("model/selected_features.json"))
assert sorted(c for c in _sf if c.startswith("aux_")) == _ax, "selected_features 불일치"
assert len(_sf) == X_full.shape[1], (len(_sf), X_full.shape[1])
# tor / t13 / phteam 은 그대로 살아 있어야 한다 (기준선이 v10wtor 다)
for _c in ("t13", "t13_post", "tor16", "tor21"):
    assert _c in X_full.columns, _c
assert "phteam" in cat_features, "phteam 누락"
print("noaux 검증 OK | aux %s | 피처 %d개 (v10wtor 는 130)" % (AUX_TARGETS, X_full.shape[1]))

'''
# ⚠️ 앵커는 **aux 블록 뒤**여야 한다. '최종 파라미터' 줄은 aux 블록 앞이라
#    (a) AUX_TARGETS 가 아직 정의되지 않았고 (NameError -- 오늘 d9/t16 을 죽인 유형)
#    (b) X_full 에 aux 컬럼이 아직 안 붙어 있어 가드가 무의미하게 통과한다.
#    5분류 검산 줄이 aux 블록 바로 뒤다.
GUARD = GUARD.replace('{TGT!r}', repr(TARGETS))
ANC = 'print("  검산 OK: 클래스0 개수 == success 개수")'
sub(find(ANC), ANC, ANC + '\n' + GUARD)
print('자체 검증 셀 삽입 (aux 블록 뒤, 학습 전)')

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

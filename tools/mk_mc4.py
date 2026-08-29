# 타겟을 5분류 -> 4분류로 (가장 약한 '둘다' 를 '몰림' 에 합친다) — v10wzc 위.
#
# 근거 (2026-08-29, 캐글 kmg/kmgb 셔플 대조군):
#   wsboth 893 / multicls 899 (+6) / mcshuf 870 (**-23**)
#   -> 표적을 쪼개는 **구조 자체가 -23 의 비용**이고, 실패 유형의 의미론이 +29 를
#      얹어 순 +6 이 된다. 다중과제 규제가 이득이라는 가설은 기각됐다.
#
# 비용이 출력 개수당 발생한다면 방향은 '더 쪼개기' 가 아니라 **'덜 쪼개기'** 다.
# 5분류의 클래스 구성에서 '둘다' 가 가장 약하다:
#
#   0 성공 52.4% | 1 몰림만 11.5% | 2 반대만 19.5% | 4 크게벗어남 13.2%
#   3 둘다 **3.4%**  <- eda41 신뢰도 0.709 / 신호 0.0077 = success 대비 **17%**
#
# 전체의 3.4% 에 의미론 기여 17% 인데 출력 하나 몫의 구조적 비용(-23/4 ~= -6)은
# 똑같이 낸다. '둘다' 는 middle 과 reverse 가 겹친 칸이므로 '몰림' 에 합치는 것이
# 자연스럽다 (둘 다 몰림 성분을 갖는다).
#
# claude.md 의 포화 패턴과도 맞는다 — 보조 타겟 1개 / 범주형 조합 2개 / 다중분류
# 5분류에서 전부 "하나 더 얹으면 음수" 였다. **하나 빼는 쪽은 아직 안 재봤다.**
#
# 기대 +2~5. 하네스마다 크기가 달라(multicls 가 CPU +35.7 / GPU +6 / LB +12.85)
# 정밀하게는 못 잡는다.
import ast
import json
import os
import sys

BASE = os.environ.get('MC4_BASE', 'aimers_v10wzc.ipynb')
OUT = os.environ.get('MC4_OUT', 'experiments/v10w/aimers_v10wmc4.ipynb')
ZIP = os.environ.get('MC4_ZIP', 'submit_v10wmc4.zip')
OLDZIP = os.environ.get('MC4_OLDZIP', 'submit_v10wzc.zip')

nb = json.load(open(BASE, encoding='utf-8'))
cells = nb['cells']
code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']


def src(i):
    return ''.join(cells[i]['source'])


def sub(i, old, new, n=1):
    s = src(i)
    if s.count(old) != n:
        sys.exit('셀 %d: 앵커 %d개 기대, %d개\n  %s' % (i, n, s.count(old), old[:160]))
    cells[i]['source'] = s.replace(old, new).splitlines(keepends=True)


def find(anchor):
    hit = [i for i in code if anchor in src(i)]
    if len(hit) != 1:
        sys.exit('앵커 %d곳: %s' % (len(hit), anchor[:110]))
    return hit[0]


# ---------------------------------------------- 1) 라벨 구성
A1 = '''Y_CLS = np.where(_y5 == 1, 0,
                 np.where((_m5 == 1) & (_r5 == 1), 3,
                          np.where(_m5 == 1, 1, np.where(_r5 == 1, 2, 4))))'''
N1 = '''# 4분류: '둘다'(3.4%, eda41 신뢰도 17%)를 '몰림' 에 합친다. 분할은 출력 개수당
# -23/4 ~= -6 의 구조적 비용을 내는데(kmg 셔플 대조군) 그 칸의 의미론이 가장 약하다.
# middle 이 1 이면 reverse 와 무관하게 1 -> '둘다' 가 자연히 '몰림' 으로 들어간다.
Y_CLS = np.where(_y5 == 1, 0,
                 np.where(_m5 == 1, 1, np.where(_r5 == 1, 2, 3)))'''
sub(find(A1), A1, N1)
print("라벨: 5분류 -> 4분류 ('둘다' 를 '몰림' 에 병합)")

# 출력 문구도 맞춘다
for i in code:
    if '"  5분류 분포:"' in src(i):
        sub(i, '"  5분류 분포:"', '"  4분류 분포:"')
        print('출력 문구 갱신')
        break

# ---------------------------------------------- 2) 학습 전 자체 검증 (4-12)
GUARD = '''
# ---- 4분류 검증 (학습 전에 터뜨린다, 4-12) ----
_u = np.unique(Y_CLS[CLS_OK]).astype(int)
assert list(_u) == [0, 1, 2, 3], _u
# 클래스 0 은 여전히 control_success 와 정확히 같아야 한다
assert (Y_CLS[CLS_OK] == 0).sum() == int(y_full.to_numpy()[CLS_OK].sum()), "클래스0 != success"
# '둘다' 였던 행이 전부 클래스 1 로 갔는가 (5분류에서 3번, 실패의 7.2%)
_both = (_m5 == 1) & (_r5 == 1) & CLS_OK
assert _both.sum() > 40000, int(_both.sum())          # eda41: 50,266행
assert (Y_CLS[_both] == 1).all(), "둘다 가 몰림으로 안 갔다"
_c = {int(k): int(v) for k, v in zip(*np.unique(Y_CLS[CLS_OK], return_counts=True))}
print(f"4분류 검증 OK: {_c} | '둘다' {int(_both.sum()):,}행이 클래스1 에 병합됨")

'''
# ⚠️ 앵커는 반드시 Y_CLS **구성 뒤**여야 한다. 처음에 'print(최종 파라미터)' 에
# 걸었더니 같은 셀 안에서 라벨 구성보다 앞이라 NameError 로 죽었다 (3분).
ANC = 'assert (Y_CLS[CLS_OK] == 0).sum() == int(_y5[CLS_OK].sum()), "클래스0 != success"'
sub(find(ANC), ANC, ANC + GUARD)
print('자체 검증 셀 삽입 (학습 전)')

# ---------------------------------------------- 3) 산출물 이름
sub(find(OLDZIP), OLDZIP, ZIP)
print('zip -> %s' % ZIP)

for i in code:
    ast.parse(src(i))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

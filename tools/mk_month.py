# game_month 복원 — v10wzc(1,121.89) 에서 DROP_CAL 한 줄만 바꾼다.
#
# 왜 (2026-08-29):
#   DROP_CAL 은 v9m(996점, 8/22)에서 +4.52 로 채택됐다. 그때는 w_n 이 없었다.
#   지금 배포본의 최대 피처군(wseason5 +61.72 / wsbat +30.11 = LB +92)이 전부
#   w_n / wb_n 위에 서 있는데, w_n 은 '시즌진행도 x 등판페이스' 의 곱이라
#   캘린더 없이는 두 인자를 분리할 수 없다. 실측(2024, R):
#
#     w_n 300~600 인 행 -> 그 투수의 시즌 총투구 중앙값
#       4월 2,397   6월 988   9월 602        <- 4배
#     월별 w_n 중앙값  3월 43  ...  10월 1,956  <- 45배
#
#   즉 같은 w_n 이 '에이스의 4월' 과 '변두리 자원의 9월' 을 구분 못 한다.
#   현재 모델에는 캘린더 정보가 하나도 없다(is_heat_wave/weekend 플래그만 있고
#   그건 순서 있는 축이 아니다).
#
#   원래 제거 논거는 "이미 어긋난 수준 정보를 확신 있게 주입한다" 였고,
#   그 병은 w5_* 가 시즌 리그평균으로 디트렌드하면서 이미 고쳐졌다.
#   claude.md 규칙: "과거에 실측된 이득도 피처가 바뀌면 다시 재야 한다"
#   (CPU 전환이 +26 -> -2.95 로 죽은 그 규칙).
#
# ⚠️ season 과 달리 외삽 문제가 없다. game_month 는 int64 이고 3~10 이 매년
#    반복되므로 학습에서 만든 분기 경계가 2025 에 그대로 유효하다 (eda31 대조).
#
# 파생(w_n / 진행도)은 만들지 않는다 — 한 행 안 파생은 3번 독립으로 0 이었고,
# 월만 있으면 트리가 스스로 만든다.
import ast
import json
import os
import sys

BASE = os.environ.get('MON_BASE', 'aimers_v10wzc.ipynb')
OUT = os.environ.get('MON_OUT', 'experiments/v10w/aimers_v10wm.ipynb')
ZIP = os.environ.get('MON_ZIP', 'submit_v10wm.zip')
OLDZIP = os.environ.get('MON_OLDZIP', 'submit_v10wzc.zip')

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


# ---------------------------------------------- 1) DROP_CAL 한 줄
OLD = "DROP_CAL = ['game_month', 'game_dayofweek']"
NEW = "DROP_CAL = ['game_dayofweek']   # game_month 복원 (w_n 정규화용, mk_month.py)"
hit = 0
for i in code:
    if OLD in src(i):
        sub(i, OLD, NEW)
        hit += 1
if hit != 1:
    sys.exit('DROP_CAL 앵커 %d개 -- 수동 확인' % hit)
print('DROP_CAL 수정 완료')

# ---------------------------------------------- 2) 자체 검증 (학습 전에 터뜨린다, 4-12)
GUARD = '''
# ---- game_month 복원 검증 (학습 전에 터뜨린다, 4-12) ----
assert "game_month" in X_full.columns, "game_month 가 X_full 에 없다 -- DROP_CAL 수정 실패"
assert "game_dayofweek" not in X_full.columns, "game_dayofweek 는 계속 빠져 있어야 한다"
assert "game_month" not in cat_features, "game_month 는 수치형이어야 한다 (범주형이면 CTR 경로가 달라진다)"
_mn, _mx = int(X_full["game_month"].min()), int(X_full["game_month"].max())
assert (_mn, _mx) == (3, 10), f"game_month 범위 {_mn}~{_mx} (3~10 기대)"
print(f"game_month 복원 OK: 수치형, {_mn}~{_mx}, 피처 {X_full.shape[1]}개")

'''
ANC = 'print("최종 파라미터:", BEST_PARAMS)'
done = False
for i in code:
    if ANC in src(i):
        sub(i, ANC, GUARD + ANC)
        done = True
        break
if not done:
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

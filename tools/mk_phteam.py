# pitcher_hand x pitcher_team_id (20칸) 를 범주형으로 — v10wt13(1,142.47) 위.
#
# 근거 (2026-08-30, tools/eda52.py + eda53.py):
#   저카디널리티 조합 105쌍을 전수 스캔했다. 알려진 정답 두 개를 정확히 짚었다 —
#   1위 season x t13 (202, 방금 +20.58 로 회수), pitcher_hand x batter_hand (49.5,
#   이미 hand4 로 모델에 있고 2023<->2024 상관 1.00).
#
#   t13 관여 행을 빼고도 남는 것은 둘뿐이고, 그중 하나만 안정적이다:
#     pitcher_hand x pitcher_team   38.1 -> 42.3   2023/2024 상관 **0.40**
#     pitcher_team x batter_team    27.8 -> 29.1   상관 **-0.14** (90셀, 기각)
#     pitcher_team x era            95.7 ->  11.2  (88% 가 t13 이었다)
#
#   천장: 강도 42 x 상관 0.40 ~= 17. t13 의 회수율 46% 적용하면 **+6~8**.
#
# ⚠️ 범주형 조합은 정확히 2개에서 포화하고 4개면 -4.61 이다 (CTR 경쟁).
#    지금 cnt12 + hand4 로 2개이므로 이건 **세 번째**다. 그 위험을 안고 재는 것이고,
#    음수면 '조합 2개 포화' 가 5분류+t13 base 에서도 성립한다는 확정 정보가 된다.
#    (t13 은 수치형 0/1 이라 CTR 을 만들지 않으므로 이 카운트에 안 들어간다.)
import ast, json, os, sys

BASE = os.environ.get('PH_BASE', 'experiments/v10w/aimers_v10wt13.ipynb')
OUT = os.environ.get('PH_OUT', 'experiments/v10w/aimers_v10wph.ipynb')
ZIP = os.environ.get('PH_ZIP', 'submit_v10wph.zip')
OLDZIP = os.environ.get('PH_OLDZIP', 'submit_v10wt13.zip')

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
        sys.exit('앵커 %d곳: %s' % (len(h), a[:90]))
    return h[0]


A1 = "    df_proc['t13'] = _t13.astype('float64')"
N1 = A1 + """
    # pitcher_hand x pitcher_team (20칸). eda53: t13 을 빼고도 강도 42.3 이고
    # 2023<->2024 상관 0.40 으로 유일하게 살아남은 조합이다.
    df_proc['phteam'] = (df_proc['pitcher_hand'].astype(str) + '|'
                         + df_proc['pitcher_team_id'].astype(str))"""
sub(find(A1), A1, N1)
print('step4 에 phteam 삽입')

CAT = "'is_full_count', 'count_advantage', 'is_waste_pitch_sit',"
n = 0
for i in code:
    c = src(i).count(CAT)
    if c:
        sub(i, CAT, "'is_full_count', 'count_advantage', 'phteam', 'is_waste_pitch_sit',", n=c)
        n += c
if n == 0:
    sys.exit('created_cat_cols 앵커를 못 찾았다')
print('범주형 등록: %d곳' % n)

GUARD = '''
# ---- phteam 검증 (학습 전에 터뜨린다, 4-12) ----
assert "phteam" in X_full.columns, "phteam 누락"
assert "phteam" in cat_features, "phteam 이 범주형으로 등록되지 않았다"
# ⚠️ eda52/53 은 R 만 봐서 10팀(20칸)이었는데 학습은 F 도 포함한다.
# F 에는 팀 22/23/25 가 더 있어 실제로는 13팀 x 2 = 26칸이다. 데이터에서 계산한다.
_nt = df_processed["pitcher_team_id"].astype(str).nunique()
_np = X_full["phteam"].astype(str).nunique()
assert _np == 2 * _nt, f"phteam {_np}칸 != 2 x {_nt}팀"
print(f"phteam 검증 OK: {_np}칸 | 피처 {X_full.shape[1]}개")

'''
ANC = 'print("최종 파라미터:", BEST_PARAMS)'
sub(find(ANC), ANC, GUARD + ANC)
print('자체 검증 셀 삽입 (학습 전)')

sub(find(OLDZIP), OLDZIP, ZIP)
print('zip -> %s' % ZIP)
for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

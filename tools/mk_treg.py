# 팀 x 체제전환 지시자 (일반화판) — mk_t13.py 를 팀·시점 파라미터로 뽑은 것.
#
#   TREG_TEAM=16 TREG_YEAR=2024 TREG_MONTH=6 TREG_F=0 \
#   TREG_BASE=experiments/v10w/aimers_v10wph.ipynb python tools/mk_treg.py
#
# t13 (TEAM=13 YEAR=2023 MONTH=5 F=1) 이 리더보드 **+20.58** 을 냈다. 그 형태는
#   (pitcher_team==t OR batter_team==t) AND (시점 > T)
# 이고, 세 조건이 겹쳐서 트리가 만들기 최대로 비쌌다 —
#   ① 두 컬럼의 OR (레벨 2개, oblivious 는 레벨을 통째로 쓴다)
#   ② 전환점이 시즌 중간인데 DROP_CAL 로 game_month 를 버렸다 (표현 불가)
#   ③ 6시즌을 뭉치면 부호가 상쇄된다 (4시즌 음수 / 2시즌 양수)
#
# ⚠️ 팀16(2024-06)은 t13 과 달리 **계단폭/잡음 비가 1.7** 로 내 문턱(3) 아래다.
#    잡음일 확률이 높다. 다만 (a) 투수측·타자측 부호가 같고 (b) 2025 가 전부 '후' 이며
#    (c) 코드가 동일해 비용이 0 이다. 기대 **+2~5**.
#
# 규정: 행 A 자기 컬럼(team_id / season / game_month / game_type) + train 라벨 통계
#       상수. 허용 범위 1·2·4번이고 행 독립 시험을 자명하게 통과한다.
#
# 수치형 0/1 로 준다 — 범주형 조합은 CTR 경쟁이 있는데 0/1 플래그는 CTR 을 안 만든다.
import ast
import json
import os
import sys

TEAM = int(os.environ.get('TREG_TEAM', '13'))
YEAR = int(os.environ.get('TREG_YEAR', '2023'))
MONTH = int(os.environ.get('TREG_MONTH', '5'))
USE_F = os.environ.get('TREG_F', '1') == '1'      # F 행을 전환 후로 볼 것인가
TAG = 't%d' % TEAM
BASE = os.environ.get('TREG_BASE', 'experiments/v10w/aimers_v10wph.ipynb')
OUT = os.environ.get('TREG_OUT', 'experiments/v10w/aimers_v10w%s.ipynb' % TAG)
ZIP = os.environ.get('TREG_ZIP', 'submit_v10w%s.zip' % TAG)
OLDZIP = os.environ.get('TREG_OLDZIP', 'submit_v10wph.zip')

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


FC = (" | (df_proc['game_type'].astype(str) == 'F')") if USE_F else ""
GFC = (" | (_gts == 'F')") if USE_F else ""
A1 = "    df_proc['cnt12'] = (b.astype(int).astype(str) + '-' + s.astype(int).astype(str))"
N1 = A1 + """
    # 팀%d x %d-%02d 체제 전환. 두 컬럼의 OR 이라 대칭트리는 레벨 두 개를 써야 만든다
    # (nosh -17.0 이 증명한 자리). 수치형 0/1 로 준다 -- CTR 경쟁 회피.
    _tg%d = ((df_proc['pitcher_team_id'].astype('int64') == %d)
            | (df_proc['batter_team_id'].astype('int64') == %d))
    df_proc['%s'] = _tg%d.astype('float64')
    # 전환 시점은 train 라벨 통계에서 고정한 상수다. 2025 행은 season>%d 이라 전부 1 ->
    # 학습 시대에 얼어붙는 성분이 없다 (auxrev 전달률 0.10 의 함정 회피).
    _sea%d = df_proc['season'].astype('int64')
    _mon%d = df_proc['game_month'].astype('int64')
    _post%d = ((_sea%d > %d) | ((_sea%d == %d) & ((_mon%d >= %d)%s)))
    df_proc['%s_post'] = (_tg%d & _post%d).astype('float64')""" % (
    TEAM, YEAR, MONTH,
    TEAM, TEAM, TEAM, TAG, TEAM,
    YEAR,
    TEAM, TEAM, TEAM, TEAM, YEAR, TEAM, YEAR, TEAM, MONTH, FC, TAG, TEAM, TEAM)
sub(find(A1), A1, N1)
print('step4 에 %s / %s_post 삽입 (팀%d, %d-%02d 전환, F절%s)'
      % (TAG, TAG, TEAM, YEAR, MONTH, '포함' if USE_F else '없음'))

GUARD = '''
# ---- %s 검증 (학습 전에 터뜨린다, 4-12) ----
for _c in ("%s", "%s_post"):
    assert _c in X_full.columns, f"{_c} 가 X_full 에 없다"
    assert _c not in cat_features, f"{_c} 는 수치형이어야 한다"
    assert set(np.unique(X_full[_c].to_numpy())) <= {0.0, 1.0}, _c
_t = X_full["%s"].to_numpy(); _p = X_full["%s_post"].to_numpy()
_sea = df_processed["season"].astype("int64").to_numpy()
print(f"  %s 커버리지 {_t.mean():.4f}")
assert 0.15 < _t.mean() < 0.40, _t.mean()
assert (_p <= _t).all(), "post 가 본체를 넘는다"
# post 마스크를 여기서 다시 계산해 정확히 대조한다.
# (t13 은 전환이 2023 이라 마지막 시즌 2024 가 전부 post 였지만, 전환이
#  마지막 학습 시즌 **안**에 있으면 그 시즌은 전/후가 섞인다. 2025 는 어느
#  쪽이든 전부 post 이므로 피처로서는 동일하게 유효하다.)
_mon = df_processed["game_month"].astype("int64").to_numpy()
_gts = df_processed["game_type"].astype(str).to_numpy()
_exp = _t.astype(bool) & ((_sea > %d) | ((_sea == %d) & ((_mon >= %d)%s)))
assert (_p.astype(bool) == _exp).all(), "post 마스크가 재계산과 다르다"
assert _p[_sea < %d].sum() == 0, "전환 연도 이전에 post 가 있다"
print(f"  post 비율 {_p.mean():.4f} | 마지막시즌 {_p[_sea == _sea.max()].mean():.4f}")
print(f"%s 검증 OK | 피처 {X_full.shape[1]}개")

''' % (TAG, TAG, TAG, TAG, TAG, TAG, YEAR, YEAR, MONTH, GFC, YEAR, TAG)
ANC = 'print("최종 파라미터:", BEST_PARAMS)'
sub(find(ANC), ANC, GUARD + ANC)
print('자체 검증 셀 삽입 (학습 전)')

sub(find(OLDZIP), OLDZIP, ZIP)
print('zip -> %s' % ZIP)

for i in code:
    ast.parse(src(i))
# 생성될 script.py 소스 문자열도 미리 파싱한다 (42분짜리 사고 방지)
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

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

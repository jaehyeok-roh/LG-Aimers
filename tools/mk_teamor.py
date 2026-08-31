# 팀 '관여' OR 플래그를 **전 팀에** 준다 (t13 의 일반화, 시점 게이트 없음).
#
#   python tools/mk_teamor.py
#
# t13(+20.58)이 통한 이유는 세 겹이었는데 그중 **첫째만** 다른 팀에도 그대로 있다:
#   ① 두 컬럼의 OR 이라 대칭트리가 레벨 두 개를 써야 만든다  <- 전 팀 공통
#   ② 전환점이 시즌 중간인데 game_month 를 버렸다            <- 13 만 해당
#   ③ 6시즌을 뭉치면 부호가 상쇄된다                          <- 13 만 해당
# 그래서 다른 팀에서 기대할 것은 **정적 경기단위 효과**뿐이고 크기도 작다.
#
# eda56 실측 (시즌 x 팀 주효과 제거 후 잔차, R 전용, 2023 -> 2024):
#   팀16 -0.0085 -> -0.0100 (천장 6.4) | 팀21 -0.0106 -> -0.0087 (4.9)
#   팀17 -0.0056 -> -0.0061 (2.4)      | 팀14 -0.0119 -> -0.0056 (2.0)
#   ... 13 을 뺀 합계 천장 약 19. t13 이 천장의 46% 를 회수했으므로 기대 +9.
#
# ⚠️ eda56 은 계단 변화가 **팀 13 하나뿐**임도 같이 확인했다 (2위 비 1.5).
#    따라서 여기에 시점 게이트를 더 붙일 이유는 없다.
#
# **수치형 0/1** 로 준다. 범주형 조합은 CTR 경쟁이 있어 4개면 -4.61 이었지만
# (bs_out/bs_cnt), 0/1 수치 플래그는 CTR 을 안 만들어 그 경쟁에 끼지 않는다.
# t13/t13_post 가 이미 같은 방식으로 들어가 있고 +20.58 을 냈다.
#
# 규정: 행 A 자기 컬럼(pitcher_team_id / batter_team_id)만 쓴다. 상수조차 없다 —
#       허용 범위 2번(자기 입력만으로 만든 파생변수)이고 행 독립 시험을 자명히 통과한다.
import ast
import json
import os
import sys

# eda56 에서 R 커버리지 3% 이상인 팀 (22/23/25 는 퓨처스 전용이라 제외).
# 13 은 t13/t13_post 로 이미 들어가 있으므로 빼지 않는다 -- 중복 컬럼이 되면
# 트리가 같은 분할을 두 번 갖게 되므로 여기서는 제외한다.
TEAMS = [int(t) for t in os.environ.get(
    'TOR_TEAMS', '12,14,15,16,17,18,19,20,21').split(',')]
BASE = os.environ.get('TOR_BASE', 'experiments/v10w/aimers_v10wph.ipynb')
OUT = os.environ.get('TOR_OUT', 'experiments/v10w/aimers_v10wtor.ipynb')
ZIP = os.environ.get('TOR_ZIP', 'submit_v10wtor.zip')
OLDZIP = os.environ.get('TOR_OLDZIP', 'submit_v10wph.zip')

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


# step4 안. t13 삽입 지점 바로 뒤에 붙인다 (같은 부류이므로 한 곳에 모은다).
A1 = "    df_proc['t13_post'] = (_t13 & _post).astype('float64')"
N1 = A1 + """
    # 전 팀 관여 OR 플래그 (mk_teamor). t13 의 일반화 -- 시점 게이트는 없다.
    # eda56: 계단 변화가 있는 팀은 13 하나뿐이고, 나머지는 정적 효과만 있다.
    _pti = df_proc['pitcher_team_id'].astype('int64')
    _bti = df_proc['batter_team_id'].astype('int64')
    for _tk in %r:
        df_proc['tor%%d' %% _tk] = ((_pti == _tk) | (_bti == _tk)).astype('float64')""" % (
    TEAMS,)
sub(find(A1), A1, N1)
print('step4 에 tor 플래그 %d개 삽입: %s' % (len(TEAMS), TEAMS))

GUARD = '''
# ---- teamor 검증 (학습 전에 터뜨린다, 4-12) ----
_TOR = %r
_pti_v = df_processed["pitcher_team_id"].astype("int64").to_numpy()
_bti_v = df_processed["batter_team_id"].astype("int64").to_numpy()
_gtv = df_processed["game_type"].astype(str).to_numpy()
for _tk in _TOR:
    _c = "tor%%d" %% _tk
    assert _c in X_full.columns, _c + " 가 X_full 에 없다"
    assert _c not in cat_features, _c + " 는 수치형이어야 한다 (CTR 경쟁 회피)"
    _v = X_full[_c].to_numpy()
    assert set(np.unique(_v)) <= {0.0, 1.0}, _c
    # 값을 여기서 다시 계산해 정확히 대조한다 (짐작 대신 재계산, 4-12)
    assert (_v.astype(bool) == ((_pti_v == _tk) | (_bti_v == _tk))).all(), _c
    _cr = float(_v[_gtv == "R"].mean())
    assert 0.15 < _cr < 0.25, "%%s R커버 %%.4f" %% (_c, _cr)
# 13 은 t13 이 이미 갖고 있으므로 중복이 없어야 한다
assert "tor13" not in X_full.columns, "tor13 이 t13 과 중복된다"
print("teamor 검증 OK: 플래그 %%d개 | 피처 %%d개" %% (len(_TOR), X_full.shape[1]))

''' % (TEAMS,)
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

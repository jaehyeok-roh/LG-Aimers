# cond_* 의 결측을 0(리그평균)으로 채운다 — 수식이 원래 그렇게 말한다.
#
#   python tools/mk_condfill.py
#
# `cond_*` 는 `sum/(count + C)` 로 리그평균(디트렌드 후 0)에 shrink 된다. 과거 표본이
# 0 인 투수는 수식대로면 `0/(0+C) = **0**` 인데, 구현은 merge 실패로 **NaN** 을 남긴다.
# 과거 투구가 1개인 투수는 ~0 을 받고 0개인 투수는 NaN 을 받는 불연속이다.
#
# 왜 해로울 수 있나: CatBoost 기본 `nan_mode='Min'` 은 NaN 을 **모든 값의 왼쪽 끝**에
# 놓는다. `cond_p` 는 0 을 중심으로 하는 편차라, 신규 투수가 '가장 나쁜 투수' 로
# 취급된다. 신규 투수는 나쁜 게 아니라 **모르는** 것이고, 올바른 사전값은 리그평균이다.
# 트리가 한 번의 분할로 분리할 수는 있지만 **대칭트리는 레벨을 통째로 쓰므로 비싸고**,
# 120피처와 경쟁하다 보면 NaN 이 '낮은 cond_p' 가지에 얹혀 갈 수 있다.
# t13(+20.58) / phteam(+7.79) 을 만든 것과 같은 '표현 비용' 논거다.
#
# 규모: cond_* 결측은 2020~2024 에서 13.8~22.7% 로 **표류 없이 안정적**이다
# (2019 는 과거 시즌이 없어 100%). 2025 도 같을 것이므로 학습/추론이 일관된다.
#
# ⚠️ 정직하게: `cond_*` 축은 국소 최적으로 확인됐다 — 서로 다른 두 변경
#    (`v10wcw` 표본두께 노출, `v10wc12` 키 정밀화)이 **둘 다 -4.75** 였다.
#    그 둘은 '정보를 더한' 쪽이고 이건 '구분을 없애는' 쪽이라 방향이 다르지만,
#    축 자체에 반대 증거가 있다. **기대 0, 상방 +4, 하방 -5.**
import ast
import json
import os
import sys

BASE = os.environ.get('CF_BASE', 'experiments/v10w/aimers_v10wph.ipynb')
OUT = os.environ.get('CF_OUT', 'experiments/v10w/aimers_v10wcf.ipynb')
ZIP = os.environ.get('CF_ZIP', 'submit_v10wcf.zip')
OLDZIP = os.environ.get('CF_OLDZIP', 'submit_v10wph.zip')

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


# ── 학습 경로: attach_cond_features ──
A1 = """        df[name] = col
        print(f"  {name}: 결측 {np.isnan(col).mean()*100:.1f}% (첫 시즌 + 신규투수)")"""
N1 = """        _nafrac = float(np.isnan(col).mean())
        # 수식대로 sum/(0+C) = 0 = 리그평균. merge 실패로 남은 NaN 을 그 값으로 채운다.
        df[name] = np.where(np.isnan(col), 0.0, col)
        print(f"  {name}: 결측 {_nafrac*100:.1f}% -> 0(리그평균) 으로 채움 (mk_condfill)")"""
sub(find(A1), A1, N1)
print('학습 경로 (attach_cond_features) 패치')

# ── 추론 경로: script.py 의 룩업 merge ──
A2 = """        if len(df_proc) != _n_before:
            raise RuntimeError("%s 병합에서 행 수가 %d -> %d 로 변함 (테이블 키 중복)"
                               % (_nm, _n_before, len(df_proc)))"""
N2 = A2 + """
        # 학습 경로와 같은 규칙: 룩업에 없는 조합은 0(리그평균) (mk_condfill).
        # ⚠️ 학습/추론 규칙 불일치는 이 프로젝트에서 36점짜리 사고였다 (트랙맨 exact/asof).
        df_proc[_nm] = pd.to_numeric(df_proc[_nm], errors="coerce").fillna(0.0)"""
sub(find(A2), A2, N2)
print('추론 경로 (script.py 룩업 merge) 패치')

GUARD = '''
# ---- condfill 검증 (학습 전에 터뜨린다, 4-12) ----
_CC = [c for c in ("cond_p", "cond_pc", "cond_ph", "cond_phc") if c in X_full.columns]
assert len(_CC) == 4, _CC
for _c in _CC:
    _v = X_full[_c].to_numpy(dtype="float64")
    assert not np.isnan(_v).any(), "%s 에 NaN 이 남아 있다" % _c
    # 0 으로 채워진 비중이 결측률(2019 100%% + 이후 14~23%%)과 맞아야 한다
    _z = float((_v == 0.0).mean())
    print("  %s: 정확히 0 인 비중 %.4f" % (_c, _z))
    assert 0.25 < _z < 0.45, "%s 0비중 %.4f -- 채우기가 안 먹었거나 과했다" % (_c, _z)
print("condfill 검증 OK | 피처 %d개" % X_full.shape[1])

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

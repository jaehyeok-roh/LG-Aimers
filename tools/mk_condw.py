# cond_* 의 **표본 신뢰도**를 노출한다 — v10wzc(1,121.89) 위.
#
# claude.md 가 '미해결로 남겨둔다' 고 적어놓은 유일한 항목이다:
#   > cond_* 의 의미가 학습 시즌마다 다르다. 2019 행은 과거 시즌 0개라 전부 NaN,
#   > 2024 행은 5개, test 는 6개다. 모델은 "1시즌으로 만든 cond_p" 와
#   > "5시즌으로 만든 cond_p" 를 같은 축에 놓는다.
#
# cond_* 는 빼면 -12.4 인 실동 피처군인데 학습/추론 간 의미가 어긋난 채다.
# 트랙맨 exact/asof 36점 사고와 같은 부류(학습·추론 규칙 불일치)다.
#
# 고치는 법: build_cond_table 이 이미 계산해놓고 **버리는** _w(과거 시즌 표본 수)를
# n/(n+C) 로 내보낸다. 그러면 트리가 '이 추정치가 얼마나 두꺼운 표본에서 왔나' 를
# 직접 보고 얇은 것을 스스로 할인한다 — prevfix 항목에서 확인된 기전이다
# ("w_n 만 있으면 트리가 스스로 '당해 투구수가 적으면 prev5 를 믿지 마라' 를 학습한다").
#
# ⚠️ 왜 n 이 아니라 n/(n+C) 인가: n 은 test(6시즌)가 학습(≤5시즌) 범위를 넘어서
#    트리 경계 밖으로 나간다. asof fresh 가 -217 로 무너진 함정이 정확히 그것이다.
#    n/(n+C) 는 [0,1) 로 유계이고 학습 안에 이미 두꺼운 표본이 많아 분포가 겹친다.
#
# script.py 는 고칠 필요가 없다 — cond 룩업을 merge(_ct, on=_keys) 로 **통째로**
# 붙이므로 CSV 에 컬럼이 하나 더 있으면 자동으로 따라온다 (배포본에서 확인함).
import ast
import json
import os
import sys

BASE = os.environ.get('CW_BASE', 'aimers_v10wzc.ipynb')
OUT = os.environ.get('CW_OUT', 'aimers_v10wcw.ipynb')
ZIP = os.environ.get('CW_ZIP', 'submit_v10wcw.zip')
OLDZIP = os.environ.get('CW_OLDZIP', 'submit_v10wzc.zip')

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
        sys.exit('셀 %d: 앵커 %d개 기대, %d개\n  %s' % (i, n, s.count(old), old[:140]))
    setsrc(i, s.replace(old, new))


def find(anchor):
    hit = [i for i in code if anchor in src(i)]
    if len(hit) != 1:
        sys.exit('앵커 %d곳: %s' % (len(hit), anchor[:100]))
    return hit[0]


# ------------------------------------------------ 1) 표본 신뢰도 컬럼을 만들어 반환
A1 = """    g[name] = g['_wd'] / (g['_w'] + C)             # 0(리그평균)으로 shrink
    return g[keys + [name]]"""
N1 = """    g[name] = g['_wd'] / (g['_w'] + C)             # 0(리그평균)으로 shrink
    # 표본 신뢰도: n/(n+C). 이 추정치가 0(리그평균)에서 얼마나 떨어져 있을 수 있는지의
    # 상한이며, 곧 '과거 표본이 얼마나 두꺼운가' 다. [0,1) 로 유계라 test 가 학습
    # 범위를 벗어나지 않는다 (asof fresh -217 함정 회피).
    g[name + '_w'] = g['_w'] / (g['_w'] + C)
    return g[keys + [name, name + '_w']]"""
i = find(A1)
sub(i, A1, N1)
print('build_cond_table: 표본 신뢰도 컬럼 추가')

# ------------------------------------------------ 2) 학습 경로에서도 붙인다
A2 = """            t = build_cond_table(past, keys, C, name, s).set_index(keys)[name]
            cur = (df['season'] == s).values
            sl = df.loc[cur, keys]
            idx = pd.MultiIndex.from_frame(sl) if len(keys) > 1 else pd.Index(sl[keys[0]])
            col[cur] = t.reindex(idx).values
        df[name] = col"""
N2 = """            _tb = build_cond_table(past, keys, C, name, s).set_index(keys)
            t, tw = _tb[name], _tb[name + '_w']
            cur = (df['season'] == s).values
            sl = df.loc[cur, keys]
            idx = pd.MultiIndex.from_frame(sl) if len(keys) > 1 else pd.Index(sl[keys[0]])
            col[cur] = t.reindex(idx).values
            colw[cur] = tw.reindex(idx).values
        df[name] = col
        df[name + '_w'] = colw"""
i2 = find(A2)
sub(i2, A2, N2)
sub(i2, """        col = np.full(len(df), np.nan)
        for s in seasons:""",
    """        col = np.full(len(df), np.nan)
        colw = np.full(len(df), np.nan)
        for s in seasons:""")
print('attach_cond_features: 학습 경로에 부착')

# ------------------------------------------------ 3) 학습 전 자체 검증 (4-12)
GUARD = '''
# ---- cond_*_w 검증 (학습 전에 터뜨린다, 4-12) ----
_cw = [c + "_w" for c in COND_COLS]
for _c in _cw:
    assert _c in X_full.columns, f"{_c} 가 X_full 에 없다"
    assert _c not in cat_features, f"{_c} 는 수치형이어야 한다"
    _v = X_full[_c].to_numpy(dtype="float64")
    _f = _v[~np.isnan(_v)]
    assert _f.min() >= 0.0 and _f.max() < 1.0, f"{_c} 범위 {_f.min():.3f}~{_f.max():.3f} ([0,1) 기대)"
print(f"cond_*_w 검증 OK: {len(_cw)}개, 전부 [0,1) 수치형 | 피처 {X_full.shape[1]}개")

'''
ANC = 'print("최종 파라미터:", BEST_PARAMS)'
i3 = find(ANC)
sub(i3, ANC, GUARD + ANC)
print('자체 검증 셀 삽입 (학습 전)')

# ------------------------------------------------ 4) 산출물 이름
i4 = find(OLDZIP)
sub(i4, OLDZIP, ZIP)
print('zip -> %s' % ZIP)

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

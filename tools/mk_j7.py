# 5분류 타겟을 **7분류**로 넓힌다 — 성공 칸을 판정(스트라이크/볼/인플레이)으로 3분할.
#
#   0 succ_strike 26.1%  1 succ_ball 15.7%  2 succ_inplay 10.6%
#   3 몰림만 11.6%  4 반대만 19.5%  5 둘다 3.4%  6 크게벗어남 13.2%
#
# 왜 (eda_judge, 2026-08-27): 판정 축의 드리프트가 실패유형 축보다 크다.
#   2019->2024 (R 전용)
#     succ_strike -0.0454   fail_strike +0.0570   <- 서로 반대, 가장 크다
#     succ_inplay -0.0254   fail_inplay +0.0143
#     succ_ball   +0.0109   fail_ball   -0.0114
#   이진이 보는 순변동 0.0599 vs 6칸 총 변동 0.1644 = **2.74배**
#   (비교) 실패유형 5분류는 2.4배였고 LB **+12.85** 였다.
#   해석: 같은 제구 성공이 갈수록 스트라이크가 아니라 볼로 불린다 = 존 체제 변화.
#
# ★ 왜 ptc6(-82.3)의 실패를 안 밟는가: 판정은 **결과의 하위 분할**이다.
#   구종은 투구 전에 정해지는, 타겟과 직교하는 축이라 용량이 "다음에 뭘 던질까" 로
#   샜다. 판정은 제구의 결과이므로 5분류와 같은 부류다.
#
# ★ 파편화 비용 0: 성공(52%)만 3등분하므로 최소 칸이 5분류와 동일한 3.41%(둘다) 그대로다.
#   (ptc15 는 최소 칸 0.47% 였다.)
#
# ⚠️ 성공 클래스가 **셋**이다. 5분류처럼 열 하나를 읽는 게 아니라 **더해야** P(success)다.
#    train_constants 에 success_cols 리스트로 저장하고 script.py 가 합산한다.
import ast
import json
import os
import sys

BASE = os.environ.get('J7_BASE', 'aimers_v10wzc.ipynb')
OUT = os.environ.get('J7_OUT', 'experiments/v10w/aimers_v10wj7.ipynb')
ZIP = os.environ.get('J7_ZIP', 'submit_v10wj7.zip')

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


def find(pat):
    for i in code:
        if pat in src(i):
            return i
    sys.exit('앵커를 못 찾았다: %s' % pat[:120])


# ------------------------------------------------- 1) 라벨 블록 교체
OLD_LBL = '''_mr = _recover_labels(df_processed, ["middle", "reverse"])
_m5, _r5 = _mr["middle"], _mr["reverse"]
_y5 = y_full.to_numpy(dtype="float64")
Y_CLS = np.where(_y5 == 1, 0,
                 np.where((_m5 == 1) & (_r5 == 1), 3,
                          np.where(_m5 == 1, 1, np.where(_r5 == 1, 2, 4))))
Y_CLS = np.where(np.isnan(_m5) | np.isnan(_r5), np.nan, Y_CLS)
CLS_OK = np.isfinite(Y_CLS)
_cnt = {int(k): int(v) for k, v in
        zip(*np.unique(Y_CLS[CLS_OK], return_counts=True))}
print("  5분류 분포:", _cnt, "| 결측 %.2f%%" % (100 * (~CLS_OK).mean()))
# 검산: 클래스 0 == control_success
assert (Y_CLS[CLS_OK] == 0).sum() == int(_y5[CLS_OK].sum()), "클래스0 != success"
print("  검산 OK: 클래스0 개수 == success 개수")'''

NEW_LBL = '''_mr = _recover_labels(df_processed, ["middle", "reverse", "ball", "strike"])
_m5, _r5 = _mr["middle"], _mr["reverse"]
_b5, _s5 = _mr["ball"], _mr["strike"]
_y5 = y_full.to_numpy(dtype="float64")

# 판정 3분할: 0 스트라이크 / 1 볼 / 2 인플레이. ball 과 strike 는 배타적이다.
_jok = np.isfinite(_b5) & np.isfinite(_s5)
assert int((_jok & (np.nan_to_num(_b5) + np.nan_to_num(_s5) > 1)).sum()) == 0, \\
    "ball 과 strike 가 동시에 1인 행이 있다 -- 판정이 분할이 아니다"
_J = np.where(_jok, np.where(_s5 == 1, 0, np.where(_b5 == 1, 1, 2)), np.nan)

# 7분류: 성공은 판정으로 3분할, 실패는 기존 4유형(번호를 3~6으로 민다)
Y_CLS = np.where(_y5 == 1, _J,
                 np.where((_m5 == 1) & (_r5 == 1), 5,
                          np.where(_m5 == 1, 3, np.where(_r5 == 1, 4, 6))))
Y_CLS = np.where(np.isnan(_m5) | np.isnan(_r5) | np.isnan(_J), np.nan, Y_CLS)
SUCCESS_CLS = [0, 1, 2]
CLS_OK = np.isfinite(Y_CLS)
_cnt = {int(k): int(v) for k, v in
        zip(*np.unique(Y_CLS[CLS_OK], return_counts=True))}
print("  7분류 분포:", _cnt, "| 결측 %.2f%%" % (100 * (~CLS_OK).mean()))
# 검산: 성공 클래스(0,1,2) 합 == control_success 합
_ns = int(np.isin(Y_CLS[CLS_OK], SUCCESS_CLS).sum())
assert _ns == int(_y5[CLS_OK].sum()), "성공 클래스 합 %d != success %d" % (
    _ns, int(_y5[CLS_OK].sum()))
print("  검산 OK: 성공 클래스 합 == success 개수 (%s)" % format(_ns, ","))'''

i = find(OLD_LBL)
sub(i, OLD_LBL, NEW_LBL)
sub(i, '# ================= 5분류 타겟 =================',
    '# ================= 7분류 타겟 (5분류 + 성공을 판정으로 3분할) =================')
print('1) 라벨 블록 -> 7분류 (셀 %d)' % i)

# ------------------------------------------------- 2) 본 학습: 성공 열 합산
i = find('        _ci = list(model.classes_).index(0)')
sub(i, '''        _ci = list(model.classes_).index(0)          # 성공 클래스의 열 위치
        if fold == 0 and seed == SEEDS[0]:
            SUCCESS_COL = _ci
            print(f"(성공 클래스 열 {_ci}, classes_={list(model.classes_)}) ", end="")
        raw = model.predict_proba(X_val)[:, _ci]''',
    '''        # ⚠️ 성공 클래스가 셋이다 -> 열들을 **더해야** P(success) 가 된다.
        _cl = list(model.classes_)
        _cix = [_cl.index(_s) for _s in SUCCESS_CLS]
        if fold == 0 and seed == SEEDS[0]:
            SUCCESS_COLS = _cix
            print(f"(성공 클래스 열 {_cix}, classes_={_cl}) ", end="")
        raw = model.predict_proba(X_val)[:, _cix].sum(axis=1)''')
sub(i, 'X, y = X_full, y_full\nSUCCESS_COL = 0',
    'X, y = X_full, y_full\nSUCCESS_COLS = list(range(len(SUCCESS_CLS)))')
print('2) 본 학습 -> 성공 열 합산 (셀 %d)' % i)

# ------------------------------------------------- 3) 오프셋 셀
i = find('        _cix = list(_m.classes_).index(0)')
sub(i, '''        _cix = list(_m.classes_).index(0)
        _rv = _m.predict_proba(_Xh.iloc[_vi])[:, _cix]
        _iso = IsotonicRegression(out_of_bounds="clip").fit(_rv, _yh.iloc[_vi].to_numpy())
        _ps.append(_iso.predict(_m.predict_proba(_Xv)[:, _cix]))''',
    '''        _cl = list(_m.classes_)
        _cx = [_cl.index(_s) for _s in SUCCESS_CLS]
        _rv = _m.predict_proba(_Xh.iloc[_vi])[:, _cx].sum(axis=1)
        _iso = IsotonicRegression(out_of_bounds="clip").fit(_rv, _yh.iloc[_vi].to_numpy())
        _ps.append(_iso.predict(_m.predict_proba(_Xv)[:, _cx].sum(axis=1)))''')
print('3) 오프셋 셀 -> 성공 열 합산 (셀 %d)' % i)

# ------------------------------------------------- 4) 상수
i = find('_tcj["success_col"] = int(SUCCESS_COL)')
sub(i, '_tcj["success_col"] = int(SUCCESS_COL)',
    '_tcj["success_cols"] = [int(_v) for _v in SUCCESS_COLS]')
sub(i, 'print("train_constants 에 success_col=%d, multiclass=True 기록" % SUCCESS_COL)',
    'print("train_constants 에 success_cols=%s 기록" % SUCCESS_COLS)')
print('4) success_cols 상수 저장 (셀 %d)' % i)

# ------------------------------------------------- 5) script.py 추론
i = find('_const.get("success_col", 1)')
sub(i, '''        # 5분류 모델이면 성공 클래스 열을 쓴다 (학습 때 저장한 상수).
        _sc = int(_const.get("success_col", 1)) if _const.get("multiclass") else 1
        raw = model.predict_proba(df_in[names])[:, _sc]''',
    '''        # 다중분류면 성공 클래스 열들을 **더한다** (학습 때 저장한 상수).
        if _const.get("multiclass"):
            _sc = [int(_v) for _v in _const["success_cols"]]
            raw = model.predict_proba(df_in[names])[:, _sc].sum(axis=1)
        else:
            raw = model.predict_proba(df_in[names])[:, 1]''')
print('5) script.py 예측 열 합산 (셀 %d)' % i)

# ------------------------------------------------- 6) zip 이름
for i in code:
    s = src(i)
    if 'ZIP_PATH' in s and 'submit_v10wzc.zip' in s:
        sub(i, 'submit_v10wzc.zip', ZIP)
        print('6) zip -> %s' % ZIP)
        break

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

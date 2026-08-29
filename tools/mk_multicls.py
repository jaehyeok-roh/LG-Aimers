# 타겟을 이진 -> **5분류**로 바꾼다 (aimers_mcbase.ipynb 위에).
#
# 0 성공 / 1 몰림만 / 2 반대만 / 3 둘다 / 4 크게벗어남.
# 라벨은 asof 인접 행 차분으로 복원한다 (eda40, success 검산 일치율 1.000000).
#
# 왜: 유형별 드리프트가 서로 반대다 (2019->2024, R 전용, 실패율 +0.0598):
#     middle +0.0484 | reverse +0.0543 | both +0.0015 | wild **-0.0414**
#     총 변동 0.144 가 순변동 0.060 을 만든다 — 이진 타겟이 2.4배를 가린다.
#
# 스크리너 실측 (wsboth 기준):
#     multicls  2024 +35.7 / 2023 +37.8   <- 두 시즌이 일치하는 유일한 후보
#     mcaux (multicls + auxrevns)  2024 +25.3 / 2023 +101.6
#     -> auxrevns(+22.5 / +55.0) 대비 증분 **2024 +2.8 / 2023 +46.6**
#   두 시즌 다 양수이고 어느 시즌에서도 지지 않는다. 다만 편차가 크다
#   (wsbat 과 같은 모양: 2024 +3.0 / 2023 +71.7 -> LB +30.11).
#
# ⚠️ 구현 요점 셋
#   1. CalibratedClassifierCV 는 다중분류에 그대로 못 쓴다.
#      성공 클래스 확률을 뽑아 **이진 y 로 IsotonicRegression 을 직접 적합**한다.
#   2. 폴드 분할은 **이진 y** 로 한다 (기존 구성과 폴드를 맞춰 비교 가능하게).
#   3. 성공 클래스의 열 인덱스를 train_constants.json 에 저장해 script.py 가 읽는다.
#      (CatBoost 의 classes_ 순서를 하드코딩하지 않는다.)
import ast
import json
import os
import sys

BASE = os.environ.get('MC_BASE', 'experiments/v10w/aimers_mcbase.ipynb')
OUT = os.environ.get('MC_OUT', 'experiments/v10w/aimers_v10wz.ipynb')

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


# ------------------------------------------------- 1) 5분류 라벨 (aux 블록 뒤)
CLSBLOCK = '''
# ================= 5분류 타겟 =================
# 0 성공 / 1 몰림만 / 2 반대만 / 3 둘다 / 4 크게벗어남
# ⚠️ middle 과 reverse 는 배타적이 아니다 (실패의 7.2%가 둘 다 1).
#    포수가 바깥에 앉았는데 가운데-몸쪽으로 몰리면 둘 다다 (eda41 교차표).
_mr = _recover_labels(df_processed, ["middle", "reverse"])
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
print("  검산 OK: 클래스0 개수 == success 개수")
# ==============================================

'''
ANCHOR = 'print("  selected_features.json 재기록 (%d개)" % len(feature_cols))'
hit = 0
for i in code:
    if ANCHOR in src(i):
        sub(i, ANCHOR, ANCHOR + '\n' + CLSBLOCK)
        hit = 1
        break
if not hit:
    sys.exit('5분류 블록 앵커를 못 찾았다')
print('1) 5분류 라벨 블록 삽입')

# ------------------------------------------------- 2) 본 학습을 다중분류로
OLD_FIT = '''        params = dict(BEST_PARAMS)
        params["random_seed"] = seed
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        raw = model.predict_proba(X_val)[:, 1]
        seed_oof_raw[seed][val_idx] = raw
        model.save_model(f"model/cb_fold_{seed}_{fold+1}.cbm")

        _c = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        _c.fit(X_val, y_val)
        iso = extract_isotonic(_c)
        cal = iso.predict(raw)'''
NEW_FIT = '''        params = dict(BEST_PARAMS)
        params["random_seed"] = seed
        params["loss_function"] = "MultiClass"
        params["eval_metric"] = "MultiClass"
        params.pop("early_stopping_rounds", None)
        # 라벨 결측 행(0.05%)은 학습에서만 뺀다
        _tk = tr_idx[CLS_OK[tr_idx]]
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[_tk], Y_CLS[_tk].astype(int), verbose=0)
        _ci = list(model.classes_).index(0)          # 성공 클래스의 열 위치
        if fold == 0 and seed == SEEDS[0]:
            SUCCESS_COL = _ci
            print(f"(성공 클래스 열 {_ci}, classes_={list(model.classes_)}) ", end="")
        raw = model.predict_proba(X_val)[:, _ci]
        seed_oof_raw[seed][val_idx] = raw
        model.save_model(f"model/cb_fold_{seed}_{fold+1}.cbm")

        # ⚠️ CalibratedClassifierCV 는 다중분류에 못 쓴다.
        #    성공 확률을 **이진 y** 로 직접 isotonic 적합한다 (채점이 P(success)).
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y_val.to_numpy())
        cal = iso.predict(raw)'''
hit = 0
for i in code:
    if OLD_FIT in src(i):
        sub(i, OLD_FIT, NEW_FIT)
        hit = 1
        break
if not hit:
    sys.exit('본 학습 앵커를 못 찾았다')
print('2) 본 학습 -> MultiClass')

# IsotonicRegression import + SUCCESS_COL 초기화
sub_done = 0
for i in code:
    s = src(i)
    if 'from sklearn.calibration import CalibratedClassifierCV' in s and 'X, y = X_full' in s:
        sub(i, 'from sklearn.calibration import CalibratedClassifierCV',
            'from sklearn.calibration import CalibratedClassifierCV\n'
            'from sklearn.isotonic import IsotonicRegression')
        sub(i, 'X, y = X_full, y_full', 'X, y = X_full, y_full\nSUCCESS_COL = 0')
        sub_done = 1
        break
if not sub_done:
    sys.exit('import 앵커를 못 찾았다')
print('3) IsotonicRegression import + SUCCESS_COL')

# ------------------------------------------------- 3) 오프셋 셀도 다중분류로
OLD_OFF = '''        _p = dict(BEST_PARAMS); _p["random_seed"] = SEEDS[0]
        _m = CatBoostClassifier(**_p)
        _m.fit(_Xh.iloc[_ti], _yh.iloc[_ti], eval_set=(_Xh.iloc[_vi], _yh.iloc[_vi]), verbose=0)
        _c = CalibratedClassifierCV(_m, method='isotonic', cv='prefit')
        _c.fit(_Xh.iloc[_vi], _yh.iloc[_vi])
        _ps.append(extract_isotonic(_c).predict(_m.predict_proba(_Xv)[:, 1]))'''
NEW_OFF = '''        _p = dict(BEST_PARAMS); _p["random_seed"] = SEEDS[0]
        _p["loss_function"] = "MultiClass"; _p["eval_metric"] = "MultiClass"
        _p.pop("early_stopping_rounds", None)
        _ch = Y_CLS[_tr_m]                       # 학습 절반의 5분류 라벨
        _ok = np.isfinite(_ch)
        _tk = _ti[_ok[_ti]]
        _m = CatBoostClassifier(**_p)
        _m.fit(_Xh.iloc[_tk], _ch[_tk].astype(int), verbose=0)
        _cix = list(_m.classes_).index(0)
        _rv = _m.predict_proba(_Xh.iloc[_vi])[:, _cix]
        _iso = IsotonicRegression(out_of_bounds="clip").fit(_rv, _yh.iloc[_vi].to_numpy())
        _ps.append(_iso.predict(_m.predict_proba(_Xv)[:, _cix]))'''
hit = 0
for i in code:
    if OLD_OFF in src(i):
        sub(i, OLD_OFF, NEW_OFF)
        hit = 1
        break
if not hit:
    sys.exit('오프셋 셀 학습 앵커를 못 찾았다')
print('4) 오프셋 셀 -> MultiClass')

# ------------------------------------------------- 4) 성공 클래스 열을 상수로 저장
# ⚠️ 오프셋 셀은 train_constants.json 을 **통째로 덮어쓴다** (ws_* 소실 사고의 원인).
#    따라서 그 셀 **맨 끝**에서 다시 읽어 병합한다.
MERGE = '''

# 5분류 정보를 train_constants.json 에 병합 (script.py 가 어느 열을 쓸지 안다).
# 이 셀이 위에서 파일을 통째로 덮어쓰므로 반드시 **맨 끝**에서 해야 한다.
with open("model/train_constants.json", "r") as f:
    _tcj = json.load(f)
_tcj["success_col"] = int(SUCCESS_COL)
_tcj["multiclass"] = True
with open("model/train_constants.json", "w") as f:
    json.dump(_tcj, f)
print("train_constants 에 success_col=%d, multiclass=True 기록" % SUCCESS_COL)
assert "ws_league_mean" in _tcj, "ws_* 상수가 날아갔다 (조립 전에 잡아야 한다)"
'''
hit = 0
for i in code:
    s = src(i)
    if 'RECENTER_OFFSET = solve_logit_offset' in s:
        setsrc(i, s + MERGE)
        hit = 1
        break
if not hit:
    sys.exit('오프셋 셀을 못 찾았다')
print('5) success_col 상수 저장 (오프셋 셀 맨 끝)')

# ------------------------------------------------- 5) script.py 추론
OLD_PRED = '        raw = model.predict_proba(df_in[names])[:, 1]'
NEW_PRED = ('        # 5분류 모델이면 성공 클래스 열을 쓴다 (학습 때 저장한 상수).\n'
            '        _sc = int(_const.get("success_col", 1)) '
            'if _const.get("multiclass") else 1\n'
            '        raw = model.predict_proba(df_in[names])[:, _sc]')
hit = 0
for i in code:
    if OLD_PRED in src(i) and 'SCRIPT_TEMPLATE' in src(i):
        sub(i, OLD_PRED, NEW_PRED)
        hit = 1
        break
if not hit:
    sys.exit('script.py 예측 앵커를 못 찾았다')
print('6) script.py 예측 열 선택')

for i in code:
    if 'ZIP_PATH' in src(i):
        s = src(i)
        for old in ('submit_v10wa.zip', 'submit_v10wq.zip', 'submit_v10wn.zip'):
            if old in s:
                sub(i, old, 'submit_v10wz.zip')
                break
        break

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

# 타겟에 **구종**을 곱한다 (aimers_mcbase.ipynb 위에).
#
#   PTC_MODE=ptc6   {성공, 실패} x {직구, 변화구, 오프스피드}   = 6분류
#   PTC_MODE=ptc15  5분류 실패유형 x 구종                        = 15분류
#
# 라벨은 asof 인접 행 차분으로 복원한다. 구종 셋은 합이 정확히 1임을 확인했다
# (fastball .5414 / breaking .2958 / offspeed .1627, 합=1 비율 1.000000).
#
# 왜: 구종별 성공률이 54.5 / 49.0 / 51.3% 로 5.5%p 벌어지는데 이진 타겟은
#     이걸 평균해 버린다. 5분류(multicls)가 LB +12.85 였던 것과 같은 논리 —
#     서로 다르게 움직이는 과정을 분리한다.
#
# ⚠️ 성공 클래스가 **여러 개**다 (구종마다 하나). 5분류와 달리 열 하나가 아니라
#    여러 열을 **더해야** P(success) 가 된다. train_constants 에 리스트로 저장한다.
#
# ⚠️ 스크리너로 판정하지 않았다. 다중분류 스크리너가 후보당 11시간이라
#    (fold 당 121분) 리더보드보다 느리다 — claude.md '스크리너는 재앙 필터로만'.
import ast
import json
import os
import sys

MODE = os.environ.get('PTC_MODE', 'ptc6')
if MODE not in ('ptc6', 'ptc15'):
    sys.exit('PTC_MODE 는 ptc6 또는 ptc15')
BASE = os.environ.get('PTC_BASE', 'experiments/v10w/aimers_mcbase.ipynb')
OUT = os.environ.get('PTC_OUT', 'aimers_%s.ipynb' % MODE)
ZIP = 'submit_%s.zip' % MODE

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


# ------------------------------------------------- 1) 라벨
PT_SRC = '''
# 구종 코드 0=직구 1=변화구 2=오프스피드. 셋 중 정확히 하나가 1임이 검증됐다.
_pt5 = _recover_labels(df_processed, ["fastball", "breaking", "offspeed"])
_f5, _b5, _o5 = _pt5["fastball"], _pt5["breaking"], _pt5["offspeed"]
_ptok = np.isfinite(_f5) & np.isfinite(_b5) & np.isfinite(_o5)
_tot = np.nan_to_num(_f5) + np.nan_to_num(_b5) + np.nan_to_num(_o5)
assert int((_ptok & (_tot != 1)).sum()) == 0, "구종 합이 1이 아닌 행 -- 복원 의심"
_PT = np.where(_ptok, np.where(_f5 == 1, 0, np.where(_b5 == 1, 1, 2)), np.nan)
_y5 = y_full.to_numpy(dtype="float64")
'''

BODY6 = PT_SRC + '''
# 0~2 = 성공 x 구종,  3~5 = 실패 x 구종
Y_CLS = _PT + 3.0 * (1.0 - _y5)
SUCCESS_CLS = [0, 1, 2]
'''

BODY15 = PT_SRC + '''
# 5분류 실패유형: 0 성공 / 1 몰림만 / 2 반대만 / 3 둘다 / 4 크게벗어남
# ⚠️ middle 과 reverse 는 배타적이 아니다 (실패의 7.2%가 둘 다 1, eda41 교차표).
_mr = _recover_labels(df_processed, ["middle", "reverse"])
_m5, _r5 = _mr["middle"], _mr["reverse"]
_C5 = np.where(_y5 == 1, 0,
               np.where((_m5 == 1) & (_r5 == 1), 3,
                        np.where(_m5 == 1, 1, np.where(_r5 == 1, 2, 4))))
_C5 = np.where(np.isnan(_m5) | np.isnan(_r5), np.nan, _C5)
Y_CLS = _C5 * 3.0 + _PT
SUCCESS_CLS = [0, 1, 2]        # _C5 == 0 인 세 칸
'''

CLSBLOCK = ('\n# ================= ' + MODE + ': 구종을 곱한 다중분류 ================='
            + (BODY6 if MODE == 'ptc6' else BODY15) + '''
CLS_OK = np.isfinite(Y_CLS)
_cnt = {int(k): int(v) for k, v in
        zip(*np.unique(Y_CLS[CLS_OK], return_counts=True))}
print("  ''' + MODE + ''' 분포:", _cnt, "| 결측 %.2f%%" % (100 * (~CLS_OK).mean()))
# 검산: 성공 클래스 합 == control_success 합
_ns = int(np.isin(Y_CLS[CLS_OK], SUCCESS_CLS).sum())
assert _ns == int(_y5[CLS_OK].sum()), "성공 클래스 합 %d != success %d" % (
    _ns, int(_y5[CLS_OK].sum()))
print("  검산 OK: 성공 클래스 합 == success 개수 (%s)" % format(_ns, ","))
# ==========================================================

''')

ANCHOR = 'print("  selected_features.json 재기록 (%d개)" % len(feature_cols))'
hit = 0
for i in code:
    if ANCHOR in src(i):
        sub(i, ANCHOR, ANCHOR + '\n' + CLSBLOCK)
        hit = 1
        break
if not hit:
    sys.exit('라벨 블록 앵커를 못 찾았다')
print('1) %s 라벨 블록 삽입' % MODE)

# ------------------------------------------------- 2) 본 학습
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
        # 라벨 결측 행은 학습에서만 뺀다
        _tk = tr_idx[CLS_OK[tr_idx]]
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[_tk], Y_CLS[_tk].astype(int), verbose=0)
        # ⚠️ 성공 클래스가 여러 개다 -> 열들을 **더해야** P(success) 가 된다.
        _cl = list(model.classes_)
        _cix = [_cl.index(_s) for _s in SUCCESS_CLS]
        if fold == 0 and seed == SEEDS[0]:
            SUCCESS_COLS = _cix
            print(f"(성공 클래스 열 {_cix}, classes_={_cl}) ", end="")
        raw = model.predict_proba(X_val)[:, _cix].sum(axis=1)
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

hit = 0
for i in code:
    s = src(i)
    if 'from sklearn.calibration import CalibratedClassifierCV' in s and 'X, y = X_full' in s:
        sub(i, 'from sklearn.calibration import CalibratedClassifierCV',
            'from sklearn.calibration import CalibratedClassifierCV\n'
            'from sklearn.isotonic import IsotonicRegression')
        sub(i, 'X, y = X_full, y_full',
            'X, y = X_full, y_full\nSUCCESS_COLS = list(range(len(SUCCESS_CLS)))')
        hit = 1
        break
if not hit:
    sys.exit('import 앵커를 못 찾았다')
print('3) IsotonicRegression import + SUCCESS_COLS')

# ------------------------------------------------- 3) 오프셋 셀
OLD_OFF = '''        _p = dict(BEST_PARAMS); _p["random_seed"] = SEEDS[0]
        _m = CatBoostClassifier(**_p)
        _m.fit(_Xh.iloc[_ti], _yh.iloc[_ti], eval_set=(_Xh.iloc[_vi], _yh.iloc[_vi]), verbose=0)
        _c = CalibratedClassifierCV(_m, method='isotonic', cv='prefit')
        _c.fit(_Xh.iloc[_vi], _yh.iloc[_vi])
        _ps.append(extract_isotonic(_c).predict(_m.predict_proba(_Xv)[:, 1]))'''
NEW_OFF = '''        _p = dict(BEST_PARAMS); _p["random_seed"] = SEEDS[0]
        _p["loss_function"] = "MultiClass"; _p["eval_metric"] = "MultiClass"
        _p.pop("early_stopping_rounds", None)
        _ch = Y_CLS[_tr_m]                       # 학습 절반의 다중분류 라벨
        _ok = np.isfinite(_ch)
        _tk = _ti[_ok[_ti]]
        _m = CatBoostClassifier(**_p)
        _m.fit(_Xh.iloc[_tk], _ch[_tk].astype(int), verbose=0)
        _cl = list(_m.classes_)
        _cx = [_cl.index(_s) for _s in SUCCESS_CLS]
        _rv = _m.predict_proba(_Xh.iloc[_vi])[:, _cx].sum(axis=1)
        _iso = IsotonicRegression(out_of_bounds="clip").fit(_rv, _yh.iloc[_vi].to_numpy())
        _ps.append(_iso.predict(_m.predict_proba(_Xv)[:, _cx].sum(axis=1)))'''
hit = 0
for i in code:
    if OLD_OFF in src(i):
        sub(i, OLD_OFF, NEW_OFF)
        hit = 1
        break
if not hit:
    sys.exit('오프셋 셀 학습 앵커를 못 찾았다')
print('4) 오프셋 셀 -> MultiClass')

# ------------------------------------------------- 4) 상수 병합
# ⚠️ 오프셋 셀은 train_constants.json 을 통째로 덮어쓴다 (ws_* 소실 사고의 원인).
#    따라서 그 셀 맨 끝에서 다시 읽어 병합한다.
MERGE = '''

# 다중분류 정보를 train_constants.json 에 병합 (script.py 가 어느 열을 더할지 안다).
# 이 셀이 위에서 파일을 통째로 덮어쓰므로 반드시 **맨 끝**에서 해야 한다.
with open("model/train_constants.json", "r") as f:
    _tcj = json.load(f)
_tcj["success_cols"] = [int(_v) for _v in SUCCESS_COLS]
_tcj["multiclass"] = True
with open("model/train_constants.json", "w") as f:
    json.dump(_tcj, f)
print("train_constants 에 success_cols=%s 기록" % SUCCESS_COLS)
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
print('5) success_cols 상수 저장')

# ------------------------------------------------- 5) script.py 추론
OLD_PRED = '        raw = model.predict_proba(df_in[names])[:, 1]'
NEW_PRED = ('        # 다중분류면 성공 클래스 열들을 **더한다** (학습 때 저장한 상수).\n'
            '        if _const.get("multiclass"):\n'
            '            _sc = [int(_v) for _v in _const["success_cols"]]\n'
            '            raw = model.predict_proba(df_in[names])[:, _sc].sum(axis=1)\n'
            '        else:\n'
            '            raw = model.predict_proba(df_in[names])[:, 1]')
hit = 0
for i in code:
    if OLD_PRED in src(i) and 'SCRIPT_TEMPLATE' in src(i):
        sub(i, OLD_PRED, NEW_PRED)
        hit = 1
        break
if not hit:
    sys.exit('script.py 예측 앵커를 못 찾았다')
print('6) script.py 예측 열 합산')

for i in code:
    if 'ZIP_PATH' in src(i):
        s = src(i)
        for old in ('submit_v10wa.zip', 'submit_v10wq.zip', 'submit_v10wn.zip'):
            if old in s:
                sub(i, old, ZIP)
                break
        break

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

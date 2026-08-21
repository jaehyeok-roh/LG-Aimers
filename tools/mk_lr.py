# 전이 손실 공격: 선형 baseline + 트리 잔차 부스팅.
#
# 문제: 학습분 스킬 2.21% vs 미래 시즌 0.80%. **64%(=141점)를 전이에서 잃는다.**
# 마감까지 필요한 +124 가 통째로 여기 있고, 다른 어떤 축에도 세 자릿수는 없다.
#
# 가설: 트리는 구조적으로 외삽을 못 한다. 2025 행의 피처값이 학습 범위 밖이면
# 가장 가까운 잎으로 보내고 끝이다. 선형항은 그 방향으로 값을 이어서 낸다.
# CatBoost 의 Pool(baseline=...) 은 **선형 예측을 출발점으로 깔고 트리가 잔차만
# 부스팅**하게 만든다. 재중심화(+18)가 이걸 '상수 하나'로 때운 조잡한 버전이고,
# 그게 먹혔다는 사실이 이 방향의 유일한 사전 증거다.
#
# 구현 요점
#   - LR 은 수치 피처만 쓴다 (범주형은 트리가 그대로 담당).
#   - 계수/스케일러를 **JSON 으로** 저장한다. pickle 이 아니라 numpy 한 줄로 재현되므로
#     sklearn 버전 차이에 안 걸린다 (평가 서버는 scikit-learn 1.8.0 고정).
#   - baseline 을 쓰면 CalibratedClassifierCV 가 깨진다 — 내부적으로 baseline 없는
#     predict_proba 를 부르기 때문이다. IsotonicRegression 직접 적합으로 바꾼다
#     (CLAUDE.md 에서 이 치환은 이미 '중립'으로 확인됨).
#   - LR 은 그 문맥의 학습분으로만 적합한다. 본 학습은 전체(~2024), 오프셋 셀은
#     ~2023. fold 루프 안에서는 전역 LR 을 쓴다 — OOF 가 약간 낙관적이 되지만
#     어차피 OOF 로 판정하지 않는다.
#
# 사용: python tools/mk_lr.py
import ast, json, os, sys   # noqa

BASE = '.kernels/s1o/aimers_s1o.ipynb'
# LR_SEASON_INTERACT=1 이면 선형항에 (시즌 x 각 피처) 상호작용을 넣는다 -> 태그 lr2
INTERACT = os.environ.get('LR_SEASON_INTERACT') == '1'
TAG = 'lr2' if INTERACT else 'lr'
nb = json.load(open(BASE, encoding='utf-8'))
C = nb['cells']


def sub(old, new, n=1):
    hit = 0
    for c in C:
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c['source'])
        if old in s:
            hit += s.count(old)
            c['source'] = s.replace(old, new).splitlines(keepends=True)
    if hit != n:
        sys.exit(f'앵커 {n}개 기대, {hit}개:\n  {old[:120]}')


# ---------- 1. 설정 ----------
sub("USE_COND_PB = False", """USE_COND_PB = False

# --- 전이 공격 (2026-08-21): 선형 baseline + 트리 잔차 ---
USE_LR_BASELINE = True
LR_C = 1.0            # L2 규제. 1.48M 행이라 약하게 걸어도 안정적이다
LR_MAX_ITER = 120     # lbfgs. 수렴 경고가 떠도 baseline 용도로는 문제 없다
# 시즌 x 피처 상호작용: 평범한 LR 은 season 이 선형항이라 2025 행 전체에 같은 상수를
# 더할 뿐이다(= 재중심화와 사실상 동일). 상호작용을 넣어야 '피처와 결과의 관계 자체가
# 해마다 어떻게 변해왔는지' 를 외삽한다. 전이 손실의 실체가 그것이라면 여기서 잡힌다.
LR_SEASON_INTERACT = """ + str(INTERACT))

# ---------- 2. LR 적합 헬퍼 (학습 셀 맨 앞) ----------
sub("""X, y = X_full, y_full  # Cell 6a에서 만든 전체 데이터 재사용""",
    '''X, y = X_full, y_full  # Cell 6a에서 만든 전체 데이터 재사용

from catboost import Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_lr_baseline(Xf, yf, save_to=None):
    """수치 피처만으로 로지스틱 회귀를 적합하고, 로짓을 내는 함수를 돌려준다.

    반환값 apply(Xg) 는 어떤 행 집합에도 쓸 수 있다. 계수·중앙값·스케일을
    JSON 으로 저장하면 script.py 가 numpy 한 줄로 같은 값을 재현한다.
    """
    num = [c for c in Xf.columns if Xf[c].dtype.name != 'category']
    A = Xf[num].astype('float64')
    med = A.median()
    A = A.fillna(med)
    mu = A.mean()
    sd = A.std().replace(0.0, 1.0).fillna(1.0)
    Z = ((A - mu) / sd).to_numpy()
    if LR_SEASON_INTERACT:
        _sc = Z[:, num.index('season')][:, None]
        Z = np.hstack([Z, Z * _sc])          # [피처, 시즌 x 피처]
    lr = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, solver='lbfgs')
    lr.fit(Z, yf)
    coef = lr.coef_[0].astype('float64')
    icpt = float(lr.intercept_[0])
    if save_to:
        with open(save_to, 'w') as _f:
            json.dump({'cols': num, 'coef': coef.tolist(), 'intercept': icpt,
                       'med': med.tolist(), 'mu': mu.tolist(), 'sd': sd.tolist(),
                       'interact': bool(LR_SEASON_INTERACT)}, _f)
        print(f"  LR baseline 저장: {save_to} (수치피처 {len(num)}개)")

    def apply(Xg):
        B = Xg[num].astype('float64').fillna(med)
        G = ((B - mu) / sd).to_numpy()
        if LR_SEASON_INTERACT:
            G = np.hstack([G, G * G[:, num.index('season')][:, None]])
        return G @ coef + icpt

    return apply


LR_APPLY = None
if USE_LR_BASELINE:
    import time as _t
    _t0 = _t.time()
    LR_APPLY = fit_lr_baseline(X, y, save_to="model/lr_baseline.json")
    BASE_ALL = LR_APPLY(X)
    _p0 = 1.0 / (1.0 + np.exp(-BASE_ALL))
    print(f"  LR 단독 Brier={brier_score_loss(y, _p0):.5f} "
          f"(상수예측 {y.mean()*(1-y.mean()):.5f}) / {_t.time()-_t0:.0f}초")
    print(f"  baseline 로짓: 평균 {BASE_ALL.mean():+.4f} 표준편차 {BASE_ALL.std():.4f}")''')

# ---------- 3. fold 루프에 baseline 주입 ----------
sub("""        params = dict(BEST_PARAMS)
        params["random_seed"] = seed
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        raw = model.predict_proba(X_val)[:, 1]
        seed_oof_raw[seed][val_idx] = raw
        model.save_model(f"model/cb_fold_{seed}_{fold+1}.cbm")

        _c = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        _c.fit(X_val, y_val)
        iso = extract_isotonic(_c)
        cal = iso.predict(raw)""",
    """        params = dict(BEST_PARAMS)
        params["random_seed"] = seed
        model = CatBoostClassifier(**params)
        if USE_LR_BASELINE:
            # baseline 을 쓰면 CalibratedClassifierCV 가 깨진다 (내부에서 baseline 없이
            # predict_proba 를 부른다) -> IsotonicRegression 직접 적합.
            _ptr = Pool(X_tr, y_tr, cat_features=cat_features, baseline=BASE_ALL[tr_idx])
            _pva = Pool(X_val, y_val, cat_features=cat_features, baseline=BASE_ALL[val_idx])
            model.fit(_ptr, eval_set=_pva, verbose=0)
            raw = model.predict_proba(_pva)[:, 1]
        else:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
            raw = model.predict_proba(X_val)[:, 1]
        seed_oof_raw[seed][val_idx] = raw
        model.save_model(f"model/cb_fold_{seed}_{fold+1}.cbm")

        if USE_LR_BASELINE:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(raw, y_val)
        else:
            _c = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            _c.fit(X_val, y_val)
            iso = extract_isotonic(_c)
        cal = iso.predict(raw)""")

# ---------- 4. 오프셋 셀: 그 문맥(~2023)으로 LR 재적합 ----------
sub("""    _skf = StratifiedKFold(n_splits=N_HOLDOUT_FOLDS, shuffle=True, random_state=SEEDS[0])
    _ps = []
    for _f, (_ti, _vi) in enumerate(_skf.split(_Xh, _yh)):
        _p = dict(BEST_PARAMS); _p["random_seed"] = SEEDS[0]
        _m = CatBoostClassifier(**_p)
        _m.fit(_Xh.iloc[_ti], _yh.iloc[_ti], eval_set=(_Xh.iloc[_vi], _yh.iloc[_vi]), verbose=0)
        _c = CalibratedClassifierCV(_m, method='isotonic', cv='prefit')
        _c.fit(_Xh.iloc[_vi], _yh.iloc[_vi])
        _ps.append(extract_isotonic(_c).predict(_m.predict_proba(_Xv)[:, 1]))
        print(f"  홀드아웃 fold {_f+1}/{N_HOLDOUT_FOLDS} 완료")""",
    """    # 배포 구조와 맞추려면 LR 도 그 문맥의 학습분(~2023)으로만 적합해야 한다.
    # 전역 LR(2024 포함)을 쓰면 홀드아웃에 누수가 생긴다.
    _lrh = fit_lr_baseline(_Xh, _yh) if USE_LR_BASELINE else None
    _bh = _lrh(_Xh) if USE_LR_BASELINE else None
    _bv = _lrh(_Xv) if USE_LR_BASELINE else None

    _skf = StratifiedKFold(n_splits=N_HOLDOUT_FOLDS, shuffle=True, random_state=SEEDS[0])
    _ps = []
    for _f, (_ti, _vi) in enumerate(_skf.split(_Xh, _yh)):
        _p = dict(BEST_PARAMS); _p["random_seed"] = SEEDS[0]
        _m = CatBoostClassifier(**_p)
        if USE_LR_BASELINE:
            _m.fit(Pool(_Xh.iloc[_ti], _yh.iloc[_ti], cat_features=cat_features,
                        baseline=_bh[_ti]),
                   eval_set=Pool(_Xh.iloc[_vi], _yh.iloc[_vi], cat_features=cat_features,
                                 baseline=_bh[_vi]), verbose=0)
            _rv = _m.predict_proba(Pool(_Xh.iloc[_vi], cat_features=cat_features,
                                        baseline=_bh[_vi]))[:, 1]
            _iso = IsotonicRegression(out_of_bounds='clip')
            _iso.fit(_rv, _yh.iloc[_vi])
            _pv = _m.predict_proba(Pool(_Xv, cat_features=cat_features, baseline=_bv))[:, 1]
            _ps.append(_iso.predict(_pv))
        else:
            _m.fit(_Xh.iloc[_ti], _yh.iloc[_ti], eval_set=(_Xh.iloc[_vi], _yh.iloc[_vi]), verbose=0)
            _c = CalibratedClassifierCV(_m, method='isotonic', cv='prefit')
            _c.fit(_Xh.iloc[_vi], _yh.iloc[_vi])
            _ps.append(extract_isotonic(_c).predict(_m.predict_proba(_Xv)[:, 1]))
        print(f"  홀드아웃 fold {_f+1}/{N_HOLDOUT_FOLDS} 완료")""")

# ---------- 5. script.py: baseline 재현 후 Pool 로 추론 ----------
sub("""    import glob
    preds = []
    cb_paths = sorted(glob.glob(os.path.join("model", "cb_fold_*.cbm")))""",
    """    # ---------- 선형 baseline 재현 ----------
    # 학습 때 저장한 계수/중앙값/스케일로 같은 로짓을 numpy 한 줄로 만든다.
    # sklearn 객체를 pickle 하지 않으므로 버전 차이에 걸리지 않는다.
    _lr_base = None
    _lrp = os.path.join("model", "lr_baseline.json")
    if os.path.exists(_lrp):
        with open(_lrp, "r") as _f:
            _lrj = json.load(_f)
        _cols = _lrj["cols"]
        for _c2 in _cols:
            if _c2 not in df_features.columns:
                df_features[_c2] = np.nan
        _A = df_features[_cols].astype("float64")
        _A = _A.fillna(pd.Series(_lrj["med"], index=_cols))
        _Z = (_A.to_numpy() - np.asarray(_lrj["mu"])) / np.asarray(_lrj["sd"])
        if _lrj.get("interact"):
            _Z = np.hstack([_Z, _Z * _Z[:, _cols.index("season")][:, None]])
        _lr_base = _Z @ np.asarray(_lrj["coef"]) + float(_lrj["intercept"])
        print("LR baseline 적용: 평균 %.4f 표준편차 %.4f" % (_lr_base.mean(), _lr_base.std()))

    import glob
    preds = []
    cb_paths = sorted(glob.glob(os.path.join("model", "cb_fold_*.cbm")))""")

sub("""        raw = model.predict_proba(df_in[names])[:, 1]""",
    """        if _lr_base is None:
            raw = model.predict_proba(df_in[names])[:, 1]
        else:
            _cf = [c for c in names if str(df_in[c].dtype) in ("category", "object")]
            raw = model.predict_proba(
                Pool(df_in[names], cat_features=_cf, baseline=_lr_base))[:, 1]""")

sub("""from catboost import CatBoostClassifier""",
    """from catboost import CatBoostClassifier, Pool""", n=3)   # Optuna 셀 + 학습 셀 + script.py 템플릿

# ---------- 6. zip 목록 ----------
sub("""       "model/feat_diff.csv", "model/feat_speed.csv", "model/feat_rp.csv"]""",
    """       "model/feat_diff.csv", "model/feat_speed.csv", "model/feat_rp.csv"]
    + (["model/lr_baseline.json"] if os.path.exists("model/lr_baseline.json") else [])""")

# ---------- 검사 ----------
for i, c in enumerate(C):
    if c['cell_type'] == 'code':
        try:
            ast.parse(''.join(c['source']))
        except SyntaxError as e:
            sys.exit(f'cell {i} 문법 오류: {e}')

src = '\n'.join(''.join(c['source']) for c in C if c['cell_type'] == 'code')
for nm, k in [('USE_LR_BASELINE', 6), ('fit_lr_baseline', 3), ('BASE_ALL', 3),
              ('lr_baseline.json', 3), ('IsotonicRegression', 4), ('Pool(', 7)]:
    if src.count(nm) < k:
        sys.exit(f'{nm} 참조 {src.count(nm)}회 (기대 {k}+) — 패치 누락')
if 'RUN_OPTUNA = False' not in src:
    sys.exit('RUN_OPTUNA 가 False 가 아니다')

D = f'.kernels/s1{TAG}'
os.makedirs(D, exist_ok=True)
for c in C:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
json.dump(nb, open(f'{D}/aimers_s1{TAG}.ipynb', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
meta = json.load(open('kernel-metadata.json'))
meta.update(id=f'homekeggle/aimers-s1{TAG}', title=f'aimers-s1{TAG}',
            code_file=f'aimers_s1{TAG}.ipynb')
json.dump(meta, open(f'{D}/kernel-metadata.json', 'w'), indent=2)
print(f'{D}/aimers_s1{TAG}.ipynb 생성 — 전 셀 문법 OK')

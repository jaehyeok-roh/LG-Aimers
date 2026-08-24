# 실패유형 보조 타겟 P(reverse|X) 를 교차적합해 피처로 얹는다 (v10wp 위에).
#
# 스크리너 실측 (wsboth 기준):
#   로컬 2024   894 -> auxrev **930 (+36)** / aux3 933 (+39)
#   캐글 2024   889 -> auxrev **929 (+40)** / auxperm(라벨 섞음) 887 (**-1.9**)
# 대조군이 기준선으로 돌아왔고 섞은 라벨의 OOF 표준편차가 0.00452 (사실상 상수)이므로
# **구조적 누수가 아니다.** 다른 하드웨어 두 곳에서 독립 재현됐다.
#
# 왜 되는가: 본 모델은 이진 y 만 보므로 P(reverse|X) 를 **배울 수 없다** (그 라벨을
# 못 보니까). 그리고 reverse 는 여섯 실패유형 중 유일하게 투수 수준 신뢰도가
# success 보다 높다 (0.941 vs 0.911, 신호 크기 102% — eda41).
#
# ⚠️ 누수 방지가 이 구현의 전부다. `reverse=1 => success=0` 이므로:
#   학습 행  -> 반드시 out-of-fold 값
#   추론 행  -> 폴드 모델들의 평균 (그 행의 라벨은 애초에 없다)
#   오프셋셀 -> 검증 시즌 라벨을 안 본 별도 판(AUX_HO)을 쓴다
# ⚠️ 행 순서: 복원은 (pitcher_id, asof_pitcher_n) 정렬이 전제인데 df_processed 는
#   time_idx 로 정렬돼 있다 (claude.md 4-15). success 검산을 assert 로 박는다.
import ast
import json
import os
import sys

BASE = os.environ.get('AR_BASE', 'aimers_v10wp.ipynb')
OUT = os.environ.get('AR_OUT', 'aimers_v10wa.ipynb')

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


# ---------------------------------------------------------------- 1) 학습측
AUXBLOCK = '''
# ================= aux_rev: 실패유형 보조 타겟 (교차적합) =================
AUX_ITERS = 300
AUX_FOLDS = 3


def _recover_reverse(df):
    """asof 인접 행 차분으로 **투구 단위** reverse 라벨을 복원한다 (train 전용).

    `asof_pitcher_n` 은 투수 내에서 정확히 +1 씩 증가하므로 인접 두 행이
    한 투구 차이다. asof 는 "직전까지" 이므로 행 i 와 i+1 로 투구 i 의 라벨이 나온다.
    ⚠️ (pitcher_id, asof_pitcher_n) 정렬이 전제다 — df_processed 는 time_idx 로
       정렬돼 있다 (claude.md 4-15). success 로 검산해 틀리면 즉시 중단한다.
    ⚠️ 규정: 차분은 train 에서만. test 인접 행 차분은 주최측이 "규칙 위반" 이라
       답한 사안이다 (2jin1 08-17). train 유래 값에는 제약이 없다 (DACON.GM 08-19).
    """
    n = df["asof_pitcher_n"].to_numpy(dtype="float64")
    g = df["pitcher_id"].to_numpy()
    o = np.lexsort((n, g))
    ns, gs = n[o], g[o]
    ok = np.r_[(gs[:-1] == gs[1:]) & (ns[1:] - ns[:-1] == 1), False]
    out = {}
    for k in ("reverse", "success"):
        cum = np.round(df["asof_pitcher_%s_rate" % k].fillna(0)
                       .to_numpy(dtype="float64")[o] * ns)
        d = np.r_[cum[1:] - cum[:-1], np.nan]
        v = np.where(ok & np.isin(d, [0.0, 1.0]), d, np.nan)
        b = np.full(len(df), np.nan)
        b[o] = v
        out[k] = b
    m = np.isfinite(out["success"])
    acc = float((out["success"][m] == df["control_success"].to_numpy()[m]).mean())
    print("  라벨 복원 검산: success 일치율 %.6f (표본 %s)"
          % (acc, format(int(m.sum()), ",")))
    if acc < 0.999:
        raise RuntimeError("복원 검산 실패 %.6f -- 행 순서를 의심할 것" % acc)
    return out["reverse"]


def _fit_aux(Xh, tgt, Xv, tag):
    """보조 모델을 교차적합한다. 학습행은 OOF, 예측행은 폴드 평균.

    ⚠️ reverse=1 이면 success=0 이므로 in-sample 적합은 타겟을 통째로 흘린다.
    """
    _p = dict(BEST_PARAMS)
    _p["iterations"] = AUX_ITERS
    _p.pop("early_stopping_rounds", None)
    _p["eval_metric"] = "Logloss"
    oof = np.full(len(Xh), np.nan)
    vp = np.zeros(len(Xv)) if Xv is not None else None
    fit = np.isfinite(tgt)
    ms = []
    _sk = StratifiedKFold(n_splits=AUX_FOLDS, shuffle=True, random_state=7)
    for _k, (_ti, _vi) in enumerate(_sk.split(Xh, np.nan_to_num(tgt, nan=0.0).astype(int))):
        _ti = _ti[fit[_ti]]
        _m = CatBoostClassifier(**_p)
        _m.fit(Xh.iloc[_ti], tgt[_ti].astype(int), verbose=0)
        oof[_vi] = _m.predict_proba(Xh.iloc[_vi])[:, 1]
        if Xv is not None:
            vp += _m.predict_proba(Xv)[:, 1] / AUX_FOLDS
        ms.append(_m)
        print("  %s aux fold %d/%d" % (tag, _k + 1, AUX_FOLDS))
    return oof, vp, ms


_rev = _recover_reverse(df_processed)
print("  reverse 율 %.4f | 결측 %.2f%%" % (np.nanmean(_rev), 100 * np.isnan(_rev).mean()))

# 배포용: 전체 train 으로 교차적합. 모델 3개를 zip 에 실어 추론 때 평균한다.
with open("model/aux_features.json", "w") as f:
    json.dump(list(feature_cols), f)
_oof, _, _auxms = _fit_aux(X_full, _rev, None, "배포")
for _k, _m in enumerate(_auxms):
    _m.save_model("model/aux_rev_%d.cbm" % _k)

# 오프셋 측정용 별도 판: 검증 시즌(HOLDOUT_SEASON) 라벨을 **안 본** 보조 모델.
# 이걸 안 하면 오프셋 셀이 낙관적으로 측정돼 재중심화 상수가 어긋난다 (134점짜리다).
_hm = (df_processed["season"] <= HOLDOUT_SEASON - 1).to_numpy()
_vm = (df_processed["season"] == HOLDOUT_SEASON).to_numpy()
_o2, _v2, _ = _fit_aux(X_full[_hm].reset_index(drop=True), _rev[_hm],
                       X_full[_vm].reset_index(drop=True), "오프셋용")
AUX_HO = np.full(len(X_full), np.nan)
AUX_HO[_hm] = _o2
AUX_HO[_vm] = _v2

X_full["aux_rev"] = _oof.astype(np.float32)
feature_cols = list(feature_cols) + ["aux_rev"]
print("  aux_rev 추가: OOF 평균 %.4f (실제 reverse 율 %.4f) | 피처 %d -> %d"
      % (np.nanmean(_oof), np.nanmean(_rev), len(feature_cols) - 1, len(feature_cols)))
assert not np.isnan(_oof).any(), "aux_rev 에 NaN 이 있다 -- 폴드가 전체를 안 덮었다"

# feature_cols 가 바뀌었으므로 selected_features.json 을 다시 쓴다.
# (원본 기록은 이 블록보다 앞에 있어서 aux_rev 가 빠져 있다.)
with open("model/selected_features.json", "w") as f:
    json.dump(list(feature_cols), f)
print("  selected_features.json 재기록 (%d개)" % len(feature_cols))
# =========================================================================

'''
# ⚠️ claude.md 4-12 — 앵커는 반드시 BEST_PARAMS 가 **완성된 뒤**여야 한다.
#    _fit_aux 가 BEST_PARAMS 를 쓰는데 그건 셀 16 **끝**에서 만들어진다.
#    앞에 넣었다가 NameError 로 GPU 35분을 태웠다 (2026-08-24 v1).
ANCHOR = 'with open("model/best_params.json", "w") as f:\n    json.dump(BEST_PARAMS, f, indent=2)'
hit = 0
for i in code:
    if ANCHOR in src(i):
        sub(i, ANCHOR, ANCHOR + AUXBLOCK)
        hit = 1
        break
if not hit:
    sys.exit('학습측 앵커를 못 찾았다')
print('1) 학습측 aux 블록 삽입 (BEST_PARAMS 뒤)')

# ---------------------------------------------------------------- 2) 오프셋 셀
OFF_OLD = '    _Xh, _yh = X[_tr_m], y[_tr_m]\n    _Xv, _yv = X[_va_m], y[_va_m]'
OFF_NEW = ('    _Xh, _yh = X[_tr_m].copy(), y[_tr_m]\n'
           '    _Xv, _yv = X[_va_m].copy(), y[_va_m]\n'
           '    # 검증 시즌 라벨을 안 본 보조 예측으로 갈아끼운다. 안 그러면 오프셋이\n'
           '    # 낙관적으로 측정돼 재중심화 상수가 어긋난다 (지금 재중심화는 +134 다).\n'
           '    _Xh["aux_rev"] = AUX_HO[_tr_m].astype(np.float32)\n'
           '    _Xv["aux_rev"] = AUX_HO[_va_m].astype(np.float32)\n'
           '    print("  오프셋용 aux_rev 교체 완료 (검증 시즌 라벨 미접촉)")')
hit = 0
for i in code:
    if OFF_OLD in src(i):
        sub(i, OFF_OLD, OFF_NEW)
        hit = 1
        break
if not hit:
    sys.exit('오프셋 셀 앵커를 못 찾았다')
print('2) 오프셋 셀 aux 교체')

# ---------------------------------------------------------------- 3) script.py
INFER = '''    # ---------- aux_rev: 보조 모델로 P(reverse|X) 예측 ----------
    # 학습 때 교차적합해 만든 피처다. 추론은 폴드 모델 3개의 평균을 쓴다.
    # 이 행의 입력만으로 계산되므로 "평가 데이터 각 행 독립" 규정을 만족한다.
    import glob as _glob
    _auxf = os.path.join("model", "aux_features.json")
    if os.path.exists(_auxf):
        with open(_auxf, "r") as f:
            _aux_cols = json.load(f)
        _apaths = sorted(_glob.glob(os.path.join("model", "aux_rev_*.cbm")))
        if not _apaths:
            raise RuntimeError("aux_features.json 은 있는데 aux_rev_*.cbm 이 없다")
        for _c in _aux_cols:
            if _c not in df_proc.columns:
                df_proc[_c] = np.nan
        _Xa = df_proc[_aux_cols].copy()
        for _c in _Xa.columns:
            if _Xa[_c].dtype.name in ["category", "object"]:
                _Xa[_c] = _Xa[_c].astype(str).astype("category")
        _acc = np.zeros(len(_Xa))
        for _p in _apaths:
            _am = CatBoostClassifier()
            _am.load_model(_p)
            _acc += _am.predict_proba(_Xa)[:, 1] / len(_apaths)
        df_proc["aux_rev"] = _acc.astype(np.float32)
        print("aux_rev 예측 완료: 모델 %d개 | 평균 %.4f" % (len(_apaths), _acc.mean()))

'''
SEL = ('    with open("model/selected_features.json", "r") as f:\n'
       '        selected_features = json.load(f)')
GUARD = ('\n    if "aux_rev" in selected_features and "aux_rev" not in df_proc.columns:\n'
         '        raise RuntimeError("selected_features 에 aux_rev 가 있는데 "\n'
         '                           "만들어지지 않았다 -- aux_features.json / "\n'
         '                           "aux_rev_*.cbm 을 확인할 것")')
hit = 0
for i in code:
    if SEL in src(i) and 'SCRIPT_TEMPLATE' in src(i):
        sub(i, SEL, INFER + SEL + GUARD)
        hit = 1
        break
if not hit:
    sys.exit('script.py 앵커를 못 찾았다')
print('3) script.py 추론 블록 + 가드 삽입')

# ---------------------------------------------------------------- 4) zip 목록
OLDZ = ('    + (["model/pitcher_appearance.csv"]\n'
        '       if os.path.exists("model/pitcher_appearance.csv") else [])\n)')
hit = 0
for i in code:
    s = src(i)
    if 'ZIP_PATH' in s:
        if OLDZ in s:
            sub(i, OLDZ, OLDZ[:-2] + '\n'
                '    + sorted(glob.glob("model/aux_rev_*.cbm"))\n'
                '    + (["model/aux_features.json"]\n'
                '       if os.path.exists("model/aux_features.json") else [])\n)')
            hit = 1
        sub(i, 'ZIP_PATH = "submit_v10wp.zip"', 'ZIP_PATH = "submit_v10wa.zip"')
        break
if not hit:
    sys.exit('zip 목록 앵커를 못 찾았다')
print('4) zip 목록에 보조 모델 추가')

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

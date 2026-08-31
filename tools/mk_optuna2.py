# Optuna 재탐색 — 5분류 120피처 영역에서 처음이다.
#
#   python tools/mk_optuna2.py
#
# 마지막 탐색은 **v4(이진 96피처)** 였고 그 이후로 `RUN_OPTUNA=False` 로 얼려뒀다.
# `optbest` 교훈이 그대로 적용된다: 다른 파라미터 영역에서 잰 효과는 안 옮겨진다
# (그때는 bc128 -12 / ctr2 -5 를 벗기면 +17 일 줄 알았는데 부호가 뒤집혔다).
# 지금 파라미터도 정확히 그 처지다 — 이진에서 고른 값을 5분류에서 쓰고 있다.
#
# 설계는 세 가지가 전부다.
#
# ① **용량 축을 고정한다** (iterations / depth / learning_rate).
#    리더보드가 이 축에서 이미 포화를 확인했고(it500 -23.07, it700 -1.47),
#    홀드아웃은 학습 데이터가 18% 적어 체계적으로 "용량을 줄여라" 에 가산점을
#    준다 (3장 'CPU 가 6배 과대평가된 이유' 와 같은 원인). 그 편향이 닿을 수
#    있는 축을 탐색 공간에서 아예 빼버린다. depth9 / it1400 은 별도 커널이
#    리더보드로 직접 재고 있으므로 중복도 아니다.
#
# ② **현행 파라미터를 trial 0 으로 강제**해 대조군으로 쓴다 (mk_optuna.py 와
#    같은 패턴). 절대값은 하네스마다 다르므로 같은 실행 안의 차이만 읽는다.
#
# ③ **탐색 1위를 그대로 믿지 않는다.** 오프셋 셀의 실행 간 노이즈가 ±13 인데
#    (GPU 비결정성, 3장) 30 trial 의 최댓값은 선택만으로 +25 가 뜬다.
#    상위 3개와 대조군을 seed 3개씩 재실행해 평균으로 판정하고, 마진(+10)
#    미만이면 **현행을 유지하고 학습을 건너뛴다.**
#
# 탐색 위치: aux_rev / Y_CLS 가 만들어진 **뒤**다. 그래야 배포와 같은 5분류로
# 잴 수 있다. 보조 모델은 현행 파라미터로 이미 적합된 상태로 두는 게 맞다 —
# 한 번에 하나만 바꾼다 (4-14).
import ast
import json
import os
import sys

BASE = os.environ.get('OPT_BASE', 'experiments/v10w/aimers_v10wph.ipynb')
OUT = os.environ.get('OPT_OUT', 'experiments/v10w/aimers_v10wopt.ipynb')
ZIP = os.environ.get('OPT_ZIP', 'submit_v10wopt.zip')
OLDZIP = os.environ.get('OPT_OLDZIP', 'submit_v10wph.zip')
N_TRIALS = int(os.environ.get('OPT_TRIALS', '30'))
SUB_FRAC = float(os.environ.get('OPT_SUB', '0.5'))
MARGIN = float(os.environ.get('OPT_MARGIN', '10'))

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


SEARCH = '''
# ================= Optuna 재탐색 (5분류 120피처, 2024 홀드아웃) =================
# 마지막 탐색은 v4(이진 96피처)였다. 여기서 처음으로 배포와 같은 영역에서 잰다.
# 용량 축(iterations/depth/learning_rate)은 **고정**한다 -- 홀드아웃이 그 축에서
# 방향조차 못 맞히고(it500 홀드아웃 +30 / LB -23), 소표본 편향이 정확히 거기
# 작용하기 때문이다. 남은 것은 규제·추정 방식뿐이고 그건 5분류에서 미측정이다.
try:
    import optuna
except ImportError:
    import subprocess as _sp, sys as _sy
    _sp.run([_sy.executable, "-m", "pip", "install", "-q", "optuna"], check=True)
    import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPT_TRIALS = %d
OPT_SUB_FRAC = %g          # 탐색용 학습 서브샘플 (~2023 안에서)
OPT_CONFIRM_SEEDS = [11, 22, 33]
OPT_MARGIN = %g            # 이만큼 못 이기면 현행 유지 (노이즈 +-13 을 감안)

_ssn = df_processed["season"].to_numpy()
_htr = np.where((_ssn <= HOLDOUT_SEASON - 1) & CLS_OK)[0]
_hva = np.where(_ssn == HOLDOUT_SEASON)[0]
_rng = np.random.default_rng(0)
_htr = np.sort(_rng.choice(_htr, int(len(_htr) * OPT_SUB_FRAC), replace=False))

# 검증행의 aux_rev 는 **검증 시즌 라벨을 안 본 판**(AUX_HO)으로 갈아끼운다.
# 배포용 aux 는 전체 교차적합이라 2024 행이 다른 폴드를 통해 2024 라벨을 본다.
X_OPT = X_full.copy()
for _k, _v in AUX_HO.items():
    if _k in X_OPT.columns:
        X_OPT[_k] = _v.astype(np.float32)
_yva = y_full.to_numpy()[_hva]
_TGT = float(_yva.mean())
print("Optuna 하네스: 학습 %%s행 (~%%d) -> 검증 %%s행 (%%d)"
      %% (format(len(_htr), ","), HOLDOUT_SEASON - 1, format(len(_hva), ","), HOLDOUT_SEASON))


def _recenter(p, target):
    """평균 예측을 target 에 맞추는 로짓 상수 시프트. 모든 trial 에 동일하게
    적용하므로 '편향이 우연히 작았던' trial 이 이기는 교락을 없앤다."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    lo, hi = -5.0, 5.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if float((1 / (1 + np.exp(-(z + m)))).mean()) < target:
            lo = m
        else:
            hi = m
    return 1 / (1 + np.exp(-(z + 0.5 * (lo + hi))))


def _run(extra, seed):
    """한 구성을 1회 적합해 2024 스킬 점수를 낸다. 실패하면 큰 음수."""
    p = dict(BEST_PARAMS)
    p.update(extra)
    p["random_seed"] = seed
    p["loss_function"] = "MultiClass"
    p["eval_metric"] = "MultiClass"
    p.pop("early_stopping_rounds", None)
    try:
        m = CatBoostClassifier(**p)
        m.fit(X_OPT.iloc[_htr], Y_CLS[_htr].astype(int), verbose=0)
        _ci = list(m.classes_).index(0)
        raw = m.predict_proba(X_OPT.iloc[_hva])[:, _ci]
    except Exception as e:
        print("   ! 실패:", str(e)[:120])
        return -1e4
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, _yva)
    pr = _recenter(iso.predict(raw), _TGT)
    return float((1 - brier_score_loss(_yva, pr) / (_TGT * (1 - _TGT))) * 100000)


def _space(t):
    return {
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": t.suggest_float("random_strength", 0.05, 5.0, log=True),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": t.suggest_categorical("border_count", _BC),
        # one_hot_max_size 는 이 프로젝트에서 **한 번도 안 건드린 축**이고,
        # cnt12(12칸)/hand4(4칸)/phteam(26칸)/base_state 를 CTR 대신 one-hot 으로
        # 돌린다. claude.md 의 'CTR 경쟁' 가설(조합 4개면 -4.61)을 직접 시험한다.
        "one_hot_max_size": t.suggest_categorical("one_hot_max_size", _OH),
        "max_ctr_complexity": t.suggest_categorical("max_ctr_complexity", _MC),
    }


_BC, _OH, _MC = [128, 254], [2, 12, 30], [1, 2]

# 대조군은 **실제로 해석된 기본값**에서 읽는다. CatBoost 는 장치·모드에 따라
# 기본값을 다르게 잡고(claude.md: CPU border_count 254 / GPU 128 로 26점 차이),
# one_hot_max_size 는 GPU 에서 2 또는 10 이다. 짐작으로 박으면 대조군이
# 배포본과 달라져서 이 실험 전체가 무의미해진다. 20 iter 짜리로 한 번 물어본다.
_probe = dict(BEST_PARAMS)
_probe.update({"iterations": 20, "loss_function": "MultiClass",
               "eval_metric": "MultiClass"})
_probe.pop("early_stopping_rounds", None)
_pm = CatBoostClassifier(**_probe)
_pm.fit(X_OPT.iloc[_htr[:20000]], Y_CLS[_htr[:20000]].astype(int), verbose=0)
_ap = _pm.get_all_params()
_CTRL = {k: BEST_PARAMS[k] for k in
         ("l2_leaf_reg", "random_strength", "bagging_temperature")}
for _k in ("border_count", "one_hot_max_size", "max_ctr_complexity"):
    _CTRL[_k] = _ap[_k]
print("해석된 기본값:", {k: _ap[k] for k in
      ("border_count", "one_hot_max_size", "max_ctr_complexity",
       "bootstrap_type", "depth", "learning_rate")})
print("대조군:", _CTRL)
# 탐색 공간에 실제 기본값이 없으면 대조군을 enqueue 할 수 없다 -> 넣어준다
for _k, _lst in (("border_count", _BC), ("one_hot_max_size", _OH),
                 ("max_ctr_complexity", _MC)):
    if _CTRL[_k] not in _lst:
        _lst.append(_CTRL[_k])
_hist = []


def _obj(t):
    e = _space(t)
    s = _run(e, 42)
    _hist.append((s, e))
    print("  trial %%2d  %%8.1f  %%s" %% (len(_hist) - 1, s, e))
    return -s


print("\\n=== 1단계: 탐색 %%d trial (1회 적합) ===" %% OPT_TRIALS)
_st = optuna.create_study(direction="minimize")
_st.enqueue_trial({k: _CTRL[k] for k in
                   ("l2_leaf_reg", "random_strength", "bagging_temperature",
                    "border_count", "one_hot_max_size", "max_ctr_complexity")})
_st.optimize(_obj, n_trials=OPT_TRIALS)

_ctrl_screen = _hist[0][0]
_top = sorted(_hist[1:], key=lambda r: -r[0])[:3]
print("\\n대조군(trial 0) %%.1f | 탐색 1~3위 %%s"
      %% (_ctrl_screen, [round(s, 1) for s, _ in _top]))

# --- 2단계: 상위 3개 + 대조군을 seed 3개씩 재실행 ---
# 탐색 1위를 그대로 믿으면 안 된다. 1회 실행 노이즈가 +-13 이라 30 trial 의
# 최댓값은 선택만으로 +25 가 뜬다 (3장 '오프셋 측정 셀에 +-13 노이즈').
print("\\n=== 2단계: 대조군 + 상위 3개를 seed %%d개씩 재실행 ==="
      %% len(OPT_CONFIRM_SEEDS))
_cands = [("현행", _CTRL)] + [("탐색%%d" %% (i + 1), e) for i, (_, e) in enumerate(_top)]
_res = []
for _nm, _e in _cands:
    _v = [_run(_e, s) for s in OPT_CONFIRM_SEEDS]
    _res.append((float(np.mean(_v)), _nm, _e, _v))
    print("  %%-6s 평균 %%8.1f  (%%s)" %% (_nm, np.mean(_v), [round(x, 1) for x in _v]))

_base = _res[0][0]
_best = max(_res[1:], key=lambda r: r[0])
_gain = _best[0] - _base
print("\\n최선 %%s: %%+.1f (홀드아웃)" %% (_best[1], _gain))

OPT_ADOPT = _gain >= OPT_MARGIN
if OPT_ADOPT:
    BEST_PARAMS.update(_best[2])
    print("채택 -> BEST_PARAMS 갱신:", _best[2])
else:
    print("마진 %%+.0f 미만 -> **현행 유지**. 학습을 건너뛴다 (GPU 45분 절약)." %% OPT_MARGIN)
    print("  = 5분류 영역에서도 하이퍼파라미터 축이 닫혀 있다는 뜻이다.")

with open("model/best_params.json", "w") as f:
    json.dump({k: v for k, v in BEST_PARAMS.items() if k != "cat_features"}, f, indent=2)
print("최종 파라미터:", {k: v for k, v in BEST_PARAMS.items() if k != "cat_features"})
''' % (N_TRIALS, SUB_FRAC, MARGIN)

# 탐색 셀은 Y_CLS / AUX_HO 가 만들어진 **뒤**에 온다 (cell 16 의 끝).
ANC = 'print("  검산 OK: 클래스0 개수 == success 개수")'
i16 = find(ANC)
sub(i16, ANC, ANC + '\n' + SEARCH)
print('탐색 블록 삽입 (Y_CLS 뒤, cell %d)' % i16)

# 학습 셀은 IsotonicRegression / brier_score_loss / CatBoostClassifier 를 쓰는데
# 탐색 셀이 그보다 앞에 있으므로 import 를 앞당긴다.
IMP = 'from sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import brier_score_loss\nfrom catboost import CatBoostClassifier'
i_imp = find(IMP)
sub(i_imp, IMP, IMP + '\nfrom sklearn.isotonic import IsotonicRegression')
print('IsotonicRegression import 앞당김 (cell %d)' % i_imp)

# 채택 안 되면 학습을 건너뛴다 -- 현행 파라미터로 30모델을 다시 만들 이유가 없다.
TR = 'print(f"\\n=== 최종 학습: {N_SPLITS}-fold x {len(SEEDS)}-seed = {N_SPLITS * len(SEEDS)}개 모델 ===")'
i_tr = find(TR)
sub(i_tr, TR, 'if not OPT_ADOPT:\n    raise SystemExit("Optuna 가 현행을 못 이겼다 -- 학습 생략. 제출할 것 없음.")\n' + TR)
print('학습 셀에 조기 종료 삽입 (cell %d)' % i_tr)

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
print('%s 생성 -- %d셀 문법 OK (trial %d / 서브 %.0f%% / 마진 %+.0f)'
      % (OUT, len(code), N_TRIALS, 100 * SUB_FRAC, MARGIN))

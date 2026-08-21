# 목적함수를 2024 홀드아웃으로 바꾼 Optuna.
#
# 왜 다시 하는가:
#   현행 파라미터는 40 trial x 33% 서브샘플 x 3-fold 로 찾은 얇은 결과다. 그런데
#   그냥 더 두껍게 돌리면 오히려 해롭다 — 목적함수가 StratifiedKFold OOF 이기 때문이다.
#   OOF 를 더 잘 최적화하면 **용량을 키우는 쪽**으로 가는데, 그건 리더보드에서 손해였다:
#     depth 9   OOF 0.24395 -> 0.24374 (개선)   리더보드 -0.57
#     v8        OOF 0.24399 (악화)              리더보드 -22.35  <- OOF 가 맞았다
#
#   그래서 목적함수를 **~2023 학습 -> 2024 예측** 으로 바꾼다. '미래 시즌 전이'를
#   직접 최적화하는 것이고, 이건 한 번도 안 해봤다.
#
# 과거 실패와의 차이:
#   예전에 2023 홀드아웃으로 시도했다가 전 trial 이 음수였다 (2019~2022 학습분에
#   2022->2023 최대 낙폭 체제가 없어서 상수 예측보다 못 맞혔다). **2024 는 2023 이
#   학습에 들어가 있어서 790 근처가 정상적으로 나온다** — 로컬 스크리너로 확인했다.
#
# 탐색에 처음 넣는 축 둘:
#   border_count       (GPU 기본 128 / CPU 254)  — 수치 피처 분할 후보 해상도
#   max_ctr_complexity (기본 1)                  — 범주형 조합 CTR. 우리가 손으로 만든
#                                                  cond_pc/ph/phc 가 +27 이었다
#
# 사용: python tools/mk_optuna.py
import ast, json, os, sys

BASE = '.kernels/s1o/aimers_s1o.ipynb'
STOP = 'BEST_PARAMS = _found'      # 이 셀까지만 쓰고 학습/오프셋/zip 셀은 버린다
OUT = 'colab/aimers_optuna_colab.ipynb'

V4 = dict(learning_rate=0.022831883708228414, depth=8,
          l2_leaf_reg=8.552069332567962, bagging_temperature=0.05636104060100738,
          random_strength=0.7731135614050382, iterations=1000,
          border_count=128, max_ctr_complexity=1)

CELL = '''# ===== Optuna: 목적함수 = 2024 홀드아웃 (~2023 학습 -> 2024 예측) =====
import json, os, time
import numpy as np
import optuna
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

OUT_DIR = "/content/drive/MyDrive/aimers_ablation"
os.makedirs(OUT_DIR, exist_ok=True)
N_TRIALS = int(os.environ.get("OPT_TRIALS", "25"))
N_FOLD = 2            # 속도. 오프셋 셀의 3-fold 보다 낮지만 trial 간에는 동일 조건
HOLD = 2024

# 4-15: run_full_pipeline 이 행 순서를 바꾸므로 season 은 df_processed 에서 꺼낸다
_season = df_processed["season"].to_numpy()
_mh, _mv = _season <= HOLD - 1, _season == HOLD
Xh = X_full[_mh].reset_index(drop=True)
yh = y_full[_mh].reset_index(drop=True)
Xv = X_full[_mv].reset_index(drop=True)
yv = y_full[_mv].reset_index(drop=True).to_numpy()
TARGET = float(yv.mean())
U = TARGET * (1 - TARGET)
print(f"학습 {len(Xh):,}행(~{HOLD-1}) -> 검증 {len(Xv):,}행({HOLD}) | fold {N_FOLD}", flush=True)


def _recenter(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(q / (1 - q))
    off = 0.0
    for _ in range(300):
        cur = 1.0 / (1.0 + np.exp(-(lo + off)))
        e = cur.mean() - TARGET
        if abs(e) < 1e-9:
            break
        off -= e * 4.0
    return 1.0 / (1.0 + np.exp(-(lo + off)))


def _score(p):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yv) ** 2).mean() / U) * 100000


def run_params(params):
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)
    ps = []
    for ti, vi in skf.split(Xh, yh):
        m = CatBoostClassifier(**params)
        m.fit(Xh.iloc[ti], yh.iloc[ti], eval_set=(Xh.iloc[vi], yh.iloc[vi]), verbose=0)
        rv = m.predict_proba(Xh.iloc[vi])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(rv, yh.iloc[vi])
        ps.append(iso.predict(m.predict_proba(Xv)[:, 1]))
    return _score(_recenter(np.mean(ps, axis=0)))


TRIALS = []


def objective(trial):
    p = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.010, 0.10, log=True),
        depth=trial.suggest_int("depth", 5, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength=trial.suggest_float("random_strength", 0.3, 3.0),
        iterations=trial.suggest_int("iterations", 400, 1600, step=200),
        # 여기 둘은 이 프로젝트에서 한 번도 탐색한 적이 없다
        border_count=trial.suggest_categorical("border_count", [128, 254]),
        max_ctr_complexity=trial.suggest_categorical("max_ctr_complexity", [1, 2]),
    )
    full = dict(p, eval_metric="Logloss", task_type="GPU", random_seed=42,
                early_stopping_rounds=50, cat_features=cat_features)
    t0 = time.time()
    try:
        s = run_params(full)
    except Exception as e:                     # GPU 가 특정 조합을 거부해도 탐색은 계속
        print(f"  trial {len(TRIALS)+1} 실패: {type(e).__name__} {str(e)[:120]}", flush=True)
        s = -1e5
    TRIALS.append({"params": p, "score": round(float(s), 1), "sec": int(time.time() - t0)})
    # 세션이 끊겨도 여기까지는 남는다
    with open(f"{OUT_DIR}/optuna_trials.json", "w") as f:
        json.dump(TRIALS, f, indent=1, ensure_ascii=False)
    best = max(t["score"] for t in TRIALS)
    print(f"  trial {len(TRIALS):>2}: {s:>7,.0f}점 (최고 {best:,.0f})  {TRIALS[-1]['sec']}초  "
          f"d{p['depth']} lr{p['learning_rate']:.4f} it{p['iterations']} "
          f"bc{p['border_count']} ctr{p['max_ctr_complexity']}", flush=True)
    return -s


study = optuna.create_study(direction="minimize")
study.enqueue_trial(__V4__)          # 첫 trial 은 현행 파라미터 = 대조군
print(f"\\n=== Optuna {N_TRIALS} trial 시작 (첫 trial 은 현행 v4 파라미터) ===", flush=True)
study.optimize(objective, n_trials=N_TRIALS)

base = TRIALS[0]["score"]
best = max(TRIALS, key=lambda t: t["score"])
print(f"\\n{'='*60}")
print(f"현행 v4 파라미터 : {base:,.0f}점")
print(f"최고             : {best['score']:,.0f}점  ({best['score']-base:+,.0f})")
print(f"  {json.dumps(best['params'], ensure_ascii=False)}")
print(f"\\n+50 미만이면 노이즈. 100인 컷은 이 지표로 +57~95 가 필요하다.")
with open(f"{OUT_DIR}/optuna_best.json", "w") as f:
    json.dump({"baseline": base, "best": best, "n_trials": len(TRIALS)}, f,
              indent=1, ensure_ascii=False)
print(f"저장: {OUT_DIR}/optuna_best.json, optuna_trials.json")
'''

nb = json.load(open(BASE, encoding='utf-8'))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
srcs = [''.join(c['source']) for c in cells]
last = next(i for i, s in enumerate(srcs) if STOP in s)


def cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


body = [cell(s) for s in srcs[:last + 1]]
body.append(cell(CELL.replace('__V4__', json.dumps(V4))))

out = {"cells": body,
       "metadata": {"accelerator": "GPU",
                    "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                    "language_info": {"name": "python"}},
       "nbformat": 4, "nbformat_minor": 0}

for c in out['cells']:
    ast.parse(''.join(c['source']))

os.makedirs('colab', exist_ok=True)
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'{OUT} 생성 — 코드 셀 {len(body)}개 (전처리 {last+1} + Optuna 1), 문법 OK')

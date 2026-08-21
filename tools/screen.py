# 큰 스윙 전용 스크리너.
#
# 왜 홀드아웃을 다시 쓰는가: 이 지표는 한 자릿수 차이의 **부호도 못 맞힌다**
# (홀드아웃 +13 이 리더보드 -3.51 이었다). 하지만 우리가 찾는 건 +3 이 아니라
# **+122** 다 — 100인 컷 1,119.20 을 v5 의 오프셋(+207)으로 역산하면 홀드아웃 912,
# 우리는 790 이다. ±15 노이즈로 122 점 신호를 놓칠 수는 없다.
#
#   규칙: **+50 미만은 전부 노이즈로 보고 버린다.** 한 자릿수는 쳐다보지 않는다.
#
# 구조는 노트북의 오프셋 측정 셀과 같다 (~2023 학습 -> 2024 예측, 3-fold, isotonic,
# 재중심화). 후보끼리 같은 기준으로 비교되므로 절대값이 아니라 base 대비 차이만 읽는다.
#
# 사용:
#   python tools/screen.py                      # 전 후보
#   python tools/screen.py base rsm70           # 일부만
#   python tools/screen.py --iters=300 base     # 빠른 연기 테스트 (판정용 아님)
import json, os, sys, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

CACHE = os.environ.get('SCREEN_CACHE', 'cache')
RESULTS = 'cache/screen_results.json'
HOLDOUT = 2024
FOLDS = 3


def load():
    X = pd.read_pickle(f'{CACHE}/X.pkl')
    for c in X.columns:                       # 메모리: float64 -> float32
        if X[c].dtype == np.float64:
            X[c] = X[c].astype(np.float32)
    y = np.load(f'{CACHE}/y.npy')
    season = np.load(f'{CACHE}/season.npy')
    meta = json.load(open(f'{CACHE}/meta.json', encoding='utf-8'))
    return X, y, season, meta


def skill(p, yv):
    r = float(yv.mean())
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - yv) ** 2).mean() / (r * (1 - r))) * 100000


def recenter(p, target):
    """평균 예측이 target 이 되게 하는 로짓 상수 시프트 (오프셋 셀과 동일)."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(q / (1 - q))
    off = 0.0
    for _ in range(300):
        cur = 1.0 / (1.0 + np.exp(-(lo + off)))
        e = cur.mean() - target
        if abs(e) < 1e-9:
            break
        off -= e * 4.0
    return 1.0 / (1.0 + np.exp(-(lo + off)))


def cv_predict(Xh, yh, Xv, params, cat, baseline=None, base_v=None, folds=FOLDS):
    """~2023 을 folds 개로 나눠 학습하고 2024 예측을 평균한다."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    out = []
    for ti, vi in skf.split(Xh, yh):
        m = CatBoostClassifier(**params)
        if baseline is None:
            m.fit(Xh.iloc[ti], yh[ti], eval_set=(Xh.iloc[vi], yh[vi]), verbose=0)
            rv = m.predict_proba(Xh.iloc[vi])[:, 1]
            pv = m.predict_proba(Xv)[:, 1]
        else:
            m.fit(Pool(Xh.iloc[ti], yh[ti], cat_features=cat, baseline=baseline[ti]),
                  eval_set=Pool(Xh.iloc[vi], yh[vi], cat_features=cat, baseline=baseline[vi]),
                  verbose=0)
            rv = m.predict_proba(Pool(Xh.iloc[vi], cat_features=cat,
                                      baseline=baseline[vi]))[:, 1]
            pv = m.predict_proba(Pool(Xv, cat_features=cat, baseline=base_v))[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip').fit(rv, yh[vi])
        out.append(iso.predict(pv))
    return np.mean(out, axis=0)


# ---------------------------------------------------------------- 후보들
def cand_base(ctx, **kw):
    """v5 파라미터 그대로. 다른 후보는 전부 이것 대비로 읽는다."""
    return cv_predict(ctx['Xh'], ctx['yh'], ctx['Xv'], ctx['params'], ctx['cat'])


def _rsm(ctx, frac):
    """랜덤 부분공간: 모델마다 피처의 일부만 쓴다.

    트리 계열을 바꿔봤자 예측 상관이 0.92 아래로 안 내려갔다(블렌드 이득 +7 에서 포화).
    피처를 갈라주면 상관이 더 떨어질 수 있고, 그게 앙상블에서 유일하게 남은 여지다.
    """
    p = dict(ctx['params'])
    p['rsm'] = frac
    p['boosting_type'] = 'Plain'      # rsm 은 Ordered 와 같이 못 쓴다
    return cv_predict(ctx['Xh'], ctx['yh'], ctx['Xv'], p, ctx['cat'])


def cand_rsm70(ctx, **kw):
    return _rsm(ctx, 0.7)


def cand_rsm40(ctx, **kw):
    return _rsm(ctx, 0.4)


def _lrbase(ctx, interact):
    """선형 baseline + 트리 잔차. 트리는 학습 범위 밖을 외삽하지 못한다.

    interact=True 면 (시즌 x 피처) 항을 추가한다. 평범한 LR 은 season 이 선형항이라
    2024 행 전체에 같은 상수를 더할 뿐이고 그건 재중심화와 같다. 상호작용이 있어야
    '피처와 결과의 관계가 해마다 어떻게 변해왔는지' 를 외삽한다.
    """
    from sklearn.linear_model import LogisticRegression
    num = [c for c in ctx['Xh'].columns if ctx['Xh'][c].dtype.name != 'category']
    si = num.index('season')

    def expand(M):
        return np.hstack([M, M * M[:, si][:, None]]) if interact else M
    A = ctx['Xh'][num].astype('float64')
    med = A.median()
    A = A.fillna(med)
    mu, sd = A.mean(), A.std().replace(0.0, 1.0).fillna(1.0)
    Z = expand(((A - mu) / sd).to_numpy())
    lr = LogisticRegression(C=1.0, max_iter=120).fit(Z, ctx['yh'])
    b_h = Z @ lr.coef_[0] + lr.intercept_[0]
    Zv = expand(((ctx['Xv'][num].astype('float64').fillna(med) - mu) / sd).to_numpy())
    b_v = Zv @ lr.coef_[0] + lr.intercept_[0]
    print(f'    LR baseline(interact={interact}): 학습 sd={b_h.std():.3f} '
          f'/ 2024 sd={b_v.std():.3f} / 2024 범위 [{b_v.min():.2f}, {b_v.max():.2f}] '
          f'(학습 [{b_h.min():.2f}, {b_h.max():.2f}])', flush=True)
    return cv_predict(ctx['Xh'], ctx['yh'], ctx['Xv'], ctx['params'], ctx['cat'],
                      baseline=b_h, base_v=b_v)


def cand_lrbase(ctx, **kw):
    return _lrbase(ctx, False)


def cand_lrbase2(ctx, **kw):
    return _lrbase(ctx, True)


def cand_blend_recent(ctx, **kw):
    """전 시즌 모델 + 최근 시즌 모델의 블렌드.

    '최근만 학습'은 단독으로 -128 이었지만 **섞는** 것은 안 해봤다. 최근 모델이
    단독으로 나쁜 이유는 데이터량이고, 그 편향은 블렌드에서 상쇄될 수 있다.
    """
    p_all = cv_predict(ctx['Xh'], ctx['yh'], ctx['Xv'], ctx['params'], ctx['cat'])
    m = ctx['sh'] >= 2022
    p_rec = cv_predict(ctx['Xh'][m], ctx['yh'][m], ctx['Xv'], ctx['params'], ctx['cat'])
    for w in (0.0, 0.2, 0.3, 0.5):
        q = recenter((1 - w) * p_all + w * p_rec, ctx['target'])
        print(f'    최근 가중 {w:.1f}: {skill(q, ctx["yv"]):,.0f}', flush=True)
    return 0.7 * p_all + 0.3 * p_rec


def _p(ctx, **over):
    p = dict(ctx['params'])
    p.update(over)
    return cv_predict(ctx['Xh'], ctx['yh'], ctx['Xv'], p, ctx['cat'])


def cand_ctr2(ctx, **kw):
    """범주형 조합 CTR.

    기본값이 max_ctr_complexity=1 이라 CatBoost 는 범주형을 하나씩만 타깃 인코딩하고
    **조합은 만들지 않는다**. 그런데 우리가 손으로 만든 cond_pc(투수x카운트) /
    cond_ph(투수x좌우) / cond_phc(셋 다) 가 +27 을 벌어준 최상위 피처이고, 그건
    정확히 수작업 CTR 조합이다. 범주형 25개 전체에 같은 일을 자동으로 시키는 것은
    한 번도 안 해봤다.
    """
    return _p(ctx, max_ctr_complexity=2)


def cand_ctr3(ctx, **kw):
    return _p(ctx, max_ctr_complexity=3)


def cand_onehot(ctx, **kw):
    """one_hot_max_size 기본값이 2 라 범주 3개 이상은 전부 CTR 로 간다.
    base_state(8) / 팀(12) / inning(12) 을 원핫으로 주면 트리가 직접 쪼갠다."""
    return _p(ctx, one_hot_max_size=16)


CANDS = {'base': cand_base, 'ctr2': cand_ctr2, 'ctr3': cand_ctr3,
         'onehot': cand_onehot, 'rsm70': cand_rsm70, 'rsm40': cand_rsm40,
         'lrbase': cand_lrbase, 'lrbase2': cand_lrbase2,
         'blend_recent': cand_blend_recent}


def main():
    iters = 1000
    args = []
    for a in sys.argv[1:]:
        if a.startswith('--iters='):
            iters = int(a.split('=')[1])
        elif not a.startswith('--'):
            args.append(a)
    names = args or list(CANDS)
    for n in names:
        if n not in CANDS:
            sys.exit(f'모르는 후보: {n} (가능: {list(CANDS)})')

    X, y, season, meta = load()
    params = dict(meta['best_params'])
    params.update(iterations=iters, cat_features=meta['cat_features'],
                  task_type='CPU', thread_count=-1, verbose=0)
    # early_stopping_rounds 는 그대로 둔다 — 오프셋 셀과 조건을 맞춰야 base 가 알려진
    # 790 근처를 재현하는지 확인할 수 있다 (4-15: 하네스가 기지의 값을 내는지 먼저 볼 것).

    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    ctx = {'Xh': X[mh].reset_index(drop=True), 'yh': y[mh],
           'Xv': X[mv].reset_index(drop=True), 'yv': y[mv],
           'sh': season[mh], 'params': params, 'cat': meta['cat_features']}
    ctx['target'] = float(ctx['yv'].mean())
    print(f"학습 {mh.sum():,}행(~{HOLDOUT-1}) -> 검증 {mv.sum():,}행({HOLDOUT}) "
          f"| 반복 {iters} | fold {FOLDS}\n", flush=True)

    res = json.load(open(RESULTS, encoding='utf-8')) if os.path.exists(RESULTS) else {}
    for n in names:
        t = time.time()
        print(f'[{n}] 시작', flush=True)
        p = recenter(CANDS[n](ctx), ctx['target'])
        s = skill(p, ctx['yv'])
        res[f'{n}@{iters}'] = round(float(s), 1)
        json.dump(res, open(RESULTS, 'w'), indent=2, sort_keys=True)
        print(f'[{n}] {s:,.0f}점  ({(time.time()-t)/60:.0f}분)\n', flush=True)

    b = res.get(f'base@{iters}')
    print('=' * 52)
    print(f'{"후보":<16}{"점수":>10}{"base 대비":>12}   판정')
    for k in sorted(res):
        if not k.endswith(f'@{iters}'):
            continue
        d = res[k] - b if b else None
        v = ('—' if d is None or k.startswith('base')
             else ('★ 추격 가능' if d >= 50 else '노이즈' if abs(d) < 50 else '음수'))
        print(f'{k.split("@")[0]:<16}{res[k]:>10,.0f}{"" if d is None else f"{d:>+12,.0f}"}   {v}')
    print('\n+50 미만은 전부 노이즈다. 100인 컷은 홀드아웃 847~885 이므로 '
          '여기서 +57~95 가 필요하다 (베이스라인 549.51 실측으로 재계산).')


if __name__ == '__main__':
    main()

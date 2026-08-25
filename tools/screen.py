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
HOLDOUT = int(os.environ.get('SCREEN_HOLDOUT', '2024'))
# 2023 을 쓸 때는 F(퓨처스) 행을 채점에서 뺀다 — 2023 에 라벨 체제가 바뀌어
# 2019~22 로 학습한 모델이 F 를 .68 로 보는데 실제는 .473 이다 (산술로 -1,800점).
# R 만 보면 2022->2023 낙폭이 -0.0006 이라 정상적인 검증 시즌이 된다.
DROP_F = os.environ.get('SCREEN_DROP_F', '1') == '1' and HOLDOUT == 2023
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


_WS = {}


def _wseason_cols():
    """당해 시즌 성적을 복원한다 (EDA 2026-08-22).

    `asof_pitcher_success_rate` 는 **커리어 누적**이라 투수마다 의미가 다르다.
    이력 있는 투수에게는 2019(리그 .5647)부터 섞인 값이고 신규 투수에게는 순수한
    당해 시즌 값이다. 2024 에서 단독 예측 스킬이 141 vs 906 으로 갈린다.

    복원:
      커리어 성공수 = asof_pitcher_success_rate x asof_pitcher_n   (그 행이 갖고 있다)
      직전 시즌까지 = 투수별 룩업 (학습 데이터로 만든다)              (test 행 통계 아님 = 합법)
      차이 -> 당해 시즌 투구수 / 성공수

    검증 완료 (train 2024): 성공수 정수 오차 0.008, 음수 0%, 범위 밖 0.0000%.
    test 5행에서도 asof_pitcher_n >= 그 투수의 train 총투구가 전부 성립.

    ⚠️ 최근성 가중 4연패(-33/-118, -62/-128, -5~-27, -7.95)와 다른 점:
       그것들은 전부 정보를 **빼거나 대체**했다. 이건 커리어 값을 그대로 두고
       컬럼을 **더한다**. 그리고 학습·추론 양쪽에서 똑같이 신선하다
       (주최측 asof 가 test 에서도 시즌 내 누적이므로) — `asof_*` fresh 설계가
       -217 로 무너진 것과 정반대 조건이다.

    ⚠️ 트리가 스스로 만들 수 없다. '그 투수의 직전 시즌까지 누적' 은 어떤 컬럼에도
       없고 다른 행에서 끌어와야 한다. 이득이 났던 cond_*(+27)/트랙맨(+9)과 같은 부류.
    """
    if _WS:
        return _WS
    import glob
    rid = np.load(f'{CACHE}/row_id.npy', allow_pickle=True)
    _na = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
    c = ([f for f in ('data/train.csv',) if os.path.exists(f)]
         + glob.glob('/kaggle/input/**/train.csv', recursive=True))
    tr = pd.read_csv(c[0], usecols=['row_id', 'season', 'pitcher_id', 'control_success',
                                    'asof_pitcher_n', 'asof_pitcher_success_rate'],
                     keep_default_na=False, na_values=_na)
    tr = tr.set_index('row_id').reindex(pd.Index(rid)).reset_index()
    assert tr['pitcher_id'].notna().all(), 'row_id 매칭 실패'

    # 시즌별 누적 룩업: (투수, 시즌) -> 그 시즌 **이전까지**의 투구수/성공수
    g = tr.groupby(['pitcher_id', 'season'])['control_success'].agg(['size', 'sum'])
    g = g.sort_index()
    cum = g.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0)
    key = pd.MultiIndex.from_arrays([tr['pitcher_id'], tr['season']])
    pn = cum['size'].reindex(key).to_numpy()
    ps = cum['sum'].reindex(key).to_numpy()

    n = tr['asof_pitcher_n'].to_numpy(dtype='float64')
    s = (tr['asof_pitcher_success_rate'].fillna(0).to_numpy(dtype='float64') * n).round()
    wn = np.maximum(n - pn, 0.0)
    ws = np.clip(s - ps, 0.0, wn)
    lg = tr.groupby('season')['control_success'].mean()
    lgv = tr['season'].map(lg).to_numpy(dtype='float64')
    prior = float(tr['control_success'].mean())

    C = 100.0
    rate = (ws + prior * C) / (wn + C)
    _WS.update(w_n=wn, w_rate=rate, w_dev=rate - lgv,
               w_share=wn / np.maximum(n, 1.0))
    print(f'    당해시즌 복원: 투구수 중앙값 {np.median(wn):.0f} | '
          f'성공률 평균 {rate.mean():.4f} | 커리어대비 비중 {_WS["w_share"].mean():.2f}',
          flush=True)
    return _WS


def cand_wseason(ctx, **kw):
    """당해 시즌 성적 4개를 추가한다."""
    W = _wseason_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for k, v in W.items():
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_wseason1(ctx, **kw):
    """성공률 하나만 (최소판). 넷 중 무엇이 일하는지 가른다."""
    W = _wseason_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    Xh['w_dev'] = W['w_dev'][mh].astype(np.float32)
    Xv['w_dev'] = W['w_dev'][mv].astype(np.float32)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


_WS5 = {}
_RATES = ['success', 'middle', 'reverse', 'ball', 'strike']


def _wseason5_cols():
    """다섯 개 asof 비율 전부를 '당해 시즌' 으로 복원한다.

    `wseason` 은 success_rate 하나만 했다. 나머지 넷(middle/reverse/ball/strike)도
    **똑같이 커리어 누적**이라 같은 수준 오염을 겪는다.

    ⚠️ 넷은 투구별 라벨이 없어서 control_success 처럼 라벨로 누적을 계산할 수 없다.
       대신 **asof 컬럼 자체에서 읽는다**: 각 (투수, 시즌)의 **첫 행**은 그 시즌
       시작 시점의 상태이므로 `asof_pitcher_n` 과 비율의 곱이 곧 직전 시즌까지의
       누적 개수다. 라벨이 필요 없고 다섯 개에 똑같이 적용된다.

       배포 시에는 train 의 2024 마지막 행에서 같은 값을 읽는다 (한 투구 오차).
    """
    if _WS5:
        return _WS5
    import glob
    rid = np.load(f'{CACHE}/row_id.npy', allow_pickle=True)
    _na = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
    cols = ['row_id', 'season', 'pitcher_id', 'asof_pitcher_n'] + \
           [f'asof_pitcher_{k}_rate' for k in _RATES]
    c = ([f for f in ('data/train.csv',) if os.path.exists(f)]
         + glob.glob('/kaggle/input/**/train.csv', recursive=True))
    tr = pd.read_csv(c[0], usecols=cols, keep_default_na=False, na_values=_na)
    tr = tr.set_index('row_id').reindex(pd.Index(rid)).reset_index()

    n = tr['asof_pitcher_n'].to_numpy(dtype='float64')
    # (투수, 시즌) 첫 행 = 그 시즌 시작 시점의 커리어 상태
    first = tr.assign(_n=n).sort_values('_n').groupby(
        ['pitcher_id', 'season'], sort=False).head(1)
    key = pd.MultiIndex.from_arrays([tr['pitcher_id'], tr['season']])
    fi = first.set_index(['pitcher_id', 'season'])
    n0 = fi['_n'].reindex(key).to_numpy()
    wn = np.maximum(n - n0, 0.0)
    out = {'w5_n': wn, 'w5_share': wn / np.maximum(n, 1.0)}
    for k in _RATES:
        col = f'asof_pitcher_{k}_rate'
        x = (tr[col].fillna(0).to_numpy(dtype='float64') * n).round()
        x0 = (fi[col].fillna(0).reindex(key).to_numpy() * n0).round()
        wx = np.clip(x - x0, 0.0, wn)
        prior = float(np.nanmean(tr[col].to_numpy(dtype='float64')))
        rate = (wx + prior * 100.0) / (wn + 100.0)
        # 시즌 리그평균을 빼서 디트렌드 (cond_* 와 같은 처리)
        lgk = tr.assign(_v=tr[col]).groupby('season')['_v'].mean()
        out[f'w5_{k}'] = rate - tr['season'].map(lgk).to_numpy(dtype='float64')
    _WS5.update(out)
    print('    당해시즌 5종 복원: 투구수 중앙값 %.0f | ' % np.median(wn)
          + ' '.join(f'{k}={np.nanmean(out["w5_"+k]):+.4f}' for k in _RATES), flush=True)
    return _WS5


_WS5B = {}


def _wseason5b_cols():
    """`_wseason5_cols` 의 디트렌드 기준을 고친 판.

    ⚠️ 원래 판은 **커리어 컬럼(asof_pitcher_*_rate)의 시즌 평균**으로 뺐다.
       그런데 우리가 만드는 값은 **당해 시즌 비율**이고 둘의 평균 차이가
       시즌마다 흔들린다 (success: 2019 -0.000 / 2023 -0.029 / 2024 -0.022).
       그래서 디트렌드가 절반만 되고 시즌 의존 잔차가 남는다.
       여기서는 **복원된 값 자체의 시즌 평균**으로 뺀다 -> 모든 시즌이 0 에 중심.
    """
    if _WS5B:
        return _WS5B
    W = dict(_wseason5_cols())          # 원판을 재사용해 wn/원시비율을 얻는다
    season = np.load(f'{CACHE}/season.npy')
    # 원판은 이미 (rate - asof컬럼 시즌평균) 이므로, 그 결과의 시즌 평균을 다시 뺀다.
    # 두 번 빼면 결국 '복원값의 시즌평균' 으로 뺀 것과 같다.
    out = {'w5_n': W['w5_n'], 'w5_share': W['w5_share']}
    for k in _RATES:
        v = np.array(W[f'w5_{k}'], dtype='float64')
        adj = v.copy()
        for s in np.unique(season):
            m = season == s
            adj[m] = v[m] - np.nanmean(v[m])
        out[f'w5_{k}'] = adj
    _WS5B.update(out)
    print('    디트렌드 재기준: ' + ' '.join(
        f'{k}={np.nanmean(out["w5_"+k]):+.5f}' for k in _RATES), flush=True)
    return _WS5B


def cand_wseason5b(ctx, **kw):
    """고친 디트렌드로 다섯 비율 전부."""
    W = _wseason5b_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for k, v in W.items():
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_wseason5(ctx, **kw):
    """다섯 비율 전부 + 표본크기 = 7개 추가."""
    W = _wseason5_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for k, v in W.items():
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_wseason4(ctx, **kw):
    """success 를 뺀 나머지 넷만. wseason(성공률)과 겹치지 않는 증분을 가른다."""
    W = _wseason5_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for k, v in W.items():
        if k == 'w5_success':
            continue
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


_PMIX = {}


def _pmix_cols():
    """상황별 **구종 성향** (트랙맨 기반, EDA 2026-08-22).

    발견 경로: train x trackman 을 투구 1:1 로 정렬해보니 **실제 구종을 알면 +478** 이다
    (변화구가 제구하기 어렵다). 실제 구종은 추론 시 못 쓰지만 '그 상황에서 그 투수가
    무엇을 던지는가' 는 트랙맨(공식 학습 데이터)에서 만들 수 있고 합법이다.

    ⚠️ 처음 쟀을 때는 -12(=0) 였다. base 에 투수-시즌 **평균 물리량 8개**가 있어서
       레퍼토리가 이미 중복돼 있었고, 정렬된 23% 에서만 쟀기 때문이다. 불공정했다.
       공정한 base 로 다시 재니 **+76** 이다:
         투수 전체 믹스 +42 / 상황 편차(pc+pch) +55 / 둘 다 +76

    왜 `cond_pc`(투수x카운트 **성적**)보다 유리한가: 성적은 잡음투성이라 카운트 12개로
    쪼개면 표본이 죽는데, **구종 선택은 결정론적에 가까워** 적은 표본으로도 잘 추정된다.

    ⚠️ staleness: 트랙맨에 2025 가 없다. 다만 성능 지표는 1년 묵으면 77% 를 잃는 반면
       **레퍼토리는 해가 바뀌어도 안정적**이라 손실이 작다 (위 +76 은 트랙맨 <=2023 으로
       2024 를 맞힌 값이므로 이미 배포 조건이다).

    leak-free: 각 행은 **자기 시즌보다 과거**의 트랙맨만 쓴다 (cond_* 와 같은 패턴).
    """
    if _PMIX:
        return _PMIX
    import glob
    _na = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']

    def _find(name):
        c = ([f'data/{name}'] if os.path.exists(f'data/{name}') else []) + \
            glob.glob(f'/kaggle/input/**/{name}', recursive=True)
        if not c:
            raise SystemExit(f'{name} 을 못 찾음')
        return c[0]

    rid = np.load(f'{CACHE}/row_id.npy', allow_pickle=True)
    tr = pd.read_csv(_find('train.csv'),
                     usecols=['row_id', 'season', 'pitcher_id', 'batter_hand',
                              'balls_before', 'strikes_before'],
                     keep_default_na=False, na_values=_na)
    tr = tr.set_index('row_id').reindex(pd.Index(rid)).reset_index()
    tm = pd.read_csv(_find('trackman_history.csv'),
                     usecols=['pitcher_trackman_id', 'season', 'balls_before',
                              'strikes_before', 'batter_hand', 'pitch_type_group'],
                     keep_default_na=False, na_values=_na)
    mp = pd.read_csv(_find('pitcher_id_mapping_v2.csv'),
                     keep_default_na=False, na_values=_na)
    tm = tm.merge(mp[['pitcher_id', 'pitcher_trackman_id', 'season']].dropna(),
                  on=['pitcher_trackman_id', 'season'], how='inner')
    T = sorted(tm['pitch_type_group'].dropna().unique())
    tm['ct'] = tm['balls_before'].astype(str) + '-' + tm['strikes_before'].astype(str)
    tr['ct'] = tr['balls_before'].astype(str) + '-' + tr['strikes_before'].astype(str)
    tr['bh'] = tr['batter_hand'].map({1: 'Left', 2: 'Right'}).fillna(
        tr['batter_hand'].astype(str))
    C = 30.0

    n = len(tr)
    out = {f'pmix_{t}': np.full(n, np.nan) for t in T}
    out.update({f'pdev_c_{t}': np.full(n, np.nan) for t in T})
    out.update({f'pdev_ch_{t}': np.full(n, np.nan) for t in T})
    out['pmix_have'] = np.zeros(n)

    for s in sorted(tr['season'].unique()):
        past = tm[tm['season'] < s]
        if len(past) == 0:
            continue
        m = (tr['season'] == s).to_numpy()
        sub = tr[m]
        allmix = past.groupby(['pitcher_id', 'pitch_type_group']).size().unstack(
            fill_value=0).reindex(columns=T, fill_value=0)
        allmix = allmix.div(allmix.sum(1).clip(lower=1), axis=0)
        P = allmix.reindex(sub['pitcher_id']).to_numpy(dtype='float64')

        def shr(keys, dcols):
            g = past.groupby(keys + ['pitch_type_group']).size().unstack(
                fill_value=0).reindex(columns=T, fill_value=0)
            b = allmix.reindex(g.index.get_level_values('pitcher_id')).to_numpy('float64')
            nn = g.sum(1).to_numpy()[:, None]
            tab = pd.DataFrame((g.to_numpy() + b * C) / (nn + C), index=g.index, columns=T)
            return tab.reindex(pd.MultiIndex.from_arrays(
                [sub[c] for c in dcols])).to_numpy(dtype='float64')

        Pc = shr(['pitcher_id', 'ct'], ['pitcher_id', 'ct'])
        Pch = shr(['pitcher_id', 'ct', 'batter_hand'], ['pitcher_id', 'ct', 'bh'])
        for i, t in enumerate(T):
            out[f'pmix_{t}'][m] = P[:, i]
            out[f'pdev_c_{t}'][m] = Pc[:, i] - P[:, i]
            out[f'pdev_ch_{t}'][m] = Pch[:, i] - P[:, i]
        out['pmix_have'][m] = (~np.isnan(P[:, 0])).astype(float)

    _PMIX.update(out)
    print(f'    구종 성향 {len(out)}개 | 커버리지 {out["pmix_have"].mean():.1%} | '
          f'구종군 {T}', flush=True)
    return _PMIX


def cand_pmix(ctx, **kw):
    """구종 성향 (전체 믹스 + 상황 편차)."""
    W = _pmix_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for k, v in W.items():
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_wspmix(ctx, **kw):
    """★ 당해 시즌 복원 + 구종 성향 — 오늘 살아남은 둘을 합친다."""
    A, B = _wseason5_cols(), _pmix_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (A, B):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


_XPROB = {}


def _xprob_cols():
    """★ step16 이 계산해놓고 **버리는** 세 확률을 그대로 내보낸다.

    step16_calc_expected_difficulty 안에는 이미
        exp_fb_prob / exp_br_prob / exp_off_prob = P(구종 | 투수, count_advantage)
    가 있는데, return 문이 이것들을 `expected_control_difficulty`
    (= 성향 x 릴리스포인트 std) 라는 **스칼라 하나**로 뭉갠 뒤 버린다.
    모델이 보는 것은 그 곱 하나뿐이다.

    여기서는 step16 과 **글자 그대로 같은 계산**(월 단위 expanding, 현재 월 제외)에
    merge_asof 도 파이프라인과 같은 방식으로 붙인다. 즉 '그냥 같이 내보내면 얼마인가'.

    `pmix` 와의 차이 — 이쪽이 싸고 저쪽이 촘촘하다:
      xprob: 카운트 4단계(count_advantage), 손 축 없음, 월 단위(더 신선)
      pmix : 카운트 12칸, x 타자 손, shrink 있음, 시즌 단위
    """
    global _XPROB
    if _XPROB:
        return _XPROB
    import glob
    _na = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']

    def _find(name):
        c = ([f'data/{name}'] if os.path.exists(f'data/{name}') else []) +             glob.glob(f'/kaggle/input/**/{name}', recursive=True)
        if not c:
            raise SystemExit(f'{name} 을 못 찾음')
        return c[0]

    def _ca(b, s):                       # step4 / step15 와 같은 정의
        pa = ((b == 0) & (s == 1)) | ((b == 0) & (s == 2)) | ((b == 1) & (s == 2))
        ba = (((b == 1) & (s == 0)) | ((b == 2) & (s == 0)) | ((b == 3) & (s == 0))
              | ((b == 2) & (s == 1)) | ((b == 3) & (s == 1)))
        nu = ((b == 1) & (s == 1)) | ((b == 2) & (s == 2))
        return np.select([pa, ba, nu], ['Pitcher', 'Batter', 'Neutral'], default='None')

    G = ['fastball', 'breaking', 'offspeed']
    EX = [f'xp_{c}' for c in G]

    rid = np.load(f'{CACHE}/row_id.npy', allow_pickle=True)
    tr = pd.read_csv(_find('train.csv'),
                     usecols=['row_id', 'season', 'game_month', 'pitcher_id',
                              'balls_before', 'strikes_before'],
                     keep_default_na=False, na_values=_na)
    tr = tr.set_index('row_id').reindex(pd.Index(rid)).reset_index()
    assert tr['pitcher_id'].notna().all(), 'row_id 매칭 실패'
    tr['pitcher_id'] = tr['pitcher_id'].astype('int64')
    tr['count_advantage'] = _ca(tr['balls_before'], tr['strikes_before'])
    tr['time_idx'] = tr['season'] * 100 + tr['game_month']
    tr['__orig'] = np.arange(len(tr))

    tm = pd.read_csv(_find('trackman_history.csv'),
                     usecols=['season', 'game_month', 'balls_before', 'strikes_before',
                              'pitcher_trackman_id', 'pitch_type_group'],
                     keep_default_na=False, na_values=_na)
    mp = pd.read_csv(_find('pitcher_id_mapping_v2.csv'),
                     keep_default_na=False, na_values=_na)
    tm = tm.merge(mp[['season', 'pitcher_trackman_id', 'pitcher_id']].dropna()
                  .drop_duplicates(), on=['season', 'pitcher_trackman_id'], how='inner')
    tm['pitcher_id'] = tm['pitcher_id'].astype('int64')
    tm['count_advantage'] = _ca(tm['balls_before'], tm['strikes_before'])
    tm['pitch_group'] = tm['pitch_type_group'].astype(str).str.lower()
    tm = tm[tm['pitch_group'].isin(G)]

    # ---- step16 과 동일: (투수 x count_advantage) 안 월 단위 누적, 현재 월 제외
    sit = tm.groupby(['season', 'game_month', 'pitcher_id', 'count_advantage',
                      'pitch_group']).size().unstack(fill_value=0).reset_index()
    for c in G:
        if c not in sit.columns:
            sit[c] = 0
    sit = sit.sort_values(by=['pitcher_id', 'count_advantage', 'season', 'game_month'])
    g = sit.groupby(['pitcher_id', 'count_advantage'])
    past = {c: (g[c].cumsum() - sit[c]).to_numpy() for c in G}
    tot = past[G[0]] + past[G[1]] + past[G[2]]
    for c in G:
        sit[f'xp_{c}'] = np.where(tot > 0, past[c] / np.maximum(tot, 1), 0.0)
    sit['time_idx'] = sit['season'] * 100 + sit['game_month']

    m = pd.merge_asof(
        tr.sort_values('time_idx'),
        sit[['time_idx', 'pitcher_id', 'count_advantage'] + EX].sort_values('time_idx'),
        on='time_idx', by=['pitcher_id', 'count_advantage'], direction='backward')
    m = m.sort_values('__orig')
    out = {k: m[k].to_numpy(dtype='float64') for k in EX}
    _XPROB.update(out)
    cov = np.isfinite(out[EX[0]]).mean()
    nz = (np.nan_to_num(out[EX[0]]) + np.nan_to_num(out[EX[1]])
          + np.nan_to_num(out[EX[2]]) > 0).mean()
    print(f'    step16 성향 {len(EX)}개 | 셀 {len(sit):,} | '
          f'merge 성공 {cov:.1%} | 과거표본 있음 {nz:.1%}', flush=True)
    return _XPROB


def _add(ctx, *tables):
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in tables:
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_xprob(ctx, **kw):
    """★ step16 이 버리는 세 확률을 '그냥 같이 내보내면' 얼마인가."""
    return _add(ctx, _xprob_cols())


def cand_wsxprob(ctx, **kw):
    """당해 시즌 복원 + step16 성향 — 배포 비용이 가장 싼 조합."""
    return _add(ctx, _wseason5_cols(), _xprob_cols())


def _read_tr(cols):
    """row_id 순서를 캐시와 맞춰 train.csv 의 일부 컬럼을 읽는다."""
    import glob
    _na = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
    rid = np.load(f'{CACHE}/row_id.npy', allow_pickle=True)
    c = ([f for f in ('data/train.csv',) if os.path.exists(f)]
         + glob.glob('/kaggle/input/**/train.csv', recursive=True))
    tr = pd.read_csv(c[0], usecols=['row_id'] + list(cols),
                     keep_default_na=False, na_values=_na)
    tr = tr.set_index('row_id').reindex(pd.Index(rid)).reset_index()
    assert tr[cols[0]].notna().all(), 'row_id 매칭 실패'
    return tr


def _count_adv(b, s):
    """step4 / step15 와 같은 정의."""
    pa = ((b == 0) & (s == 1)) | ((b == 0) & (s == 2)) | ((b == 1) & (s == 2))
    ba = (((b == 1) & (s == 0)) | ((b == 2) & (s == 0)) | ((b == 3) & (s == 0))
          | ((b == 2) & (s == 1)) | ((b == 3) & (s == 1)))
    nu = ((b == 1) & (s == 1)) | ((b == 2) & (s == 2))
    return np.select([pa, ba, nu], ['Pitcher', 'Batter', 'Neutral'], default='None')


_WS5GT = {}


def _ws5gt_cols():
    """wseason5 와 같되 디트렌드를 **(시즌 x game_type)** 으로 한다.

    근거: game_type=F 는 퓨처스(2군)이고 2023 에 라벨 체제가 바뀌었다.
      F 성공률  2019 .689  2020 .588  2021 .704  2022 .709 | 2023 .473  2024 .459
      R 성공률       .550       .527       .513       .504 |      .503       .490
    2022->2023 낙폭이 전체로는 -0.0290 인데 **R 만 보면 -0.0006** 이다.
    즉 우리가 '리그 드리프트' 로 알고 있던 최대 낙폭이 거의 전부 F 체제 변경이었다.

    그런데 디트렌드는 '그 시즌 리그평균' 하나만 빼므로, 2019~22 의 F 행은
    **+0.12 ~ +0.18 의 오차**를 받는다. 그 오염이 방금 +61.72 를 벌어준
    피처 안에 들어 있다.
    """
    if _WS5GT:
        return _WS5GT
    W = dict(_wseason5_cols())            # 복원값은 그대로 쓰고 디트렌드만 갈아끼운다
    tr = _read_tr(['season', 'game_type'] + [f'asof_pitcher_{k}_rate' for k in _RATES])
    for k in _RATES:
        col = f'asof_pitcher_{k}_rate'
        # 기존: season 평균을 뺐다 -> 되돌린 뒤 (season x game_type) 평균을 뺀다
        old = tr.groupby('season')[col].transform('mean').to_numpy('float64')
        new = tr.groupby(['season', 'game_type'])[col].transform('mean').to_numpy('float64')
        W[f'w5_{k}'] = W[f'w5_{k}'] + old - new
    _WS5GT.update(W)
    d = np.abs(_WS5GT['w5_success'] - _WS5_RAW_SUCCESS) if _WS5_RAW_SUCCESS is not None else None
    print('    디트렌드 (시즌 x game_type) 로 교체'
          + ('' if d is None else f' | success 이동 평균 {d.mean():.4f} 최대 {d.max():.4f}'),
          flush=True)
    return _WS5GT


_WS5_RAW_SUCCESS = None
_CONDGT = {}
_CONDC = {'cond_p': 200.0, 'cond_pc': 100.0, 'cond_ph': 100.0, 'cond_phc': 50.0}


def _condgt_cols():
    """cond_* 네 개를 **(시즌 x game_type)** 디트렌드로 다시 만든다.

    현행과 같은 구조: 시즌 편차를 sum/(count+C) 로 0 에 shrink, 각 행은
    **자기 시즌보다 과거** 시즌만으로 인코딩(leak-free). 바뀌는 것은 디트렌드 기준뿐.
    2024 홀드아웃 투수 편차와의 상관: 현행 +0.260 -> 제안 +0.477 (가중 +0.420 -> +0.624).
    """
    if _CONDGT:
        return _CONDGT
    tr = _read_tr(['season', 'game_type', 'pitcher_id', 'batter_hand',
                   'balls_before', 'strikes_before', 'control_success'])
    tr['ca'] = _count_adv(tr['balls_before'], tr['strikes_before'])
    lg = tr.groupby(['season', 'game_type'])['control_success'].transform('mean')
    tr['dev'] = tr['control_success'] - lg

    SPEC = {'cond_p': ['pitcher_id'],
            'cond_pc': ['pitcher_id', 'ca'],
            'cond_ph': ['pitcher_id', 'batter_hand'],
            'cond_phc': ['pitcher_id', 'batter_hand', 'ca']}
    for name, keys in SPEC.items():
        g = tr.groupby(keys + ['season'])['dev'].agg(['sum', 'size']).sort_index()
        # 자기 시즌 **이전까지**의 누적 (leak-free)
        cum = g.groupby(level=list(range(len(keys)))).cumsum()
        cum = cum.groupby(level=list(range(len(keys)))).shift(1)
        C = _CONDC[name]
        val = cum['sum'] / (cum['size'] + C)
        idx = pd.MultiIndex.from_arrays([tr[k] for k in keys] + [tr['season']])
        _CONDGT[name] = val.reindex(idx).to_numpy(dtype='float64')
    miss = np.isnan(_CONDGT['cond_p']).mean()
    print(f'    cond_* 4개 재생성 (시즌 x game_type 디트렌드) | cond_p 결측 {miss:.1%}',
          flush=True)
    return _CONDGT


def cand_ws5gt(ctx, **kw):
    """★ wseason5 의 디트렌드를 (시즌 x game_type) 으로. 비교 기준은 wseason5 885.6."""
    return _add(ctx, _ws5gt_cols())


def cand_condgt(ctx, **kw):
    """★ wseason5(현행 디트렌드) + cond_* 를 (시즌 x game_type) 으로 재생성."""
    W = _wseason5_cols()
    G = _condgt_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for k, v in W.items():                      # 새 컬럼 추가
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
    for k, v in G.items():                      # 기존 cond_* 를 **덮어쓴다**
        assert k in Xh.columns, f'{k} 가 캐시에 없다'
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]} (cond_* 4개 덮어씀)', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_bothgt(ctx, **kw):
    """★ 둘 다 — wseason5 도 cond_* 도 (시즌 x game_type) 디트렌드."""
    W = _ws5gt_cols()
    G = _condgt_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W_ in (W, G):
        for k, v in W_.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


_WSBAT = {}
_BRATES = ['success', 'middle']


def _wsbat_cols():
    """타자측 두 비율을 '당해 시즌' 으로 복원한다 — `wseason5` 와 같은 수법.

    `asof_batter_success_rate` / `_middle_rate` 도 **커리어 누적**이고 분모는
    `asof_batter_n` 이다. 투수측과 구조가 완전히 같으므로 같은 복원이 성립한다.

    기대는 작다: eda10 의 선형 대리모형에서 투수 5개가 +83(실측 +72.2)인데
    타자 2개는 **+10** 이었다. 타자 축의 진짜 신호가 0.138% 뿐이기 때문이다
    (투수 0.828%). 다만 코드가 이미 있고 '새 정보' 부류(전달률 0.85)라 값이 싸다.
    """
    if _WSBAT:
        return _WSBAT
    cols = ['season', 'batter_id', 'asof_batter_n'] +            [f'asof_batter_{k}_rate' for k in _BRATES]
    tr = _read_tr(cols)
    n = tr['asof_batter_n'].to_numpy(dtype='float64')
    first = tr.assign(_n=n).sort_values('_n').groupby(
        ['batter_id', 'season'], sort=False).head(1)
    key = pd.MultiIndex.from_arrays([tr['batter_id'], tr['season']])
    fi = first.set_index(['batter_id', 'season'])
    n0 = fi['_n'].reindex(key).to_numpy()
    wn = np.maximum(n - n0, 0.0)
    out = {'wb_n': wn, 'wb_share': wn / np.maximum(n, 1.0)}
    for k in _BRATES:
        col = f'asof_batter_{k}_rate'
        x = (tr[col].fillna(0).to_numpy(dtype='float64') * n).round()
        x0 = (fi[col].fillna(0).reindex(key).to_numpy() * n0).round()
        wx = np.clip(x - x0, 0.0, wn)
        prior = float(np.nanmean(tr[col].to_numpy(dtype='float64')))
        rate = (wx + prior * 100.0) / (wn + 100.0)
        lgk = tr.assign(_v=tr[col]).groupby('season')['_v'].mean()
        out[f'wb_{k}'] = rate - tr['season'].map(lgk).to_numpy(dtype='float64')
    _WSBAT.update(out)
    print('    타자 당해시즌 복원: 타석수 중앙값 %.0f | ' % np.median(wn)
          + ' '.join(f'{k}={np.nanmean(out["wb_"+k]):+.4f}' for k in _BRATES), flush=True)
    return _WSBAT


def cand_wsbat(ctx, **kw):
    """wseason5 + 타자측 당해 시즌 복원. 비교 기준은 wseason5."""
    return _add(ctx, _wseason5_cols(), _wsbat_cols())


_TSK = {}


def _tskill_cols():
    """★ 구종별 제구 실력 x 상황별 구종 성향 — 1:1 정렬을 처음으로 실제로 쓴다.

    eda14/eda26: **그 투구의 구종을 알면 +428~478.** 추론 시점엔 못 쓴다.
    `pmix`(구종 **성향**만) 는 0 이었다 — 성향은 투수 수준 상수라 이미 base 에 있다.

    **빠져 있던 조각은 성향이 아니라 '구종별 실력' 이다.**
    슬라이더는 흔들리는데 직구는 정확한 투수가 있다. 그 정보는 어디에도 없다 —
    투구별 구종 라벨이 필요한데, 그것을 **1:1 정렬(cache/raw/aligned.parquet)** 이 준다.

        tskill[투수, 구종] = 그 투수의 그 구종 제구 편차   (과거 시즌만, 0 으로 shrink)
        mix[구종 | 투수, 카운트, 손]                      (트랙맨, 과거 시즌만)
        exp = Σ mix x tskill                              <- 상황별 기대 제구
        adj = exp − (투수 전체 믹스로 잰 exp)              <- ★ 상황 조정분

    `adj` 가 핵심이다. 카운트가 믹스를 바꾸고 구종마다 실력이 다르므로
    **트리가 만들 수 없는 상호작용**이다 (구종별 결과 데이터를 아예 못 본다).
    """
    if _TSK:
        return _TSK
    import glob
    _na = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']

    def _find(name, sub='data/'):
        c = ([f'{sub}{name}'] if os.path.exists(f'{sub}{name}') else []) +             glob.glob(f'/kaggle/input/**/{name}', recursive=True)
        if not c:
            raise SystemExit(f'{name} 을 못 찾음')
        return c[0]

    C_SK, C_MIX = 60.0, 30.0
    d = _read_tr(['season', 'game_type', 'pitcher_id', 'batter_hand',
                  'balls_before', 'strikes_before'])
    d['ct'] = _count_adv(d['balls_before'], d['strikes_before'])
    d['bh'] = d['batter_hand'].map({1: 'Left', 2: 'Right'}).fillna(
        d['batter_hand'].astype(str))

    # ---- (1) 구종별 실력: 정렬된 투구에서만 얻을 수 있다
    # aligned.parquet 은 94MB 로컬 파일이라 캐글 커널엔 없다. 미리 뽑아둔
    # 룩업(target_season x pitcher x 구종 -> skill, 382KB)을 먼저 찾는다.
    T = ['fastball', 'breaking', 'offspeed']
    TAB, a = None, None
    for c in ('cache/raw/tskill_table.csv', 'tskill_table.csv',
              'model/tskill_table.csv'):
        if os.path.exists(c):
            TAB = pd.read_csv(c)
            print(f'    구종별 실력 룩업 {c} ({len(TAB):,}행)', flush=True)
            break
    if TAB is None:
        ap = _find('aligned.parquet', 'cache/raw/')
        a = pd.read_parquet(ap, columns=['season', 'game_type', 'pitcher_id',
                                         'pitch_type_group', 'control_success'])
        a['pg'] = a['pitch_type_group'].astype(str).str.lower()
        a = a[a['pg'].isin(T)].copy()
        a['dev'] = a['control_success'] - a.groupby(
            ['season', 'game_type'])['control_success'].transform('mean')

    # ---- (2) 상황별 구종 믹스: 트랙맨 전체 (정렬 안 된 것도 쓴다)
    tm = pd.read_csv(_find('trackman_history.csv'),
                     usecols=['season', 'balls_before', 'strikes_before',
                              'batter_hand', 'pitcher_trackman_id', 'pitch_type_group'],
                     keep_default_na=False, na_values=_na)
    mp = pd.read_csv(_find('pitcher_id_mapping_v2.csv'),
                     keep_default_na=False, na_values=_na)
    tm = tm.merge(mp[['season', 'pitcher_trackman_id', 'pitcher_id']].dropna()
                  .drop_duplicates(), on=['season', 'pitcher_trackman_id'], how='inner')
    tm['ct'] = _count_adv(tm['balls_before'], tm['strikes_before'])
    tm['pg'] = tm['pitch_type_group'].astype(str).str.lower()
    tm = tm[tm['pg'].isin(T)]

    n = len(d)
    exp = np.full(n, np.nan)
    exp0 = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    have = np.zeros(n)
    for s in sorted(d['season'].unique()):
        m = (d['season'] == s).to_numpy()
        sub = d[m]
        pt = tm[tm['season'] < s]
        if not len(pt):
            continue
        if TAB is not None:                     # 미리 뽑아둔 룩업 (누수 없음: target_season 기준)
            q = TAB[TAB['target_season'] == s]
            if not len(q):
                continue
            sk = q.pivot(index='pitcher_id', columns='pg',
                         values='skill').reindex(columns=T)
        else:
            pa = a[a['season'] < s]
            if not len(pa):
                continue
            g = pa.groupby(['pitcher_id', 'pg'])['dev'].agg(['sum', 'size'])
            sk = (g['sum'] / (g['size'] + C_SK)).unstack().reindex(columns=T)
        # 믹스: 투수 전체 / 투수 x 카운트 x 손
        def _mix(keys, dcols):
            gg = pt.groupby(keys + ['pg']).size().unstack(fill_value=0).reindex(
                columns=T, fill_value=0)
            allm = pt.groupby(['pitcher_id', 'pg']).size().unstack(
                fill_value=0).reindex(columns=T, fill_value=0)
            allm = allm.div(allm.sum(1).clip(lower=1), axis=0)
            base = (allm if keys == ['pitcher_id']
                    else allm.reindex(gg.index.get_level_values('pitcher_id')).to_numpy())
            nn = gg.sum(1).to_numpy()[:, None]
            tab = pd.DataFrame((gg.to_numpy() + np.asarray(base) * C_MIX) / (nn + C_MIX),
                               index=gg.index, columns=T)
            idx = (pd.Index(sub[dcols[0]]) if len(dcols) == 1
                   else pd.MultiIndex.from_arrays([sub[c] for c in dcols]))
            return tab.reindex(idx).to_numpy('float64')
        M1 = _mix(['pitcher_id'], ['pitcher_id'])
        M3 = _mix(['pitcher_id', 'ct', 'batter_hand'], ['pitcher_id', 'ct', 'bh'])
        SK = sk.reindex(pd.Index(sub['pitcher_id'])).to_numpy('float64')
        ok = np.isfinite(SK).all(1) & np.isfinite(M1).all(1) & np.isfinite(M3).all(1)
        e3, e1 = (M3 * SK).sum(1), (M1 * SK).sum(1)
        exp[m] = np.where(ok, e3, np.nan)
        exp0[m] = np.where(ok, e1, np.nan)
        spread[m] = np.where(ok, np.nanstd(SK, axis=1), np.nan)
        have[m] = ok.astype(float)
    _TSK.update({'tsk_exp': exp, 'tsk_adj': exp - exp0,
                 'tsk_spread': spread, 'tsk_have': have})
    print(f'    구종별 실력 x 상황믹스 | 커버리지 {have.mean():.1%} | '
          f'adj std {np.nanstd(exp - exp0):.5f} | spread 중앙 {np.nanmedian(spread):.4f}',
          flush=True)
    return _TSK


def cand_tskill(ctx, **kw):
    """★ wseason5 + 구종별 실력 x 상황 믹스. 비교 기준은 wseason5 885.6."""
    return _add(ctx, _wseason5_cols(), _tskill_cols())


_WS5C = {}


def _ws5_C(C):
    """`wseason5` 를 임의의 shrink 상수 C 로 다시 만든다.

    현행 C=100 은 근거 없이 정한 값이고 **한 번도 안 돌려봤다.**
    C 는 '당해 시즌 성적을 몇 구부터 믿을 것인가' 를 정한다 —
        rate = (당해 성공수 + 리그평균 x C) / (당해 투구수 + C)
    당해 투구수 중앙값이 526 이므로 C=100 은 대부분의 행에서 약하게 당기지만,
    시즌 초 구간(7.4% 가 0~60구)에서는 거의 리그평균으로 눌러버린다.

    group_oof 가 '당해 시즌 폼' 축 전체를 301점으로 쟀다. 그 축의 **추정기 정확도**가
    곧바로 점수이므로 C 는 유일하면서 값싼 손잡이다.
    모델 규제가 아니라 새 정보의 추정 방식이라 홀드아웃의 소표본 편향을 덜 받는다.
    """
    if C in _WS5C:
        return _WS5C[C]
    base = _wseason5_cols()                    # 복원값(wx, wn)은 C 와 무관하다
    cols = ['season', 'pitcher_id', 'asof_pitcher_n'] +            [f'asof_pitcher_{k}_rate' for k in _RATES]
    tr = _read_tr(cols)
    n = tr['asof_pitcher_n'].to_numpy(dtype='float64')
    first = tr.assign(_n=n).sort_values('_n').groupby(
        ['pitcher_id', 'season'], sort=False).head(1)
    key = pd.MultiIndex.from_arrays([tr['pitcher_id'], tr['season']])
    fi = first.set_index(['pitcher_id', 'season'])
    n0 = fi['_n'].reindex(key).to_numpy()
    wn = np.maximum(n - n0, 0.0)
    out = {'w5_n': wn, 'w5_share': wn / np.maximum(n, 1.0)}
    for k in _RATES:
        col = f'asof_pitcher_{k}_rate'
        x = (tr[col].fillna(0).to_numpy(dtype='float64') * n).round()
        x0 = (fi[col].fillna(0).reindex(key).to_numpy() * n0).round()
        wx = np.clip(x - x0, 0.0, wn)
        prior = float(np.nanmean(tr[col].to_numpy(dtype='float64')))
        rate = (wx + prior * C) / (wn + C)
        lgk = tr.assign(_v=tr[col]).groupby('season')['_v'].mean()
        out[f'w5_{k}'] = rate - tr['season'].map(lgk).to_numpy(dtype='float64')
    d = np.abs(out['w5_success'] - base['w5_success'])
    print(f'    C={C:g} | success 이동 평균 {d.mean():.4f} 최대 {d.max():.4f} '
          f'| std {np.nanstd(out["w5_success"]):.4f}', flush=True)
    _WS5C[C] = out
    return out


def cand_ws5c30(ctx, **kw):
    """wseason5, shrink C=30 (당해 시즌을 더 빨리 믿는다)."""
    return _add(ctx, _ws5_C(30.0))


def cand_ws5c300(ctx, **kw):
    """wseason5, shrink C=300 (더 보수적으로 당긴다)."""
    return _add(ctx, _ws5_C(300.0))


def cand_ws5c10(ctx, **kw):
    """wseason5, shrink C=10 (거의 날것)."""
    return _add(ctx, _ws5_C(10.0))


def cand_ws5nocond(ctx, **kw):
    """★ wseason5 + `cond_*` 네 개 **제거**. claude.md 가 미검증으로 남긴 가설이다.

    근거 둘이 같은 곳을 가리킨다:
      1) 당해 시즌 복원이 들어간 뒤로 `cond_*` 는 거의 잉여일 수 있다.
         eda11 이 네 번 독립적으로 '투수의 과거 이력은 당해 시즌 예측에 거의 쓸모없다'
         고 쟀다 (eda7 -2 / eda9 -0 / group_oof B~C / eda11 -20).
      2) `cond_*` 는 시즌 리그평균으로만 디트렌드하는데 2019~22 F 행은
         +0.12~0.18 오차를 받는다. 디트렌드를 '고치는' 시도는 실패했지만
         (ws5gt -17.8 / condgt -8.2 / bothgt -1.8) **빼는 것은 다른 개입**이다.
    """
    W = _wseason5_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    drop = [c for c in ctx['Xh'].columns if c.startswith('cond_')]
    Xh = ctx['Xh'].drop(columns=drop).copy()
    Xv = ctx['Xv'].drop(columns=drop).copy()
    for k, v in W.items():
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
    print(f'    cond_* {len(drop)}개 제거 {drop} | 피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}',
          flush=True)
    cat = [c for c in ctx['cat'] if c in Xh.columns]
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], cat)


def _season_map(ctx, fn, tag):
    """`season` 을 거칠게 만든 판. wseason5 위에서 잰다.

    배포 모델의 season 분기 경계는 [2019.5 ... 2023.5] 로 **최대가 2023.5** 다
    (eda31, 배포 zip 에서 직접 읽음). 따라서 2025 는 2024 와 **비트 단위로 같은 잎**
    으로 간다 — 모델은 2025 에 '2024 전용으로 배운 것' 을 그대로 적용한다.
    그게 추세면 좋고 2024 만의 우연이면 해롭다.

    트리에서 그걸 건드리는 유일한 방법은 **마지막 시즌을 특별 취급하지 못하게
    경계를 없애는 것**이다. season 을 지우는 것(-424)과 현행 사이의 중간 지점.

    ⚠️ 기대값은 낮게 잡는다. -424 는 season 이 많은 것을 담고 있다는 뜻이라
    거칠게 만들수록 잃을 것이 크다.
    """
    W = _wseason5_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for k, v in W.items():
        Xh[k] = v[mh].astype(np.float32)
        Xv[k] = v[mv].astype(np.float32)
    sh, sv = fn(season[mh]), fn(season[mv])
    Xh['season'] = sh
    Xv['season'] = sv
    print(f'    {tag} | 학습 season 값 {sorted(set(sh.tolist()))} '
          f'-> 검증 {sorted(set(sv.tolist()))}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_seasonlast2(ctx, **kw):
    """마지막 두 학습 시즌을 병합 — '최근 체제' 를 두 배 데이터로 배우게 한다."""
    L = HOLDOUT - 1
    return _season_map(ctx, lambda s: np.where(s >= L - 1, L - 1, s),
                       f'{L-1}·{L} 병합')


def cand_seasonpair(ctx, **kw):
    """두 시즌씩 짝지어 3구간으로 (2019-20 / 2021-22 / 2023-24)."""
    return _season_map(ctx, lambda s: ((s - 2019) // 2) * 2 + 2019, '2시즌 묶음')


_CONDSIT = {}


def _condsit_cols():
    """★ 투수 x 상황 조건부 통계 — 주자·이닝·점수차. 주최측이 힌트를 준 축이다.

    문제 소개 자료의 '코치의 네 가지 질문' = 주자 상황 / 볼카운트 / 경기 후반 / 최근 흐름.
    우리 `cond_*` 는 **카운트와 좌우뿐**이다. 주자·이닝 축은 만든 적이 없다.

    eda32 반쪽 분할 신뢰도 (투수 주효과를 뺀 뒤, 잡음을 걷어낸 신호 std):
        batter_hand  0.0227   <- cond_ph (기존)
        점수차 구간   0.0180   = 교정점의 125%
        base_state   0.0162   = 113%
        이닝 구간     0.0139   =  97%
        count_adv    0.0144   <- cond_pc (기존, 리더보드 +27)

    구조는 기존 cond_* 와 완전히 동일하다: 시즌 리그평균으로 디트렌드 -> sum/(count+C)
    로 0 에 shrink -> 각 행은 **자기 시즌보다 과거** 시즌만으로 인코딩(leak-free).

    ⚠️ 세 축은 서로 겹친다 (점수차 ~ li ~ 주자). 그리고 오늘 '상관이 오르는데 점수는
    내려가는' 사례를 겪었다 (ws5gt: 상관 +0.260->+0.477, 점수 -17.8).
    **신뢰도는 대리지표다. 판정은 이 스크리너 실측으로만 한다.**
    """
    if _CONDSIT:
        return _CONDSIT
    tr = _read_tr(['season', 'pitcher_id', 'inning', 'base_state',
                   'score_diff_pitcher_team', 'control_success'])
    lg = tr.groupby('season')['control_success'].transform('mean')
    tr['dev'] = tr['control_success'] - lg
    tr['bs'] = tr['base_state'].astype(str)
    tr['inn'] = np.clip((tr['inning'] - 1) // 3, 0, 2)
    tr['sd'] = np.clip(np.sign(tr['score_diff_pitcher_team'])
                       * np.minimum(np.abs(tr['score_diff_pitcher_team']), 4), -4, 4)

    SPEC = {'cond_p_bs': (['pitcher_id', 'bs'], 100.0),
            'cond_p_inn': (['pitcher_id', 'inn'], 100.0),
            'cond_p_sd': (['pitcher_id', 'sd'], 100.0)}
    for name, (keys, C) in SPEC.items():
        g = tr.groupby(keys + ['season'])['dev'].agg(['sum', 'size']).sort_index()
        lv = list(range(len(keys)))
        cum = g.groupby(level=lv).cumsum().groupby(level=lv).shift(1)
        val = cum['sum'] / (cum['size'] + C)
        idx = pd.MultiIndex.from_arrays([tr[k] for k in keys] + [tr['season']])
        _CONDSIT[name] = val.reindex(idx).to_numpy(dtype='float64')
    print('    투수x상황 조건부 3종 | ' + ' '.join(
        f'{k}: 결측 {np.isnan(v).mean():.1%}' for k, v in _CONDSIT.items()), flush=True)
    return _CONDSIT


def cand_condsit(ctx, **kw):
    """★ wseason5 + 투수x(주자·이닝·점수차) 조건부 통계."""
    return _add(ctx, _wseason5_cols(), _condsit_cols())


_CONDB = {}


def _condb_cols():
    """★ 타자측 조건부통계 — 2024 홀드아웃에서 +1 로 기각했던 것을 되살린다.

    claude.md 는 "cond_p(투수) +27 vs cond_b(타자) +1 — 27배 비대칭.
    제구 성공에 타자는 거의 기여하지 않는다" 고 단정했다.
    **그 근거가 전부 2024 홀드아웃이었고, wsbat 이 그것을 깨뜨렸다**
    (2024 +3.0 / 2023 +71.7 / 리더보드 **+30.11**).

    구조는 cond_p 가족과 동일: 시즌 리그평균 디트렌드 -> sum/(count+C) 로 0 에 shrink
    -> 각 행은 자기 시즌보다 **과거** 시즌만으로 인코딩(leak-free).
    """
    if _CONDB:
        return _CONDB
    tr = _read_tr(['season', 'batter_id', 'batter_hand', 'pitcher_hand',
                   'balls_before', 'strikes_before', 'control_success'])
    tr['dev'] = tr['control_success'] - tr.groupby('season')['control_success'].transform('mean')
    tr['ca'] = _count_adv(tr['balls_before'], tr['strikes_before'])
    tr['ph'] = tr['pitcher_hand'].astype(str)

    SPEC = {'cond_b': (['batter_id'], 200.0),
            'cond_bc': (['batter_id', 'ca'], 100.0),
            'cond_bp': (['batter_id', 'ph'], 100.0)}
    for name, (keys, C) in SPEC.items():
        g = tr.groupby(keys + ['season'])['dev'].agg(['sum', 'size']).sort_index()
        lv = list(range(len(keys)))
        cum = g.groupby(level=lv).cumsum().groupby(level=lv).shift(1)
        val = cum['sum'] / (cum['size'] + C)
        idx = pd.MultiIndex.from_arrays([tr[k] for k in keys] + [tr['season']])
        _CONDB[name] = val.reindex(idx).to_numpy(dtype='float64')
    print('    타자 조건부 3종 | ' + ' '.join(
        f'{k}: 결측 {np.isnan(v).mean():.1%} std {np.nanstd(v):.4f}'
        for k, v in _CONDB.items()), flush=True)
    return _CONDB


def cand_wsboth(ctx, **kw):
    """★ 새 기준선 — wseason5 + wsbat. 리더보드 1088.83 의 구성이다.
    앞으로 후보는 전부 이 위에서 재야 한다."""
    return _add(ctx, _wseason5_cols(), _wsbat_cols())


def cand_condb(ctx, **kw):
    """★ wsboth + 타자측 조건부통계 3종."""
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), _condb_cols())


def cand_wsbsit(ctx, **kw):
    """wsboth + 투수x상황(주자·이닝·점수차) — eda32 가 고른 축."""
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), _condsit_cols())


def cand_wsbpmix(ctx, **kw):
    """재심: 구종 성향을 새 기준선 위에서 (2024 에서 +0.9 로 기각했던 것)."""
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), _pmix_cols())


def cand_wsbtsk(ctx, **kw):
    """재심: 구종별 실력 x 상황믹스 (2024 에서 -9.3 으로 기각)."""
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), _tskill_cols())


_PGF = {}
_PG_SPEC = [('asof_pitcher_prev1_game_success_rate', 1),
            ('asof_pitcher_prev3_game_success_rate', 3),
            ('asof_pitcher_prev5_game_success_rate', 5),
            ('asof_pitcher_prev1_game_middle_rate', 1),
            ('asof_pitcher_prev3_game_middle_rate', 3),
            ('asof_pitcher_prev5_game_middle_rate', 5)]


def _prevfix_cols():
    """★ prev-game 지표의 시즌 경계 오염을 고친다 — 두 겹 전부.

    `asof_pitcher_prev{1,3,5}_game_*` 는 **시즌 경계를 넘는다.**
      prev5 에 작년분이 섞인 행 22.6% / prev3 14.5% / prev1 5.3%
      당해 0~60구 & 이력 있는 투수 82,826행에서 prev5 단독 스킬 **-3,224** (적극적 독)

    오염이 두 겹이다:
      (A) 수준 — 작년 값이 작년 리그 수준을 달고 온다
      (B) 관련성 — 작년 마지막 5경기는 올해 폼과 사실상 무관하다

    2026-08-22 시도는 (A)만 고쳐서 +6 이었다. 그 측정은 **`wseason5` 이전**이라
    모델이 `w_n`(당해 시즌 투구수)을 몰랐다 — 즉 (B)를 배울 재료가 없었다.
    지금은 있다. 그리고 +6 은 wsbat(+3.0 -> LB +30.11)과 같은 거짓 음성 구간이다.

    여기서는 (B)를 **직접** 준다: prev-N 창 중 작년분 비율.
        등판수 추정 = w_n / (그 투수의 등판당 평균 투구수, train 룩업)
        cross_N     = clip(N - 등판수추정, 0, N) / N
    자기 행의 asof + train 룩업만 쓰므로 규정상 안전하다 (wseason 과 같은 논리).
    """
    if _PGF:
        return _PGF
    cols = (['season', 'game_month', 'game_dayofweek', 'game_type', 'inning',
             'pitcher_id', 'asof_pitcher_n'] + [c for c, _ in _PG_SPEC])
    tr = _read_tr(cols)

    # 경기 복원 -> 투수별 등판당 평균 투구수 (train 룩업)
    # ⚠️ 복원은 **원본 행 순서**에 의존한다. _read_tr 은 캐시 순서(파이프라인이
    #    time_idx 로 정렬한 것)로 리인덱스하므로 여기서 원본을 따로 읽어야 한다.
    #    (claude.md 4-15: run_full_pipeline 은 행 순서를 바꾼다)
    import glob as _g
    _na2 = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
    _c = ([f for f in ('data/train.csv',) if os.path.exists(f)]
          + _g.glob('/kaggle/input/**/train.csv', recursive=True))
    raw = pd.read_csv(_c[0], usecols=['season', 'game_month', 'game_dayofweek',
                                      'game_type', 'inning', 'pitcher_id'],
                      keep_default_na=False, na_values=_na2)
    k = raw[['season', 'game_month', 'game_dayofweek', 'game_type']].astype(str).agg(
        '|'.join, axis=1)
    gid = ((k != k.shift()) | (raw['inning'].diff() < 0)).cumsum()
    ap = raw.assign(g=gid).groupby(['pitcher_id', 'g']).size()
    avg = ap.groupby(level=0).mean()
    print(f'    경기 복원 {gid.nunique():,}개 | 등판 {len(ap):,}개 | '
          f'등판당 투구수 중앙 {avg.median():.1f}', flush=True)
    del raw

    # 당해 시즌 투구수 (wseason 과 같은 방식)
    n = tr['asof_pitcher_n'].to_numpy(dtype='float64')
    o = np.argsort(n, kind='stable')
    first = tr.iloc[o].groupby(['pitcher_id', 'season'], sort=False).head(1)
    key = pd.MultiIndex.from_arrays([tr['pitcher_id'], tr['season']])
    n0 = first.set_index(['pitcher_id', 'season'])['asof_pitcher_n'].reindex(
        key).to_numpy(dtype='float64')
    wn = np.maximum(n - n0, 0.0)
    gest = wn / np.maximum(tr['pitcher_id'].map(avg).to_numpy(dtype='float64'), 1.0)
    _PGF['pg_gest'] = gest

    lg_by_season = {c: tr.groupby('season')[c].mean() for c, _ in _PG_SPEC}
    seen = set()
    for c, w in _PG_SPEC:
        cross = np.clip(w - gest, 0.0, w) / w
        if w not in seen:
            _PGF[f'pg_cross{w}'] = cross          # (B) 관련성: 작년분 비율
            seen.add(w)
        m = lg_by_season[c]
        cur = tr['season'].map(m).to_numpy(dtype='float64')
        prv = tr['season'].sub(1).map(m).to_numpy(dtype='float64')
        prv = np.where(np.isfinite(prv), prv, cur)
        blend = cur * (1.0 - cross) + prv * cross   # (A) 수준: 가리키는 시즌으로 디트렌드
        _PGF['pg_' + c.replace('asof_pitcher_', '')] =             tr[c].to_numpy(dtype='float64') - blend
    print(f'    prev-game 보정 {len(_PGF)}개 | 작년분 섞인 행 '
          f'prev1 {(_PGF["pg_cross1"] > 0).mean():.1%} '
          f'prev3 {(_PGF["pg_cross3"] > 0).mean():.1%} '
          f'prev5 {(_PGF["pg_cross5"] > 0).mean():.1%}', flush=True)
    return _PGF


def cand_prevfix(ctx, **kw):
    """★ wsboth + prev-game 시즌 경계 보정 (수준 + 관련성 둘 다)."""
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), _prevfix_cols())


_BTTM = {}


def _bttm_cols():
    """★ 타자가 상대해온 투구 프로파일 (트랙맨). 두 겹으로 미검증인 축이다.

    (1) **타자측 트랙맨은 독립 검증된 적이 없다.** 2026-08-18 에 zone_speed 와
        **묶어서** 제출해 930.00(-2.96) 으로 롤백했는데, 그 뒤 zone_speed 단독이
        929.61(-3.35) 로 거의 같은 낙폭을 냈다 -> 원인은 zone_speed 였고 타자 피처는
        결백이다. claude.md 가 '독립적으로 재검증 필요' 라고 적어뒀는데 안 했다.

    (2) **수준(level)을 한 번도 안 썼다.** eda13: 트랙맨 `_mean` 이 `_std` 의 2배를
        설명하는데 우리 트랙맨 피처 20개 중 18개가 std 다. 타자측은 그마저 없다.

    그리고 wsbat 이 리더보드 +30.11 로 **타자 축이 살아있음을 증명했다.**

    구조: (타자, 시즌) 단위 집계, 각 행은 **자기 시즌보다 과거** 트랙맨만 (leak-free).
    ⚠️ 트랙맨에 2025 가 없으므로 배포 시 1년 묵은 값을 쓴다 (투수측은 여기서 77% 를
       잃었다). 다만 '상대해온 구질' 은 리그가 그 타자를 어떻게 다루는지라 성적보다
       안정적일 수 있다 — 그게 이 후보의 유일한 근거다.
    """
    if _BTTM:
        return _BTTM
    import glob
    _na = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']

    def _find(name, sub='data/'):
        c = ([f'{sub}{name}'] if os.path.exists(f'{sub}{name}') else []) +             glob.glob(f'/kaggle/input/**/{name}', recursive=True)
        if not c:
            raise SystemExit(f'{name} 을 못 찾음')
        return c[0]

    M = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break']
    T = ['fastball', 'breaking', 'offspeed']
    d = _read_tr(['season', 'batter_id'])
    tm = pd.read_csv(_find('trackman_history.csv'),
                     usecols=['season', 'batter_trackman_id', 'pitch_type_group'] + M,
                     keep_default_na=False, na_values=_na)
    for c in ('cache/raw/gold_batter_map.csv', 'gold_batter_map.csv',
              'model/gold_batter_map.csv'):
        if os.path.exists(c):
            mp = pd.read_csv(c, keep_default_na=False, na_values=_na)
            break
    else:
        raise SystemExit('gold_batter_map.csv 를 못 찾음')
    mp = mp[['batter_id', 'batter_trackman_id']].dropna().drop_duplicates(
        subset=['batter_trackman_id'])
    n0 = len(tm)
    tm = tm.merge(mp, on='batter_trackman_id', how='inner')
    if len(tm) > n0:
        raise SystemExit(f'트랙맨 병합 팽창 {n0:,} -> {len(tm):,}')
    tm['pg'] = tm['pitch_type_group'].astype(str).str.lower()

    NAMES = ([f'bt_{c}_mean' for c in M] + ['bt_speed_std', 'bt_spin_std']
             + [f'bt_mix_{t}' for t in T] + ['bt_n'])
    out = {k: np.full(len(d), np.nan) for k in NAMES}
    have = np.zeros(len(d))
    for s in sorted(d['season'].unique()):
        m = (d['season'] == s).to_numpy()
        past = tm[tm['season'] < s]
        if not len(past):
            continue
        g = past.groupby('batter_id')
        agg = g[M].mean()
        agg.columns = [f'bt_{c}_mean' for c in M]
        agg['bt_speed_std'] = g['rel_speed'].std()
        agg['bt_spin_std'] = g['spin_rate'].std()
        mix = past.groupby(['batter_id', 'pg']).size().unstack(fill_value=0)
        mix = mix.reindex(columns=T, fill_value=0)
        mix = mix.div(mix.sum(1).clip(lower=1), axis=0)
        mix.columns = [f'bt_mix_{t}' for t in T]
        agg = agg.join(mix)
        agg['bt_n'] = g.size()
        sub = agg.reindex(pd.Index(d.loc[m, 'batter_id']))
        for k in NAMES:
            out[k][m] = sub[k].to_numpy(dtype='float64')
        have[m] = np.isfinite(sub['bt_n'].to_numpy(dtype='float64')).astype(float)
    out['bt_have'] = have
    _BTTM.update(out)
    print(f'    타자 피격 프로파일 {len(NAMES)+1}개 | 커버리지 {have.mean():.1%} | '
          f'구속 평균 {np.nanmean(out["bt_rel_speed_mean"]):.1f}', flush=True)
    return _BTTM


def cand_bttm(ctx, **kw):
    """★ wsboth + 타자가 상대해온 투구 프로파일 (트랙맨 수준 + 배합)."""
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), _bttm_cols())


_SOFT = {}
_TM_MEAS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
            'extension', 'rel_height', 'rel_side', 'zone_speed']


def cv_predict_soft(Xh, yh, ysoft, Xv, params, cat, folds=FOLDS):
    """cv_predict 와 같되 **soft target** 으로 학습한다.

    - 분할은 hard label 로 (StratifiedKFold 는 이진이 필요하다)
    - 학습은 soft target + CrossEntropy (CatBoost 가 분수 타겟을 받는 손실)
    - isotonic 보정은 **hard label** 로 맞춘다 (우리가 맞히려는 것은 실제 결과다)
    """
    p2 = dict(params)
    p2['loss_function'] = 'CrossEntropy'
    p2.pop('eval_metric', None)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    out = []
    for ti, vi in skf.split(Xh, yh):
        m = CatBoostClassifier(**p2)
        m.fit(Xh.iloc[ti], ysoft[ti], verbose=0)
        rv = m.predict_proba(Xh.iloc[vi])[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip').fit(rv, yh[vi])
        out.append(iso.predict(m.predict_proba(Xv)[:, 1]))
    return np.mean(out, axis=0)


def _distill_target(ctx, alpha=1.0, teacher_iters=1000):
    """★ LUPI 증류 — teacher 가 그 투구의 트랙맨을 보고, student 는 그 확률을 배운다.

    주최측 확인 (aimers_qna.md, seopseopi 08-17 -> DACON.GM 08-19):
      Q1 "train x trackman 매칭 부분집합에서 teacher(현재 투구 구종·TrackMan 포함)를
          학습하고 student 는 pre-pitch 만 받아 증류. 제출은 student 만."
      Q3 "매칭이 **부분적**일 때도 동일하게 허용되는가"
      A  **"모두 가능합니다."** + "학습 데이터에서 도출된 값에는 제약 사항 없습니다."

    구조:
      teacher = student 피처 + 그 투구의 측정 8개 + 구종군   (정렬된 행에서만)
      soft    = alpha * teacher_OOF + (1-alpha) * hard       (정렬 안 된 행은 hard)
      student = soft 를 CrossEntropy 로 학습, pre-pitch 만 입력

    ⚠️ teacher 는 **OOF** 로 예측해야 한다. 자기가 학습한 행을 그대로 채점하면
       과적합된 값이 soft target 에 들어가 student 가 잡음을 배운다.

    ⚠️ 이론적 기대는 낮다. teacher 가 보는 정보는 student 피처에 대한 조건부
       기대값을 바꾸지 않으므로 표적이 같고, 이득 경로는 **분산 감소**뿐인데
       우리는 in-sample 2.232% ~= OOF 2.21% 로 분산에 묶여 있지 않다.
       유일한 근거는 teacher/student 예측 상관 0.76 (트리 계열은 0.92~0.97).
    """
    ck = (alpha, teacher_iters)
    if ck in _SOFT:
        return _SOFT[ck]
    import glob
    cand = ['cache/raw/aligned.parquet', 'aligned.parquet',
            'cache/raw/aligned_slim.parquet', 'aligned_slim.parquet']
    ap = next((c for c in cand if os.path.exists(c)), None)
    if ap is None:
        g = (glob.glob('/kaggle/input/**/aligned_slim.parquet', recursive=True)
             + glob.glob('/kaggle/input/**/aligned.parquet', recursive=True))
        if not g:
            raise SystemExit('aligned(_slim).parquet 을 못 찾음')
        ap = g[0]
    print(f'    정렬 파일 {ap}', flush=True)
    a = pd.read_parquet(ap, columns=['row_id', 'pitch_type_group'] + _TM_MEAS)
    rid = np.load(f'{CACHE}/row_id.npy', allow_pickle=True)
    a = a.drop_duplicates(subset=['row_id']).set_index('row_id').reindex(pd.Index(rid))
    have = a[_TM_MEAS[0]].notna().to_numpy()

    season = np.load(f'{CACHE}/season.npy')
    y = np.load(f'{CACHE}/y.npy')
    mh = season <= HOLDOUT - 1
    Xh = ctx['Xh']
    yh = ctx['yh']
    hv = have[mh]                                  # 학습 절반 안에서 정렬된 행
    print(f'    정렬된 행: 전체 {have.mean():.1%} | 학습절반 {hv.mean():.1%} '
          f'({hv.sum():,}행)', flush=True)

    # teacher 입력 = student 피처 + privileged
    P = a.loc[:, _TM_MEAS].to_numpy(dtype='float64')[mh][hv]
    pt = pd.get_dummies(a['pitch_type_group'].astype(str), prefix='pt'
                        ).astype('float32').to_numpy()[mh][hv]
    Xt = Xh[hv].reset_index(drop=True).copy()
    for i, c in enumerate(_TM_MEAS):
        Xt['tm_' + c] = P[:, i].astype(np.float32)
    for j in range(pt.shape[1]):
        Xt[f'tm_pt{j}'] = pt[:, j]
    yt = yh[hv]

    tp = dict(ctx['params'])
    tp.update(iterations=teacher_iters)
    oof = np.zeros(len(Xt))
    for ti, vi in StratifiedKFold(3, shuffle=True, random_state=7).split(Xt, yt):
        m = CatBoostClassifier(**tp)
        m.fit(Xt.iloc[ti], yt[ti], verbose=0)
        oof[vi] = m.predict_proba(Xt.iloc[vi])[:, 1]
    r = float(yt.mean())
    U = r * (1 - r)
    tsk = (1 - ((np.clip(oof, 1e-6, 1 - 1e-6) - yt) ** 2).mean() / U) * 100000
    print(f'    teacher OOF 스킬 {tsk:,.0f} (정렬분 {len(Xt):,}행, 반복 {teacher_iters})',
          flush=True)

    soft = yh.astype('float64').copy()
    soft[hv] = alpha * oof + (1.0 - alpha) * yt
    print(f'    soft target: 평균 {soft.mean():.4f} std {soft.std():.4f} '
          f'| hard 평균 {yh.mean():.4f}', flush=True)
    _SOFT[ck] = soft
    return soft


def cand_distill(ctx, **kw):
    """★ 증류 (alpha=1.0) — 정렬된 행은 teacher 확률, 나머지는 hard."""
    W, B = _wseason5_cols(), _wsbat_cols()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for D in (W, B):
        for k, v in D.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    c2 = dict(ctx, Xh=Xh)
    soft = _distill_target(c2, alpha=float(kw.get('alpha', 1.0)))
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]} (student 는 pre-pitch 만)',
          flush=True)
    return cv_predict_soft(Xh, ctx['yh'], soft, Xv, ctx['params'], ctx['cat'])


def cand_distill5(ctx, **kw):
    """증류 alpha=0.5 — teacher 확률과 hard 를 반씩 섞는다."""
    return cand_distill(ctx, alpha=0.5)


def cand_nogt(ctx, **kw):
    """`game_type` 제거.

    전이 안정성 감사에서 순열 중요도가 **-554(2023) / -479(2024)** 로 압도적 최악이었다.
    즉 모델이 이 피처를 쓰는 방식이 미래 시즌에서 크게 해롭다. 원인을 데이터에서 찾았다:

      game_type=F (전체 10.9%) 성공률
        2019 .6892  2020 .5878  2021 .7038  2022 .7087   <- 리그평균보다 훨씬 높음
        2023 .4729  2024 .4593                            <- 한 시즌에 -0.236 폭락
      game_type=R (89.1%)  2022 .5037 -> 2023 .5031       <- 거의 안 움직임

    **드리프트가 아니라 체제 변화이고 부호가 뒤집혔다** (F 가 R 보다 +0.20 -> -0.03).
    2019~2022 로 배운 "F 는 높다" 를 2025 에 적용하면 10.9% 의 행에서 크게 틀린다.
    `season` 이 있으니 트리가 구분할 수는 있지만 2025 는 학습에 없는 값이라
    부분적으로 섞일 위험이 있다. 월/요일 제거(+4.52 리더보드)와 같은 부류의 후보다.
    """
    drop = ['game_type']
    Xh = ctx['Xh'].drop(columns=drop, errors='ignore')
    Xv = ctx['Xv'].drop(columns=drop, errors='ignore')
    cat = [c for c in ctx['cat'] if c not in drop]
    p = dict(ctx['params'])
    p['cat_features'] = cat
    print(f'    피처 {ctx["Xh"].shape[1]} -> {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, p, cat)



_TRACKMAN = ['expected_control_difficulty', 'past_fb_speed_mean'] + [
    f'past_{g}_{m}_std' for g in ('fastball', 'breaking', 'offspeed')
    for m in ('rel_height', 'rel_side', 'extension', 'spin_rate',
              'vert_break', 'horz_break')]
# 바닥 중요도 플래그들. is_same_hand 는 **제외** — 2.59% 로 4위이고, 이건
# pitcher_hand x batter_hand 인데 max_ctr_complexity=1 이라 CatBoost 가
# 범주형 2-way 조합을 자동으로 못 만든다. '트리가 스스로 만든다' 규칙의 예외다.
_LOWFLAGS = ['is_sac_fly_threat', 'is_must_strike_sit', 'is_first_pitch',
             'is_garbage_time', 'is_risp', 'is_veteran', 'is_self_risp',
             'is_steal_threat_sit', 'is_pure_starter', 'is_heating_up',
             'is_rookie', 'is_full_count', 'is_cooling_down',
             'is_weekend_day_game']


def _drop_on_wsboth(ctx, drop, why):
    """wsboth(= wseason5 + wsbat) 위에서 컬럼군을 빼고 잰다.

    ⚠️ 같은 실행 안에 `wsboth` 를 같이 돌려야 짝지어 읽을 수 있다.
    """
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (_wseason5_cols(), _wsbat_cols()):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    n0 = Xh.shape[1]
    Xh = Xh.drop(columns=drop, errors='ignore')
    Xv = Xv.drop(columns=drop, errors='ignore')
    cat = [c for c in ctx['cat'] if c not in set(drop)]
    p = dict(ctx['params'])
    p['cat_features'] = cat
    print(f'    {why}: 피처 {n0} -> {Xh.shape[1]} ({n0 - Xh.shape[1]}개 제거)', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, p, cat)


def cand_notm(ctx, **kw):
    """트랙맨 20개를 전부 뺀다 (wsboth 위에서).

    근거 네 겹:
      1. 배포 모델 중요도의 **12.9%** — game_type+season, w_*/wb_* 다음 덩어리.
      2. eda39 전이 안정성: 19개 전부 상관 -0.20 ~ -0.76.
         '위험(중요도 x (1-상관))' 상위 16개 중 **12개**가 트랙맨이다.
      3. 배포 조건에서 77% 가 죽는다 (2026-08-22 실측):
         같은 시즌 증분 R2 +0.047 -> 1년 묵으면 **+0.011**.
         트랙맨에 2025 가 없어서 test 는 무조건 1년 묵은 값을 받는다.
      4. **+3.56 은 v5 에서 잰 값이다.** 오늘 wbc20 이 -2.95 로 CPU 이득을
         날린 것과 같은 구조 — 트랙맨은 투수 질의 *대리변수*였고, 지금은
         w_success/wb_success 가 그걸 직접 훨씬 잘 추정한다.
         (zone_speed 가 -3.35 로 죽은 이유도 rel_speed 와 중복이어서다.)

    ⚠️ 반대 증거: 피처 제거는 20전 1승이다 (월/요일 +4.52 가 유일).
       그리고 game_type 은 이 표에서 최악인데 제거하면 -46.7 이다.
       **두 시즌 합의로만 판정한다.**
    """
    return _drop_on_wsboth(ctx, _TRACKMAN, '트랙맨 제거')


def cand_notmstd(ctx, **kw):
    """트랙맨 중 **표준편차 18개만** 뺀다 (수준 2개는 남긴다).

    2026-08-22 EDA: 투수-시즌 제구 편차 설명력이 _mean 8개 +0.053 vs
    _std 8개 +0.025 로 **평균이 표준편차의 2배**다. 우리 20개 중 18개가 std 다.
    notm 이 음수고 이게 양수면 '수준은 살리고 흔들림만 버려라' 가 답이다.
    """
    return _drop_on_wsboth(ctx, [c for c in _TRACKMAN if c.endswith('_std')],
                           '트랙맨 std 18개 제거')


def cand_noflag(ctx, **kw):
    """중요도 바닥의 step1~13 플래그 14개를 뺀다 (wsboth 위에서).

    합쳐서 중요도 2.0% 뿐이다. claude.md 는 step1~13 파생 51개의 기여가
    사실상 0 이라고 이미 결론냈다 (피처 구성 감사: 원본 760 -> 현행 795,
    그 35 는 cond_*(+27) + 트랙맨(+9) 으로 이미 설명된다).

    ⚠️ 기대값은 0 이다. 중요도가 낮다는 것은 '해롭다' 가 아니라 '안 쓴다' 는
       뜻이고, 안 쓰는 것을 빼면 아무 일도 안 일어난다. 그래도 재는 이유는
       바닥 14개를 확실히 닫아두기 위해서다.
    """
    return _drop_on_wsboth(ctx, _LOWFLAGS, '바닥 플래그 14개 제거')


def _addcat_on_wsboth(ctx, build, why):
    """wsboth(= wseason5 + wsbat) 위에 **범주형** 컬럼을 얹고 잰다.

    ⚠️ 같은 실행에 `wsboth` 를 같이 돌려야 짝지어 읽을 수 있다.
    ⚠️ CatBoost 는 cat_features 에 float NaN 이 있으면 에러다 (4-4). 전부 str 로 만든다.
    """
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (_wseason5_cols(), _wsbat_cols()):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    n0 = Xh.shape[1]
    cat = list(ctx['cat'])
    for name, fn in build.items():
        for D in (Xh, Xv):
            D[name] = fn(D).astype(str)
        cat.append(name)
    p = dict(ctx['params'])
    p['cat_features'] = cat
    ex = {k: int(Xh[k].nunique()) for k in build}
    print(f'    {why}: 피처 {n0} -> {Xh.shape[1]} | 칸수 {ex}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, p, cat)


def _s(D, c):
    return D[c].astype(str)


def cand_nosh(ctx, **kw):
    """`is_same_hand` 를 뺀다 — **범주형 조합 축이 존재하는지** 를 재는 실험.

    is_same_hand 는 배포 모델 중요도 **2.59%(4위)** 로 pitcher_hand(0.55)와
    batter_hand(1.19) 둘을 합친 것보다 크다. 그런데 중요도는 '많이 쓴다' 는
    뜻이지 '점수를 준다' 는 뜻이 아니다 (game_type 은 1위인데 순열 중요도가 -554).

    이 하나가 결정한다:
      뺐는데 크게 음수  -> 손수 만든 범주형 조합에 값어치가 있다. 더 만든다.
      뺐는데 0          -> 트리가 두 손 컬럼으로 이미 만들고 있었다. 축 종료.
                          (step6/13 의 상황 플래그들이 전부 바닥인 것과 일관)
    """
    return _drop_on_wsboth(ctx, ['is_same_hand'], 'is_same_hand 제거')


def cand_cnt12(ctx, **kw):
    """정확한 12칸 카운트를 범주형 하나로 준다.

    지금 모델이 카운트에 대해 가진 것:
      balls_before / strikes_before  (**수치형** — 임계값 분할만 된다)
      count_advantage 4칸            — 거친 요약. ⚠️ 'None' 이 **0-0 과 3-2 를
                                       같이 담는다** (4-3). 야구에서 가장 다른
                                       두 카운트가 한 칸에 들어가 있다.
      is_full_count / is_first_pitch — 그 두 칸을 따로 빼낸 땜질
    합쳐서 중요도 3.07% 를 쓰면서도 '0-2 와 2-0 은 다르다' 를 직접 말하지 못한다.

    12칸이면 행당 12만 개라 CTR 추정이 충분히 안정적이다.
    """
    return _addcat_on_wsboth(
        ctx, {'cnt12': lambda D: _s(D, 'balls_before') + '-' + _s(D, 'strikes_before')},
        '정확한 12칸 카운트')


def cand_catx(ctx, **kw):
    """손수 고른 저카디널리티 범주형 조합 셋.

    `ctr2`(모든 2-way 자동 생성) 는 **-5** 였다. 25개에서 300쌍을 만들면
    대부분 노이즈이고 고카디널리티라 희석된다. 여기서는 **기전이 확실한
    저카디널리티** 셋만 고른다:

      hand4  (2x2=4)   pitcher_hand x batter_hand. is_same_hand 의 **무손실판** —
                       지금은 좌투vs우타와 우투vs좌타가 '다름' 한 칸에 뭉개져 있는데
                       실제 플래툰 효과는 비대칭이다. eda32 에서 batter_hand 축이
                       신호 std 0.0227 로 상황 축 중 1위였다 (count_adv 의 158%).
      bs_out (8x3=24)  base_state x outs. 야구의 표준 상태변수(득점 기대값 격자)이고
                       비가법성이 명확하다 — 3루주자는 2아웃이면 희생플라이가 안 되고,
                       1루주자는 아웃카운트에 따라 포스 상황이 달라진다.
                       eda32 에서 base_state 0.0162 (count_adv 의 113%).
      bs_cnt (8x4=32)  base_state x count_advantage. 주자가 있으면 불리한 카운트에서
                       거르는 선택지가 생겨 카운트의 의미가 달라진다.
    """
    return _addcat_on_wsboth(ctx, {
        'hand4': lambda D: _s(D, 'pitcher_hand') + _s(D, 'batter_hand'),
        'bs_out': lambda D: _s(D, 'base_state') + '|' + _s(D, 'outs_before'),
        'bs_cnt': lambda D: _s(D, 'base_state') + '|' + _s(D, 'count_advantage'),
    }, '범주형 조합 3종')


_MIXR = ['fastball', 'breaking', 'offspeed']
_WSMIX = {}


def _wsmix_cols():
    """**당해 시즌 구종 배합**을 복원한다 — 분해 가능한 마지막 컬럼군.

    asof 컬럼은 19개(비율 16 + 개수 3)이고, 그중 '비율 x 개수 = 누적' 구조라
    당해 시즌으로 분해되는 것은 열 개다:
        투수 결과 5 (success/reverse/middle/ball/strike)  -> wseason5, LB +61.72
        타자 2      (success/middle)                       -> wsbat,    LB +30.11
        **구종 3    (fastball/breaking/offspeed)           -> 미측정**
    나머지 6(prev{1,3,5}_game_{success,middle})은 분모가 없어 분해가 불가능하다.

    ★ 분모가 같다: asof_pitcher_pitchmix_n 이 asof_pitcher_n 과 **모든 행에서 동일**
      (최대차 0). 세 비율 합은 0.9995 = 완전 분할. 즉 wseason5 와 완전히 같은 기계다.

    ★ 움직인다: 같은 투수 인접 시즌 fastball_rate 변화가 std 0.0486,
      |변화| 중앙 0.0155. 부상/보직 변경/신구종으로 배합은 실제로 해마다 바뀐다.

    ⚠️ 반대 증거 둘:
      - eda10 대리모형이 이 셋을 **+0** 이라 했다 (투수 5개는 +83 -> 실측 +72.2 로
        잘 맞혔으므로 이 대리모형은 신뢰도가 있는 편이다).
      - teacher 실험: **실제로 던진 구종**을 아는 것은 +478 인데 **성향**은 정확히 0.
        배합은 성향이라 이미 base 에 녹아 있다.
      다만 그 둘은 '수준(career mix)' 에 대한 이야기이고, 이건 **당해 시즌의 변화**다.
      그리고 wsbat 이 2024 대리·홀드아웃에서 각각 +10/+3.0 인데 LB 는 +30.11 이었다.
    """
    if _WSMIX:
        return _WSMIX
    # _read_tr 이 row_id 를 붙이고 캐시 순서로 재정렬까지 한다
    cols = ['season', 'pitcher_id', 'asof_pitcher_n'] +            [f'asof_pitcher_{k}_rate' for k in _MIXR]
    tr = _read_tr(cols)
    n = tr['asof_pitcher_n'].to_numpy(dtype='float64')
    first = tr.assign(_n=n).sort_values('_n').groupby(
        ['pitcher_id', 'season'], sort=False).head(1)
    key = pd.MultiIndex.from_arrays([tr['pitcher_id'], tr['season']])
    fi = first.set_index(['pitcher_id', 'season'])
    n0 = fi['_n'].reindex(key).to_numpy()
    wn = np.maximum(n - n0, 0.0)
    out = {}
    for k in _MIXR:
        col = f'asof_pitcher_{k}_rate'
        x = (tr[col].fillna(0).to_numpy(dtype='float64') * n).round()
        x0 = (fi[col].fillna(0).reindex(key).to_numpy() * n0).round()
        wx = np.clip(x - x0, 0.0, wn)
        prior = float(np.nanmean(tr[col].to_numpy(dtype='float64')))
        rate = (wx + prior * 100.0) / (wn + 100.0)
        lgk = tr.assign(_v=tr[col]).groupby('season')['_v'].mean()
        out[f'wm_{k}'] = rate - tr['season'].map(lgk).to_numpy(dtype='float64')
    _WSMIX.update(out)
    print('    당해시즌 구종배합 복원: '
          + ' '.join(f'{k}={np.nanmean(out["wm_"+k]):+.4f}' for k in _MIXR), flush=True)
    return _WSMIX


def cand_wsmix(ctx, **kw):
    """wsboth + 당해 시즌 구종 배합 3개."""
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), _wsmix_cols())


def cand_wsmixd(ctx, **kw):
    """wsboth + 구종 배합의 **커리어 대비 변화량** 3개.

    수준(당해 배합)이 아니라 **변화**(당해 - 커리어)를 준다. teacher 실험이
    '성향은 0' 이라고 한 것은 수준에 대한 이야기이므로, 변화는 별개일 수 있다.
    ⚠️ 단 커리어 배합은 그 행에 이미 있으므로(asof_pitcher_{k}_rate)
       wsmix 가 양수면 이건 한 행 안 뺄셈이라 0 일 가능성이 높다 (diff +7 노이즈).
    """
    M = _wsmix_cols()
    # ⚠️ _read_tr 의 assert 는 cols[0] 의 NaN 을 row_id 매칭 실패로 오판한다.
    #    구종 비율은 정상적으로 NaN 이 있으므로 결측 없는 컬럼을 앞에 둔다.
    tr = _read_tr(['season'] + [f'asof_pitcher_{k}_rate' for k in _MIXR])
    D = {}
    for k in _MIXR:
        car = tr[f'asof_pitcher_{k}_rate'].fillna(0).to_numpy(dtype='float64')
        lg = float(np.nanmean(car))
        D[f'wmd_{k}'] = (M[f'wm_{k}'] + lg) - car
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), D)


_FAILL = ['middle', 'reverse']
# 보조 타겟으로 쓸 수 있는 라벨 전부. eda41 투수 수준 신호 크기(success=100%):
#   reverse 102% | ball 61% | strike 55% | middle 45% | wild 42%
_AUXL = ['middle', 'reverse', 'ball', 'strike']
_CONDF = {}


def _fail_labels():
    """투구 단위 실패유형 라벨을 asof 차분으로 복원한다 (train 전용).

    검증(eda40, 147만 행):
      투수 내 인접 행 asof_pitcher_n 증분 +1        비율 1.000000
      다섯 라벨 전부 누적개수 차분이 {0,1}           비율 1.000000
      ★ 복원 success vs control_success 일치율      1.000000

    asof 는 '직전까지' 이므로 행 i 와 i+1 로 **투구 i** 의 라벨이 나온다.
    성공 행에서 middle/reverse 는 정확히 0 이다 (실패에서만 발생, 완전 중첩).

    ⚠️ 규정: 차분은 **train 에서만** 한다. test 에서 인접 행을 차분하는 것은
       주최측이 명시적으로 '규칙 위반' 이라 답한 사안이다 (2jin1 08-17).
       train 유래 값에는 제약이 없다 (DACON.GM 08-19).
    ⚠️ 행 순서: 반드시 (pitcher_id, asof_pitcher_n) 으로 정렬해야 한다.
       run_full_pipeline 은 time_idx 로 정렬하고 원복하지 않는다 (claude.md 4-15).
       pf_* 에서 이 함정에 그대로 빠졌었다.
    """
    if _CONDF:
        return _CONDF
    tr = _read_tr(['season', 'pitcher_id', 'asof_pitcher_n', 'control_success',
                   'balls_before', 'strikes_before', 'batter_hand']
                  + [f'asof_pitcher_{k}_rate' for k in _AUXL])
    o = np.lexsort((tr['asof_pitcher_n'].to_numpy(), tr['pitcher_id'].to_numpy()))
    n = tr['asof_pitcher_n'].to_numpy(dtype='float64')[o]
    g = tr['pitcher_id'].to_numpy()[o]
    ok = np.r_[(g[:-1] == g[1:]) & (n[1:] - n[:-1] == 1), False]
    out = {}
    for k in _AUXL:
        cum = np.round(tr[f'asof_pitcher_{k}_rate'].fillna(0)
                       .to_numpy(dtype='float64')[o] * n)
        d = np.r_[cum[1:] - cum[:-1], np.nan]
        v = np.where(ok & np.isin(d, [0.0, 1.0]), d, np.nan)
        back = np.full(len(tr), np.nan)
        back[o] = v
        out[k] = back
    # 검산: success 도 같은 방식으로 복원해 정답과 대조한다
    ns = tr['asof_pitcher_n'].to_numpy(dtype='float64')[o]
    cs = np.round(_read_tr(['season', 'asof_pitcher_success_rate'])
                  ['asof_pitcher_success_rate'].fillna(0)
                  .to_numpy(dtype='float64')[o] * ns)
    ds = np.r_[cs[1:] - cs[:-1], np.nan]
    m = ok & np.isin(ds, [0.0, 1.0])
    acc = float((ds[m] == tr['control_success'].to_numpy()[o][m]).mean())
    print(f'    라벨 복원 검산: success 일치율 {acc:.6f} (표본 {m.sum():,})', flush=True)
    if acc < 0.999:
        raise RuntimeError(f'복원 검산 실패 {acc:.6f} — 행 순서를 의심할 것')
    _CONDF['_tr'] = tr
    _CONDF.update(out)
    return _CONDF


def _condfail_cols(keys=('p', 'pc', 'ph')):
    """실패유형 라벨로 cond_* 와 **완전히 같은 설계**의 조건부 통계를 만든다.

    설계는 claude.md 2장 '조건부 투수통계' 그대로다:
      1) 시즌 리그평균을 빼서 디트렌드
      2) sum / (count + C) 로 0(리그평균)에 shrink  — 표본 적으면 자동으로 0
      3) **leak-free**: 시즌 S 행은 시즌 < S 데이터로만 인코딩 (2019 행은 NaN)
    C 값도 동일: pitcher 200 / xカ운트 100 / x좌우 100.

    배포 시에는 cond_* 와 똑같이 룩업 테이블 merge 라 test 다른 행을 안 본다.
    """
    F = _fail_labels()
    tr = F['_tr']
    sea = tr['season'].to_numpy()
    ca = _count_adv(tr['balls_before'].to_numpy(), tr['strikes_before'].to_numpy())
    KEY = {'p': [tr['pitcher_id'].to_numpy()],
           'pc': [tr['pitcher_id'].to_numpy(), ca],
           'ph': [tr['pitcher_id'].to_numpy(), tr['batter_hand'].astype(str).to_numpy()]}
    CC = {'p': 200.0, 'pc': 100.0, 'ph': 100.0}
    seasons = np.array(sorted(set(sea.tolist())))
    out = {}
    for k in _FAILL:
        v = F[k]
        lg = pd.Series(v).groupby(sea).transform('mean').to_numpy()
        dv = v - lg                                   # 디트렌드
        for kk in keys:
            idx = pd.MultiIndex.from_arrays(KEY[kk] + [sea]) if False else None
            df = pd.DataFrame({'d': dv, 's': sea})
            for i, a in enumerate(KEY[kk]):
                df[f'k{i}'] = a
            kc = [c for c in df.columns if c.startswith('k')]
            gg = df.dropna(subset=['d']).groupby(kc + ['s'])['d'].agg(['sum', 'size'])
            res = np.full(len(tr), np.nan)
            for S in seasons[1:]:                     # 시즌 S 는 < S 로만 인코딩
                past = gg[gg.index.get_level_values('s') < S].groupby(level=kc).sum()
                cur = df[df['s'] == S]
                j = cur.set_index(kc).index
                sm = past['sum'].reindex(j).to_numpy()
                cn = past['size'].reindex(j).to_numpy()
                val = np.where(np.isnan(cn), np.nan,
                               np.nan_to_num(sm) / (np.nan_to_num(cn) + CC[kk]))
                res[df['s'].to_numpy() == S] = val
            out[f'cf_{kk}_{k}'] = res
    print('    실패유형 조건부: ' + ' '.join(
        f'{c}(결측 {np.isnan(v).mean():.0%})' for c, v in out.items()), flush=True)
    return out


def cand_condfail(ctx, **kw):
    """wsboth + 실패유형 조건부 통계 6개 (middle/reverse x p/pc/ph).

    ★ 팀원(송도원) 제안. 근거:
      wseason5(success 포함 5종) +72.2  vs  wseason4(success 제외 4종) +50.5
      -> 실패유형은 success 없이도 +50.5 를 낸다 = 독립 정보가 많다.
      그리고 cond_*(success 기준, 투수 x 카운트/좌우)는 리더보드 **+27** 이었다.
      같은 비율이 조건부에서도 성립하면 두 자릿수가 나온다.

    ⚠️ 반대 증거: wsbsit(투수x상황 3종)은 -18.6/-7.3 으로 두 시즌 합의 기각이었다.
       다만 그건 success 기준이고 축도 카운트/좌우 밖이었다.
    ⚠️ 무조건부 수준(w_middle/w_reverse, asof_*_rate)은 **이미 모델에 있다.**
       여기서 새로운 것은 **조건부** 부분뿐이므로 그 marginal 만 잡힌다.
    """
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), _condfail_cols())


def cand_condfailc(ctx, **kw):
    """위와 같되 **카운트 축만** (cf_pc_middle / cf_pc_reverse 둘).

    6개를 한 번에 넣으면 어느 축이 기여했는지 모른다 (v8 의 -22.35 를 넷에
    귀속시키지 못했던 실수). 카운트는 eda32 에서 우리가 이미 쓰는 축이고
    cond_pc 로 검증된 경로다.
    """
    return _add(ctx, _wseason5_cols(), _wsbat_cols(), _condfail_cols(keys=('pc',)))


def cand_hand4(ctx, **kw):
    """`pitcher_hand x batter_hand` 4칸만 (is_same_hand 의 무손실판).

    `catx`(hand4+bs_out+bs_cnt)가 2024 +8.3 / 2023 +0.7 로 갈렸다. 셋 중
    누가 끌었는지 분해한다 (v8 에서 넷을 한 번에 넣어 -22.35 를 귀속 못 한 실수).

    hand4 가 최우선 후보인 이유: `nosh`(is_same_hand 제거)가 **-17.0 / -13.0**
    두 시즌 합의로 음수다. 즉 이 축은 확실히 실재하고, is_same_hand 는 그 축의
    **손실 압축**이다 — 좌투vs우타와 우투vs좌타를 '다름' 한 칸에 뭉갠다.
    실제 플래툰 효과는 비대칭이므로 4칸이 무손실이다.
    """
    return _addcat_on_wsboth(ctx, {
        'hand4': lambda D: _s(D, 'pitcher_hand') + _s(D, 'batter_hand')}, 'hand4 단독')


def cand_bsx(ctx, **kw):
    """`base_state x outs` + `base_state x count_advantage` 둘만."""
    return _addcat_on_wsboth(ctx, {
        'bs_out': lambda D: _s(D, 'base_state') + '|' + _s(D, 'outs_before'),
        'bs_cnt': lambda D: _s(D, 'base_state') + '|' + _s(D, 'count_advantage'),
    }, 'base_state 조합 2종')


def cand_cnt12h(ctx, **kw):
    """`cnt12` + `hand4` — 둘이 쌓이는지 본다 (제출본 후보).

    둘 다 '저카디널리티 x 기전 확실' 조건을 만족하고 서로 다른 축이다
    (카운트 vs 손). 가법이면 제출본은 이 구성이 된다.
    """
    return _addcat_on_wsboth(ctx, {
        'cnt12': lambda D: _s(D, 'balls_before') + '-' + _s(D, 'strikes_before'),
        'hand4': lambda D: _s(D, 'pitcher_hand') + _s(D, 'batter_hand')}, 'cnt12+hand4')


def cand_cnth(ctx, **kw):
    """`cnt12` + `cnt12 x batter_hand` (24칸).

    같은 실행의 `cnt12` 와 비교해 **marginal 만** 읽는다.
    기전: 같은 카운트라도 좌타/우타 상대 전략이 다르다 (백도어 브레이킹볼,
    몸쪽 승부 등). `cond_phc`(투수x손x카운트)는 **투수 수준**이라 리그 수준의
    손x카운트 상호작용은 아직 어디에도 없다.
    24칸이면 행당 6만 개라 CTR 이 안정적이다.
    """
    return _addcat_on_wsboth(ctx, {
        'cnt12': lambda D: _s(D, 'balls_before') + '-' + _s(D, 'strikes_before'),
        'cnt_bh': lambda D: (_s(D, 'balls_before') + '-' + _s(D, 'strikes_before')
                             + '|' + _s(D, 'batter_hand')),
    }, 'cnt12 + cnt12xbatter_hand')


def cand_cntg(ctx, **kw):
    """`cnt12` + `cnt12 x game_type` (24칸).

    F(퓨처스)는 2023 에 판정 체제가 바뀌었고(.7087 -> .4729) 카운트별 스트라이크
    존 운용이 R 과 다를 수 있다. `game_type` 은 중요도 1위(9.13%)인데 순열 중요도는
    -554 로 최악이다 — 모델이 이 피처를 **쓰는 방식**이 나쁘다는 뜻이므로,
    쓸 방향을 명시적으로 좁혀주는 조합이 도움이 될 수 있다.
    ⚠️ `nogt`(제거)는 -46.7 이었다. 제거가 아니라 **정제**가 이 후보의 논지다.
    """
    return _addcat_on_wsboth(ctx, {
        'cnt12': lambda D: _s(D, 'balls_before') + '-' + _s(D, 'strikes_before'),
        'cnt_gt': lambda D: (_s(D, 'balls_before') + '-' + _s(D, 'strikes_before')
                             + '|' + _s(D, 'game_type')),
    }, 'cnt12 + cnt12xgame_type')


_AUXC = {}


def _aux_targets():
    """투구 단위 실패유형 라벨 (eda40/41). 배타적이 아니므로 2비트로 다룬다.

    교차표(147만 행): success=1 에서 middle/reverse 가 1인 행은 **0개** — 완전 중첩.
    다만 middle 과 reverse 는 **동시에 1일 수 있다** (실패의 7.2%).
    포수가 바깥에 앉았는데 가운데-몸쪽으로 몰리면 둘 다다.
    따라서 wild = 실패 & !middle & !reverse (1-s-m-r 은 틀렸다).

    유형별 드리프트가 서로 반대다 (2019->2024, R 전용, 실패율 +0.0598):
        middle +0.0484 | reverse +0.0543 | both +0.0015 | wild **-0.0414**
    총 변동 0.144 가 순변동 0.060 을 만든다 — 이진 타겟이 2.4배를 가린다.
    투수 수준 신뢰도: reverse **0.941** > success 0.911 (신호 크기는 102%).
    """
    if _AUXC:
        return _AUXC
    F = _fail_labels()
    tr = F['_tr']
    y = tr['control_success'].to_numpy(dtype='float64')
    m, r = F['middle'], F['reverse']
    good = ~(np.isnan(m) | np.isnan(r))
    _AUXC['middle'] = np.where(good, m, np.nan)
    _AUXC['reverse'] = np.where(good, r, np.nan)
    _AUXC['wild'] = np.where(good, (1 - y) * (1 - np.nan_to_num(m)) * (1 - np.nan_to_num(r)), np.nan)
    # 5분류: 0 성공 / 1 몰림만 / 2 반대만 / 3 둘다 / 4 크게벗어남
    cls = np.where(y == 1, 0,
                   np.where((m == 1) & (r == 1), 3,
                            np.where(m == 1, 1, np.where(r == 1, 2, 4))))
    _AUXC['cls'] = np.where(good, cls, np.nan)
    print('    보조 타겟: ' + ' '.join(
        f'{k}={np.nanmean(_AUXC[k]):.4f}' for k in ('middle', 'reverse', 'wild'))
        + f' | 결측 {np.isnan(_AUXC["cls"]).mean():.2%}', flush=True)
    return _AUXC


def _stack_aux(Xh, Xv, tgt, params, cat, folds=3, iters=300):
    """보조 타겟 예측을 **교차적합**해서 피처로 만든다.

    ⚠️ `reverse=1 => success=0` 이므로 in-sample 적합은 타겟을 통째로 흘린다.
       학습 절반은 반드시 out-of-fold 값을, 검증 절반은 폴드 평균을 받아야 한다.
    ⚠️ 보조 라벨이 NaN 인 행(투수의 마지막 투구 등)은 학습에서 빼고 예측만 받는다.
    """
    p2 = dict(params)
    p2['iterations'] = iters
    p2.pop('early_stopping_rounds', None)
    oof = np.full(len(Xh), np.nan)
    vp = np.zeros(len(Xv))
    fit = np.isfinite(tgt)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
    for ti, vi in skf.split(Xh, np.nan_to_num(tgt, nan=0.0).astype(int)):
        ti = ti[fit[ti]]
        m = CatBoostClassifier(**p2)
        m.fit(Xh.iloc[ti], tgt[ti].astype(int), verbose=0)
        oof[vi] = m.predict_proba(Xh.iloc[vi])[:, 1]
        vp += m.predict_proba(Xv)[:, 1] / folds
    return oof, vp


def cand_auxrev(ctx, **kw):
    """wsboth + 교차적합 P(reverse|X) 하나를 피처로.

    `reverse` 를 고른 이유: 투수 수준 신뢰도가 **0.941 로 success(0.911)보다 높고**
    신호 크기도 102% 다 (eda41). 이 데이터에서 가장 안정적인 투수 특성이다.
    이진 y 만 보는 본 모델은 P(reverse|X) 를 **배울 수 없다** — 그 라벨을 못 보니까.
    ⚠️ 이득 경로는 '더 나은 기저함수'(추정 효율)다. 우리는 거기 안 묶여 있으므로
       (in-sample 2.232% ~= OOF 2.21%) 기대값을 낮게 잡을 것.
    """
    A = _aux_targets()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (_wseason5_cols(), _wsbat_cols()):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    o, v = _stack_aux(Xh, Xv, A['reverse'][mh], ctx['params'], ctx['cat'])
    Xh['aux_rev'] = o.astype(np.float32)
    Xv['aux_rev'] = v.astype(np.float32)
    print(f'    aux_rev: OOF 평균 {np.nanmean(o):.4f} / 검증 평균 {v.mean():.4f} '
          f'| 실제 reverse 율 {np.nanmean(A["reverse"][mh]):.4f}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_aux3(ctx, **kw):
    """wsboth + P(reverse) + P(middle) + P(wild) 셋 다."""
    A = _aux_targets()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (_wseason5_cols(), _wsbat_cols()):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    for t in ('reverse', 'middle', 'wild'):
        o, v = _stack_aux(Xh, Xv, A[t][mh], ctx['params'], ctx['cat'])
        Xh['aux_' + t] = o.astype(np.float32)
        Xv['aux_' + t] = v.astype(np.float32)
    print(f'    보조 피처 3개 추가 -> 피처 {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_multicls(ctx, **kw):
    """타겟을 5분류로 바꾸고 '성공' 클래스 확률을 쓴다 (구조판).

    0 성공 / 1 몰림만 / 2 반대만 / 3 둘다 / 4 크게벗어남.
    유형별 드리프트가 서로 반대이므로(middle +0.048 vs wild -0.041) 이진 타겟은
    상쇄된 순변동만 본다. 나누면 세 과정이 분리된다.
    ⚠️ isotonic 은 이진 y 로 적합한다 (채점이 P(success) 니까).
    """
    A = _aux_targets()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (_wseason5_cols(), _wsbat_cols()):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    cls = A['cls'][mh]
    yh = ctx['yh']
    fit = np.isfinite(cls)
    p2 = dict(ctx['params'])
    p2['loss_function'] = 'MultiClass'
    p2['eval_metric'] = 'MultiClass'
    p2.pop('early_stopping_rounds', None)
    out = []
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for ti, vi in skf.split(Xh, yh):
        ti = ti[fit[ti]]
        m = CatBoostClassifier(**p2)
        m.fit(Xh.iloc[ti], cls[ti].astype(int), verbose=0)
        ci = list(m.classes_).index(0)
        rv = m.predict_proba(Xh.iloc[vi])[:, ci]
        iso = IsotonicRegression(out_of_bounds='clip').fit(rv, yh[vi])
        out.append(iso.predict(m.predict_proba(Xv)[:, ci]))
    print(f'    5분류 학습 완료 (성공 클래스 인덱스 {ci})', flush=True)
    return np.mean(out, axis=0)


def cand_auxperm(ctx, **kw):
    """`auxrev` 의 **대조군** — reverse 라벨을 무작위로 섞고 똑같이 돌린다.

    로컬 2024 에서 auxrev 가 wsboth 894 -> **930 (+36)** 이 나왔다.
    wsbat 이후 최대치라 먼저 누수를 의심해야 한다 (`reverse=1 => success=0`).

    라벨을 섞으면 보조 모델이 배울 게 없으므로 aux 피처는 사실상 상수가 된다.
      기준선 근처(~894)  -> auxrev 의 +36 은 **진짜 신호**다.
      여전히 크게 양수    -> 파이프라인 어딘가에 구조적 누수가 있다.

    ⚠️ 섞는 것은 **학습 절반 안에서만**. 검증 절반은 애초에 라벨을 안 쓴다.
    """
    A = _aux_targets()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (_wseason5_cols(), _wsbat_cols()):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    t = A['reverse'][mh].copy()
    rng = np.random.default_rng(123)
    rng.shuffle(t)                      # 라벨만 섞는다 (X 는 그대로)
    o, v = _stack_aux(Xh, Xv, t, ctx['params'], ctx['cat'])
    Xh['aux_rev'] = o.astype(np.float32)
    Xv['aux_rev'] = v.astype(np.float32)
    print(f'    [대조군] 섞은 라벨 | OOF 표준편차 {np.nanstd(o):.5f} '
          f'(auxrev 진본은 훨씬 클 것) / 검증 평균 {v.mean():.4f}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_cnt12b(ctx, **kw):
    """`cnt12` + `bs_out` + `bs_cnt` — 범주형 축의 제출 후보 구성.

    분해 결과 (2023): bsx **+16.2** / cnt12 +3.4 / hand4 **-11.4**.
    `catx`(hand4+bsx)가 2023 에서 +0.7 로 무너진 것은 hand4 가 상쇄했기 때문이다
    (-11.4 + 16.2 = +4.8, 실측 +0.7). hand4 를 빼고 둘만 합친다.
    """
    return _addcat_on_wsboth(ctx, {
        'cnt12': lambda D: _s(D, 'balls_before') + '-' + _s(D, 'strikes_before'),
        'bs_out': lambda D: _s(D, 'base_state') + '|' + _s(D, 'outs_before'),
        'bs_cnt': lambda D: _s(D, 'base_state') + '|' + _s(D, 'count_advantage'),
    }, 'cnt12 + base_state 조합 2종')


def cand_catall(ctx, **kw):
    """범주형 조합 넷 전부: cnt12 + hand4 + bs_out + bs_cnt.

    분해 결과 (홀드아웃 대비, 2024 / 2023):
        cnt12  +8.3 / +3.4     hand4  +6.5 / -11.4
        bsx    +3.8 / +16.2    cnt12h +19.0 / +4.3    catx +8.3 / +0.7
    hand4 는 **단독으로는 뒤집히는데 cnt12 와 함께면 안전하다** (증분 +10.7 / +0.9).
    넷을 다 넣었을 때 가법인지 상쇄인지가 제출 구성을 정한다.
    ⚠️ ctr2(모든 2-way 자동 생성)는 -5 였다. 칸이 늘수록 CTR 추정이 나빠지므로
       무한정 더하면 안 된다 — 이게 그 한계를 재는 실험이다.
    """
    return _addcat_on_wsboth(ctx, {
        'cnt12': lambda D: _s(D, 'balls_before') + '-' + _s(D, 'strikes_before'),
        'hand4': lambda D: _s(D, 'pitcher_hand') + _s(D, 'batter_hand'),
        'bs_out': lambda D: _s(D, 'base_state') + '|' + _s(D, 'outs_before'),
        'bs_cnt': lambda D: _s(D, 'base_state') + '|' + _s(D, 'count_advantage'),
    }, '범주형 조합 4종 전부')


def cand_auxcnt(ctx, **kw):
    """`auxrev` + `cnt12` + `hand4` — 두 축이 쌓이는지 (제출 구성 후보).

    두 축은 기전이 완전히 다르다:
      auxrev  라벨 채널 — 본 모델이 볼 수 없는 P(reverse|X) 를 기저함수로 준다
      cnt12h  범주형 조합 — 트리가 만들 수 있지만 비싼 상호작용을 CTR 하나로 준다
    가법이면 제출본은 이 구성이다.
    ⚠️ auxrev 의 +36 은 **아직 대조군(auxperm) 통과 전**이다. 누수로 판명되면 폐기.
    """
    A = _aux_targets()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (_wseason5_cols(), _wsbat_cols()):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    o, v = _stack_aux(Xh, Xv, A['reverse'][mh], ctx['params'], ctx['cat'])
    Xh['aux_rev'] = o.astype(np.float32)
    Xv['aux_rev'] = v.astype(np.float32)
    cat = list(ctx['cat'])
    for name, fn in {
            'cnt12': lambda D: _s(D, 'balls_before') + '-' + _s(D, 'strikes_before'),
            'hand4': lambda D: _s(D, 'pitcher_hand') + _s(D, 'batter_hand')}.items():
        Xh[name] = fn(Xh).astype(str)
        Xv[name] = fn(Xv).astype(str)
        cat.append(name)
    p = dict(ctx['params'])
    p['cat_features'] = cat
    print(f'    aux + 범주형 2종 -> 피처 {Xh.shape[1]}', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, p, cat)


def cand_auxrevns(ctx, **kw):
    """`auxrev` 에서 보조 모델의 **season / game_type 을 제거**한 판.

    리더보드 실측 진단(2026-08-25): auxrev 는 스크리너 +40.2 인데 LB **+4.14**
    (전달률 0.10). 원인은 보조 모델이 무엇을 보느냐였다:
        game_type 15.94% | season 15.13% | w_reverse 14.10%
        wb_success 5.74% | asof_pitcher_reverse_rate 4.38%
    **절반이 season x game_type(드리프트)** 이고 나머지는 본 모델이 이미 가진 피처다.
    `season` 경계가 2023.5 에서 끝나므로(eda31) aux_rev 는 **2024 수준에 고정된 값**을
    2025 에 뱉는다 -- 외삽이 안 된다.

    그 둘을 보조 모델에서 빼면 aux_rev 가 드리프트 대신 **순수한 투수·상황 reverse
    성향**만 담는다. 본 모델은 season/game_type 을 직접 갖고 있으므로 잃는 것이 없다.
    ⚠️ 남는 것도 결국 w_reverse 의 압축이라 기대는 낮다. 두 시즌 합의로만 판정할 것.
    """
    A = _aux_targets()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (_wseason5_cols(), _wsbat_cols()):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    drop = [c for c in ('season', 'game_type') if c in Xh.columns]
    ap = dict(ctx['params'])
    ap['cat_features'] = [c for c in ctx['cat'] if c not in drop]
    print(f'    보조 모델에서 제거: {drop}', flush=True)
    o, v = _stack_aux(Xh.drop(columns=drop), Xv.drop(columns=drop),
                      A['reverse'][mh], ap, ap['cat_features'])
    Xh['aux_rev'] = o.astype(np.float32)
    Xv['aux_rev'] = v.astype(np.float32)
    print(f'    aux_rev(ns): OOF 평균 {np.nanmean(o):.4f} / 검증 평균 {v.mean():.4f}',
          flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def _auxns(ctx, targets):
    """보조 타겟 여러 개를 **season / game_type 없이** 교차적합해 피처로 얹는다.

    ★ 2026-08-25 리더보드 실측이 이 설계를 확정했다:
        auxrev  (season 포함)  스크리너 +40.2 -> LB **+4.14**  전달률 0.10
        auxrevns(season 제거)  스크리너 +22.5 -> LB **+13.50** 전달률 **0.60**
      홀드아웃 숫자는 절반인데 리더보드는 3배다. `season` 분기 경계가 2023.5 에서
      끝나므로(eda31) season 을 재료로 만든 파생피처는 **학습 시대에 얼어붙은 값**을
      2025 에 뱉는다. 2023 홀드아웃에서 auxrev 가 -18.1 이었던 것이 그 증거다.

    eda41 투수 수준 신호 크기 (success=100%):
        reverse 102% | ball 61% | strike 55% | middle 45% | wild 42%
    지금 배포본은 reverse 하나만 쓴다.
    """
    A = _aux_targets()
    season = np.load(f'{CACHE}/season.npy')
    mh, mv = season <= HOLDOUT - 1, season == HOLDOUT
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for W in (_wseason5_cols(), _wsbat_cols()):
        for k, v in W.items():
            Xh[k] = v[mh].astype(np.float32)
            Xv[k] = v[mv].astype(np.float32)
    drop = [c for c in ('season', 'game_type') if c in Xh.columns]
    ap = dict(ctx['params'])
    ap['cat_features'] = [c for c in ctx['cat'] if c not in drop]
    Ah, Av = Xh.drop(columns=drop), Xv.drop(columns=drop)
    for t in targets:
        o, v = _stack_aux(Ah, Av, A[t][mh], ap, ap['cat_features'])
        Xh['aux_' + t] = o.astype(np.float32)
        Xv['aux_' + t] = v.astype(np.float32)
        print(f'    aux_{t}: OOF {np.nanmean(o):.4f} / 검증 {v.mean():.4f} '
              f'(실제 {np.nanmean(A[t][mh]):.4f})', flush=True)
    print(f'    보조 {len(targets)}개 -> 피처 {Xh.shape[1]} (보조모델에서 {drop} 제거)',
          flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_auxns3(ctx, **kw):
    """reverse + ball + strike (신호 102% / 61% / 55%). 현행 배포본 = reverse 하나."""
    return _auxns(ctx, ['reverse', 'ball', 'strike'])


def cand_auxns4(ctx, **kw):
    """reverse + ball + strike + middle (넷 다)."""
    return _auxns(ctx, ['reverse', 'ball', 'strike', 'middle'])


def cand_auxnsb(ctx, **kw):
    """reverse + ball 둘만 — 신호 상위 두 개. 셋이 희석되면 이게 답이다."""
    return _auxns(ctx, ['reverse', 'ball'])

def cand_dropoldF(ctx, **kw):
    """체제 변화 **이전**의 F 행만 학습에서 뺀다 (game_type 은 유지).

    `nogt` 는 피처를 통째로 버려서 새 체제의 F 신호(-0.03)까지 잃는다.
    이건 낡은 규칙을 가르치는 행만 버리고 피처는 남긴다.
    '데이터 양이 최신성을 이긴다'(-62/-128)와 다른 점: 오래된 행이라서 버리는 게
    아니라 **라벨 관계가 깨진 것이 확인된** 행이라서 버린다.
    """
    gt = ctx['Xh']['game_type'].astype(str).to_numpy()
    keep = ~((ctx['sh'] <= 2022) & (gt == 'F'))
    print(f'    학습행 {len(keep):,} -> {keep.sum():,} '
          f'(구체제 F {(~keep).sum():,}행 제거)', flush=True)
    return cv_predict(ctx['Xh'][keep].reset_index(drop=True), ctx['yh'][keep],
                      ctx['Xv'], ctx['params'], ctx['cat'])


def _mono_map(cols, aggressive):
    """부호가 확실한 피처에만 단조 제약을 건다.

    왜 이게 전이(transfer)에 듣는가: 트리는 학습 범위 밖을 외삽하지 못하고, 범위 **안**
    에서도 축을 아무 모양으로나 쪼갤 수 있다. 2025 는 리그 성공률이 내려가 투수 통계
    분포가 통째로 왼쪽으로 밀리므로, 지금 별로 안 쓰이던 저구간의 **우연한 굴곡**이
    그때는 대량으로 쓰인다. 단조 제약은 그 굴곡을 금지한다.

    선형 baseline(lrbase +7 / lrbase2 -183)과 정반대 접근이다. 그쪽은 외삽하는 항을
    **더했고**, 이건 잘못 외삽할 자유도를 **뺀다**. 항을 더하면 공선성으로 불안정해지지만
    (lrbase2 의 +0.120/-0.118 상쇄 쌍), 자유도를 빼는 쪽은 그런 실패 모드가 없다.

    ⚠️ 단조 제약은 **편(partial)** 단조성이다. 다른 피처를 고정했을 때의 방향이
    확실한 것만 넣어야 한다. 그래서 보수판(mono)은 '성공률' 계열만 건다.
    """
    m = {}
    pos = ['asof_pitcher_success_rate', 'smoothed_pitcher_success_rate',
           'asof_batter_success_rate',
           'asof_pitcher_prev1_game_success_rate',
           'asof_pitcher_prev3_game_success_rate',
           'asof_pitcher_prev5_game_success_rate',
           'cond_p', 'cond_pc', 'cond_ph', 'cond_phc']
    for c in pos:
        if c in cols:
            m[c] = 1
    if aggressive:
        # 실패 유형 비율과 난이도 지수 — 방향은 자명하나 편효과는 덜 확실하다.
        neg = [c for c in cols
               if c.endswith('_middle_rate') or c.endswith('_reverse_rate')]
        neg += [c for c in cols if c.startswith('past_') and c.endswith('_std')]
        neg += [c for c in cols if c == 'expected_control_difficulty']
        for c in neg:
            m[c] = -1
    return m


def _mono(ctx, aggressive):
    mm = _mono_map(list(ctx['Xh'].columns), aggressive)
    print(f'    단조 제약 {len(mm)}개 '
          f'(+{sum(v > 0 for v in mm.values())} / -{sum(v < 0 for v in mm.values())})',
          flush=True)
    # monotone_constraints 는 Ordered 부스팅과 같이 못 쓴다.
    return _p(ctx, monotone_constraints=mm, boosting_type='Plain')


def cand_mono(ctx, **kw):
    return _mono(ctx, False)


def cand_mono2(ctx, **kw):
    return _mono(ctx, True)


def cand_plain(ctx, **kw):
    """mono 의 대조군. boosting_type='Plain' 자체의 효과를 분리한다 —
    이걸 안 재면 mono 의 차이가 제약 때문인지 부스팅 방식 때문인지 알 수 없다."""
    return _p(ctx, boosting_type='Plain')


def _add_diffs(D):
    """드리프트에 불변인 '차이' 피처.

    트리는 축에 평행한 분할만 하므로 두 피처의 **뺄셈**을 표현하려면 분할이 수십 개
    필요하다. 그래서 명시적으로 주는 게 통한다.

    그리고 차이는 시즌 드리프트에 불변이다 — 리그 전체가 1%p 내려가면 두 항이 같이
    내려가고 차이는 그대로다. 즉 2019 년에 배운 관계가 2025 년에도 성립한다.
    이게 우리가 전이에서 잃고 있는 바로 그것이다.

    ⚠️ 과거에 기각된 것들과 다르다: `asof_*` 시즌 디트렌드(-13)와 백분위 변환(+0)은
    **수준 정규화**였고, 이건 두 피처 **사이의** 차이다.

    현행 momentum_short/mid 는 prev1-prev3, prev1-prev5 로 **최근끼리의 차이**뿐이고
    '자기 커리어 평균 대비' 는 없다 (step10 확인함).
    """
    P = 'asof_pitcher_success_rate'
    out = {}
    for n in (1, 3, 5):
        c = f'asof_pitcher_prev{n}_game_success_rate'
        if c in D and P in D:
            out[f'd_form{n}'] = D[c] - D[P]            # 최근 폼 vs 자기 커리어
    if 'asof_pitcher_prev3_game_middle_rate' in D and 'asof_pitcher_middle_rate' in D:
        out['d_middle'] = (D['asof_pitcher_prev3_game_middle_rate']
                           - D['asof_pitcher_middle_rate'])
    if 'asof_batter_success_rate' in D and P in D:
        out['d_pb'] = D[P] - D['asof_batter_success_rate']   # 투타 실력차
    if 'smoothed_pitcher_success_rate' in D and P in D:
        out['d_smooth'] = D['smoothed_pitcher_success_rate'] - D[P]
    return out


def cand_diff(ctx, **kw):
    """차이 피처를 얹는다. 캐시의 기존 컬럼만으로 만들어지므로 파이프라인 변경이 없다."""
    Xh, Xv = ctx['Xh'].copy(), ctx['Xv'].copy()
    for k, v in _add_diffs(Xh).items():
        Xh[k] = v
    for k, v in _add_diffs(Xv).items():
        Xv[k] = v
    print(f'    차이 피처 {Xh.shape[1] - ctx["Xh"].shape[1]}개 추가 '
          f'-> {Xh.shape[1]}개', flush=True)
    return cv_predict(Xh, ctx['yh'], Xv, ctx['params'], ctx['cat'])


def cand_bc128(ctx, **kw):
    """결정적 실험: CPU(816)가 GPU(~790)보다 높은 이유가 border_count 인가.

    CatBoost 기본값이 CPU 254 / GPU 128 이다. CPU 를 128 로 낮춰서 790 근처로
    떨어지면 범인 확정이고, 그러면 본 학습(GPU)에 border_count=254 한 줄만 넣어도
    +25 다. 이 축은 이 프로젝트에서 한 번도 건드린 적이 없다.
    """
    return _p(ctx, border_count=128)


def cand_bc512(ctx, **kw):
    """반대 방향. 254 가 좋으면 512 는 더 좋은가 (해상도 축의 기울기)."""
    return _p(ctx, border_count=512)


def cand_optbest(ctx, **kw):
    """GPU Optuna 20 trial 최고 조합을 같은 3-fold CPU 하네스에서 재본다.
    거기서는 대조군 766 -> 805 (+39) 였다."""
    return _p(ctx, learning_rate=0.013514413699827602, depth=6,
              l2_leaf_reg=9.842179645955403, bagging_temperature=0.38201955123496467,
              random_strength=1.0055668048760378, iterations=1400,
              border_count=128, max_ctr_complexity=2)


def _optp(ctx, **over):
    """optbest 조합에서 일부만 되돌린다."""
    p = dict(learning_rate=0.013514413699827602, depth=6,
             l2_leaf_reg=9.842179645955403, bagging_temperature=0.38201955123496467,
             random_strength=1.0055668048760378, iterations=1400,
             border_count=128, max_ctr_complexity=2)
    p.update(over)
    return _p(ctx, **p)


def cand_opt254(ctx, **kw):
    """optbest 에서 border_count 만 CPU 기본값으로."""
    return _optp(ctx, border_count=254)


def cand_opt254c1(ctx, **kw):
    """optbest 에서 bc/ctr 을 둘 다 CPU 기본값으로 = '용량 파라미터만' 남긴 판.

    이게 진짜 질문이다: CPU(MVS + bc254) 하네스에서 **용량 축의 최적점이 다른 자리에
    있는가**. base 는 d8/lr0.0228/it1000, 이건 d6/lr0.0135/it1400 이다.

    ⚠️ 결과가 크게 양수여도 그 자체로는 제출 근거가 못 된다. 이 조합은 총 학습량이
    base 보다 17% 적고 깊이도 2단계 얕다 — 홀드아웃이 체계적으로 가산점을 준다고
    문서화된 **규제 방향**이고, 그 편향이 만든 반복수 500 은 홀드아웃 +30 / 리더보드
    -23 이었다. 이 축은 리더보드로만 판정한다.
    """
    return _optp(ctx, border_count=254, max_ctr_complexity=1)


def cand_seasonbase(ctx, **kw):
    """시즌 절편을 baseline 으로 빼고 트리는 편차만 학습하게 한다.

    `season` 제거가 -424 였다는 건 트리가 **시즌 수준을 배우는 데 상당한 용량을 쓴다**
    는 뜻이다. 그 수준을 절편으로 미리 주면 그 용량이 풀린다.

    lrbase2(-183)와 결정적으로 다른 점: 선형항이 **시즌당 절편 하나**뿐이라
    `smoothed +0.120 / cond_p -0.118` 같은 공선 쌍이 생길 수 없다. 재중심화(+18)가
    이것을 상수 하나로 때운 crude 버전이다.

    ⚠️ 검증 시즌 절편은 정답을 보면 안 된다 — 학습 시즌만으로 외삽한다
    (최근 3년 선형, CLAUDE.md 실측 오차 0.0016).
    """
    sh, yh = ctx['sh'], ctx['yh']
    rate = {int(s): float(yh[sh == s].mean()) for s in np.unique(sh)}
    ss = sorted(rate)
    a, b = np.polyfit(ss[-3:], [rate[s] for s in ss[-3:]], 1)
    nxt = float(np.clip(a * HOLDOUT + b, 0.30, 0.70))
    print(f'    시즌 절편 {[f"{s}:{rate[s]:.4f}" for s in ss]}'
          f' -> {HOLDOUT} 외삽 {nxt:.4f} (실제 {ctx["target"]:.4f})', flush=True)

    def lg(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    bh = lg(np.array([rate[int(s)] for s in sh], dtype=np.float64))
    bv = np.full(len(ctx['Xv']), float(lg(nxt)), dtype=np.float64)
    return cv_predict(ctx['Xh'], yh, ctx['Xv'], ctx['params'], ctx['cat'],
                      baseline=bh, base_v=bv)


def cand_bayes(ctx, **kw):
    """결정적 실험 2: CPU(816) vs GPU(~790) 의 원인이 bootstrap_type 인가.

    CatBoost 기본값이 CPU=MVS / GPU=Bayesian 이고 **MVS 는 CPU 전용**이다.
    우리는 bagging_temperature(=Bayesian 전용 파라미터)를 넘기고 있는데 CPU 는 그걸
    무시하고 MVS 를 쓴다. 즉 같은 코드가 두 기계에서 다른 알고리즘을 돌려왔다.
    CPU 를 강제로 Bayesian 으로 바꿔 790 근처로 떨어지면 범인 확정이고,
    그러면 **본 학습을 CPU 로 돌려야 한다**는 결론이 된다 (GPU 는 MVS 를 못 쓴다).
    """
    return _p(ctx, bootstrap_type='Bayesian')


def cand_mvs_bc128(ctx, **kw):
    """MVS 는 유지하고 border_count 만 GPU 값으로. 두 요인을 분리한다."""
    return _p(ctx, border_count=128, bootstrap_type='MVS')


CANDS = {'base': cand_base, 'diff': cand_diff, 'bayes': cand_bayes, 'mvs_bc128': cand_mvs_bc128,
         'bc128': cand_bc128, 'bc512': cand_bc512,
         'optbest': cand_optbest, 'ctr2': cand_ctr2, 'ctr3': cand_ctr3,
         'onehot': cand_onehot, 'rsm70': cand_rsm70, 'rsm40': cand_rsm40,
         'lrbase': cand_lrbase, 'lrbase2': cand_lrbase2,
         'blend_recent': cand_blend_recent,
         'mono': cand_mono, 'mono2': cand_mono2, 'plain': cand_plain,
         'wseason': cand_wseason, 'wseason1': cand_wseason1,
         'wseason5': cand_wseason5, 'wseason4': cand_wseason4,
         'wseason5b': cand_wseason5b,
         'nogt': cand_nogt, 'dropoldF': cand_dropoldF,
         'pmix': cand_pmix, 'wspmix': cand_wspmix,
         'xprob': cand_xprob, 'wsxprob': cand_wsxprob,
         'ws5gt': cand_ws5gt, 'condgt': cand_condgt,
         'wsbat': cand_wsbat,
         'tskill': cand_tskill,
         'ws5c10': cand_ws5c10, 'ws5c30': cand_ws5c30,
         'ws5c300': cand_ws5c300,
         'ws5nocond': cand_ws5nocond,
         'seasonlast2': cand_seasonlast2, 'seasonpair': cand_seasonpair,
         'condsit': cand_condsit,
         'wsboth': cand_wsboth, 'condb': cand_condb,
         'wsbsit': cand_wsbsit, 'wsbpmix': cand_wsbpmix,
         'wsbtsk': cand_wsbtsk,
         'prevfix': cand_prevfix,
         'bttm': cand_bttm,
         'distill': cand_distill, 'distill5': cand_distill5,
         'bothgt': cand_bothgt,
         'opt254': cand_opt254, 'opt254c1': cand_opt254c1,
         'seasonbase': cand_seasonbase,
         'notm': cand_notm, 'notmstd': cand_notmstd,
         'noflag': cand_noflag,
         'nosh': cand_nosh, 'cnt12': cand_cnt12, 'catx': cand_catx,
         'wsmix': cand_wsmix, 'wsmixd': cand_wsmixd,
         'condfail': cand_condfail, 'condfailc': cand_condfailc,
         'hand4': cand_hand4, 'bsx': cand_bsx, 'cnt12h': cand_cnt12h,
         'cnth': cand_cnth, 'cntg': cand_cntg,
         'auxrev': cand_auxrev, 'aux3': cand_aux3,
         'multicls': cand_multicls, 'auxperm': cand_auxperm,
         'cnt12b': cand_cnt12b, 'catall': cand_catall, 'auxcnt': cand_auxcnt, 'auxrevns': cand_auxrevns,
         'auxns3': cand_auxns3, 'auxns4': cand_auxns4,
         'auxnsb': cand_auxnsb}


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
    ctx['score_mask'] = None
    if DROP_F:
        gt = X.loc[mv, 'game_type'].astype(str).to_numpy()
        ctx['score_mask'] = gt != 'F'
        print(f"⚠️ {HOLDOUT} 검증에서 F(퓨처스) {(~ctx['score_mask']).sum():,}행을 "
              f"채점에서 뺀다 ({(~ctx['score_mask']).mean():.1%})", flush=True)
    ctx['target'] = float(ctx['yv'][ctx['score_mask']].mean() if ctx['score_mask']
                          is not None else ctx['yv'].mean())
    print(f"학습 {mh.sum():,}행(~{HOLDOUT-1}) -> 검증 {mv.sum():,}행({HOLDOUT}) "
          f"| 반복 {iters} | fold {FOLDS}\n", flush=True)

    res = json.load(open(RESULTS, encoding='utf-8')) if os.path.exists(RESULTS) else {}
    for n in names:
        t = time.time()
        print(f'[{n}] 시작', flush=True)
        p = recenter(CANDS[n](ctx), ctx['target'])
        k_ = ctx['score_mask']
        s = skill(p, ctx['yv']) if k_ is None else skill(p[k_], ctx['yv'][k_])
        res[f'{n}@{iters}' + ('' if HOLDOUT == 2024 else f'#{HOLDOUT}')] = round(float(s), 1)
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

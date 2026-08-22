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
         'opt254': cand_opt254, 'opt254c1': cand_opt254c1,
         'seasonbase': cand_seasonbase}


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

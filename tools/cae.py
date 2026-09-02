# CAE (Command Above Expected) — 지표를 만들고 **진짜인지 검증한다**.
#
#   python tools/mk_cae.py            # 먼저 홀드아웃 예측을 만들고 (캐글 GPU)
#   python tools/cae.py [cae_pred.npz]
#
# 정의:  CAE(투수) = mean( 실제 성공 - 모델 기대 성공 )
#        "같은 상황을 던진 평균 투수 대비 얼마나 더 성공시켰나"
#
# 주최측 자료의 STEP 03 이 '투수 제구력 평가' 이고, 문제의 동기가
# "이 투수는 정말 제구가 좋은 투수일까? ERA/BB%/K% 는 결과만 알려준다" 였다.
# 그래서 예측(STEP 01)을 재료로 삼아 상황을 보정한 제구력 지표를 만든다.
#
# ⚠️ 기대값은 `~Y-1 학습 -> Y 예측` 홀드아웃에서만 온다. 배포 모델은 Y 를 학습에
#    포함하므로 그대로 쓰면 암기로 CAE 분산이 부풀고 신뢰도가 가짜로 높아진다.
#
# 지표를 제안하는 것과 **믿을 수 있음을 보이는 것**은 다르다. 그래서 셋을 잰다:
#   ① 반쪽 분할 신뢰도 — 실력인가 잡음인가
#   ② 연도 간 이행    — 작년 CAE 가 올해를 예측하는가 (선수 지표의 진짜 시험)
#   ③ 안정화 표본     — 몇 구부터 믿어도 되는가 (현장이 실제로 묻는 것)
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import relia

NPZ = sys.argv[1] if len(sys.argv) > 1 else 'kout_cae/cae_pred.npz'
OUTD = 'out'
_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
RNG = np.random.default_rng(0)

z = np.load(NPZ, allow_pickle=True)
seasons = sorted({k.split('_')[0] for k in z.files})
print('예측 시즌:', seasons)

tr = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
tr['_i'] = np.arange(len(tr))


# ───────────────────────── 투구 단위 라벨 복원 ─────────────────────────
# asof_pitcher_n 이 투수 내에서 정확히 +1 씩 증가하므로 인접 두 행의 누적 개수
# 차분이 그 투구의 라벨이다. success 는 정답이 있어 **검산이 된다** (eda40/eda63).
def recover(df, keys):
    n = df['asof_pitcher_n'].to_numpy(dtype='float64')
    g = df['pitcher_id'].to_numpy()
    o = np.lexsort((n, g))
    ns, gs = n[o], g[o]
    ok = np.r_[(gs[:-1] == gs[1:]) & (ns[1:] - ns[:-1] == 1), False]
    out = {}
    for k in list(keys) + ['success']:
        cum = np.round(df['asof_pitcher_%s_rate' % k].fillna(0)
                       .to_numpy(dtype='float64')[o] * ns)
        d = np.r_[cum[1:] - cum[:-1], np.nan]
        v = np.where(ok & np.isin(d, [0.0, 1.0]), d, np.nan)
        b = np.full(len(df), np.nan)
        b[o] = v
        out[k] = b
    m = np.isfinite(out['success'])
    acc = float((out['success'][m] == df['control_success'].to_numpy()[m]).mean())
    print('라벨 복원 검산: success 일치율 %.6f (표본 %s)'
          % (acc, format(int(m.sum()), ',')))
    if acc < 0.999:
        raise RuntimeError('복원 검산 실패 %.6f -- 행 정렬을 의심할 것' % acc)
    return out


LAB = recover(tr, ['middle', 'reverse'])


# ─────────────────── 상황 전용 기대값 (CAE 의 기준선) ───────────────────
# ⚠️ 이것이 이 지표의 핵심 설계다. 처음엔 배포 모델의 예측을 기대값으로 썼는데
#    신뢰도가 0.200 으로 원시 성공률(0.589)보다 **낮게** 나왔다. 당연하다 —
#    배포 모델의 입력에 asof_pitcher_* / w_success / cond_p 가 들어 있어서
#    **투수 실력이 이미 기대값에 들어가 있다.** 그걸 빼면 실력이 아니라 잔차만 남는다.
#
# OAA 계열 지표가 하는 것은 그 반대다: **상황만으로** 기대값을 만들고, 실제와의
# 차이를 선수의 몫으로 돌린다. 그래서 투수·타자 정체성과 모든 누적 통계를 뺀
# 상황 전용 모델을 따로 학습한다. 선발과 마무리를, 4월과 9월을 같은 자에 올리는
# 것이 목적이다.
SIT_NUM = ['inning', 'balls_before', 'strikes_before', 'outs_before',
           'score_diff_pitcher_team', 'li', 'runner_on_1b', 'runner_on_2b',
           'runner_on_3b']
SIT_CAT = ['top_bottom', 'base_state', 'batter_hand', 'game_type']


def situation_expectation(hold):
    """~hold-1 로 학습해 hold 시즌의 상황 난이도를 예측한다 (투수 정보 없음)."""
    from catboost import CatBoostClassifier, Pool
    cols = [c for c in SIT_NUM + SIT_CAT if c in tr.columns]
    X = tr[cols].copy()
    for c in SIT_CAT:
        if c in X.columns:
            X[c] = X[c].astype(str)
    m_tr = (tr.season < hold).to_numpy()
    m_va = (tr.season == hold).to_numpy()
    y = tr.control_success.to_numpy()
    cat = [c for c in SIT_CAT if c in X.columns]
    mdl = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.08,
                             loss_function='Logloss', verbose=0,
                             random_seed=42, cat_features=cat)
    mdl.fit(Pool(X[m_tr], y[m_tr], cat_features=cat))
    p = mdl.predict_proba(Pool(X[m_va], cat_features=cat))[:, 1]
    # 시즌 드리프트는 상황이 아니라 리그 수준이다. 기대값의 평균을 그 시즌
    # 실제 평균에 맞춰 수준 차이가 CAE 로 새지 않게 한다 (지표는 시즌 내 비교용).
    p = p + (y[m_va].mean() - p.mean())
    print('  상황 전용 기대값 %d: 학습 %s행 -> 예측 %s행 | 평균 %.4f (실제 %.4f)'
          % (hold, format(int(m_tr.sum()), ','), format(int(m_va.sum()), ','),
             p.mean(), y[m_va].mean()))
    out = np.full(len(tr), np.nan)
    out[m_va] = p
    return out


print('\n상황 전용 기대값 학습 (투수·타자 정체성과 누적 통계 전부 제외)')
SIT = {s: situation_expectation(int(s)) for s in seasons}


# 반쪽 분할 신뢰도는 tools/relia.py 한 곳에만 둔다.
# 그 구현은 tools/cae_selftest.py 가 eda33 의 알려진 값(0.637 / -0.008 / 0.227)을
# 소수 셋째 자리까지 재현하는 것으로 교정돼 있다. 여기서 다시 짜지 않는다.
half_split = relia.reliab_arr


RES = {}
CAE = {}
for s in seasons:
    rid = z['%s_row_id' % s].astype(str)
    pred = z['%s_pred' % s].astype('float64')
    yv = z['%s_y' % s].astype('float64')
    d = pd.DataFrame({'row_id': rid, 'pred': pred, 'y': yv}).merge(
        tr[['row_id', 'pitcher_id', 'batter_id', 'season', 'game_type', 'inning',
            'balls_before', 'strikes_before', 'outs_before', 'base_state',
            'score_diff_pitcher_team', 'li', '_i']], on='row_id', how='left')
    assert d['pitcher_id'].notna().all(), 'row_id 조인 실패'
    assert (d['season'] == int(s)).all(), '시즌 불일치'
    d['exp_sit'] = SIT[s][d['_i'].to_numpy()]  # 상황 전용 기대값
    d['resid'] = d['y'] - d['exp_sit']         # <- CAE 의 원자
    d['resid_full'] = d['y'] - d['pred']       # (대조) 배포 모델 기준 잔차
    for k in ('middle', 'reverse'):
        d[k] = LAB[k][d['_i'].to_numpy()]
    # 시간 순으로 한 번만 정렬한다. 뒤에서 다시 정렬하면 이미 뽑아둔
    # half/pid 배열과 어긋나 조용히 틀린 값이 나온다.
    CAE[s] = d.sort_values('_i').reset_index(drop=True)
    print('\n%s: %s행 | 평균예측 %.4f | 실제 %.4f | 잔차평균 %+.5f'
          % (s, format(len(d), ','), d.pred.mean(), d.y.mean(), d.resid.mean()))

Y = seasons[-1]
d = CAE[Y]
half = RNG.random(len(d)) < 0.5
pid = d.pitcher_id.to_numpy()

# ── ① 신뢰도: CAE 가 원시 성공률보다 나은 특성 추정치인가 ──
print('\n' + '=' * 62)
print('① 반쪽 분할 신뢰도 (%s, 투수당 %d구 이상)' % (Y, 30))
print('%-26s%10s%12s%10s' % ('지표', '신뢰도', '신호 std', '투수수'))
rows = {}
for nm, v in (('원시 성공률', d.y.to_numpy()),
              ('CAE (상황 보정)', d.resid.to_numpy()),
              ('(대조) 배포모델 잔차', d.resid_full.to_numpy())):
    r, sig, n = half_split(v, pid, half)
    rows[nm] = dict(rel=r, sig=sig, n=n)
    print('%-26s%10.3f%12.4f%10d' % (nm, r, sig, n))
RES['reliability'] = rows

# ── ② 연도 간 이행: 작년 CAE 가 올해 CAE 를 예측하는가 ──
if len(seasons) >= 2:
    P = seasons[-2]
    a = CAE[P].groupby('pitcher_id')['resid'].agg(['mean', 'size'])
    b = CAE[Y].groupby('pitcher_id')['resid'].agg(['mean', 'size'])
    j = a.join(b, lsuffix='_p', rsuffix='_y', how='inner')
    ay = CAE[P].groupby('pitcher_id')['y'].mean()
    by = CAE[Y].groupby('pitcher_id')['y'].mean()
    j['raw_p'], j['raw_y'] = ay.reindex(j.index), by.reindex(j.index)
    print('\n② 연도 간 이행  %s -> %s' % (P, Y))
    print('%-26s%12s%10s' % ('지표', '상관', '투수수'))
    tr_rows = {}
    for lo in (50, 100, 150, 200, 300, 500, 800):
        k = j[(j['size_p'] >= lo) & (j['size_y'] >= lo)]
        if len(k) < 20:
            continue
        rc = float(np.corrcoef(k['mean_p'], k['mean_y'])[0, 1])
        rr = float(np.corrcoef(k['raw_p'], k['raw_y'])[0, 1])
        tr_rows['%d구 이상' % lo] = dict(cae=rc, raw=rr, n=len(k))
        print('%-26s CAE %+.3f · 원시 %+.3f   n=%d' % ('%d구 이상' % lo, rc, rr, len(k)))
    RES['transfer'] = tr_rows

# ── ③ 안정화 표본: 몇 구부터 믿어도 되는가 ──
print('\n③ 안정화 표본 (반쪽 분할 신뢰도 vs 최소 투구수)')
print('%-14s%12s%10s' % ('최소 투구수', '신뢰도', '투수수'))
st = {}
for lo in (20, 50, 100, 200, 400, 800):
    r, sig, n = half_split(d.resid.to_numpy(), pid, half, min_n=lo)
    if np.isfinite(r):
        st[lo] = dict(rel=r, n=n)
        print('%-14d%12.3f%10d' % (lo, r, n))
RES['stability'] = st
_ok = [k for k, v in st.items() if v['rel'] >= 0.5]
print('  -> 신뢰도 0.5 를 넘는 최소 표본: %s'
      % (('%d구' % min(_ok)) if _ok else '이 범위에서는 도달 못 함'))

# ── ④ 코치의 질문 넷 ──
# 주최 자료가 던진 네 질문에 어느 것이 측정 가능한지 숫자로 답한다.
print('\n④ 코치의 질문 넷 — 상황축별 CAE 신뢰도')
sd = d.score_diff_pitcher_team.to_numpy()
axes = {
    '주자 (클러치)': d.base_state.astype(str).to_numpy(),
    '카운트 (불리)': np.where(d.balls_before < d.strikes_before, 'P',
                          np.where(d.balls_before > d.strikes_before, 'B', 'N')),
    '이닝 (경기 후반)': pd.cut(d.inning, [0, 3, 6, 99],
                          labels=['1-3', '4-6', '7+']).astype(str).to_numpy(),
    '점수차': pd.cut(pd.Series(sd), [-99, -3, -1, 1, 3, 99]).astype(str).to_numpy(),
}
print('%-20s%12s%12s' % ('축', '축내 신뢰도', '해석'))
qa = {}
for nm, ax in axes.items():
    rr = []
    for v in pd.unique(ax):
        m = ax == v
        if m.sum() < 20000:
            continue
        r, _, n = half_split(d.resid.to_numpy()[m], pid[m], half[m], min_n=20)
        if np.isfinite(r):
            rr.append(r)
    if rr:
        mr = float(np.mean(rr))
        qa[nm] = dict(rel=mr, cells=len(rr))
        print('%-20s%12.3f%12s' % (nm, mr, '측정 가능' if mr > 0.15 else '약하다'))

# 최근 흐름: 시즌 내 전후반 편차가 **실력 변화인가 표본 잡음인가**
#
# ⚠️ 여기서 투수 주효과를 반드시 먼저 뺀다. 안 빼면 H1 과 H2 가 둘 다 그 투수의
#    실력 수준을 담으므로 상관이 높게 나오고 "폼이 측정된다" 는 반대 결론이 된다.
#    잴 것은 '그 투수 자신의 평균 대비 전반/후반이 얼마나 달랐나' 이고, 그 편차를
#    무작위 반쪽 둘에서 따로 재서 서로 재현되는지 본다.
rk = d.groupby('pitcher_id').cumcount()
tot = d.groupby('pitcher_id')['_i'].transform('size')
first = (rk < tot / 2).to_numpy()
d['half_season'] = np.where(first, 'H1', 'H2')
d['h'] = RNG.integers(0, 2, len(d))
r_form, n_form, _o, _s, cov_f = relia.reliab(
    d, ['pitcher_id', 'half_season'], ['pitcher_id'], val='resid')
qa['최근 흐름 (시즌 내 전후반)'] = dict(rel=r_form, cells=n_form, cov=cov_f)
print('%-20s%12.3f%12s' % ('최근 흐름', r_form, '측정 가능' if r_form > 0.15 else '**잡음**'))
print('  (투수 주효과를 뺀 뒤의 전반/후반 편차다. 0 이하면 "최근 폼" 은 실력')
print('   변화가 아니라 표본 잡음이라는 뜻이다. eda33 의 -0.089 와 같은 계산)')
RES['coach_questions'] = qa

# ── ⑤ 실패 유형 프로파일 ──
print('\n⑤ 실패 유형 — "얼마나" 가 아니라 "어떻게" 실패하나')
print('%-16s%12s%12s%10s' % ('유형', '신뢰도', '신호 std', '투수수'))
prof = {}
mi = d['middle'].to_numpy() == 1
re_ = d['reverse'].to_numpy() == 1
fin = np.isfinite(d['middle'].to_numpy()) & np.isfinite(d['reverse'].to_numpy())
yy = d.y.to_numpy()
for nm, v in (('성공', yy),
              ('몰림', np.where(fin, mi & ~re_, np.nan).astype('float64')),
              ('반대방향', np.where(fin, re_ & ~mi, np.nan).astype('float64')),
              ('둘다', np.where(fin, mi & re_, np.nan).astype('float64')),
              ('크게벗어남', np.where(fin, (yy == 0) & ~mi & ~re_, np.nan).astype('float64'))):
    r, sig, n = half_split(v, pid, half, min_n=100)
    if np.isfinite(r):
        prof[nm] = dict(rel=r, sig=sig, n=n)
        print('%-16s%12.3f%12.4f%10d' % (nm, r, sig, n))
RES['failure_profile'] = prof

# ── 저장 ──
os.makedirs(OUTD, exist_ok=True)
json.dump(RES, open(os.path.join(OUTD, 'cae_metrics.json'), 'w'),
          ensure_ascii=False, indent=2, default=float)
# 시즌별 투수 표. exp 는 **상황 전용 기대값**이다 (배포 모델 예측이 아니다).
for _s, _d in CAE.items():
    t = _d.groupby('pitcher_id').agg(
        n=('resid', 'size'), cae=('resid', 'mean'), raw=('y', 'mean'),
        exp=('exp_sit', 'mean')).query('n >= 100').sort_values('cae', ascending=False)
    t.to_csv(os.path.join(OUTD, 'cae_pitchers_%s.csv' % _s), encoding='utf-8')
    print('out/cae_pitchers_%s.csv  투수 %d명' % (_s, len(t)))
tab = pd.read_csv(os.path.join(OUTD, 'cae_pitchers_%s.csv' % Y), index_col=0)
print('\n상위 5 / 하위 5 (300구 이상)')
print(pd.concat([tab.head(5), tab.tail(5)]).round(4).to_string())

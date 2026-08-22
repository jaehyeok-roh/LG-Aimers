# '당해 시즌 성적 복원' 을 본 학습 노트북에 붙인 제출용 노트북을 만든다.
#
# 배경은 tools/screen.py 의 _wseason_cols docstring 참고.
# 핵심: `asof_pitcher_*_rate` 는 **커리어 누적**이라 투수마다 섞인 시즌 수가 다르다.
# 그 행의 (rate x n) 에서 **그 시즌 시작 시점의 커리어 누적**을 빼면 당해 시즌 값이 남는다.
#
# ⚠️ 학습과 추론이 반드시 같은 규칙을 타야 한다 (트랙맨 exact/asof 36점 사고와 같은 유형):
#     학습: 행이 속한 (투수, 시즌) 의 **첫 행** asof 값  = 그 시즌 시작 시점 상태
#     추론: train 마지막 시즌의 그 투수 **마지막 행** asof 값 (+1투구)
#   둘 다 "대상 시즌이 시작될 때의 커리어 상태" 로 동일하다. train 에 없는 투수는 0.
#
# 사용:
#   WS_RATES=success python tools/mk_ws.py      # 성공률만
#   python tools/mk_ws.py                        # 다섯 개 전부 (기본)
import ast, json, os, sys

BASE = os.environ.get('WS_BASE', 'aimers_v9m.ipynb')
OUT = os.environ.get('WS_OUT', 'aimers_v10w.ipynb')
RATES = os.environ.get('WS_RATES', 'success,middle,reverse,ball,strike').split(',')

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
        sys.exit(f'셀 {i}: 앵커 {n}개 기대, {s.count(old)}개\n  {old[:100]}')
    setsrc(i, s.replace(old, new))


# ---------------------------------------------------------------- 1) 피처 함수
FUNCS = f'''

# ======== 당해 시즌 성적 복원 (2026-08-22 EDA) ========
# asof_pitcher_*_rate 는 커리어 누적이라 투수마다 섞인 시즌 수가 다르다.
# 이력 있는 투수에게는 2019(리그 .5647)부터 섞인 낡은 값이고 신규 투수에게는
# 순수한 당해 시즌 값이다 — 2024 단독 예측 스킬이 141 vs 906 으로 갈린다.
# 그 시즌 시작 시점의 커리어 누적을 빼면 모든 투수에게 깨끗한 당해 시즌 값이 남는다.
WS_RATES = {RATES!r}
WS_C = 100.0                      # 표본이 적을 때 리그평균으로 shrink
WS_COLS = ['w_n', 'w_share'] + ['w_' + _k for _k in WS_RATES]


def _ws_rate_cols():
    return ['asof_pitcher_%s_rate' % _k for _k in WS_RATES]


def ws_season_means(df):
    """시즌별 리그평균 (각 비율마다). 디트렌드 기준이 된다."""
    return {{c: df.groupby('season')[c].mean().to_dict() for c in _ws_rate_cols()}}


def ws_next_season_mean(means, target_season):
    """대상 시즌의 리그평균을 학습 시즌만으로 외삽 (최근 3시즌 선형).
    2024 를 이 방식으로 맞히면 오차 0.0016 이다 (CLAUDE.md 1장)."""
    out = {{}}
    for c, d in means.items():
        ss = sorted(d)
        k = ss[-3:]
        a, b = np.polyfit(k, [d[s] for s in k], 1)
        out[c] = float(np.clip(a * target_season + b, 0.0, 1.0))
    return out


def ws_apply(df, prior_n, prior_x, lg):
    """행별로 당해 시즌 값을 복원해 WS_COLS 를 붙인다.

    prior_n : Series/array — 그 행의 '대상 시즌 시작 시점' 커리어 투구수
    prior_x : dict[rate_col] -> array — 같은 시점의 누적 개수
    lg      : dict[rate_col] -> 그 행이 속한 시즌의 리그평균 (array 또는 scalar)
    """
    n = df['asof_pitcher_n'].to_numpy(dtype='float64')
    wn = np.maximum(n - np.asarray(prior_n, dtype='float64'), 0.0)
    df['w_n'] = wn
    df['w_share'] = wn / np.maximum(n, 1.0)
    for c, k in zip(_ws_rate_cols(), WS_RATES):
        x = (df[c].fillna(0).to_numpy(dtype='float64') * n).round()
        wx = np.clip(x - np.asarray(prior_x[c], dtype='float64'), 0.0, wn)
        base = np.asarray(lg[c], dtype='float64')
        df['w_' + k] = (wx + base * WS_C) / (wn + WS_C) - base
    return df


def attach_wseason(df):
    """학습용. 각 행을 '그 시즌 시작 시점' 기준으로 분해한다 (leak-free)."""
    df = df.copy()
    means = ws_season_means(df)
    order = np.argsort(df['asof_pitcher_n'].to_numpy(dtype='float64'), kind='stable')
    first = df.iloc[order].groupby(['pitcher_id', 'season'], sort=False).head(1)
    key = pd.MultiIndex.from_arrays([df['pitcher_id'], df['season']])
    fi = first.set_index(['pitcher_id', 'season'])
    n0 = fi['asof_pitcher_n'].reindex(key).to_numpy(dtype='float64')
    px = {{c: (fi[c].fillna(0).reindex(key).to_numpy(dtype='float64') * n0).round()
          for c in _ws_rate_cols()}}
    lg = {{c: df['season'].map(means[c]).to_numpy(dtype='float64') for c in _ws_rate_cols()}}
    df = ws_apply(df, n0, px, lg)
    print("  당해시즌 복원: 투구수 중앙값 %.0f | " % np.nanmedian(df['w_n'])
          + " ".join("%s=%+.4f" % (k, np.nanmean(df['w_' + k])) for k in WS_RATES))
    return df


def build_pitcher_prior(df):
    """추론용 룩업. 각 투수의 **마지막 학습 시즌 끝** 시점 커리어 상태.
    test 행의 asof 에서 이걸 빼면 대상 시즌(2025) 값이 남는다."""
    order = np.argsort(df['asof_pitcher_n'].to_numpy(dtype='float64'), kind='stable')
    last = df.iloc[order].groupby('pitcher_id', sort=False).tail(1)
    n = last['asof_pitcher_n'].to_numpy(dtype='float64')
    out = pd.DataFrame({{'pitcher_id': last['pitcher_id'].to_numpy(),
                        'w_n0': n + 1.0}})       # 그 마지막 투구 자신도 포함
    for c in _ws_rate_cols():
        out['w_x0__' + c] = (last[c].fillna(0).to_numpy(dtype='float64') * n).round()
    return out.reset_index(drop=True)
'''

i3 = code[3]
sub(i3, "COND_COLS = [name for _, _, name in COND_SPECS]",
    "COND_COLS = [name for _, _, name in COND_SPECS]\n" + FUNCS)

# ---------------------------------------------------------------- 2) 파이프라인
i4 = code[4]
sub(i4, "        df_proc = attach_cond_features(df_proc)",
    "        df_proc = attach_cond_features(df_proc)\n"
    "        df_proc = attach_wseason(df_proc)")

# ---------------------------------------------------------------- 3) 룩업 저장
i7 = code[7]
SAVE = '''
# ---- 당해 시즌 복원용 룩업 + 대상 시즌 리그평균 외삽 ----
_pp = build_pitcher_prior(df_train)
_pp.to_csv("model/pitcher_prior.csv", index=False)
_tgt = int(df_train['season'].max()) + 1
_ws_lg = ws_next_season_mean(ws_season_means(df_train), _tgt)
with open("model/train_constants.json", "r") as f:
    _tc = json.load(f)
_tc["ws_target_season"] = _tgt
_tc["ws_league_mean"] = _ws_lg
_tc["ws_rates"] = WS_RATES
_tc["ws_C"] = WS_C
with open("model/train_constants.json", "w") as f:
    json.dump(_tc, f)
print(f"pitcher_prior.csv  {len(_pp):,}행 | {_tgt} 리그평균 외삽 "
      + " ".join(f"{k.split('_')[-2]}={v:.4f}" for k, v in _ws_lg.items()))
'''
setsrc(i7, src(i7) + SAVE)

# ---------------------------------------------------------------- 4) script.py
i11 = code[11]
INFER = '''
    # ---------- 당해 시즌 성적 복원 ----------
    # 학습과 같은 규칙: '대상 시즌이 시작될 때의 커리어 상태' 를 빼서 당해 시즌만 남긴다.
    # 학습은 (투수, 시즌) 첫 행에서, 추론은 train 마지막 시즌 마지막 행에서 그 값을 얻는다.
    _ws_rates = _tc.get("ws_rates")
    if _ws_rates:
        _wsC = float(_tc.get("ws_C", 100.0))
        _wslg = _tc["ws_league_mean"]
        _pp = pd.read_csv(os.path.join("model", "pitcher_prior.csv"),
                          keep_default_na=False, na_values=_NA)
        _pp["pitcher_id"] = _pp["pitcher_id"].astype(df_proc["pitcher_id"].dtype)
        _nb = len(df_proc)
        df_proc = df_proc.merge(_pp, on="pitcher_id", how="left")
        if len(df_proc) != _nb:
            raise RuntimeError("pitcher_prior 병합에서 행 수가 변함")
        _n = df_proc["asof_pitcher_n"].to_numpy(dtype="float64")
        _n0 = df_proc["w_n0"].fillna(0.0).to_numpy(dtype="float64")   # 신규 투수는 0
        _wn = np.maximum(_n - _n0, 0.0)
        df_proc["w_n"] = _wn
        df_proc["w_share"] = _wn / np.maximum(_n, 1.0)
        for _k in _ws_rates:
            _c = "asof_pitcher_%s_rate" % _k
            _x = (df_proc[_c].fillna(0).to_numpy(dtype="float64") * _n).round()
            _x0 = df_proc["w_x0__" + _c].fillna(0.0).to_numpy(dtype="float64")
            _wx = np.clip(_x - _x0, 0.0, _wn)
            _b = float(_wslg[_c])
            df_proc["w_" + _k] = (_wx + _b * _wsC) / (_wn + _wsC) - _b
        df_proc = df_proc.drop(columns=[c for c in df_proc.columns
                                        if c == "w_n0" or c.startswith("w_x0__")])
        print("당해시즌 복원 완료: 투구수 중앙값 %.0f / 신규투수 비율 %.1f%%"
              % (float(np.median(_wn)), 100.0 * float((_n0 == 0).mean())))

'''
sub(i11, "    df_proc = step14_convert_to_category(df_proc)",
    INFER + "    df_proc = step14_convert_to_category(df_proc)")

# script.py 는 prior_mean 하나만 뽑아 쓴다 -> 딕셔너리 전체를 _tc 로 남긴다
_OLD_TC = '        prior_mean = float(json.load(f)["prior_mean"])'
_NEW_TC = ('        _tc = json.load(f)\n'
           '    prior_mean = float(_tc["prior_mean"])')
sub(i11, _OLD_TC, _NEW_TC)

# ---------------------------------------------------------------- 5) zip 목록
i12 = code[12]
_OLD_REQ = '       if os.path.exists(f"model/{n}.csv")]\n)'
_NEW_REQ = ('       if os.path.exists(f"model/{n}.csv")]\n'
            '    + (["model/pitcher_prior.csv"]\n'
            '       if os.path.exists("model/pitcher_prior.csv") else [])\n)')
sub(i12, _OLD_REQ, _NEW_REQ)

for _i in code:
    if 'ZIP_PATH = "submit_tuned.zip"' in src(_i):
        sub(_i, 'ZIP_PATH = "submit_tuned.zip"', 'ZIP_PATH = "submit_v10w.zip"')
        break

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'{OUT} 생성 — 비율 {RATES}, 코드셀 {len(code)}개 문법 OK')

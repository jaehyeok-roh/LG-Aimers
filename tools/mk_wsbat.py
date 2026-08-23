# 타자측 '당해 시즌 성적 복원' 을 v10w 노트북 위에 얹는다.
#
# `asof_batter_success_rate` / `_middle_rate` 도 분모가 `asof_batter_n` 인
# **커리어 누적**이라 투수측과 구조가 완전히 같다. mk_ws.py 의 직역이다.
#
# ⚠️ 판정이 시즌마다 갈린다:  2024 홀드아웃 +3.0 / 2023 홀드아웃 +71.7
#    두 시즌이 정면으로 어긋나 리더보드로만 가를 수 있다. 그래서 만든다.
import ast
import json
import os
import sys

BASE = os.environ.get('WB_BASE', 'aimers_v10w.ipynb')
OUT = os.environ.get('WB_OUT', 'aimers_v10wb.ipynb')
RATES = ['success', 'middle']

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

# ======== 타자측 당해 시즌 복원 (2026-08-23) ========
# 투수측(w_*)과 완전히 같은 수법. asof_batter_*_rate 도 커리어 누적이다.
WB_RATES = {RATES!r}
WB_C = 100.0
WB_COLS = ['wb_n', 'wb_share'] + ['wb_' + _k for _k in WB_RATES]


def _wb_rate_cols():
    return ['asof_batter_%s_rate' % _k for _k in WB_RATES]


def wb_season_means(df):
    return {{c: df.groupby('season')[c].mean().to_dict() for c in _wb_rate_cols()}}


def wb_apply(df, prior_n, prior_x, lg):
    n = df['asof_batter_n'].to_numpy(dtype='float64')
    wn = np.maximum(n - np.asarray(prior_n, dtype='float64'), 0.0)
    df['wb_n'] = wn
    df['wb_share'] = wn / np.maximum(n, 1.0)
    for c, k in zip(_wb_rate_cols(), WB_RATES):
        x = (df[c].fillna(0).to_numpy(dtype='float64') * n).round()
        wx = np.clip(x - np.asarray(prior_x[c], dtype='float64'), 0.0, wn)
        base = np.asarray(lg[c], dtype='float64')
        df['wb_' + k] = (wx + base * WB_C) / (wn + WB_C) - base
    return df


def attach_wsbat(df):
    """학습용. 각 행을 '그 시즌 시작 시점' 기준으로 분해 (leak-free)."""
    df = df.copy()
    means = wb_season_means(df)
    order = np.argsort(df['asof_batter_n'].to_numpy(dtype='float64'), kind='stable')
    first = df.iloc[order].groupby(['batter_id', 'season'], sort=False).head(1)
    key = pd.MultiIndex.from_arrays([df['batter_id'], df['season']])
    fi = first.set_index(['batter_id', 'season'])
    n0 = fi['asof_batter_n'].reindex(key).to_numpy(dtype='float64')
    px = {{c: (fi[c].fillna(0).reindex(key).to_numpy(dtype='float64') * n0).round()
          for c in _wb_rate_cols()}}
    lg = {{c: df['season'].map(means[c]).to_numpy(dtype='float64')
          for c in _wb_rate_cols()}}
    df = wb_apply(df, n0, px, lg)
    print("  타자 당해시즌 복원: 타석수 중앙값 %.0f | " % np.nanmedian(df['wb_n'])
          + " ".join("%s=%+.4f" % (k, np.nanmean(df['wb_' + k])) for k in WB_RATES))
    return df


def build_batter_prior(df):
    """추론용 룩업. 각 타자의 마지막 학습 시즌 끝 시점 커리어 상태."""
    order = np.argsort(df['asof_batter_n'].to_numpy(dtype='float64'), kind='stable')
    last = df.iloc[order].groupby('batter_id', sort=False).tail(1)
    n = last['asof_batter_n'].to_numpy(dtype='float64')
    out = pd.DataFrame({{'batter_id': last['batter_id'].to_numpy(),
                        'wb_n0': n + 1.0}})
    for c in _wb_rate_cols():
        out['wb_x0__' + c] = (last[c].fillna(0).to_numpy(dtype='float64') * n).round()
    return out.reset_index(drop=True)
'''
sub(code[3], "def attach_wseason(df):", FUNCS.rstrip() + "\n\n\ndef attach_wseason(df):")

# ---------------------------------------------------------------- 2) 파이프라인
sub(code[4], "        df_proc = attach_wseason(df_proc)",
    "        df_proc = attach_wseason(df_proc)\n"
    "        df_proc = attach_wsbat(df_proc)")

# ---------------------------------------------------------------- 3) 룩업 저장
SAVE = '''
# ---- 타자측 당해 시즌 복원용 룩업 ----
_bp = build_batter_prior(df_train)
_bp.to_csv("model/batter_prior.csv", index=False)
_wb_lg = ws_next_season_mean(wb_season_means(df_train), _tgt)
with open("model/train_constants.json", "r") as f:
    _tc = json.load(f)
_tc["wb_league_mean"] = _wb_lg
_tc["wb_rates"] = WB_RATES
_tc["wb_C"] = WB_C
with open("model/train_constants.json", "w") as f:
    json.dump(_tc, f)
print(f"batter_prior.csv  {len(_bp):,}행 | "
      + " ".join(f"{k}={v:.4f}" for k, v in _wb_lg.items()))
'''
setsrc(code[7], src(code[7]) + SAVE)

# 오프셋 셀이 파일을 덮어쓰므로 wb_* 도 같이 다시 넣는다 (ws_* 와 같은 사고 경로)
_A = '_tc["ws_C"] = WS_C\nwith open("model/train_constants.json", "w") as f:'
_B = ('_tc["ws_C"] = WS_C\n'
      '_tc["wb_league_mean"] = ws_next_season_mean('
      'wb_season_means(df_train), _tc["ws_target_season"])\n'
      '_tc["wb_rates"] = WB_RATES\n'
      '_tc["wb_C"] = WB_C\n'
      'with open("model/train_constants.json", "w") as f:')
sub(code[10], _A, _B)
_ASSERT = ('assert "ws_league_mean" in json.load(open("model/train_constants.json")), '
           '"ws 상수 유실"')
sub(code[10], _ASSERT, _ASSERT + '\n' +
    'assert "wb_league_mean" in json.load(open("model/train_constants.json")), '
    '"wb 상수 유실"')

# ---------------------------------------------------------------- 4) script.py
INFER = '''
    # ---------- 타자측 당해 시즌 복원 ----------
    _wb_rates = _tc.get("wb_rates")
    if _wb_rates:
        _wbC = float(_tc.get("wb_C", 100.0))
        _wblg = _tc["wb_league_mean"]
        _bp = pd.read_csv(os.path.join("model", "batter_prior.csv"),
                          keep_default_na=False, na_values=_NA)
        _bp["batter_id"] = _bp["batter_id"].astype(df_proc["batter_id"].dtype)
        _nb2 = len(df_proc)
        df_proc = df_proc.merge(_bp, on="batter_id", how="left")
        if len(df_proc) != _nb2:
            raise RuntimeError("batter_prior 병합에서 행 수가 변함")
        _n = df_proc["asof_batter_n"].to_numpy(dtype="float64")
        _n0 = df_proc["wb_n0"].fillna(0.0).to_numpy(dtype="float64")
        _wn = np.maximum(_n - _n0, 0.0)
        df_proc["wb_n"] = _wn
        df_proc["wb_share"] = _wn / np.maximum(_n, 1.0)
        for _k in _wb_rates:
            _c = "asof_batter_%s_rate" % _k
            _x = (df_proc[_c].fillna(0).to_numpy(dtype="float64") * _n).round()
            _x0 = df_proc["wb_x0__" + _c].fillna(0.0).to_numpy(dtype="float64")
            _wx = np.clip(_x - _x0, 0.0, _wn)
            _b = float(_wblg[_c])
            df_proc["wb_" + _k] = (_wx + _b * _wbC) / (_wn + _wbC) - _b
        df_proc = df_proc.drop(columns=[c for c in df_proc.columns
                                        if c == "wb_n0" or c.startswith("wb_x0__")])
        print("타자 당해시즌 복원 완료: 타석수 중앙값 %.0f / 신규타자 비율 %.1f%%"
              % (float(np.median(_wn)), 100.0 * float((_n0 == 0).mean())))

'''
sub(code[11], "    df_proc = step14_convert_to_category(df_proc)",
    INFER + "    df_proc = step14_convert_to_category(df_proc)")

# ---------------------------------------------------------------- 5) zip 목록
_OLD_REQ = ('    + (["model/pitcher_prior.csv"]\n'
            '       if os.path.exists("model/pitcher_prior.csv") else [])\n)')
sub(code[12], _OLD_REQ,
    _OLD_REQ[:-2] + '\n'
    '    + (["model/batter_prior.csv"]\n'
    '       if os.path.exists("model/batter_prior.csv") else [])\n)')

for _i in code:
    if 'ZIP_PATH = "submit_v10w.zip"' in src(_i):
        sub(_i, 'ZIP_PATH = "submit_v10w.zip"', 'ZIP_PATH = "submit_v10wb.zip"')
        break

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'{OUT} 생성 — 타자 비율 {RATES}, 코드셀 {len(code)}개 문법 OK')

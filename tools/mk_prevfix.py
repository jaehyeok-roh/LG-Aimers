# prev-game 지표의 시즌 경계 오염 보정을 v10wb 노트북 위에 얹는다.
#
# `asof_pitcher_prev{1,3,5}_game_*` 는 시즌 경계를 넘는다 (실측):
#   prev5 에 작년분 섞인 행 22.6% / prev3 14.5% / prev1 5.3%
#   당해 0~60구 & 이력있는 투수 82,826행에서 prev5 단독 스킬 **-3,224**
#
# 오염이 두 겹인데 2026-08-22 시도는 (A)만 고쳐 +6 이었다:
#   (A) 수준   — 작년 값이 작년 리그 수준을 달고 온다
#   (B) 관련성 — 작년 마지막 5경기는 올해 폼과 사실상 무관하다
# 그 측정은 wseason5 **이전**이라 모델이 w_n 을 몰랐다 = (B)를 배울 재료가 없었다.
#
# 여기서는 (B)를 직접 준다:
#   등판수추정 = w_n / (그 투수의 등판당 평균 투구수, train 룩업)
#   cross_N    = clip(N - 등판수추정, 0, N) / N
# 자기 행의 asof + train 룩업만 쓰므로 규정상 안전하다 (wseason 과 같은 논리).
import ast
import json
import os
import sys

BASE = os.environ.get('PF_BASE', 'aimers_v10wb.ipynb')
OUT = os.environ.get('PF_OUT', 'aimers_v10wp.ipynb')

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
FUNCS = '''

# ======== prev-game 시즌 경계 보정 (2026-08-23) ========
# prev5 에 작년분이 섞인 행이 22.6% 다. 그 구간에서 이 피처는 적극적인 독이다
# (당해 0~60구 & 이력있는 투수 82,826행에서 prev5 단독 스킬 -3,224).
PF_SPEC = [('asof_pitcher_prev1_game_success_rate', 1),
           ('asof_pitcher_prev3_game_success_rate', 3),
           ('asof_pitcher_prev5_game_success_rate', 5),
           ('asof_pitcher_prev1_game_middle_rate', 1),
           ('asof_pitcher_prev3_game_middle_rate', 3),
           ('asof_pitcher_prev5_game_middle_rate', 5)]
PF_COLS = (['pf_gest'] + ['pf_cross%d' % w for w in (1, 3, 5)]
           + ['pf_' + c.replace('asof_pitcher_', '') for c, _ in PF_SPEC])


def pf_build_appearance(df):
    """추론용 룩업: 투수별 **등판당 평균 투구수**.

    train 을 경기 단위로 복원해서 센다. (season, month, dow, game_type) 변화
    또는 이닝 하락에서 자르면 R 이 시즌마다 정확히 720경기가 나온다 (KBO 정규시즌).
    ⚠️ 복원은 **원본 행 순서**에 의존한다 — 정렬된 프레임을 넣으면 안 된다.
    """
    k = df[['season', 'game_month', 'game_dayofweek', 'game_type']].astype(str).agg(
        '|'.join, axis=1)
    gid = ((k != k.shift()) | (df['inning'].diff() < 0)).cumsum()
    ap = df.assign(_g=gid).groupby(['pitcher_id', '_g']).size()
    avg = ap.groupby(level=0).mean().rename('pf_avg_pa').reset_index()
    print("  경기 복원 %d개 | 등판 %d개 | 등판당 투구수 중앙 %.1f"
          % (gid.nunique(), len(ap), avg['pf_avg_pa'].median()))
    return avg


def pf_season_means(df):
    return {c: df.groupby('season')[c].mean().to_dict() for c, _ in PF_SPEC}


def pf_apply(df, avg_pa, lg_cur, lg_prv):
    """행별로 prev-game 지표를 보정한다.

    avg_pa : array — 그 투수의 등판당 평균 투구수
    lg_cur : dict[col] -> array/scalar — 그 행이 속한 시즌의 리그평균
    lg_prv : dict[col] -> array/scalar — 그 직전 시즌의 리그평균
    """
    wn = df['w_n'].to_numpy(dtype='float64')          # wseason 이 만들어둔 당해 투구수
    gest = wn / np.maximum(np.asarray(avg_pa, dtype='float64'), 1.0)
    df['pf_gest'] = gest
    seen = set()
    for c, w in PF_SPEC:
        cross = np.clip(w - gest, 0.0, w) / w
        if w not in seen:
            df['pf_cross%d' % w] = cross              # (B) 관련성: 작년분 비율
            seen.add(w)
        a = np.asarray(lg_cur[c], dtype='float64')
        b = np.asarray(lg_prv[c], dtype='float64')
        b = np.where(np.isfinite(b), b, a)
        blend = a * (1.0 - cross) + b * cross         # (A) 수준: 가리키는 시즌으로
        df['pf_' + c.replace('asof_pitcher_', '')] = \\
            df[c].to_numpy(dtype='float64') - blend
    return df


def attach_prevfix(df):
    """학습용. attach_wseason 뒤에 불러야 한다 (w_n 이 필요하다)."""
    df = df.copy()
    means = pf_season_means(df)
    avg = pf_build_appearance(df).set_index('pitcher_id')['pf_avg_pa']
    a = df['pitcher_id'].map(avg).to_numpy(dtype='float64')
    a = np.where(np.isfinite(a), a, np.nanmedian(a))
    lg_cur = {c: df['season'].map(means[c]).to_numpy(dtype='float64')
              for c, _ in PF_SPEC}
    lg_prv = {c: df['season'].sub(1).map(means[c]).to_numpy(dtype='float64')
              for c, _ in PF_SPEC}
    df = pf_apply(df, a, lg_cur, lg_prv)
    print("  prev-game 보정: 작년분 섞인 행 "
          + " ".join("prev%d %.1f%%" % (w, 100.0 * (df['pf_cross%d' % w] > 0).mean())
                     for w in (1, 3, 5)))
    return df
'''

sub(code[3], "def attach_wsbat(df):", FUNCS.rstrip() + "\n\n\ndef attach_wsbat(df):")

# ---------------------------------------------------------------- 2) 파이프라인
sub(code[4], "        df_proc = attach_wsbat(df_proc)",
    "        df_proc = attach_wsbat(df_proc)\n"
    "        df_proc = attach_prevfix(df_proc)")

# ---------------------------------------------------------------- 3) 룩업 저장
SAVE = '''
# ---- prev-game 보정용 룩업 + 대상 시즌 리그평균 외삽 ----
_pf = pf_build_appearance(df_train)
_pf.to_csv("model/pitcher_appearance.csv", index=False)
_pf_means = pf_season_means(df_train)
_pf_cur = ws_next_season_mean(_pf_means, _tgt)                 # 2025 외삽
_pf_prv = {c: float(_pf_means[c][max(_pf_means[c])]) for c, _ in PF_SPEC}  # 2024 실측
with open("model/train_constants.json", "r") as f:
    _tc = json.load(f)
_tc["pf_league_cur"] = _pf_cur
_tc["pf_league_prev"] = _pf_prv
_tc["pf_spec"] = [[c, w] for c, w in PF_SPEC]
with open("model/train_constants.json", "w") as f:
    json.dump(_tc, f)
print(f"pitcher_appearance.csv  {len(_pf):,}행 | {_tgt} 리그평균 외삽 완료")
'''
setsrc(code[7], src(code[7]) + SAVE)

# 오프셋 셀이 파일을 덮어쓰므로 pf_* 도 다시 넣는다 (ws_*/wb_* 와 같은 사고 경로)
_A = '_tc["wb_C"] = WB_C\nwith open("model/train_constants.json", "w") as f:'
_B = ('_tc["wb_C"] = WB_C\n'
      '_pf_m = pf_season_means(df_train)\n'
      '_tc["pf_league_cur"] = ws_next_season_mean(_pf_m, _tc["ws_target_season"])\n'
      '_tc["pf_league_prev"] = {c: float(_pf_m[c][max(_pf_m[c])]) for c, _ in PF_SPEC}\n'
      '_tc["pf_spec"] = [[c, w] for c, w in PF_SPEC]\n'
      'with open("model/train_constants.json", "w") as f:')
sub(code[10], _A, _B)
_ASSERT = ('assert "wb_league_mean" in json.load(open("model/train_constants.json")), '
           '"wb 상수 유실"')
sub(code[10], _ASSERT, _ASSERT + '\n' +
    'assert "pf_league_cur" in json.load(open("model/train_constants.json")), '
    '"pf 상수 유실"')

# ---------------------------------------------------------------- 4) script.py
INFER = '''
    # ---------- prev-game 시즌 경계 보정 ----------
    # 반드시 당해 시즌 복원(w_n) 뒤에 와야 한다.
    _pf_spec = _tc.get("pf_spec")
    if _pf_spec:
        _pfc, _pfp = _tc["pf_league_cur"], _tc["pf_league_prev"]
        _pa = pd.read_csv(os.path.join("model", "pitcher_appearance.csv"),
                          keep_default_na=False, na_values=_NA)
        _pa["pitcher_id"] = _pa["pitcher_id"].astype(df_proc["pitcher_id"].dtype)
        _n3 = len(df_proc)
        df_proc = df_proc.merge(_pa, on="pitcher_id", how="left")
        if len(df_proc) != _n3:
            raise RuntimeError("pitcher_appearance 병합에서 행 수가 변함")
        _av = df_proc["pf_avg_pa"].to_numpy(dtype="float64")
        _med = float(np.nanmedian(_av))
        _av = np.where(np.isfinite(_av), _av, _med)          # 신규 투수는 중앙값
        _wn = df_proc["w_n"].to_numpy(dtype="float64")
        _gest = _wn / np.maximum(_av, 1.0)
        df_proc["pf_gest"] = _gest
        _seen = set()
        for _c, _w in _pf_spec:
            _cross = np.clip(_w - _gest, 0.0, _w) / _w
            if _w not in _seen:
                df_proc["pf_cross%d" % _w] = _cross
                _seen.add(_w)
            _a, _b = float(_pfc[_c]), float(_pfp[_c])
            _blend = _a * (1.0 - _cross) + _b * _cross
            df_proc["pf_" + _c.replace("asof_pitcher_", "")] = \\
                df_proc[_c].to_numpy(dtype="float64") - _blend
        df_proc = df_proc.drop(columns=["pf_avg_pa"])
        print("prev-game 보정 완료: 작년분 섞인 행 "
              + " ".join("prev%d %.1f%%" % (_w, 100.0 * float(
                  (df_proc["pf_cross%d" % _w] > 0).mean())) for _w in (1, 3, 5)))

'''
sub(code[11], "    df_proc = step14_convert_to_category(df_proc)",
    INFER + "    df_proc = step14_convert_to_category(df_proc)")

# ---------------------------------------------------------------- 5) zip 목록
_OLD_REQ = ('    + (["model/batter_prior.csv"]\n'
            '       if os.path.exists("model/batter_prior.csv") else [])\n)')
sub(code[12], _OLD_REQ,
    _OLD_REQ[:-2] + '\n'
    '    + (["model/pitcher_appearance.csv"]\n'
    '       if os.path.exists("model/pitcher_appearance.csv") else [])\n)')

for _i in code:
    if 'ZIP_PATH = "submit_v10wb.zip"' in src(_i):
        sub(_i, 'ZIP_PATH = "submit_v10wb.zip"', 'ZIP_PATH = "submit_v10wp.zip"')
        break

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'{OUT} 생성 — prev-game 보정 {len(code)}셀 문법 OK')

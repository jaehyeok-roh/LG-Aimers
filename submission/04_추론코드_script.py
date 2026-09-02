import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import traceback
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ================= 학습과 문자 단위로 동일한 전처리 =================

def step1_basic_features(df):
    df_proc = df.copy()
    df_proc['is_weekend_day_game'] = np.where(
        (df_proc['game_month'].isin([4, 5, 9, 10])) & (df_proc['game_dayofweek'].isin([5, 6])), 1.0, 0.0)
    df_proc['is_heat_wave_game'] = np.where(df_proc['game_month'].isin([7, 8]), 1.0, 0.0)
    return df_proc


def step2_pitcher_role_features(df):
    df_proc = df.copy()
    df_proc['is_pure_starter'] = np.where(df_proc['inning'] == 1, 1.0, 0.0)
    df_proc['is_long_relief'] = np.where(
        (df_proc['inning'] > 1) & (df_proc['asof_pitcher_n'] >= (df_proc['inning'] - 1) * 12), 1.0, 0.0)
    df_proc['is_short_relief'] = np.where(
        (df_proc['inning'] > 1) & (df_proc['asof_pitcher_n'] < (df_proc['inning'] - 1) * 12), 1.0, 0.0)
    return df_proc


def step3_matchup_features(df):
    df_proc = df.copy()
    if 'pitcher_hand' in df_proc.columns and 'batter_hand' in df_proc.columns:
        df_proc['is_same_hand'] = np.where(df_proc['pitcher_hand'] == df_proc['batter_hand'], 1.0, 0.0)
    return df_proc


def step4_refined_count_features(df):
    df_proc = df.copy()
    b, s = df_proc['balls_before'], df_proc['strikes_before']
    df_proc['is_first_pitch'] = np.where((b == 0) & (s == 0), 1.0, 0.0)
    df_proc['is_full_count'] = np.where((b == 3) & (s == 2), 1.0, 0.0)
    pitcher_ahead = ((b == 0) & (s == 1)) | ((b == 0) & (s == 2)) | ((b == 1) & (s == 2))
    batter_ahead = ((b == 1) & (s == 0)) | ((b == 2) & (s == 0)) | ((b == 3) & (s == 0)) | ((b == 2) & (s == 1)) | ((b == 3) & (s == 1))
    neutral = ((b == 1) & (s == 1)) | ((b == 2) & (s == 2))
    df_proc['count_advantage'] = np.select(
        [pitcher_ahead, batter_ahead, neutral], ['Pitcher', 'Batter', 'Neutral'], default='None')
    # 정확한 12칸 카운트. count_advantage 의 'None' 이 0-0 과 3-2 를 같이 담고
    # is_first_pitch / is_full_count 가 그걸 되돌리려는 땜질이라, 12칸을 직접 준다.
    # 대칭트리는 balls/strikes 두 레벨로 이걸 만들 수 **있지만** 비싸다 —
    # 범주형으로 주면 CTR 하나로 끝난다 (nosh -17.0/-13.0 이 이 축을 증명했다).
    df_proc['cnt12'] = (b.astype(int).astype(str) + '-' + s.astype(int).astype(str))
    # 팀13 관여 x 2023-05 체제 전환 (tools/eda51.py). 두 컬럼의 OR 이라 대칭트리는
    # 레벨 두 개를 써야 만든다 — nosh(-17.0)가 증명한 자리. 수치형으로 준다(CTR 경쟁 회피).
    _t13 = ((df_proc['pitcher_team_id'].astype('int64') == 13)
            | (df_proc['batter_team_id'].astype('int64') == 13))
    df_proc['t13'] = _t13.astype('float64')
    # pitcher_hand x pitcher_team (20칸). eda53: t13 을 빼고도 강도 42.3 이고
    # 2023<->2024 상관 0.40 으로 유일하게 살아남은 조합이다.
    df_proc['phteam'] = (df_proc['pitcher_hand'].astype(str) + '|'
                         + df_proc['pitcher_team_id'].astype(str))
    # 전환 시점은 train 라벨 통계에서 고정한 상수다 (2023-04 -0.0220 -> 2023-05 +0.0469).
    # 2025 행은 season>2023 이라 전부 1 -> 학습 시대에 얼어붙는 성분이 없다.
    _sea = df_proc['season'].astype('int64')
    _mon = df_proc['game_month'].astype('int64')
    _post = ((_sea > 2023)
             | ((_sea == 2023) & ((_mon >= 5) | (df_proc['game_type'].astype(str) == 'F'))))
    df_proc['t13_post'] = (_t13 & _post).astype('float64')
    # 전 팀 관여 OR 플래그 (mk_teamor). t13 의 일반화 -- 시점 게이트는 없다.
    # eda56: 계단 변화가 있는 팀은 13 하나뿐이고, 나머지는 정적 효과만 있다.
    _pti = df_proc['pitcher_team_id'].astype('int64')
    _bti = df_proc['batter_team_id'].astype('int64')
    for _tk in [12, 14, 15, 16, 17, 18, 19, 20, 21]:
        df_proc['tor%d' % _tk] = ((_pti == _tk) | (_bti == _tk)).astype('float64')
    # hand4 = pitcher_hand x batter_hand (4칸). is_same_hand 는 좌투vs우타와
    # 우투vs좌타를 한 칸에 뭉갠다. 단독으로는 뒤집히지만(2024 +6.5 / 2023 -11.4)
    # cnt12 와 함께면 두 시즌 다 양수다 (cnt12h +19.0 / +4.3, 조합 중 최대).
    df_proc['hand4'] = (df_proc['pitcher_hand'].astype(str)
                        + df_proc['batter_hand'].astype(str))
    df_proc['is_waste_pitch_sit'] = np.where(((b == 0) & (s == 2)) | ((b == 1) & (s == 2)), 1.0, 0.0)
    df_proc['is_must_strike_sit'] = np.where(((b == 3) & (s == 0)) | ((b == 3) & (s == 1)), 1.0, 0.0)
    return df_proc


def step5_pitches_per_inning(df):
    df_proc = df.copy()
    df_proc['pitches_per_inning'] = df_proc['asof_pitcher_n'] / df_proc['inning'].clip(lower=1)
    return df_proc


def step6_combined_runner_features(df):
    df_proc = df.copy()
    df_proc['is_risp'] = df_proc['base_state'].astype(str).apply(
        lambda x: 1.0 if ('2' in x) or ('3' in x) else 0.0)
    df_proc['is_strict_inherited_runner'] = np.where(
        (df_proc['inning'] > 1) & (df_proc['asof_pitcher_n'] < 5) & (df_proc['num_runners_on'] > 0), 1.0, 0.0)
    df_proc['is_self_risp'] = np.where(
        (df_proc['asof_pitcher_n'] >= 15) & (df_proc['is_risp'] == 1.0), 1.0, 0.0)
    li_filled = df_proc['li'].fillna(0)
    df_proc['risp_pressure_index'] = df_proc['is_risp'] * li_filled
    df_proc['is_steal_threat_sit'] = np.where(
        (df_proc['runner_on_1b'] == 1) & (df_proc['runner_on_2b'] == 0)
        & (df_proc['score_diff_pitcher_team'].abs() <= 3), 1.0, 0.0)
    return df_proc


def step7_bayesian_smoothing(df, prior_mean=0.64):
    df_proc = df.copy()
    C = 50
    if 'asof_pitcher_success_rate' in df_proc.columns and 'asof_pitcher_n' in df_proc.columns:
        n = df_proc['asof_pitcher_n']
        curr = df_proc['asof_pitcher_success_rate']
        df_proc['smoothed_pitcher_success_rate'] = (n * curr + C * prior_mean) / (n + C)
    return df_proc


def step8_batter_toughness_features(df):
    df_proc = df.copy()
    if 'asof_batter_success_rate' in df_proc.columns and 'asof_batter_middle_rate' in df_proc.columns:
        df_proc['tough_batter_index'] = (1.0 - df_proc['asof_batter_success_rate']) * (1.0 - df_proc['asof_batter_middle_rate'])
    return df_proc


def step9_garbage_time_features(df):
    df_proc = df.copy()
    df_proc['is_garbage_time'] = np.where(df_proc['score_diff_pitcher_team'].abs() >= 7, 1.0, 0.0)
    df_proc['garbage_time_index'] = df_proc['score_diff_pitcher_team'].abs() / (10 - df_proc['inning']).clip(lower=1)
    return df_proc


def step10_recent_form_momentum(df):
    df_proc = df.copy()
    tc = ['asof_pitcher_prev1_game_success_rate',
          'asof_pitcher_prev3_game_success_rate',
          'asof_pitcher_prev5_game_success_rate']
    if all(c in df_proc.columns for c in tc):
        p1, p3, p5 = df_proc[tc[0]], df_proc[tc[1]], df_proc[tc[2]]
        df_proc['momentum_short'] = p1 - p3
        df_proc['momentum_mid'] = p1 - p5
        df_proc['is_heating_up'] = np.where((p1 > p3) & (p3 > p5), 1.0, 0.0)
        df_proc['is_cooling_down'] = np.where((p1 < p3) & (p3 < p5), 1.0, 0.0)
    return df_proc


def step11_veteran_and_pressure_features(df):
    df_proc = df.copy()
    df_proc['is_rookie'] = np.where(df_proc['asof_pitcher_n'] < 684, 1.0, 0.0)
    df_proc['is_veteran'] = np.where(df_proc['asof_pitcher_n'] > 3725, 1.0, 0.0)
    li_filled = df_proc['li'].fillna(0)
    df_proc['rookie_crisis_risk'] = df_proc['is_rookie'] * li_filled
    df_proc['veteran_clutch_ability'] = df_proc['is_veteran'] * li_filled
    return df_proc


def step12_first_pitch_tendency(df):
    df_proc = df.copy()
    if 'asof_pitcher_fastball_rate' in df_proc.columns and 'asof_pitcher_strike_rate' in df_proc.columns:
        if 'is_first_pitch' in df_proc.columns:
            df_proc['first_pitch_fastball_strike_idx'] = (
                df_proc['is_first_pitch'] * df_proc['asof_pitcher_fastball_rate'] * df_proc['asof_pitcher_strike_rate'])
    return df_proc


def step13_sac_fly_threat(df):
    df_proc = df.copy()
    is_3b = df_proc['base_state'].astype(str).apply(lambda x: 1.0 if '3' in x else 0.0)
    df_proc['is_sac_fly_threat'] = np.where(
        (is_3b == 1.0) & (df_proc['outs_before'] < 2)
        & (df_proc['score_diff_pitcher_team'].abs() <= 3), 1.0, 0.0)
    return df_proc


def step14_convert_to_category(df):
    df_proc = df.copy()
    original_cat_cols = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
                         'pitcher_hand', 'batter_hand', 'base_state', 'stadium',
                         'pitch_name', 'top_bottom', 'game_type']
    created_cat_cols = ['is_weekend_day_game', 'is_heat_wave_game', 'is_pure_starter',
                        'is_long_relief', 'is_short_relief', 'is_same_hand', 'is_first_pitch',
                        'is_full_count', 'count_advantage', 'phteam', 'is_waste_pitch_sit',
                        'is_must_strike_sit', 'is_risp', 'is_strict_inherited_runner',
                        'is_self_risp', 'is_steal_threat_sit', 'is_sac_fly_threat',
                        'is_garbage_time', 'is_rookie', 'is_veteran',
                        'is_heating_up', 'is_cooling_down', 'cnt12', 'hand4']
    all_cat_cols = [c for c in original_cat_cols + created_cat_cols if c in df_proc.columns]
    for c in all_cat_cols:
        df_proc[c] = df_proc[c].astype('category')
    return df_proc

# ====================================================================


def main():
    data_dir = None
    for path in ["data", "open", "./data", "./open", "open/data"]:
        if os.path.exists(os.path.join(path, "test.csv")):
            data_dir = path
            break
    if data_dir is None:
        raise FileNotFoundError("평가용 데이터를 찾을 수 없습니다.")

    df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    row_ids = df_test['row_id'].copy() if 'row_id' in df_test.columns else df_test.index

    constants_path = os.path.join("model", "train_constants.json")
    if not os.path.exists(constants_path):
        raise RuntimeError("model/train_constants.json이 없습니다. prior_mean을 알 수 없어 중단합니다.")
    with open(constants_path, "r") as f:
        _tc = json.load(f)
    prior_mean = float(_tc["prior_mean"])

    df_proc = step1_basic_features(df_test)
    df_proc = step2_pitcher_role_features(df_proc)
    df_proc = step3_matchup_features(df_proc)
    df_proc = step4_refined_count_features(df_proc)
    df_proc = step5_pitches_per_inning(df_proc)
    df_proc = step6_combined_runner_features(df_proc)
    df_proc = step7_bayesian_smoothing(df_proc, prior_mean=prior_mean)
    df_proc = step8_batter_toughness_features(df_proc)
    df_proc = step9_garbage_time_features(df_proc)
    df_proc = step10_recent_form_momentum(df_proc)
    df_proc = step11_veteran_and_pressure_features(df_proc)
    df_proc = step12_first_pitch_tendency(df_proc)
    df_proc = step13_sac_fly_threat(df_proc)

    if 'count_advantage' not in df_proc.columns:
        b, s = df_proc['balls_before'], df_proc['strikes_before']
        p_ahead = ((b == 0) & (s == 1)) | ((b == 0) & (s == 2)) | ((b == 1) & (s == 2))
        b_ahead = ((b == 1) & (s == 0)) | ((b == 2) & (s == 0)) | ((b == 3) & (s == 0)) | ((b == 2) & (s == 1)) | ((b == 3) & (s == 1))
        neu = ((b == 1) & (s == 1)) | ((b == 2) & (s == 2))
        df_proc['count_advantage'] = np.select([p_ahead, b_ahead, neu],
                                                ['Pitcher', 'Batter', 'Neutral'], default='None')

    # ---------- 트랙맨 병합 ----------
    # 학습 때 쓴 방식(train_constants.json의 trackman_mode)을 그대로 따라간다.
    #   asof  : merge_asof backward. 2025 test 행은 가장 최근(2024) 값을 받는다.
    #   exact : (season, month) 정확 일치 merge + fillna(0).
    #           트랙맨에 2025가 없으므로 test에서는 전부 0이 된다 (900점 버전의 동작).
    with open(constants_path, "r") as f:
        _const = json.load(f)
    trackman_mode = _const.get("trackman_mode", "asof")

    _NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
    fd_path = os.path.join("model", "feat_diff.csv")
    fs_path = os.path.join("model", "feat_speed.csv")
    fr_path = os.path.join("model", "feat_rp.csv")
    has_tm = all(os.path.exists(p) for p in [fd_path, fs_path, fr_path])

    if has_tm:
        # 'None'은 0-0/3-2 카운트를 뜻하는 실제 문자열인데 pandas 기본 설정은
        # 이를 NaN으로 읽어버린다. 그러면 by= 매칭이 전량 실패한다.
        feat_diff = pd.read_csv(fd_path, keep_default_na=False, na_values=_NA)
        feat_speed = pd.read_csv(fs_path, keep_default_na=False, na_values=_NA)
        feat_rp = pd.read_csv(fr_path, keep_default_na=False, na_values=_NA)
        feat_diff['count_advantage'] = feat_diff['count_advantage'].astype(str)
        rp_value_cols = [c for c in feat_rp.columns if c.startswith('past_')]

        if trackman_mode == "asof":
            df_proc['time_idx'] = df_proc['season'] * 100 + df_proc['game_month']
            df_proc['__orig'] = np.arange(len(df_proc))
            df_proc = df_proc.sort_values('time_idx')
            feat_diff = feat_diff.sort_values('time_idx')
            feat_speed = feat_speed.sort_values('time_idx')
            feat_rp = feat_rp.sort_values('time_idx')

            df_proc = pd.merge_asof(
                df_proc,
                feat_diff[['time_idx', 'pitcher_id', 'count_advantage', 'expected_control_difficulty']],
                on='time_idx', by=['pitcher_id', 'count_advantage'], direction='backward')
            df_proc = pd.merge_asof(
                df_proc, feat_speed[['time_idx', 'pitcher_id', 'past_fb_speed_mean']],
                on='time_idx', by='pitcher_id', direction='backward')
            df_proc = pd.merge_asof(
                df_proc, feat_rp[['time_idx', 'pitcher_id'] + rp_value_cols],
                on='time_idx', by='pitcher_id', direction='backward')

            df_proc = df_proc.sort_values('__orig').drop(columns=['__orig', 'time_idx'])
        else:
            df_proc = pd.merge(
                df_proc,
                feat_diff[['season', 'game_month', 'pitcher_id', 'count_advantage', 'expected_control_difficulty']],
                on=['season', 'game_month', 'pitcher_id', 'count_advantage'], how='left')
            df_proc = pd.merge(
                df_proc, feat_speed[['season', 'game_month', 'pitcher_id', 'past_fb_speed_mean']],
                on=['season', 'game_month', 'pitcher_id'], how='left')
            df_proc = pd.merge(
                df_proc, feat_rp[['season', 'game_month', 'pitcher_id'] + rp_value_cols],
                on=['season', 'game_month', 'pitcher_id'], how='left')
            for c in ['expected_control_difficulty', 'past_fb_speed_mean'] + rp_value_cols:
                if c in df_proc.columns:
                    df_proc[c] = df_proc[c].fillna(0)

    # ---------- 조건부 투수통계 병합 ----------
    # 학습 때 저장한 룩업 테이블을 그대로 붙인다. 'None'(0-0/3-2 카운트) 보존을 위해
    # na_values 를 반드시 명시해야 한다 (기본 read_csv 는 NaN 으로 읽어 매칭이 전량 실패).
    for _nm, _keys in [("cond_p",   ["pitcher_id"]),
                       ("cond_pc",  ["pitcher_id", "count_advantage"]),
                       ("cond_ph",  ["pitcher_id", "batter_hand"]),
                       ("cond_phc", ["pitcher_id", "batter_hand", "count_advantage"]),
                       ("cond_pb",  ["pitcher_id", "batter_id"])]:
        _cp = os.path.join("model", _nm + ".csv")
        if not os.path.exists(_cp):
            continue
        _ct = pd.read_csv(_cp, keep_default_na=False, na_values=_NA)
        for _k in _keys:
            if _ct[_k].dtype == object or df_proc[_k].dtype == object:
                _ct[_k] = _ct[_k].astype(str)
                df_proc[_k] = df_proc[_k].astype(str)
        _n_before = len(df_proc)
        df_proc = df_proc.merge(_ct, on=_keys, how="left")
        if len(df_proc) != _n_before:
            raise RuntimeError("%s 병합에서 행 수가 %d -> %d 로 변함 (테이블 키 중복)"
                               % (_nm, _n_before, len(df_proc)))


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
            df_proc["pf_" + _c.replace("asof_pitcher_", "")] = \
                df_proc[_c].to_numpy(dtype="float64") - _blend
        df_proc = df_proc.drop(columns=["pf_avg_pa"])
        print("prev-game 보정 완료: 작년분 섞인 행 "
              + " ".join("prev%d %.1f%%" % (_w, 100.0 * float(
                  (df_proc["pf_cross%d" % _w] > 0).mean())) for _w in (1, 3, 5)))

    df_proc = step14_convert_to_category(df_proc)

    # ---------- aux_rev: 보조 모델로 P(reverse|X) 예측 ----------
    # 학습 때 교차적합해 만든 피처다. 추론은 폴드 모델 3개의 평균을 쓴다.
    # 이 행의 입력만으로 계산되므로 "평가 데이터 각 행 독립" 규정을 만족한다.
    import glob as _glob
    _auxf = os.path.join("model", "aux_features.json")
    if os.path.exists(_auxf):
        with open(_auxf, "r") as f:
            _aux_cols = json.load(f)
        _tf = os.path.join("model", "aux_targets.json")
        _atgts = json.load(open(_tf)) if os.path.exists(_tf) else ["rev"]
        for _c in _aux_cols:
            if _c not in df_proc.columns:
                df_proc[_c] = np.nan
        _Xa = df_proc[_aux_cols].copy()
        for _c in _Xa.columns:
            if _Xa[_c].dtype.name in ["category", "object"]:
                _Xa[_c] = _Xa[_c].astype(str).astype("category")
        for _t in _atgts:
            _apaths = sorted(_glob.glob(os.path.join("model", "aux_%s_*.cbm" % _t)))
            if not _apaths:
                raise RuntimeError("aux_%s_*.cbm 이 없다" % _t)
            _acc = np.zeros(len(_Xa))
            for _p in _apaths:
                _am = CatBoostClassifier()
                _am.load_model(_p)
                _acc += _am.predict_proba(_Xa)[:, 1] / len(_apaths)
            df_proc["aux_" + _t] = _acc.astype(np.float32)
            print("aux_%s 예측 완료: 모델 %d개 | 평균 %.4f"
                  % (_t, len(_apaths), _acc.mean()))

    with open("model/selected_features.json", "r") as f:
        selected_features = json.load(f)
    _miss = [c for c in selected_features
             if c.startswith("aux_") and c not in df_proc.columns]
    if _miss:
        raise RuntimeError("selected_features 에 %s 가 있는데 만들어지지 "
                           "않았다 -- aux_features.json / aux_targets.json "
                           "/ aux_*.cbm 을 확인할 것" % _miss)
    for col in selected_features:
        if col not in df_proc.columns:
            df_proc[col] = np.nan
    df_features = df_proc[selected_features].copy()

    # CatBoost는 cat_features에 실제 NaN을 허용하지 않는다 (학습과 동일 처리)
    for col in df_features.columns:
        if df_features[col].dtype.name in ['category', 'object']:
            df_features[col] = df_features[col].astype(str).astype('category')

    # ---------- 추론: 모델 전체 평균 ----------
    # StratifiedKFold(shuffle=True) x 여러 seed로 학습했으므로 모든 모델이
    # 대등한 실력을 가진다 -> 균등 평균이 순수한 분산 감소로 이어진다.
    # 파일명에서 seed/fold 조합을 실제로 스캔한다 (개수를 하드코딩하지 않음 ->
    # N_SPLITS/SEEDS를 나중에 바꿔도 script.py 수정이 필요 없다).
    import glob
    preds = []
    cb_paths = sorted(glob.glob(os.path.join("model", "cb_fold_*.cbm")))
    for cb_path in cb_paths:
        stem = os.path.splitext(os.path.basename(cb_path))[0]  # cb_fold_{seed}_{fold}
        suffix = stem[len("cb_fold_"):]  # {seed}_{fold}
        model = CatBoostClassifier()
        model.load_model(cb_path)
        names = list(model.feature_names_)
        df_in = df_features.copy()
        for col in names:
            if col not in df_in.columns:
                df_in[col] = np.nan
        # 5분류 모델이면 성공 클래스 열을 쓴다 (학습 때 저장한 상수).
        _sc = int(_const.get("success_col", 1)) if _const.get("multiclass") else 1
        raw = model.predict_proba(df_in[names])[:, _sc]
        iso_path = os.path.join("model", "isotonic_fold_%s.pkl" % suffix)
        if os.path.exists(iso_path):
            raw = joblib.load(iso_path).predict(raw)
        preds.append(raw)

    if len(preds) == 0:
        raise RuntimeError(
            "모델을 하나도 로드하지 못했습니다. cwd=%s, model=%s"
            % (os.getcwd(), sorted(os.listdir('model')) if os.path.isdir('model') else '(없음)'))

    final_preds = np.mean(preds, axis=0)
    if np.isnan(final_preds).any():
        final_preds = np.nan_to_num(final_preds, nan=prior_mean)

    # ---------- 재중심화 ----------
    # 학습 시점에 홀드아웃(~Y-1 학습 -> Y 예측)으로 측정해 박아둔 고정 로짓 오프셋.
    # test 를 전혀 참조하지 않으므로 '평가 데이터 전체를 보고 만든 사후 보정값'이 아니다.
    _off = float(_const.get("recenter_offset", 0.0))
    if _off != 0.0:
        _q = np.clip(final_preds, 1e-6, 1 - 1e-6)
        final_preds = 1.0 / (1.0 + np.exp(-(np.log(_q / (1 - _q)) + _off)))

    final_preds = np.clip(final_preds, 0.01, 0.99)

    os.makedirs("output", exist_ok=True)
    submission = pd.DataFrame({"row_id": row_ids, "control_success": final_preds})

    sample_path = os.path.join(data_dir, "sample_submission.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        sample['row_id'] = sample['row_id'].astype(str)
        submission['row_id'] = submission['row_id'].astype(str)
        sample = sample.drop(columns=['control_success'], errors='ignore')
        sample = sample.merge(submission, on='row_id', how='left')
        sample['control_success'] = sample['control_success'].fillna(prior_mean)
        sample.to_csv("output/submission.csv", index=False)
    else:
        submission.to_csv("output/submission.csv", index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        os.makedirs("output", exist_ok=True)
        with open("output/error_log.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise

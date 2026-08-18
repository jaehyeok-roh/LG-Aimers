"""pitcher_id <-> pitcher_trackman_id 매핑 재구축.

주최측이 준 pitcher_id_mapping.csv 는 구종비율 하나로만 매칭돼 있어 약 91% 가 틀렸다
(시즌 간 일관성 1.9%, 2024 커버리지 28%). 이 스크립트는 두 단계로 다시 만든다.

  1단계 팀   : (월 x 요일 x 공수) 63차원 투구량 프로파일로 헝가리안 매칭.
               검증 = 10개 팀이 6시즌 내내 같은 프랜차이즈로 대응되는가 (10/10 통과).
               ※ 월 단위 9차원으로는 실패한다 - 팀별 월간 분포가 거의 같아 비용이 평평해진다.
  2단계 투수 : 팀-시즌 안에서 등판 프로파일 + 이닝 분포 + 구종배합 + 총투구량으로 매칭.
               손(L/R)은 하드제약. 마지막에 시즌 간 다수결로 교정.
               검증 = 시즌 간 일관성 (매칭에 시즌 간 정보를 안 쓰므로 순환이 아님). 90.9%.

출력: data/pitcher_id_mapping_v2.csv  (season, pitcher_id, pitcher_trackman_id, cost, margin, conf)

실행: PYTHONUTF8=1 python tools/rebuild_pitcher_mapping.py
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# 'None' 은 count_advantage 의 실제 값이므로 NaN 으로 읽히면 안 된다 (claude.md 4-3)
_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
MINOR_PREFIX = ('MIN_', 'KBO_', 'ACE_')   # 2군 / 올스타 / 기타 - 1군 프로파일에서 제외
SEASONS = range(2019, 2025)
OUT = 'data/pitcher_id_mapping_v2.csv'


def load():
    tr = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA, usecols=[
        'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
        'pitcher_id', 'pitcher_hand', 'pitcher_team_id', 'asof_pitcher_pitchmix_n',
        'asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate'])
    tm = pd.read_csv('data/trackman_history.csv', usecols=[
        'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
        'pitcher_trackman_id', 'pitcher_hand', 'pitcher_team', 'pitch_type_group'])
    # 손 코딩이 다르다: train 은 1=Left/2=Right 정수, trackman 은 'Left'/'Right' 문자열
    tr['pitcher_hand'] = tr.pitcher_hand.map({1: 'L', 2: 'R'})
    tm['pitcher_hand'] = tm.pitcher_hand.map({'Left': 'L', 'Right': 'R'})
    tr['tb'] = tr.top_bottom
    tm['tb'] = tm.top_bottom.map({'Top': 'T', 'Bottom': 'B'})
    tm['grp'] = tm.pitch_type_group.astype(str).str.lower()
    tm['team'] = tm.pitcher_team.replace({'SK_WYV': 'SSG_LAN'})   # 2021 개명, 같은 프랜차이즈
    tm['is_major'] = ~tm.pitcher_team.str.startswith(MINOR_PREFIX, na=False)
    return tr, tm


def cells(df, key):
    """(엔티티 x 월_요일_공수) 투구량 행렬"""
    d = df.assign(c=df.game_month.astype(str) + '_' + df.game_dayofweek.astype(str) + '_' + df.tb)
    return d.pivot_table(index=key, columns='c', aggfunc='size', fill_value=0).astype(float)


def match_teams(tr, tm):
    """team_id -> 프랜차이즈 코드. 시즌마다 독립적으로 풀고 일관성으로 검증한다."""
    major = tm[tm.is_major]
    rows = []
    for s in SEASONS:
        pa = cells(tr[tr.season == s], 'pitcher_team_id')
        pb = cells(major[major.season == s], 'team')
        pa, pb = pa.div(pa.sum(1), axis=0), pb.div(pb.sum(1), axis=0)
        cols = sorted(set(pa.columns) & set(pb.columns))
        A, B = pa[cols].values, pb[cols].values
        C = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(C)
        rows += [dict(season=s, tid=pa.index[i], code=pb.index[j]) for i, j in zip(r, c)]

    piv = pd.DataFrame(rows).pivot(index='tid', columns='season', values='code')
    stable = int((piv.nunique(axis=1) == 1).sum())
    print(f'[팀] 6시즌 내내 동일 프랜차이즈: {stable}/{len(piv)}')
    if stable != len(piv):
        raise RuntimeError('팀 매칭이 시즌 간 불일치. 프로파일 차원을 다시 확인할 것.\n'
                           + piv.to_string())
    return piv.iloc[:, 0].to_dict()


def train_season_mix(sub):
    """train 의 누적 asof 비율에서 '그 시즌만'의 구종배합을 복원한다."""
    g = sub.sort_values('asof_pitcher_pitchmix_n').groupby('pitcher_id')
    n0, n1 = g.asof_pitcher_pitchmix_n.first(), g.asof_pitcher_pitchmix_n.last()
    out = {c: g[col].last() * n1 - g[col].first() * n0 for c, col in
           [('fastball', 'asof_pitcher_fastball_rate'),
            ('breaking', 'asof_pitcher_breaking_rate'),
            ('offspeed', 'asof_pitcher_offspeed_rate')]}
    M = pd.DataFrame(out)
    return M.div(M.sum(1).replace(0, np.nan), axis=0)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def match_pitchers(tr, tm, team_of):
    major = tm[tm.is_major]
    mixsrc = tm[tm.grp.isin(['fastball', 'breaking', 'offspeed'])]   # 배합은 2군 포함 (주최측과 동일)
    tr = tr.assign(team=tr.pitcher_team_id.map(team_of))
    rows = []
    for s in SEASONS:
        a_all, b_all = tr[tr.season == s], major[major.season == s]
        mix_a_all = train_season_mix(a_all)
        mix_b_all = pd.crosstab(mixsrc[mixsrc.season == s].pitcher_trackman_id,
                                mixsrc[mixsrc.season == s].grp, normalize='index')
        for team in sorted(set(team_of.values())):
            a, b = a_all[a_all.team == team], b_all[b_all.team == team]
            if a.empty or b.empty:
                continue
            Pa, Pb = cells(a, 'pitcher_id'), cells(b, 'pitcher_trackman_id')
            Ia = a.assign(i=a.inning.clip(1, 10)).pivot_table(
                index='pitcher_id', columns='i', aggfunc='size', fill_value=0
            ).reindex(columns=range(1, 11), fill_value=0).astype(float)
            Ib = b.assign(i=b.inning.clip(1, 10)).pivot_table(
                index='pitcher_trackman_id', columns='i', aggfunc='size', fill_value=0
            ).reindex(columns=range(1, 11), fill_value=0).astype(float)
            cols = sorted(set(Pa.columns) & set(Pb.columns))
            MIX = ['fastball', 'breaking', 'offspeed']
            ma = mix_a_all.reindex(Pa.index).reindex(columns=MIX).fillna(0.34).values
            mb = mix_b_all.reindex(Pb.index).reindex(columns=MIX).fillna(0.34).values
            ta, tb = Pa.values.sum(1), Pb.values.sum(1)

            c_sched = 1 - unit(Pa[cols].values) @ unit(Pb[cols].values).T   # 커버리지 차이에 강건
            c_inn = ((unit(Ia.values)[:, None, :] - unit(Ib.values)[None, :, :]) ** 2).sum(-1)
            c_mix = ((ma[:, None, :] - mb[None, :, :]) ** 2).sum(-1)
            c_tot = (np.log1p(ta)[:, None] - np.log1p(tb)[None, :]) ** 2 * 0.05
            ha = a.groupby('pitcher_id').pitcher_hand.first().reindex(Pa.index).values
            hb = b.groupby('pitcher_trackman_id').pitcher_hand.first().reindex(Pb.index).values
            C = c_sched + c_inn + 2.0 * c_mix + c_tot + 100 * (ha[:, None] != hb[None, :])

            for i, j in zip(*linear_sum_assignment(C)):
                srt = np.sort(C[i])
                rows.append(dict(season=s, pitcher_id=Pa.index[i],
                                 pitcher_trackman_id=Pb.index[j], cost=C[i, j],
                                 margin=srt[1] - srt[0] if len(srt) > 1 else np.inf,
                                 n_tm=tb[j]))
    return pd.DataFrame(rows)


def main():
    tr, tm = load()
    team_of = match_teams(tr, tm)
    print('     ' + '  '.join(f'{k}:{v}' for k, v in sorted(team_of.items())))

    res = match_pitchers(tr, tm, team_of)
    # 트레이드 선수는 여러 팀에서 후보가 나오므로 시즌별 1:1 로 정리
    best = res.sort_values('cost').groupby(['season', 'pitcher_id'], as_index=False).first()
    best = best.sort_values('cost').groupby(['season', 'pitcher_trackman_id'], as_index=False).first()

    # 시즌 간 다수결(투구량 가중)로 교정
    vote = best.groupby(['pitcher_trackman_id', 'pitcher_id']).n_tm.sum().reset_index()
    win = (vote.sort_values('n_tm', ascending=False)
              .groupby('pitcher_trackman_id', as_index=False).first()
              .rename(columns={'pitcher_id': 'vote_pid'})[['pitcher_trackman_id', 'vote_pid']])
    best = best.merge(win, on='pitcher_trackman_id')

    # 검증은 반드시 다수결 '이전' 값으로 해야 한다. 교정 후에는 정의상 100%라 증거가 못 된다.
    g = best.groupby('pitcher_trackman_id').pitcher_id
    multi = g.nunique()[g.size() > 1]
    print(f'[검증] 교정 전 시즌간 일관성 {(multi == 1).mean() * 100:.1f}%  '
          f'(2시즌+ 등장 {len(multi)}명) — 매칭에 시즌간 정보를 안 쓰므로 순환 아님')
    print(f'[투수] 시즌간 다수결 교정 {int((best.pitcher_id != best.vote_pid).sum())} / {len(best)}쌍')
    best['pitcher_id'] = best.vote_pid

    out = best[['season', 'pitcher_id', 'pitcher_trackman_id', 'cost', 'margin']].copy()
    out['conf'] = np.where(out.cost <= out.cost.quantile(0.75), 'high',
                    np.where(out.cost <= out.cost.quantile(0.90), 'mid', 'low'))
    out.sort_values(['season', 'pitcher_id']).to_csv(OUT, index=False)

    cov = tr.groupby('season').apply(
        lambda d: d.pitcher_id.isin(set(out[out.season == d.name].pitcher_id)).mean())
    print(f'[커버리지] ' + '  '.join(f'{s}:{v * 100:.1f}%' for s, v in cov.items()))
    print(f'저장: {OUT} ({len(out)}행)')


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    main()

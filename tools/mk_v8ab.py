# v8 이 리더보드에서 -22.35 였다. 네 가지를 한꺼번에 넣어서 범인을 특정할 수 없다.
# 코드는 v8 과 완전히 동일하게 두고 **설정 플래그 하나씩만** 켜서 4개 커널을 만든다.
#
#   m : DROP_CAL 만            (월/요일 제거)          홀드아웃 +28
#   r : USE_REST_FOUL 만       (휴식·등판밀도·파울 7개) 홀드아웃 +20
#   b : USE_COND_PB 만         (투수x타자 맞대결)       홀드아웃 +16
#   d : COND_DECAY=0.25 만     (조건부통계 시즌 감쇠)   홀드아웃 +13
#
# 끄는 방법이 전부 '값' 이라 코드 경로는 네 커널이 동일하다 (v8 과도 동일).
#   COND_DECAY = 1.0 이면 w=1 이고 정규화도 항등이라 원래 식과 **수식적으로 같다**.
#   USE_COND_PB = False 면 cond_pb.csv 가 안 생기는데, script.py 병합 루프와 zip
#   REQUIRED 목록은 둘 다 os.path.exists 가드가 이미 있어서 그냥 건너뛴다 (확인함).
#
# 사용: python mk_v8ab.py m r b d
import ast, json, os, sys

NB = 'aimers_tuned_ensemble.ipynb'

S1 = os.environ.get('S1') == '1'      # 1-seed(10모델) 축소판. 부호만 보면 되는 절개용.
PREFIX = 's1' if S1 else 'v8'

VARIANTS = {
    #        DROP_CAL                        DECAY  REST   PB
    'o': ("[]",                               "1.0", "False", "False"),  # 전부 끔 = v5 기준선
    'm': ("['game_month', 'game_dayofweek']", "1.0", "False", "False"),
    'r': ("[]",                               "1.0", "True",  "False"),
    'b': ("[]",                               "1.0", "False", "True"),
    'd': ("[]",                               "0.25", "False", "False"),
}


def build(tag):
    drop_cal, decay, rest, pb = VARIANTS[tag]
    nb = json.load(open(NB, encoding='utf-8'))
    C = nb['cells']

    def sub(old, new, n=1, cells=None):
        hit = 0
        for i, c in enumerate(C):
            if c['cell_type'] != 'code' or (cells and i not in cells):
                continue
            s = ''.join(c['source'])
            if old in s:
                hit += s.count(old)
                c['source'] = s.replace(old, new).splitlines(keepends=True)
        if hit != n:
            sys.exit(f'[{tag}] 앵커 {n}개 기대, {hit}개:\n  {old[:110]}')

    # ---------- 1. 설정 ----------
    sub("USE_RELEASE_DYNAMICS = False", f'''USE_RELEASE_DYNAMICS = False

# --- v8 절개 실험 ({tag}): 네 항목 중 하나만 켠다. 나머지 셋은 v5 와 동일해진다. ---
DROP_CAL = {drop_cal}
COND_DECAY = {decay}          # 1.0 이면 감쇠 없음 = v5 와 수식적으로 동일
USE_REST_FOUL = {rest}
USE_COND_PB = {pb}''')

    # ---------- 2. 휴식/파울 테이블 ----------
    sub("def step15_prep_trackman_data(trackman_df, pitcher_map_df):",
        '''def build_rest_foul(tm):
    """등판 간 휴식 / 등판 밀도 / 파울 성향. 키와 컬럼 접두사를 feat_rp 와 맞춰
    저장·추론 경로를 그대로 재사용한다."""
    KEY = ['season', 'game_month', 'pitcher_id']
    t = tm.copy()
    t['_d'] = pd.to_datetime(t['game_date'], format='%m/%d/%Y', errors='coerce')
    out = (t.groupby(['pitcher_id', 'season', 'trackman_game_id'])
             .agg(_d=('_d', 'first'), n_pitch=('_d', 'size'),
                  game_month=('game_month', 'first')).reset_index()
             .sort_values(['pitcher_id', 'season', '_d']))
    out['rest'] = out.groupby(['pitcher_id', 'season'])['_d'].diff().dt.days
    mon = out.groupby(KEY).agg(
        rest_mean=('rest', 'mean'), rest_min=('rest', 'min'),
        b2b_rate=('rest', lambda s: float((s <= 1).mean()) if s.notna().any() else np.nan),
        n_out=('trackman_game_id', 'size'), pitch_per_out=('n_pitch', 'mean')).reset_index()
    t['_foul'] = t['pitch_of_pa'] - t['balls_before'] - t['strikes_before'] - 1
    fl = t.groupby(KEY).agg(foul_mean=('_foul', 'mean'),
                            pa_len=('pitch_of_pa', 'mean')).reset_index()
    mon = mon.merge(fl, on=KEY, how='outer')
    vals = ['rest_mean', 'rest_min', 'b2b_rate', 'n_out', 'pitch_per_out',
            'foul_mean', 'pa_len']
    # step17/18 과 동일한 leak-free 패턴: 그 달 '이전' 값만 쓴다
    mon = mon.sort_values(['pitcher_id', 'season', 'game_month'])
    g = mon.groupby('pitcher_id')
    for c in vals:
        mon['past_' + c] = g[c].transform(lambda s: s.shift(1).expanding().mean())
    return mon[KEY + ['past_' + c for c in vals]]


def step15_prep_trackman_data(trackman_df, pitcher_map_df):''')

    # ---------- 3. 조건부통계: 감쇠 + 맞대결 ----------
    sub("""    (['pitcher_id', 'batter_hand', 'count_advantage'],   50, 'cond_phc'),
]""",
        """    (['pitcher_id', 'batter_hand', 'count_advantage'],   50, 'cond_phc'),
]
if USE_COND_PB:
    COND_SPECS.append((['pitcher_id', 'batter_id'], 20, 'cond_pb'))""")

    sub("""def build_cond_table(src, keys, C, name):
    g = src.groupby(keys, observed=True)['_dev'].agg(['sum', 'count']).reset_index()
    g[name] = g['sum'] / (g['count'] + C)          # 0(리그평균)으로 shrink
    return g[keys + [name]]""",
        """def build_cond_table(src, keys, C, name, target_season):
    # 시즌 감쇠. 가중치 합을 행 수에 맞춰 정규화하므로 C 의 의미는 감쇠값과 무관하다.
    # COND_DECAY = 1.0 이면 w 가 전부 1 이라 원래 식(sum/(count+C))과 완전히 같다.
    w = COND_DECAY ** ((target_season - 1) - src['season'].to_numpy())
    w = w * (len(src) / w.sum())
    t = src[keys].copy()
    t['_w'] = w
    t['_wd'] = w * src['_dev'].to_numpy()
    g = t.groupby(keys, observed=True)[['_wd', '_w']].sum().reset_index()
    g[name] = g['_wd'] / (g['_w'] + C)             # 0(리그평균)으로 shrink
    return g[keys + [name]]""")

    sub("            t = build_cond_table(past, keys, C, name).set_index(keys)[name]",
        "            t = build_cond_table(past, keys, C, name, s).set_index(keys)[name]")

    sub("""    return {name: build_cond_table(d, keys, C, name) for keys, C, name in COND_SPECS}""",
        """    _ts = int(d['season'].max()) + 1        # 추론 대상 시즌(2025)이 감쇠 기준점
    out = {name: build_cond_table(d, keys, C, name, _ts) for keys, C, name in COND_SPECS}
    for _n, _t in out.items():
        print(f"  룩업 {_n}: {len(_t):,}행")
    return out""")

    # ---------- 4. 파이프라인에 휴식/파울 병합 ----------
    _PIPE = [i for i, c in enumerate(C)
             if c['cell_type'] == 'code' and 'def run_full_pipeline' in ''.join(c['source'])]
    assert len(_PIPE) == 1, f'run_full_pipeline 셀 {len(_PIPE)}개'
    sub("""    rp_value_cols = [c for c in feat_rp.columns if c.startswith('past_')]""",
        """    if USE_REST_FOUL:
        _rf = build_rest_foul(tm_base)
        _n0 = len(feat_rp)
        feat_rp = feat_rp.merge(_rf, on=['season', 'game_month', 'pitcher_id'], how='outer')
        print(f"  휴식·파울 {len(_rf):,}행 -> feat_rp {_n0:,} -> {len(feat_rp):,}행 "
              f"(신규 {len([c for c in _rf.columns if c.startswith('past_')])}개)")
    rp_value_cols = [c for c in feat_rp.columns if c.startswith('past_')]""", cells=_PIPE)

    # ---------- 5. 피처에서 월/요일 제외 ----------
    sub("""drop_cols += DEAD_FEATURES   # Cell 0 에서 정의 (커리어누적 오해로 죽은 피처들)""",
        """drop_cols += DEAD_FEATURES   # Cell 0 에서 정의 (커리어누적 오해로 죽은 피처들)
drop_cols += DROP_CAL        # 절개 실험: 변형 m 에서만 비어있지 않다""")

    # ---------- 6. script.py 에 cond_pb 병합 추가 (파일이 없으면 자동 skip) ----------
    sub("""                       ("cond_phc", ["pitcher_id", "batter_hand", "count_advantage"])]:""",
        """                       ("cond_phc", ["pitcher_id", "batter_hand", "count_advantage"]),
                       ("cond_pb",  ["pitcher_id", "batter_id"])]:""")

    # ---------- 7. zip 목록 (os.path.exists 가드가 이미 있다) ----------
    sub("""    + [f"model/{n}.csv" for n in ["cond_p", "cond_pc", "cond_ph", "cond_phc"]""",
        """    + [f"model/{n}.csv" for n in ["cond_p", "cond_pc", "cond_ph", "cond_phc", "cond_pb"]""")

    if S1:
        sub("SEEDS = [42, 202, 2024]       # seed 앙상블 (3개)",
            "SEEDS = [42]                  # 절개용 축소 (10모델)")

    # ---------- 검사 ----------
    for i, c in enumerate(C):
        if c['cell_type'] == 'code':
            try:
                ast.parse(''.join(c['source']))
            except SyntaxError as e:
                sys.exit(f'[{tag}] cell {i} 문법 오류: {e}')

    src = '\n'.join(''.join(c['source']) for c in C if c['cell_type'] == 'code')
    for nm, k in [('DROP_CAL', 2), ('COND_DECAY', 2), ('USE_REST_FOUL', 2),
                  ('USE_COND_PB', 2), ('build_rest_foul', 2), ('target_season', 2)]:
        if src.count(nm) < k:
            sys.exit(f'[{tag}] {nm} 참조 {src.count(nm)}회 (기대 {k}+) — 패치 누락')
    if 'build_cond_table(past, keys, C, name)' in src or \
            'build_cond_table(d, keys, C, name)' in src:
        sys.exit(f'[{tag}] 구 시그니처 호출이 남아 있다')
    if 'RUN_OPTUNA = False' not in src:
        sys.exit(f'[{tag}] RUN_OPTUNA 가 False 가 아니다')

    D = f'.kernels/{PREFIX}{tag}'
    os.makedirs(D, exist_ok=True)
    for c in C:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    json.dump(nb, open(f'{D}/aimers_{PREFIX}{tag}.ipynb', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    meta = json.load(open('kernel-metadata.json'))
    meta.update(id=f'homekeggle/aimers-{PREFIX}{tag}', title=f'aimers-{PREFIX}{tag}',
                code_file=f'aimers_{PREFIX}{tag}.ipynb')
    json.dump(meta, open(f'{D}/kernel-metadata.json', 'w'), indent=2)
    print(f'{D}/aimers_{PREFIX}{tag}.ipynb  —  DROP_CAL={drop_cal} DECAY={decay} '
          f'REST={rest} PB={pb} SEEDS={"[42]" if S1 else "3개"}')


for t in sys.argv[1:]:
    build(t)

# 재중심화 상수를 count_advantage 4구간마다 따로 둔다 — v10wzc(1,121.89) 위.
#
# 근거 (2026-08-29, tools/eda43.py + eda44.py -> 캐글 kseg3):
#   네 해를 각각 '~Y-1 학습 -> Y 예측' 으로 만들고 구간별 편향을 쟀다.
#   count_advantage 만 네 해 내내 부호가 안 바뀐다:
#
#              2021      2022      2023      2024
#     Batter  +0.0047   +0.0041   +0.0026   +0.0063     <- 4년 전부 양수
#     Pitcher -0.0024   -0.0028   -0.0032   -0.0067     <- 4년 전부 음수
#
#   그리고 '전 해 상수를 다음 해에 적용' 했을 때 세 전이 전부 양수인 축도 이것뿐이다:
#     count_adv  +0.3 / +1.9 / +6.7
#     wn         +6.8 / +34.8 / **-261.5**      <- 2023 편향 +0.0669 가 2024 에 +0.0009 로 붕괴
#     game_type  -316.8 / -70.7 / **-1431.5**   <- F 체제 변경
#
#   ⚠️ 한 쌍만 봤으면 wn 을 채택했을 것이다. eda38 이후 '한 시즌 앞' 검증이
#      224점 / 46점 / 1,431점짜리 실수를 막았다.
#
# 규정: 행 A 는 자기 count_advantage 만 보고 상수 하나를 더한다. 상수는 홀드아웃
#       (학습 데이터)에서만 만든다 -> 행 독립 시험을 자명하게 통과한다.
#       ⚠️ test 평균으로 재정규화하면 안 된다(주최측 금지). 대신 **학습 빈도**로
#          가중 중심화해서 전역 평균이 흔들리지 않게 한다.
#
# 기대: 세 전이 평균 +3, 최근 전이 +6.7. 상수라 전달률은 1.0 에 가깝다.
import ast
import json
import os
import sys

BASE = os.environ.get('SO_BASE', 'aimers_v10wzc.ipynb')
OUT = os.environ.get('SO_OUT', 'experiments/v10w/aimers_v10wso.ipynb')
ZIP = os.environ.get('SO_ZIP', 'submit_v10wso.zip')
OLDZIP = os.environ.get('SO_OLDZIP', 'submit_v10wzc.zip')

nb = json.load(open(BASE, encoding='utf-8'))
cells = nb['cells']
code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']


def src(i):
    return ''.join(cells[i]['source'])


def sub(i, old, new, n=1):
    s = src(i)
    if s.count(old) != n:
        sys.exit('셀 %d: 앵커 %d개 기대, %d개\n  %s' % (i, n, s.count(old), old[:150]))
    cells[i]['source'] = s.replace(old, new).splitlines(keepends=True)


def find(anchor):
    hit = [i for i in code if anchor in src(i)]
    if len(hit) != 1:
        sys.exit('앵커 %d곳: %s' % (len(hit), anchor[:110]))
    return hit[0]


# ------------------------------------------------ 1) 구간별 상수 측정 (오프셋 셀)
A1 = "    RECENTER_OFFSET = solve_logit_offset(_ph, _actual)"
N1 = '''    RECENTER_OFFSET = solve_logit_offset(_ph, _actual)

    # ---- count_advantage 4구간 상수 (kseg3: 네 해 내내 부호 일관) ----
    # 전역 상수를 먼저 적용한 뒤, 구간마다 '예측평균 = 실제평균' 이 되는 상수를 더 잰다.
    _qh = np.clip(_ph, 1e-6, 1 - 1e-6)
    _loh = np.log(_qh / (1 - _qh)) + RECENTER_OFFSET
    _segv = df_processed.loc[_va_m, "count_advantage"].astype(str).to_numpy()
    _raw = {}
    for _s in sorted(set(_segv)):
        _m = (_segv == _s)
        _raw[_s] = solve_logit_offset(1.0 / (1.0 + np.exp(-_loh[_m])),
                                      float(_yv.to_numpy()[_m].mean()))
    # 학습 빈도로 가중 중심화 -> 전역 평균이 흔들리지 않는다.
    # ⚠️ test 평균으로 재정규화하면 주최측이 금지한 '평가 데이터 전체 분포로 보정' 이 된다.
    _segt = df_processed.loc[_tr_m, "count_advantage"].astype(str)
    _w = _segt.value_counts(normalize=True).to_dict()
    _mu = sum(_w.get(_s, 0.0) * _v for _s, _v in _raw.items())
    SEG_OFFSET = {_s: float(_v - _mu) for _s, _v in _raw.items()}

    _adj = _loh + np.array([SEG_OFFSET.get(_s, 0.0) for _s in _segv])
    _pseg = 1.0 / (1.0 + np.exp(-_adj))
    print("\\n  count_advantage 구간 상수 (학습빈도 가중 중심화):")
    for _s in sorted(SEG_OFFSET):
        print(f"    {_s:<10} 원값 {_raw[_s]:+.4f} -> 중심화 {SEG_OFFSET[_s]:+.4f}"
              f"  (학습비중 {_w.get(_s, 0.0):.1%})")'''
sub(find(A1), A1, N1)
print('오프셋 셀: 구간 상수 측정 삽입')

# 이득 출력은 기존 점수 출력 바로 뒤에 붙인다
A2 = '    print(f"  홀드아웃 환산점수: 보정전 {_sk(_ph):,.0f} -> 보정후 {_sk(_after):,.0f}  ({_sk(_after)-_sk(_ph):+,.0f})")'
N2 = A2 + '''
    print(f"  구간 상수까지: {_sk(_pseg):,.0f}  ({_sk(_pseg)-_sk(_after):+,.1f})")
    # 전역 평균이 유지되는지 확인 (중심화가 제대로 됐다는 뜻)
    assert abs(_pseg.mean() - _after.mean()) < 2e-3, (_pseg.mean(), _after.mean())'''
sub(find(A2), A2, N2)
print('오프셋 셀: 이득 출력 + 중심화 검증')

# ------------------------------------------------ 2) 상수 저장
A3 = '''    json.dump({"prior_mean": PRIOR_MEAN, "trackman_mode": TRACKMAN_MODE,
               "recenter_offset": RECENTER_OFFSET}, f)'''
N3 = '''    json.dump({"prior_mean": PRIOR_MEAN, "trackman_mode": TRACKMAN_MODE,
               "recenter_offset": RECENTER_OFFSET,
               "seg_offset_count_adv": SEG_OFFSET}, f)'''
sub(find(A3), A3, N3)
print('train_constants 에 seg_offset_count_adv 저장')

# ------------------------------------------------ 3) 추론에서 적용 (script.py)
A4 = '''    _off = float(_const.get("recenter_offset", 0.0))
    if _off != 0.0:
        _q = np.clip(final_preds, 1e-6, 1 - 1e-6)
        final_preds = 1.0 / (1.0 + np.exp(-(np.log(_q / (1 - _q)) + _off)))'''
N4 = '''    _off = float(_const.get("recenter_offset", 0.0))
    _segoff = _const.get("seg_offset_count_adv") or {}
    if _off != 0.0 or _segoff:
        _q = np.clip(final_preds, 1e-6, 1 - 1e-6)
        _lo = np.log(_q / (1 - _q)) + _off
        if _segoff:
            # 행 A 는 자기 count_advantage 만 본다 (다른 행을 보지 않는다).
            # 학습에 없던 값은 0 -> 전역 상수만 적용된다.
            _sv = df_proc["count_advantage"].astype(str).to_numpy()
            assert len(_sv) == len(_lo), (len(_sv), len(_lo))   # 4-15 행 순서
            _lo = _lo + np.array([float(_segoff.get(_s, 0.0)) for _s in _sv])
            print("구간 상수 적용:", {k: round(v, 4) for k, v in _segoff.items()})
        final_preds = 1.0 / (1.0 + np.exp(-_lo))'''
sub(find(A4), A4, N4)
print('script.py: 행별 구간 상수 적용')

# ------------------------------------------------ 4) 산출물 이름
sub(find(OLDZIP), OLDZIP, ZIP)
print('zip -> %s' % ZIP)

for i in code:
    ast.parse(src(i))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

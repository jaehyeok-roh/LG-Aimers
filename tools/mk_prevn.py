# prev{1,3,5}_game 비율의 **분모**(그 구간 투구수)를 실수값에서 역산해 피처로 준다.
#
# claude.md 는 이 6개 컬럼을 "분모가 test 에 없어서 분해 불가능" 으로 닫아뒀다.
# 틀렸다 — 분모는 **값 안에 있다.** 비율이 성공수/투구수 인 유리수이고 소수 6자리로
# 저장돼 있는데, 분모 400 이하 기약분수 사이 최소 간격 1/(400*399)=6.3e-6 이
# 반올림 오차 5e-7 의 12배라 역산이 사실상 유일하다.
# 같은 윈도우에 분수가 둘(success, middle)이라 최소공배수로 약분 모호성을 지운다.
#
# 정확도 (train 경기 복원 4,868경기 / 45,862등판으로 대조):
#     prev1  success 단독 61.2% -> lcm 82.5%      (약수 보장 100.0%)
#     prev3              60.7% -> lcm 82.9%
#     prev5              55.7% -> lcm 76.8%
#
# 왜 새 정보인가
#   1. 트리가 못 만든다. 실수 -> 분모는 연분수 전개다. 0.5 -> 2 인데 0.500001 -> 없음
#      이라 임계값 분할로는 어떤 깊이로도 근사되지 않는다. claude.md 의
#      '한 행 안 파생은 트리가 스스로 만든다'(3번 확인) 의 예외다.
#   2. 규정상 가장 깨끗하다. 행 A 자기 값 하나만 쓴다 (허용 2번). test 다른 행도,
#      train 룩업도 안 본다. 행 독립 시험은 자명하게 통과한다.
#   3. 모델이 지금 못 보는 것을 준다. claude.md: "prev1_game 은 전 구간에서 음수다
#      (한 경기 ~20구 = 노이즈)" -- 원인이 분모를 몰라서다. 3구 등판의 1.000 과
#      100구 등판의 0.55 를 같은 축에 놓고 있었다.
#
# 주는 것은 6개뿐이다: pn{1,3,5} (투구수) + ps{1,3,5} (성공 개수).
# 차이/비율(pn3-pn1 등)은 **넣지 않는다** — 트리가 임계값으로 근사할 수 있는
# 한 행 안 파생이고 그 부류는 세 번 다 0 이었다.
import ast
import json
import os
import sys

BASE = os.environ.get('PN_BASE', 'aimers_v10wz.ipynb')
OUT = os.environ.get('PN_OUT', 'aimers_v10wpn.ipynb')
ZIP = os.environ.get('PN_ZIP', 'submit_v10wpn.zip')

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
        sys.exit('셀 %d: 앵커 %d개 기대, %d개\n  %s' % (i, n, s.count(old), old[:140]))
    setsrc(i, s.replace(old, new))


def find(pat, where=None):
    for i in (where or code):
        if pat in src(i):
            return i
    sys.exit('앵커를 못 찾았다: %s' % pat[:120])


# 학습과 추론이 **같은 소스**를 써야 한다 (claude.md 2장 설계 원칙).
# STEPS_SRC 와 똑같이 문자열 하나로 두고, 노트북은 exec, script.py 는 텍스트 삽입한다.
BODY = '''

# 윈도우별 분모 상한. 상한이 작으면 긴 구간(선발 5등판 = 450구+)을 통째로 놓친다.
# 정답 대조 실측: 상한을 올려도 정확도는 83% 로 평평하고 **초과(오답) 0.0%** 다.
PREVN_MAXD = {1: 250, 3: 700, 5: 1200}
PREVN_TOL = 5e-7


def _prevn_denom(v, maxd):
    """실수 배열 -> 기약분수 분모 (0 = 역산 실패). 고유값만 계산해 되뿌린다."""
    from fractions import Fraction
    out = np.zeros(len(v), dtype="float64")
    fin = np.isfinite(v)
    if not fin.any():
        return out
    u, inv = np.unique(v[fin], return_inverse=True)
    d = np.empty(len(u), dtype="float64")
    for _i, _x in enumerate(u):
        _k = round(float(_x), 6)
        _f = Fraction(_k).limit_denominator(maxd)
        d[_i] = _f.denominator if abs(float(_f) - _k) <= PREVN_TOL else 0.0
    out[fin] = d[inv]
    return out


def attach_prevn(df):
    """prev{1,3,5}_game 의 분모(투구수)와 분자(성공수)를 복원해 6개 컬럼을 붙인다.

    ⚠️ 비율이 정확히 0 이나 1 이면 분모가 1 로 떨어져 정보가 없다 -> NaN 으로 둔다.
    ⚠️ 역산은 틀리지 않고 **약분될** 뿐이다 (참값이 항상 배수, 실측 100.0%).
       따라서 값은 '투구수의 약수' 이고 82% 는 투구수 자체다.
    """
    for _w in (1, 3, 5):
        _cs = "asof_pitcher_prev%d_game_success_rate" % _w
        _cm = "asof_pitcher_prev%d_game_middle_rate" % _w
        _md = PREVN_MAXD[_w]
        _rs = df[_cs].to_numpy(dtype="float64")
        _ds = _prevn_denom(_rs, _md)
        _dm = (_prevn_denom(df[_cm].to_numpy(dtype="float64"), _md)
               if _cm in df.columns else np.zeros(len(df)))
        _a = _ds.astype("int64")
        _b = _dm.astype("int64")
        _g = np.gcd(_a, _b)
        _n = np.where((_a > 0) & (_b > 0),
                      _a.astype("float64") * _b / np.maximum(_g, 1),
                      np.maximum(_ds, _dm))
        _n = np.where(_n <= 1, np.nan, _n)
        df["pn%d" % _w] = _n
        df["ps%d" % _w] = np.round(_rs * _n)
    _v = df["pn1"].to_numpy(dtype="float64")
    print("  prev 분모 역산: pn1 결측 %.1f%% | 중앙 %.0f | pn5 중앙 %.0f"
          % (100.0 * float(np.isnan(_v).mean()), float(np.nanmedian(_v)),
             float(np.nanmedian(df["pn5"].to_numpy(dtype="float64")))))
    return df
'''

# ---------------------------------------------------------------- 1) 함수 정의
FUNCS = ('\n\n# 학습 노트북과 script.py 가 공유하는 소스. 문자열 하나가 진실이다.\n'
         'PREVN_SRC = ' + repr(BODY) + '\n'
         'exec(PREVN_SRC)\n')
i_f = find('COND_COLS = [name for _, _, name in COND_SPECS]')
sub(i_f, 'COND_COLS = [name for _, _, name in COND_SPECS]',
    'COND_COLS = [name for _, _, name in COND_SPECS]\n' + FUNCS)
print('1) PREVN_SRC 정의 + exec 삽입 (셀 %d)' % i_f)

# ---------------------------------------------------------------- 2) 학습 파이프라인
i_p = find('        df_proc = attach_wseason(df_proc)')
sub(i_p, '        df_proc = attach_wseason(df_proc)',
    '        df_proc = attach_wseason(df_proc)\n'
    '        df_proc = attach_prevn(df_proc)')
print('2) 학습 파이프라인에 연결 (셀 %d)' % i_p)

# ---------------------------------------------------------------- 3) 추론(script.py)
i_s = find('SCRIPT_TEMPLATE')
INFER = '''    # ---------- prev-game 분모 역산 ----------
    # 학습과 **같은 함수**를 쓴다 (STEPS_SRC 로 삽입돼 있다).
    # 행 A 자기 값만 보므로 행 독립 원칙에 걸릴 여지가 없다.
    df_proc = attach_prevn(df_proc)

'''
sub(i_s, '    df_proc = step14_convert_to_category(df_proc)',
    INFER + '    df_proc = step14_convert_to_category(df_proc)')
print('3) script.py 에 연결 (셀 %d)' % i_s)

# script.py 는 `__STEPS__` 자리에 STEPS_SRC 를 텍스트로 넣는다.
# attach_prevn 은 그 밖에 정의돼 있으므로 같은 자리에 **함께** 실어야 한다.
sub(i_s, 'SCRIPT_TEMPLATE.replace("__STEPS__", STEPS_SRC)',
    'SCRIPT_TEMPLATE.replace("__STEPS__", STEPS_SRC + "\\n" + PREVN_SRC)')
print('4) script.py 본문에 PREVN_SRC 동봉')

# ---------------------------------------------------------------- 4) zip 이름
for i in code:
    s = src(i)
    if 'ZIP_PATH' in s:
        for old in ('submit_v10wz.zip', 'submit_v10wq.zip', 'submit_ptc6.zip'):
            if old in s:
                sub(i, old, ZIP)
                print('5) zip -> %s' % ZIP)
                break
        break

for i in code:
    ast.parse(src(i))
json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

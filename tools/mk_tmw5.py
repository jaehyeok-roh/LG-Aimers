# pitcher_team x w_success 5분위 (60칸) 를 범주형으로 — v10wph(1,150.26) 위.
#
# 근거 (2026-08-31, tools/eda54.py):
#   eda52 는 **원본 범주형끼리**만 스캔했다. 배포 점수의 92점을 지고 있는 w5_*/wb_*
#   를 구간화해 교차한 적은 없었다. 같은 방법(강도 x 2023<->2024 상관)으로 재니:
#
#     축A                축B     강도   상관    곱
#     pitcher_team     w5q      51.8   0.76   **39**   <- phteam 의 2.6배
#     w5q              wnq      40.5   0.67     27
#     w5q              wb5q     29.0   0.48     14
#     (교정점) hand4            49.5   1.00     50    <- 모델에 있음
#     (교정점) phteam           38.1   0.40     15    -> 실측 **+7.79**
#
#   phteam 의 회수율(곱 15 -> +7.79)을 적용하면 **+20 급**이다.
#
# ⚠️ 다만 w5q 는 수치형 파생이라 트리가 임계값으로 싸게 쪼갠다. phteam(범주형 x 범주형)
#    만큼 비싸지 않을 수 있고, 그러면 회수율이 낮아진다. 기대는 **+5~20** 으로 넓게 잡는다.
#
# ⚠️⚠️ w_* 는 STEPS_SRC 밖이라 **학습(셀07)과 추론(셀22)에 따로 있다.**
#      트랙맨 exact/asof 36점 사고가 정확히 '두 경로가 다른 규칙을 쓴' 유형이다.
#      그래서 양쪽에 **같은 리터럴 경계**를 박고, 학습 전 가드로 칸 수/결측을 확인한다.
#      경계는 train 전체에서 뽑은 5분위이며 상수로 고정한다 (t13 의 2023-05 와 같은 부류).
import ast
import json
import os
import sys

BASE = os.environ.get('TW_BASE', 'experiments/v10w/aimers_v10wph.ipynb')
OUT = os.environ.get('TW_OUT', 'experiments/v10w/aimers_v10wtw.ipynb')
ZIP = os.environ.get('TW_ZIP', 'submit_v10wtw.zip')
OLDZIP = os.environ.get('TW_OLDZIP', 'submit_v10wph.zip')

# train 전체 w_success 의 5분위 경계 (tools/eda54.py 로 산출)
EDGES = [-0.047673, -0.022461, -0.001398, 0.024163]

nb = json.load(open(BASE, encoding='utf-8'))
cells = nb['cells']
code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']
src = lambda i: ''.join(cells[i]['source'])


def sub(i, old, new, n=1):
    s = src(i)
    if s.count(old) != n:
        sys.exit('셀 %d: 앵커 %d개 기대, %d개\n  %s' % (i, n, s.count(old), old[:140]))
    cells[i]['source'] = s.replace(old, new).splitlines(keepends=True)


def find(a, want=1):
    h = [i for i in code if a in src(i)]
    if len(h) != want:
        sys.exit('앵커 %d곳(기대 %d): %s' % (len(h), want, a[:90]))
    return h


BIN = ("np.digitize(%s, %r).astype(int).astype(str)" % ('{V}', EDGES))

# ---------------------------------------------- 1) 학습 경로 (셀 07)
A1 = """        df['w_' + k] = (wx + base * WS_C) / (wn + WS_C) - base"""
N1 = A1 + """
    # pitcher_team x w_success 5분위 (60칸). eda54: 강도 51.8 / 2023<->2024 상관 0.76
    # = phteam(곱 15 -> +7.79)의 2.6배. 경계는 train 전체 5분위를 상수로 고정한다.
    df['tmw5'] = (df['pitcher_team_id'].astype(str) + '|'
                  + """ + BIN.replace('{V}', "df['w_success'].to_numpy(dtype='float64')") + """)"""
sub(find(A1)[0], A1, N1)
print('학습 경로(셀07)에 tmw5 삽입')

# ---------------------------------------------- 2) 추론 경로 (script.py, 셀 22)
# ⚠️ 앵커는 w_success 를 만드는 루프 **뒤**여야 하고, 그 블록은 `if _ws_rates:` 안이라
# 들여쓰기가 **8칸**이다. 처음에 w_share 줄(루프 앞, 4칸으로 삽입)에 걸었다가
# IndentationError 로 42분을 태웠다.
# ⚠️ 짧은 앵커는 타자측 블록에도 있어 2곳이 잡힌다. 투수측 전용 문구를 포함한다.
A2 = ('        print("당해시즌 복원 완료: 투구수 중앙값 %.0f / 신규투수 비율 %.1f%%"\n'
      '              % (float(np.median(_wn)), 100.0 * float((_n0 == 0).mean())))')
N2 = A2 + """
        # 학습과 **같은 리터럴 경계**를 쓴다 (36점짜리 exact/asof 사고 유형 회피).
        df_proc["tmw5"] = (df_proc["pitcher_team_id"].astype(str) + "|"
                           + """ + BIN.replace('{V}', 'df_proc["w_success"].to_numpy(dtype="float64")') + """)"""
sub(find(A2)[0], A2, N2)
print('추론 경로(셀22)에 tmw5 삽입 — 같은 경계 리터럴')

# ---------------------------------------------- 3) 범주형 등록
CAT = "'is_full_count', 'count_advantage', 'phteam', 'is_waste_pitch_sit',"
n = 0
for i in code:
    c = src(i).count(CAT)
    if c:
        sub(i, CAT, "'is_full_count', 'count_advantage', 'phteam', 'tmw5', 'is_waste_pitch_sit',", n=c)
        n += c
if n == 0:
    sys.exit('created_cat_cols 앵커를 못 찾았다')
print('범주형 등록: %d곳' % n)

# ---------------------------------------------- 4) 학습 전 가드 (4-12)
GUARD = '''
# ---- tmw5 검증 (학습 전에 터뜨린다, 4-12) ----
assert "tmw5" in X_full.columns, "tmw5 누락"
assert "tmw5" in cat_features, "tmw5 가 범주형으로 등록되지 않았다"
_v = X_full["tmw5"].astype(str)
assert not _v.str.contains("nan").any(), "tmw5 에 nan 문자열"
_nt = df_processed["pitcher_team_id"].astype(str).nunique()
_nb = _v.str.split("|").str[1].nunique()
assert _nb == 5, f"w5 구간 {_nb}개 (5 기대)"
assert _v.nunique() <= 5 * _nt, f"tmw5 {_v.nunique()}칸 > 5 x {_nt}"
print(f"tmw5 검증 OK: {_v.nunique()}칸 (팀 {_nt} x 구간 {_nb}) | 피처 {X_full.shape[1]}개")

'''
ANC = 'print("최종 파라미터:", BEST_PARAMS)'
sub(find(ANC)[0], ANC, GUARD + ANC)
print('자체 검증 셀 삽입 (학습 전)')

sub(find(OLDZIP)[0], OLDZIP, ZIP)
print('zip -> %s' % ZIP)
for i in code:
    ast.parse(src(i))

# ⚠️ 노트북 셀이 문법 OK 여도 **그 안의 script.py 소스 문자열**은 따로다.
# 배포본은 학습이 끝난 뒤에야 그걸 ast.parse 하므로, 거기서 터지면 GPU 42분이 날아간다.
# 여기서 미리 뽑아 파싱한다 (실제로 이걸로 42분짜리 IndentationError 를 겪었다).
_n_checked = 0
for i in code:
    for node in ast.walk(ast.parse(src(i))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str)                 and 'def main(' in node.value and 'df_proc' in node.value:
            ast.parse(node.value)          # 여기서 터지면 빌드가 멈춘다
            _n_checked += 1
if _n_checked == 0:
    sys.exit('script.py 소스 문자열을 못 찾았다 -- 검사가 안 됐다')
print('생성될 script.py 문법 OK (%d개 소스)' % _n_checked)

json.dump(nb, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('%s 생성 -- %d셀 문법 OK' % (OUT, len(code)))

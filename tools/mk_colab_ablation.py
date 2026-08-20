# 절개 실험을 코랩에서 **무인으로 연속 실행**하기 위한 빌더.
#
# 캐글은 커널 하나에 실행 하나라 변형마다 노트북을 따로 올려야 하는데, 코랩은 세션 하나가
# 몇 시간 살아있으므로 한 노트북에서 여러 변형을 순서대로 돌리는 게 낫다. 다만 같은
# 프로세스에서 반복하면 전역이 남는다 (예: feat_rp 에 지난 회차의 past_rest_* 가 그대로
# 붙어 있다). 그래서 **변형마다 새 프로세스**로 돌린다.
#
# 구조:
#   1) s1o 노트북(= v5 기준선, 전 플래그 off)을 .py 로 합친다.
#   2) 네 플래그를 환경변수에서 읽도록 바꾼다  -> 스크립트 하나로 모든 변형을 표현
#   3) 그 .py 를 base64 로 드라이버 노트북에 심는다 (파일 업로드 불필요)
#   4) 드라이버가 변형마다 별도 디렉터리에서 subprocess 로 실행하고,
#      끝나는 즉시 submit_s1{tag}.zip 을 드라이브에 복사한다
#      -> 세션이 중간에 끊겨도 그때까지 끝난 것은 남는다
#
# 사용: python tools/mk_colab_ablation.py r b d
import base64, json, os, sys

BASE = '.kernels/s1o/aimers_s1o.ipynb'
OUT = 'colab/aimers_ablation_colab.ipynb'

# 변형 정의: (AB_DROP_CAL, AB_DECAY, AB_REST, AB_PB)
VAR = {
    'o': ('[]',                                 '1.0',  '0', '0'),
    'm': ('["game_month", "game_dayofweek"]',   '1.0',  '0', '0'),
    'r': ('[]',                                 '1.0',  '1', '0'),
    'b': ('[]',                                 '1.0',  '0', '1'),
    'd': ('[]',                                 '0.25', '0', '0'),
}
DESC = {'o': 'v5 기준선 (전부 off)', 'm': '월/요일 제거', 'r': '휴식·등판밀도·파울',
        'b': 'cond_pb 맞대결', 'd': 'cond_* 감쇠 0.25'}

tags = sys.argv[1:] or ['r', 'b', 'd']
for t in tags:
    assert t in VAR, t

# ---------- 1) 노트북 -> .py ----------
nb = json.load(open(BASE, encoding='utf-8'))
parts = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    parts.append(f'# ===== cell {i} =====\n' + ''.join(c['source']))
script = '\n\n'.join(parts) + '\n'

# ---------- 2) 플래그를 환경변수로 ----------
old_flags = [l for l in script.split('\n')
             if l.startswith(('DROP_CAL =', 'COND_DECAY =', 'USE_REST_FOUL =',
                              'USE_COND_PB =', 'SEEDS ='))]
assert len(old_flags) == 5, f'플래그 줄 {len(old_flags)}개 (5개 기대): {old_flags}'
NEW = '''DROP_CAL = __import__("json").loads(__import__("os").environ.get("AB_DROP_CAL", "[]"))
COND_DECAY = float(__import__("os").environ.get("AB_DECAY", "1.0"))
USE_REST_FOUL = __import__("os").environ.get("AB_REST", "0") == "1"
USE_COND_PB = __import__("os").environ.get("AB_PB", "0") == "1"
SEEDS = [42]'''
for j, l in enumerate(old_flags):
    script = script.replace(l, NEW if j == 0 else '', 1)
for nm in ('AB_DROP_CAL', 'AB_DECAY', 'AB_REST', 'AB_PB'):
    assert nm in script, nm
# 드라이브 어디에 데이터를 뒀는지 모르므로 탐색 깊이를 넓힌다 (0단계 + 재귀 폴백 추가).
PAT_FIX = [
    ('    "/content/drive/MyDrive/*/train.csv",',
     '    "/content/drive/MyDrive/train.csv",\n'
     '    "/content/drive/MyDrive/*/train.csv",'),
    ('    "/content/*/train.csv",',
     '    "/content/*/train.csv",\n'
     '    "/content/drive/MyDrive/**/train.csv",      # 마지막 폴백 (느릴 수 있다)'),
]
for _o, _n in PAT_FIX:
    assert script.count(_o) == 1, _o
    script = script.replace(_o, _n)

compile(script, 'ablation.py', 'exec')          # 문법 검사

b64 = base64.b64encode(script.encode('utf-8')).decode('ascii')
b64_lines = '\n'.join(b64[i:i + 100] for i in range(0, len(b64), 100))

# ---------- 3) 드라이버 노트북 ----------
SETUP = '''# --- 코랩 준비: 드라이브 마운트 + catboost ---
import os, subprocess, sys
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
try:
    import catboost  # noqa: F401
except ImportError:
    print("catboost 설치 중...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost"], check=True)

# GPU 가드 — 없으면 즉시 중단. 조용히 CPU 로 몇 시간 태우는 사고를 막는다.
_r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True, text=True)
if _r.returncode != 0:
    raise RuntimeError("GPU 가 없다. 런타임 > 런타임 유형 변경 > T4 GPU 로 바꿀 것.")
print("GPU:", _r.stdout.strip(), flush=True)
'''

WRITE = f'''# --- 학습 스크립트 풀기 (변형 네 개를 환경변수로 전환하는 단일 스크립트) ---
import base64, pathlib
_B64 = """{b64_lines}"""
_src = base64.b64decode("".join(_B64.split())).decode("utf-8")
pathlib.Path("/content/ablation.py").write_text(_src, encoding="utf-8")
compile(_src, "ablation.py", "exec")
print(f"ablation.py {{len(_src):,}}자, 문법 OK")
'''

runs = ',\n    '.join(
    f'("{t}", {{"AB_DROP_CAL": \'{VAR[t][0]}\', "AB_DECAY": "{VAR[t][1]}", '
    f'"AB_REST": "{VAR[t][2]}", "AB_PB": "{VAR[t][3]}"}}),   # {DESC[t]}'
    for t in tags)

DRIVE = '/content/drive/MyDrive/aimers_ablation'
RUN = f'''# --- 변형을 순서대로 실행. 하나 끝날 때마다 즉시 드라이브에 저장한다 ---
# 세션이 중간에 끊겨도 그때까지 끝난 변형의 zip 은 드라이브에 남는다.
import os, shutil, subprocess, sys, time

RUNS = [
    {runs}
]
OUT = "{DRIVE}"
os.makedirs(OUT, exist_ok=True)

for tag, env in RUNS:
    dst = f"{{OUT}}/submit_s1{{tag}}.zip"
    if os.path.exists(dst):
        print(f"[{{tag}}] 이미 있음 — 건너뜀", flush=True)
        continue
    wd = f"/content/run_{{tag}}"
    os.makedirs(wd, exist_ok=True)          # 변형마다 별도 CWD (model/ 이 섞이지 않게)
    e = dict(os.environ, **env)
    t0 = time.time()
    print(f"\\n{{'='*60}}\\n[{{tag}}] 시작  {{env}}\\n{{'='*60}}", flush=True)
    log = f"{{OUT}}/log_s1{{tag}}.txt"
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen([sys.executable, "-u", "/content/ablation.py"],
                             cwd=wd, env=e, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace")
        for line in p.stdout:
            lf.write(line); lf.flush()
            if any(k in line for k in ("fold", "오프셋", "홀드아웃 환산", "Error",
                                       "Traceback", "통과", "생성 완료", "룩업", "휴식")):
                print(f"  [{{tag}}] {{line.rstrip()}}", flush=True)
        rc = p.wait()
    m = (time.time() - t0) / 60
    src = f"{{wd}}/submit_tuned.zip"
    if rc == 0 and os.path.exists(src):
        shutil.copy(src, dst)
        print(f"[{{tag}}] 완료 {{m:.0f}}분 -> {{dst}}  ({{os.path.getsize(dst)/1e6:.0f}}MB)", flush=True)
    else:
        print(f"[{{tag}}] 실패 rc={{rc}} ({{m:.0f}}분). 로그: {{log}}", flush=True)
    shutil.rmtree(wd, ignore_errors=True)   # 디스크 확보 (변형당 model/ 20MB + 중간 산출물)

print("\\n전부 종료. 드라이브:", os.listdir(OUT), flush=True)
'''


def cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


out = {"cells": [cell(SETUP), cell(WRITE), cell(RUN)],
       "metadata": {"accelerator": "GPU",
                    "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                    "language_info": {"name": "python"}},
       "nbformat": 4, "nbformat_minor": 0}

import ast
for c in out['cells']:
    ast.parse(''.join(c['source']))

os.makedirs('colab', exist_ok=True)
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'{OUT} 생성 — 변형 {tags}, 스크립트 {len(script):,}자, 문법 OK')

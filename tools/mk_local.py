# 본 학습을 **로컬 CPU** 로 돌리기 위한 스크립트 생성기.
#
# 왜 CPU 인가: 2026-08-21 스크리너에서 같은 파라미터·같은 분할인데 CPU 816 / GPU ~790
# 이 나왔다. 원인은 CatBoost 가 두 기계에서 기본값을 다르게 잡기 때문이다.
#   border_count        CPU 254 / GPU 128    -> bc128 실험에서 -12 로 확인됨
#   bootstrap_type      CPU MVS / GPU Bayesian  (MVS 는 CPU 전용이라 GPU 로는 못 옮긴다)
# 앞의 절반은 GPU 에 border_count=254 한 줄로 회수되지만, 뒤의 절반은 CPU 로 학습해야만
# 가진다. 로컬 16코어 실측으로 30모델에 5.6시간이라 하룻밤이면 끝난다.
#
# s1o 노트북(= v5, 전 플래그 off)의 코드 셀을 전부 .py 로 합치고 아래만 바꾼다:
#   task_type GPU -> CPU, thread_count=-1, border_count 명시
#   DATA_DIR -> ./data
# 학습/오프셋/script.py 생성/zip 은 **검증된 원본 코드 경로를 그대로 쓴다**.
#
# 사용:
#   python tools/mk_local.py            # 만들기만
#   python out/train_local.py           # 실행 (5~6시간)
import ast, json, os, sys

BASE = os.environ.get('LOCAL_BASE', '.kernels/s1o/aimers_s1o.ipynb')
OUT_DIR = 'out'
OUT = f'{OUT_DIR}/train_local.py'

nb = json.load(open(BASE, encoding='utf-8'))
parts = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    parts.append(f'# ===== cell {i} =====\n' + ''.join(c['source']))
src = '\n\n'.join(parts) + '\n'


def sub(old, new, n=1):
    global src
    if src.count(old) != n:
        sys.exit(f'앵커 {n}개 기대, {src.count(old)}개:\n  {old[:110]}')
    src = src.replace(old, new)


# ---------- 1. GPU -> CPU ----------
# 파라미터 딕셔너리에 박힌 task_type 을 전부 바꾼다 (Optuna objective / BEST_PARAMS).
sub('"task_type": "GPU",', '"task_type": "CPU", "thread_count": -1,')
sub('BEST_PARAMS["task_type"] = "GPU"',
    '''BEST_PARAMS["task_type"] = "CPU"
BEST_PARAMS["thread_count"] = -1
# CPU 기본값이지만 명시해 둔다. GPU 로 되돌릴 때 이 줄만 살리면 절반은 회수된다.
BEST_PARAMS["border_count"] = int(os.environ.get("LOCAL_BORDER", "254"))
# bagging_temperature 는 Bayesian 전용이라 CPU(MVS)에서는 무시된다.
# MVS 를 쓰려면 넘기지 않는 편이 명확하다 — 넘겨도 동작은 같지만 로그가 헷갈린다.
BEST_PARAMS.pop("bagging_temperature", None)
print("CPU 학습 파라미터:", {k: v for k, v in BEST_PARAMS.items() if k != "cat_features"})''')

# ---------- 2. 데이터 경로 ----------
# out/ 에서 실행하므로 상대경로는 어긋난다. 빌드 시점의 절대경로를 박아 넣고,
# 원본 리졸버의 초기값(DATA_DIR = None)을 그것으로 바꾼다 -> 탐색 루프가 즉시 통과한다.
_DATA = os.path.abspath('data').replace(os.sep, '/')
if not os.path.isfile(f'{_DATA}/train.csv'):
    sys.exit(f'{_DATA}/train.csv 가 없다 — 레포 루트에서 실행할 것')
sub('DATA_DIR = None', f'DATA_DIR = "{_DATA}"   # 빌드 시점에 박은 절대경로')

# ---------- 3. 산출물 이름 ----------
sub('ZIP_PATH = "submit_tuned.zip"', 'ZIP_PATH = "submit_local_cpu.zip"')

# ---------- 검사 ----------
if 'task_type": "GPU"' in src or '= "GPU"' in src:
    sys.exit('GPU 문자열이 남아 있다')
compile(src, OUT, 'exec')

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(src)
print(f'{OUT} 생성 — {len(src):,}자, 문법 OK')
print('실행:  cd out && PYTHONUTF8=1 python train_local.py')
print('  * model/ 과 submit_local_cpu.zip 이 out/ 아래에 만들어진다')
print('  * 30모델 기준 로컬 16코어에서 약 5~6시간')

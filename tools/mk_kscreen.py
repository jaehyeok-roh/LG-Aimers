# 캐글 **CPU** 커널로 스크리너를 돌린다.
#
# 캐글 CPU 세션은 GPU 쿼터를 쓰지 않는다 (세션당 12시간, 4코어/30GB). 게다가 CPU 라서
# bootstrap_type=MVS / border_count=254 — 2026-08-21 에 확인한 '좋은 기본값' 쪽이므로
# 로컬 CPU 스크리너(기준선 816)와 직접 비교할 수 있다. 코랩 GPU 로 돌리면 기본값이
# 달라 절대값을 섞어 읽을 수 없다.
#
# 구조:
#   1) s1o 노트북의 코드 셀을 feature-selection 까지 실행 -> X_full / y_full / cat_features
#   2) make_cache 와 같은 레이아웃으로 cache/ 에 저장
#   3) tools/screen.py 를 base64 로 심어 풀고 subprocess 로 실행
#      -> 로컬에서 검증된 **같은 코드**가 돈다. 후보 정의를 두 벌 관리하지 않는다.
#
# 사용: python tools/mk_kscreen.py base diff
import ast, base64, json, os, sys

BASE = '.kernels/s1o/aimers_s1o.ipynb'
STOP = 'BEST_PARAMS = _found'
EXTRA_ENV = json.loads(os.environ.get('KS_ENV', '{}'))  # 커널 안으로 넘길 환경변수
# 작은 부속 파일을 base64 로 커널에 같이 싣는다 (예: 구종별 실력 룩업 382KB)
EXTRA_FILES = [f for f in os.environ.get('KS_FILES', '').split(',') if f]
USE_GPU = os.environ.get('KS_GPU', '0') == '1'
TAG = os.environ.get('KS_TAG', 'kscreen')   # 커널을 동시에 여러 개 띄우려면 바꾼다
SCRIPT = os.environ.get('KS_SCRIPT', 'tools/screen.py')   # 다른 도구도 같은 커널로 돌린다
CANDS = sys.argv[1:] or ['base', 'diff']

screen_src = open(SCRIPT, encoding='utf-8').read()
compile(screen_src, 'screen.py', 'exec')
b64 = base64.b64encode(screen_src.encode('utf-8')).decode('ascii')
b64 = '\n'.join(b64[i:i + 100] for i in range(0, len(b64), 100))

SAVE = ('_USE_GPU = %r\n' % USE_GPU) + '''# ===== 캐시 저장 (tools/make_cache.py 와 같은 레이아웃) =====
import json
import os
import numpy as np
# 큰 캐시는 /tmp 에 둔다 — /kaggle/working 에 두면 커널 출력에 통째로
# 실려서 결과 회수가 수 분씩 걸린다 (835MB).
CDIR = "/tmp/cache" if os.path.isdir("/kaggle") else "cache"
os.makedirs(CDIR, exist_ok=True)
os.makedirs("cache", exist_ok=True)
_season = df_processed["season"].to_numpy()
X_full.to_pickle(f"{CDIR}/X.pkl")
np.save(f"{CDIR}/y.npy", y_full.to_numpy())
np.save(f"{CDIR}/season.npy", _season)
np.save(f"{CDIR}/row_id.npy", df_processed["row_id"].to_numpy())
_bp = {k: v for k, v in BEST_PARAMS.items() if k != "cat_features"}
# ⚠️ KS_GPU=1 로 GPU 커널을 띄웠으면 학습도 GPU 로 해야 한다. 예전엔 여기서
# task_type 을 CPU 로 못박아서, GPU 머신 위에서 CPU 학습이 돌아 쿼터만 태웠다
# (2026-08-29 kmcs: 5분류 2후보를 CPU 로 돌려 13시간 예상, 12시간 제한에 걸릴 뻔).
_bp["task_type"] = "GPU" if os.path.isdir("/kaggle") and _USE_GPU else "CPU"
if _bp["task_type"] == "CPU":
    _bp.pop("bagging_temperature", None)   # Bayesian 전용 — MVS 에서는 무시된다
json.dump({"cat_features": list(cat_features), "best_params": _bp,
           "n_rows": int(len(X_full)), "n_feats": int(X_full.shape[1])},
          open(f"{CDIR}/meta.json", "w"), indent=2, ensure_ascii=False)
# 재구축한 투수 매핑을 파일로 떨군다 — 캐글 데이터셋엔 없고
# 노트북이 메모리에서 만들기 때문에 screen.py 가 못 찾는다.
os.makedirs("data", exist_ok=True)
pitcher_id_mapping.to_csv("data/pitcher_id_mapping_v2.csv", index=False)
print("매핑 덤프", len(pitcher_id_mapping), flush=True)
print("캐시 저장 완료", X_full.shape, flush=True)
'''

EXTRA_B64 = {f: __import__('base64').b64encode(open(f,'rb').read()).decode()
             for f in EXTRA_FILES}

RUN = f'''# ===== 스크리너 실행 =====
import base64, pathlib, subprocess, sys, os, json
_B64 = """{b64}"""
pathlib.Path("screen.py").write_text(
    base64.b64decode("".join(_B64.split())).decode("utf-8"), encoding="utf-8")
print("screen.py 풀기 완료", flush=True)
import base64 as _b64
for _p, _d in {EXTRA_B64!r}.items():
    pathlib.Path(_p).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(_p).write_bytes(_b64.b64decode(_d))
    print("부속 파일", _p, pathlib.Path(_p).stat().st_size, "바이트", flush=True)

CANDS = {CANDS!r}
_env = dict(os.environ, SCREEN_CACHE=("/tmp/cache" if os.path.isdir("/kaggle") else "cache"))
_env.update({EXTRA_ENV!r})
print("추가 환경변수", {EXTRA_ENV!r}, flush=True)
p = subprocess.Popen([sys.executable, "-u", "screen.py"] + CANDS,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, encoding="utf-8", errors="replace", env=_env)
for line in p.stdout:
    print(line.rstrip(), flush=True)
rc = p.wait()
print("종료 코드", rc, flush=True)
if os.path.exists("cache/screen_results.json"):
    print(json.dumps(json.load(open("cache/screen_results.json")),
                     indent=1, ensure_ascii=False))
'''

nb = json.load(open(BASE, encoding='utf-8'))
srcs = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
last = next(i for i, s in enumerate(srcs) if STOP in s)


def cell(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.splitlines(keepends=True)}


cells = [cell(s) for s in srcs[:last + 1]] + [cell(SAVE), cell(RUN)]
for c in cells:
    ast.parse(''.join(c['source']))

out = {"cells": cells,
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                    "language_info": {"name": "python"}},
       "nbformat": 4, "nbformat_minor": 0}

D = f'.kernels/{TAG}'
os.makedirs(D, exist_ok=True)
json.dump(out, open(f'{D}/aimers_{TAG}.ipynb', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
meta = json.load(open('kernel-metadata.json'))
meta.update(id=f'your-kaggle-id/aimers-{TAG}', title=f'aimers-{TAG}',
            code_file=f'aimers_{TAG}.ipynb', enable_gpu=USE_GPU)
# ⚠️ 기본은 CPU (쿼터 안 씀, 동시 5개). 다중분류는 CPU 가 25배 느려서
#    (5분류 3폴드에 400분) KS_GPU=1 로 켤 것. 단 GPU 는 기본값이 달라
#    (bootstrap_type Bayesian, border_count 128) CPU 결과와 절대값을 섞지 말 것 —
#    같은 GPU 실행 안에서 짝지어 비교해야 한다.
json.dump(meta, open(f'{D}/kernel-metadata.json', 'w'), indent=2)
print(f'{D}/aimers_{TAG}.ipynb — 후보 {CANDS}, 셀 {len(cells)}개, enable_gpu={USE_GPU}, 문법 OK')

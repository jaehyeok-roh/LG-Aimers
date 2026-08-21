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
TAG = 'kscreen'
CANDS = sys.argv[1:] or ['base', 'diff']

screen_src = open('tools/screen.py', encoding='utf-8').read()
compile(screen_src, 'screen.py', 'exec')
b64 = base64.b64encode(screen_src.encode('utf-8')).decode('ascii')
b64 = '\n'.join(b64[i:i + 100] for i in range(0, len(b64), 100))

SAVE = '''# ===== 캐시 저장 (tools/make_cache.py 와 같은 레이아웃) =====
import os
import numpy as np
os.makedirs("cache", exist_ok=True)
_season = df_processed["season"].to_numpy()
X_full.to_pickle("cache/X.pkl")
np.save("cache/y.npy", y_full.to_numpy())
np.save("cache/season.npy", _season)
np.save("cache/row_id.npy", df_processed["row_id"].to_numpy())
_bp = {k: v for k, v in BEST_PARAMS.items() if k != "cat_features"}
_bp["task_type"] = "CPU"          # 캐글 CPU 커널이다
_bp.pop("bagging_temperature", None)   # Bayesian 전용 — MVS 에서는 무시된다
json.dump({"cat_features": list(cat_features), "best_params": _bp,
           "n_rows": int(len(X_full)), "n_feats": int(X_full.shape[1])},
          open("cache/meta.json", "w"), indent=2, ensure_ascii=False)
print("캐시 저장 완료", X_full.shape, flush=True)
'''

RUN = f'''# ===== 스크리너 실행 =====
import base64, pathlib, subprocess, sys, os, json
_B64 = """{b64}"""
pathlib.Path("screen.py").write_text(
    base64.b64decode("".join(_B64.split())).decode("utf-8"), encoding="utf-8")
print("screen.py 풀기 완료", flush=True)

CANDS = {CANDS!r}
p = subprocess.Popen([sys.executable, "-u", "screen.py"] + CANDS,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, encoding="utf-8", errors="replace")
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
meta.update(id=f'homekeggle/aimers-{TAG}', title=f'aimers-{TAG}',
            code_file=f'aimers_{TAG}.ipynb', enable_gpu=False)   # ← GPU 쿼터 안 씀
json.dump(meta, open(f'{D}/kernel-metadata.json', 'w'), indent=2)
print(f'{D}/aimers_{TAG}.ipynb — 후보 {CANDS}, 셀 {len(cells)}개, enable_gpu=False, 문법 OK')

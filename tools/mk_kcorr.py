# 예측 상관 측정을 **캐글 CPU 커널**로 돌린다 (로컬 CPU 를 쉬게 한다).
#
# 제출 zip 들은 `homekeggle/aimers-zips` 데이터셋으로 올리고, train.csv/트랙맨은
# 기존 `homekeggle/aimers` 를 그대로 쓴다. pred_corr.py 를 base64 로 심어
# 로컬에서 검증된 **같은 코드**가 돌게 한다.
#
# 사용: python tools/mk_kcorr.py submit_local_cpu.zip submit_v9m.zip ...
#        (인자는 데이터셋 안에서의 **파일 이름**이지 로컬 경로가 아니다)
import ast, base64, json, os, sys

TAG = os.environ.get('KC_TAG', 'kcorr')
DS = os.environ.get('KC_DS', 'homekeggle/aimers-zips')
NAMES = sys.argv[1:] or ['submit_local_cpu.zip', 'submit_v9m.zip', 'submit_s1lr.zip']
ROWS = os.environ.get('KC_ROWS', '150000')

src = open('tools/pred_corr.py', encoding='utf-8').read()
compile(src, 'pred_corr.py', 'exec')
b64 = base64.b64encode(src.encode('utf-8')).decode('ascii')
b64 = '\n'.join(b64[i:i + 100] for i in range(0, len(b64), 100))

CELL = f'''import base64, pathlib, os, shutil, subprocess, sys, glob
_B64 = """{b64}"""
pathlib.Path("pred_corr.py").write_text(
    base64.b64decode("".join(_B64.split())).decode("utf-8"), encoding="utf-8")

# 마운트 경로를 **가정하지 않는다**. 처음 push 때 '/kaggle/input/aimers-zips' 를
# 하드코딩했다가 FileNotFoundError 로 죽었다.
print("input 마운트:", sorted(os.listdir("/kaggle/input")), flush=True)


def _tree():
    for r, d, f in os.walk("/kaggle/input"):
        if r.count("/") < 7:
            print("   ", r, "->", sorted(d)[:6], sorted(f)[:6], flush=True)


# ⚠️ 이 계정은 /kaggle/input/datasets/... 로 한 단계 더 깊이 마운트된다.
# 깊이를 가정하지 말고 재귀로 찾는다 (2단계까지만 보다가 두 번 죽었다).
_c = glob.glob("/kaggle/input/**/train.csv", recursive=True)
_z = glob.glob("/kaggle/input/**/submit_*.zip", recursive=True)
if not _c or not _z:
    _tree()
    raise SystemExit(f"못 찾음 — train.csv {{len(_c)}}개 / zip {{len(_z)}}개")
DATA = os.path.dirname(_c[0])
DS_DIR = os.path.dirname(_z[0])
print("데이터 폴더:", DATA, flush=True)
print("zip 폴더:", DS_DIR, sorted(os.listdir(DS_DIR)), flush=True)

# 캐글 input 은 읽기 전용이라 zip 을 작업 폴더로 복사한다
names = {NAMES!r}
paths = []
for n in names:
    s = os.path.join(DS_DIR, n)
    if not os.path.exists(s):
        print("없음, 건너뜀:", n, flush=True)
        continue
    shutil.copy(s, n)
    paths.append(n)
print("대상 zip", paths, flush=True)

env = dict(os.environ, PC_DATA=DATA, PYTHONUTF8="1")
p = subprocess.Popen([sys.executable, "-u", "pred_corr.py"] + paths + ["--rows={ROWS}"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, encoding="utf-8", errors="replace", env=env)
for line in p.stdout:
    print(line.rstrip(), flush=True)
print("종료 코드", p.wait(), flush=True)
'''

ast.parse(CELL)
cells = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
          "source": CELL.splitlines(keepends=True)}]
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
            code_file=f'aimers_{TAG}.ipynb', enable_gpu=False,
            dataset_sources=['homekeggle/aimers', DS])
json.dump(meta, open(f'{D}/kernel-metadata.json', 'w'), indent=2)
print(f'{D}/aimers_{TAG}.ipynb — zip {NAMES}, {ROWS}행, enable_gpu=False, 문법 OK')

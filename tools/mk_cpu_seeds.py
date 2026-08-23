# 본 학습을 캐글 **CPU** 커널 3개로 쪼개 돌린다 (seed 하나씩).
#
# 왜 CPU: 같은 파라미터인데 CPU 가 GPU 보다 홀드아웃 +26 (리더보드 실측 +5.17).
# CatBoost 기본값이 다르기 때문 — border_count 254/128, bootstrap_type MVS/Bayesian.
# **MVS 는 CPU 전용**이라 GPU 로는 절반만 회수된다.
#
# 왜 쪼개나: 캐글 CPU 는 4코어라 1모델 ~27분. 30모델이면 13.5시간으로 세션 한도(12h)
# 를 넘는다. seed 하나(10모델)면 ~4.5시간이라 넉넉히 들어가고, 세 커널이 **병렬**로
# 돌아 벽시계로도 4.5시간이다. 로컬 CPU 5.6시간짜리와 결과가 같다.
#
# 병합 근거: 전처리가 결정적이라 룩업 테이블이 커널 간 동일하다 (CLAUDE.md 4-12 선례).
# 모델 파일만 모아서 zip 을 다시 조립하면 된다.
#
# 사용:
#   python tools/mk_cpu_seeds.py                 # 커널 3개 생성
#   python tools/mk_cpu_seeds.py --push          # 생성 + push
import ast, json, os, subprocess, sys

BASE = os.environ.get('CS_BASE', 'aimers_v10w.ipynb')
PFX = os.environ.get('CS_PREFIX', 'cpu')   # 커널 이름 접두어
TAGS = {'a': 42, 'b': 202, 'c': 2024}
PUSH = '--push' in sys.argv

for tag, seed in TAGS.items():
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
            sys.exit(f'[{tag}] 셀 {i}: 앵커 {n}개 기대, {s.count(old)}개\n  {old[:90]}')
        setsrc(i, s.replace(old, new))

    # --- seed 하나만 ---
    done = False
    for i in code:
        if 'SEEDS = [42, 202, 2024]' in src(i):
            sub(i, 'SEEDS = [42, 202, 2024]', f'SEEDS = [{seed}]')
            done = True
            break
    if not done:
        sys.exit('SEEDS 앵커를 못 찾음')

    # --- GPU -> CPU (MVS + border_count 254) ---
    for i in code:
        s = src(i)
        if '"task_type": "GPU",' in s:
            sub(i, '"task_type": "GPU",', '"task_type": "CPU", "thread_count": -1,',
                s.count('"task_type": "GPU",'))
        if 'BEST_PARAMS["task_type"] = "GPU"' in s:
            sub(i, 'BEST_PARAMS["task_type"] = "GPU"',
                'BEST_PARAMS["task_type"] = "CPU"\n'
                'BEST_PARAMS["thread_count"] = -1\n'
                '# CPU 기본값이지만 명시한다 (GPU 는 128 이라 -12)\n'
                'BEST_PARAMS["border_count"] = 254\n'
                '# bagging_temperature 는 Bayesian 전용 — MVS 를 쓰려면 넘기지 않는다\n'
                'BEST_PARAMS.pop("bagging_temperature", None)\n'
                'print("CPU 파라미터:", {k: v for k, v in BEST_PARAMS.items() '
                'if k != "cat_features"})')

    # --- zip 이름 ---
    for i in code:
        if 'ZIP_PATH = ' in src(i):
            s = src(i)
            j = s.index('ZIP_PATH = ')
            k = s.index('\n', j)
            sub(i, s[j:k], f'ZIP_PATH = "submit_cpu_{tag}.zip"')
            break

    for i in code:
        ast.parse(src(i))
    if 'task_type": "GPU"' in json.dumps(nb):
        sys.exit(f'[{tag}] GPU 문자열이 남아 있다')

    D = f'.kernels/{PFX}{tag}'
    os.makedirs(D, exist_ok=True)
    name = f'aimers_cpu{tag}'
    json.dump(nb, open(f'{D}/{name}.ipynb', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    meta = json.load(open('kernel-metadata.json'))
    meta.update(id=f'homekeggle/aimers-{PFX}{tag}', title=f'aimers-{PFX}{tag}',
                code_file=f'{name}.ipynb', enable_gpu=False)
    json.dump(meta, open(f'{D}/kernel-metadata.json', 'w'), indent=2)
    print(f'{D}/{name}.ipynb — seed {seed}, CPU, 문법 OK')

    if PUSH:
        r = subprocess.run(['kaggle', 'kernels', 'push', '-p', '.'], cwd=D,
                           capture_output=True, text=True,
                           encoding='utf-8', errors='replace',
                           env=dict(os.environ, PYTHONUTF8='1'))
        print('   ', (r.stdout or r.stderr).strip().splitlines()[-1])

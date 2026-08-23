# ws_* 상수가 script.py 까지 실제로 전달되는지 **대조군**으로 확인한다.
#
# 4-12 교훈 3: 상수가 조용히 무시돼도 크래시는 안 난다. 그래서 '있다' 를 확인하는
# 것으로는 부족하고, **없앤 판과 예측이 달라지는지**를 봐야 한다.
# 같으면 script.py 가 그 상수를 안 쓰고 있다는 뜻이다 (= wseason 이 죽어 있다).
import json, os, shutil, subprocess, sys, tempfile, zipfile
import numpy as np, pandas as pd

ZIP = sys.argv[1] if len(sys.argv) > 1 else 'out/submit_cpu30.zip'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
PREFIX = sys.argv[3] if len(sys.argv) > 3 else 'ws_'   # 'wb_' 로 타자측도 검사
work = tempfile.mkdtemp(prefix='wslive_')
_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
tr = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
sl = tr[tr['season'] == 2024].head(N).drop(columns=['control_success']).copy()
del tr


def run(strip_ws, tag):
    d = os.path.join(work, tag)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(d)
    if strip_ws:
        p = os.path.join(d, 'model', 'train_constants.json')
        tc = json.load(open(p, encoding='utf-8'))
        for k in [k for k in tc if k.startswith(PREFIX)]:
            tc.pop(k)
        json.dump(tc, open(p, 'w', encoding='utf-8'), indent=2)
    dd = os.path.join(d, 'data'); os.makedirs(dd, exist_ok=True)
    sl.to_csv(os.path.join(dd, 'test.csv'), index=False)
    pd.DataFrame({'row_id': sl['row_id'], 'control_success': .5}).to_csv(
        os.path.join(dd, 'sample_submission.csv'), index=False)
    shutil.copy('data/trackman_history.csv', os.path.join(dd, 'trackman_history.csv'))
    r = subprocess.run([sys.executable, '-u', 'script.py'], cwd=d, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    out = os.path.join(d, 'output', 'submission.csv')
    if not os.path.exists(out):
        print((r.stdout or '')[-600:]); print((r.stderr or '')[-800:])
        raise RuntimeError(f'{tag} 실패')
    return pd.read_csv(out).set_index('row_id')['control_success']


print(f'zip {ZIP} | {N}행\n', flush=True)
a = run(False, 'with_ws')
print(f'  {PREFIX}* 있음  평균 {a.mean():.6f}  std {a.std():.6f}', flush=True)
b = run(True, 'without_ws')
print(f'  {PREFIX}* 없음  평균 {b.mean():.6f}  std {b.std():.6f}', flush=True)

d = (a - b).abs()
print(f'\n행별 차이: 평균 {d.mean():.6f}  최대 {d.max():.6f}  0이 아닌 행 {(d>1e-9).mean():.1%}')
print('=' * 60)
if d.max() < 1e-9:
    print(f'❌ 예측이 완전히 같다 — script.py 가 {PREFIX}* 를 쓰지 않는다.')
    print('   wseason 이 죽은 채로 제출되는 상태다. 절대 올리지 말 것.')
    sys.exit(1)
print(f'✅ {PREFIX}* 가 살아 있다 (예측이 실제로 달라진다)')
print('=' * 60)

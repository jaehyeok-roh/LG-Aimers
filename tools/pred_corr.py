# 제출 zip 여러 개의 **예측 상관**을 제출 없이 잰다.
#
# 왜: 블렌드 이득은 점수가 아니라 상관이 결정한다. 트리 계열을 아무리 바꿔도 예측 상관이
# 0.92 아래로 안 내려가서 블렌드가 +7 에서 포화했다 (fam3). 상관 0.85 아래가 나오는
# 조합을 찾는 것이 앙상블에 남은 유일한 여지다.
#
# 각 zip 의 script.py 를 train 한 조각(= test 로 위장)에 돌려 output/submission.csv 를
# 받는다. 패치가 필요 없다 — script.py 가 이미 row_id + 확률을 쓴다.
#
# ⚠️ 이 행들은 두 모델 **모두**의 학습에 들어가 있다. 따라서
#   - 점수(skill)는 in-sample 이라 의미 없다. 참고로만 찍는다.
#   - 상관은 의미가 있으나 실제 test 보다 **높게** 나오는 쪽으로 편향된다
#     (둘 다 같은 정답에 잘 맞춰져 있으므로). 즉 여기서 0.85 면 실전은 그 이하다.
#
# 사용: python tools/pred_corr.py A.zip B.zip [C.zip ...] [--rows=150000] [--season=2024]
import os, shutil, subprocess, sys, tempfile, zipfile
import numpy as np
import pandas as pd

zips, ROWS, SEASON = [], 150000, 2024
DATA = os.environ.get('PC_DATA', 'data')   # 캐글에서는 /kaggle/input/aimers
for a in sys.argv[1:]:
    if a.startswith('--rows='):
        ROWS = int(a.split('=')[1])
    elif a.startswith('--season='):
        SEASON = int(a.split('=')[1])
    else:
        zips.append(a)
if len(zips) < 2:
    sys.exit('zip 을 2개 이상 넘길 것')
zips = [z for z in zips if os.path.exists(z) or sys.stderr.write(f'없음(건너뜀): {z}\n')]

root = tempfile.mkdtemp(prefix='predcorr_')
print(f'작업 폴더 {root}\n시즌 {SEASON} / {ROWS:,}행 / zip {len(zips)}개\n', flush=True)

# ---------- 위장 test.csv 를 한 번만 만든다 ----------
tr = pd.read_csv(f'{DATA}/train.csv')
sl = tr[tr['season'] == SEASON].head(ROWS).copy()
truth = sl[['row_id', 'control_success']].rename(columns={'control_success': 'y'})
shared = os.path.join(root, '_data')
os.makedirs(shared, exist_ok=True)
sl.drop(columns=['control_success']).to_csv(f'{shared}/test.csv', index=False)
pd.DataFrame({'row_id': sl['row_id'], 'control_success': 0.5}).to_csv(
    f'{shared}/sample_submission.csv', index=False)
if os.path.exists(f'{DATA}/trackman_history.csv'):
    shutil.copy(f'{DATA}/trackman_history.csv', f'{shared}/trackman_history.csv')
del tr, sl
print(f'위장 test.csv {len(truth):,}행 준비\n', flush=True)

# ---------- zip 별로 실행 ----------
preds = {}
for z in zips:
    name = os.path.splitext(os.path.basename(z.rstrip('/\\')))[0]
    w = os.path.join(root, name)
    if os.path.isdir(z):
        # 캐글은 데이터셋에 올린 zip 을 **자동으로 풀어** 디렉터리로 마운트한다.
        shutil.copytree(z, w)          # input 은 읽기 전용이라 복사해야 한다
    else:
        os.makedirs(w, exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(w)
    n_cbm = len([f for f in os.listdir(os.path.join(w, 'model')) if f.endswith('.cbm')])
    shutil.copytree(shared, os.path.join(w, 'data'))
    print(f'[{name}] 모델 {n_cbm}개 — 실행 중...', flush=True)
    r = subprocess.run([sys.executable, '-u', 'script.py'], cwd=w,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = os.path.join(w, 'output', 'submission.csv')
    if r.returncode != 0 or not os.path.exists(out):
        print((r.stdout or '')[-800:])
        print((r.stderr or '')[-1200:])
        print(f'[{name}] 실패 — 건너뜀\n', flush=True)
        continue
    s = pd.read_csv(out)
    preds[name] = s.set_index('row_id')['control_success']
    print(f'[{name}] 완료  평균 {preds[name].mean():.4f}  '
          f'표준편차 {preds[name].std():.4f}\n', flush=True)

if len(preds) < 2:
    sys.exit('성공한 zip 이 2개 미만이다')

# ---------- 상관 행렬 ----------
P = pd.DataFrame(preds).join(truth.set_index('row_id')['y'], how='inner')
y = P.pop('y').to_numpy()
names = list(P.columns)
print('=' * 60)
print(f'공통 {len(P):,}행\n')

print(f'{"":<24}' + ''.join(f'{n[:12]:>14}' for n in names))
C = P.corr()
for a in names:
    print(f'{a[:22]:<24}' + ''.join(f'{C.loc[a, b]:>14.4f}' for b in names))

r = float(y.mean())


def skill(p):
    return (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / (r * (1 - r))) * 100000


print(f'\n(참고, in-sample 이라 의미 없음) ' +
      '  '.join(f'{n[:12]}={skill(P[n].to_numpy()):,.0f}' for n in names))

print('\n판정 기준: 상관 0.85 아래여야 블렌드에 여지가 있다.')
print('  트리 계열끼리는 최적점에서 0.92~0.97 이었고 블렌드 이득이 +7 에서 포화했다.')
print('  ⚠️ in-sample 상관은 실전보다 높게 나온다 — 여기서 0.85 면 실전은 그 이하다.')
for i, a in enumerate(names):
    for b in names[i + 1:]:
        c = C.loc[a, b]
        v = '★ 블렌드 후보' if c < 0.85 else '포화 구간(기대 +7 이하)'
        print(f'  {a[:16]} x {b[:16]}: {c:.4f}  {v}')
print(f'\n작업 폴더는 남겨둔다: {root}')

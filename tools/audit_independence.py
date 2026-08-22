# 주최측이 명시한 '평가 데이터 독립 예측 원칙' 을 직접 검증한다.
#
# 공지(2026-08-13, talkboard/417123) 원문:
#   "어떤 행의 예측값은 아래 두 경우 모두 동일해야 합니다.
#      - test.csv 에 해당 행 1개만 있는 경우
#      - test.csv 에 전체 평가 데이터가 함께 있는 경우
#    두 경우의 예측값이 달라진다면, 평가 데이터의 다른 행이 추론에 영향을 준 것으로 본다."
#
# 그리고 "제출 코드 모니터링 과정에서 ... 실격 사례도 지속적으로 발생" 이라고 했다.
# 우리 파이프라인은 설계상 행 독립이지만 **한 군데 의심스러운 곳**이 있다:
#   step14_convert_to_category 의 `astype(str).astype('category')` 는
#   카테고리 목록이 **그 배치에 어떤 값이 있느냐**에 따라 달라진다.
#   CatBoost 가 문자열이 아니라 코드를 쓴다면 배치 크기에 따라 예측이 달라진다.
#
# 여기서는 실제로 돌려서 확인한다.
#   1) N행짜리 test 로 예측
#   2) 그중 몇 행을 **1행짜리 test** 로 각각 예측
#   3) 두 값이 같은가
import io, os, shutil, subprocess, sys, tempfile, zipfile
import numpy as np
import pandas as pd

ZIP = sys.argv[1] if len(sys.argv) > 1 else 'out/submit_v10w_fixed.zip'
NROW = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
NPROBE = int(sys.argv[3]) if len(sys.argv) > 3 else 6

work = tempfile.mkdtemp(prefix='indep_')
root = os.path.join(work, 'sub')
with zipfile.ZipFile(ZIP) as z:
    z.extractall(root)
print(f'zip {ZIP}\n작업 폴더 {work}\n', flush=True)

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
tr = pd.read_csv('data/train.csv', keep_default_na=False, na_values=_NA)
sl = tr[tr['season'] == 2024].head(NROW).drop(columns=['control_success']).copy()
del tr


def run(df, tag):
    """주어진 test 로 script.py 를 돌리고 예측을 돌려준다."""
    d = os.path.join(work, tag)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(root, d)
    dd = os.path.join(d, 'data')
    os.makedirs(dd, exist_ok=True)
    df.to_csv(os.path.join(dd, 'test.csv'), index=False)
    pd.DataFrame({'row_id': df['row_id'], 'control_success': 0.5}).to_csv(
        os.path.join(dd, 'sample_submission.csv'), index=False)
    if os.path.exists('data/trackman_history.csv'):
        shutil.copy('data/trackman_history.csv', os.path.join(dd, 'trackman_history.csv'))
    r = subprocess.run([sys.executable, '-u', 'script.py'], cwd=d,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = os.path.join(d, 'output', 'submission.csv')
    if r.returncode != 0 or not os.path.exists(out):
        print((r.stdout or '')[-700:])
        print((r.stderr or '')[-900:])
        raise RuntimeError(f'{tag} 실행 실패')
    return pd.read_csv(out).set_index('row_id')['control_success']


print(f'[전체] {len(sl):,}행으로 예측', flush=True)
full = run(sl, 'full')
print(f'  평균 {full.mean():.6f}  표준편차 {full.std():.6f}\n', flush=True)

rng = np.random.default_rng(0)
idx = rng.choice(len(sl), size=NPROBE, replace=False)
print(f'{"row_id":<18}{"전체 배치":>14}{"1행 단독":>14}{"차이":>14}')
diffs = []
for k, i in enumerate(idx):
    one = sl.iloc[[i]]
    rid = one['row_id'].iloc[0]
    p1 = run(one, f'one{k}')
    a, b = float(full.loc[rid]), float(p1.loc[rid])
    diffs.append(abs(a - b))
    print(f'{rid:<18}{a:>14.8f}{b:>14.8f}{a-b:>+14.2e}', flush=True)

m = max(diffs)
print('\n' + '=' * 62)
if m < 1e-9:
    print(f'✅ 통과 — 최대 차이 {m:.2e} (부동소수 수준)')
    print('   행 독립 예측 원칙을 만족한다.')
else:
    print(f'⚠️ 불일치 — 최대 차이 {m:.2e}')
    print('   test 의 다른 행이 예측에 영향을 준다. 규칙 위반 소지가 있으니 원인을 찾을 것.')
print('=' * 62)
print(f'\n작업 폴더는 남겨둔다: {work}')

# 학습 경로 vs 추론 경로 피처 대조 (행 단위 1:1).
#
# 왜: script.py 는 학습과 **완전히 다른 코드 경로**로 피처를 다시 만든다. 지금까지의
# 검증 셀은 "안 죽는가 / 분포가 상식적인가" 만 봤고 값이 같은지는 한 번도 안 봤다.
# 이런 불일치는 크래시 없이 점수만 깎는다 — 트랙맨 exact(897) vs asof(933) 의 36점이
# 통째로 그 유형이었다. CLAUDE.md 4-7 이 "같은 row_id 기준 1:1 비교" 를 요구하는데
# 피처에 대해서는 지키지 않고 있었다.
#
# 방법: train 의 한 시즌을 test 인 척 넣어 제출 zip 의 script.py 를 돌리고,
#       그때 만들어진 df_features 를 cache/X.pkl 의 같은 row_id 와 컬럼별로 비교한다.
#
# 사용: python tools/audit_parity.py [zip경로] [시즌] [행수]
import io, json, os, shutil, subprocess, sys, tempfile, zipfile
import numpy as np
import pandas as pd

ZIP = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\nojh4\Downloads\submit_v9m.zip'
SEASON = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
NROW = int(sys.argv[3]) if len(sys.argv) > 3 else 60000

work = tempfile.mkdtemp(prefix='parity_')
print(f'작업 폴더 {work}\nzip {ZIP} / 시즌 {SEASON} / {NROW:,}행\n', flush=True)

# ---------- 1) 제출 zip 풀기 ----------
with zipfile.ZipFile(ZIP) as z:
    z.extractall(work)
sp = os.path.join(work, 'script.py')
assert os.path.exists(sp), 'script.py 가 zip 에 없다'

# ---------- 2) script.py 가 피처를 덤프하도록 패치 ----------
src = io.open(sp, encoding='utf-8').read()
ANCHOR = '    # ---------- 추론: 모델 전체 평균 ----------'
assert src.count(ANCHOR) == 1, '덤프 지점 앵커를 못 찾음'
src = src.replace(ANCHOR, '''    # --- 감사용 덤프 (원본 로직은 건드리지 않는다) ---
    _dump = df_features.copy()
    _dump.insert(0, "row_id", row_ids)
    _dump.to_pickle("infer_features.pkl")
    print("감사 덤프: infer_features.pkl", _dump.shape)

''' + ANCHOR)
io.open(sp, 'w', encoding='utf-8').write(src)

# ---------- 3) train 한 조각을 test 로 위장 ----------
data = os.path.join(work, 'data')
os.makedirs(data, exist_ok=True)
tr = pd.read_csv('data/train.csv')
sl = tr[tr['season'] == SEASON].head(NROW).copy()
row_ids = sl['row_id'].to_numpy()
sl.drop(columns=['control_success']).to_csv(os.path.join(data, 'test.csv'), index=False)
pd.DataFrame({'row_id': row_ids, 'control_success': 0.5}).to_csv(
    os.path.join(data, 'sample_submission.csv'), index=False)
for f in ('trackman_history.csv',):
    if os.path.exists(f'data/{f}'):
        shutil.copy(f'data/{f}', os.path.join(data, f))
print(f'위장 test.csv {len(sl):,}행 준비', flush=True)

# ---------- 4) 실행 ----------
r = subprocess.run([sys.executable, '-u', 'script.py'], cwd=work,
                   capture_output=True, text=True, encoding='utf-8', errors='replace')
print('--- script.py 출력 ---')
print((r.stdout or '')[-1500:])
if r.returncode != 0:
    print((r.stderr or '')[-2000:])
    sys.exit(f'script.py 실패 (rc={r.returncode})')

inf = pd.read_pickle(os.path.join(work, 'infer_features.pkl'))

# ---------- 5) 학습 경로 피처 ----------
X = pd.read_pickle('cache/X.pkl')
# 4-15: 캐시의 행 순서는 df_processed 순서다. row_id 를 따로 저장하지 않았으므로
# season 마스크로는 못 맞춘다 -> 학습 경로의 row_id 가 필요하다.
rid_path = 'cache/row_id.npy'
if not os.path.exists(rid_path):
    sys.exit('cache/row_id.npy 가 없다. tools/make_cache.py 를 다시 돌려 row_id 를 저장할 것')
X = X.copy()
X.insert(0, 'row_id', np.load(rid_path))

m = inf.merge(X, on='row_id', suffixes=('_inf', '_tr'), how='inner')
print(f'\n대조 행 {len(m):,} / 추론 {len(inf):,} / 학습 {len(X):,}')
assert len(m) == len(inf), '일부 row_id 가 학습 캐시에 없다'

cols = [c for c in inf.columns if c != 'row_id' and c in X.columns]
print(f'공통 피처 {len(cols)}개 (추론 {inf.shape[1]-1} / 학습 {X.shape[1]-1})')
only_inf = sorted(set(inf.columns) - set(X.columns) - {'row_id'})
only_tr = sorted(set(X.columns) - set(inf.columns) - {'row_id'})
if only_inf:
    print('  추론에만 있음:', only_inf)
if only_tr:
    print('  학습에만 있음:', only_tr)

# ---------- 6) 컬럼별 비교 ----------
bad = []
for c in cols:
    a, b = m[f'{c}_inf'], m[f'{c}_tr']
    if str(a.dtype) in ('category', 'object') or str(b.dtype) in ('category', 'object'):
        sa, sb = a.astype(str), b.astype(str)
        diff = int((sa != sb).sum())
        if diff:
            ex = m.loc[sa != sb, ['row_id']].head(2).values.tolist()
            bad.append((c, diff, diff / len(m), '범주', str(sa[sa != sb].iloc[0]),
                        str(sb[sa != sb].iloc[0]), ex))
    else:
        fa, fb = pd.to_numeric(a, errors='coerce'), pd.to_numeric(b, errors='coerce')
        na = fa.isna() ^ fb.isna()
        close = np.isclose(fa.fillna(-9e18), fb.fillna(-9e18), rtol=1e-5, atol=1e-8)
        mis = (~close) | na
        diff = int(mis.sum())
        if diff:
            i = mis.idxmax()
            bad.append((c, diff, diff / len(m), '수치', f'{fa[i]}', f'{fb[i]}',
                        [[int(m.loc[i, 'row_id'])]]))

print('\n' + '=' * 68)
if not bad:
    print('✅ 공통 피처 전부 일치 — 학습/추론 경로에 불일치 없음')
else:
    bad.sort(key=lambda t: -t[2])
    print(f'⚠️ 불일치 {len(bad)}개 컬럼')
    print(f'{"컬럼":<38}{"불일치":>9}{"비율":>8}   추론 vs 학습 (예시)')
    for c, n, p, kind, va, vb, ex in bad:
        print(f'{c:<38}{n:>9,}{p:>7.1%}   {va} vs {vb}   row_id={ex[0][0] if ex else "?"}')
print('=' * 68)
print(f'\n작업 폴더는 남겨둔다: {work}')

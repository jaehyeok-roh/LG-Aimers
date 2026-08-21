# 전처리 결과를 한 번만 만들어 디스크에 캐시한다.
#
# 큰 스윙 스크리닝의 병목은 전처리(로컬 CPU 로 수 분)인데, 후보 대부분은 **학습만**
# 바꾼다 (하이퍼파라미터, 앙상블 구성, 블렌드, baseline). 그래서 X/y/season 을 한 번
# 만들어 두면 후보당 비용이 '학습 시간' 으로 떨어진다.
#
# s1o 노트북(= v5, 전 플래그 off)의 코드 셀을 feature-selection 셀까지만 실행한다.
# 학습·오프셋·zip 셀은 건드리지 않는다.
#
# 사용: python tools/make_cache.py [출력디렉터리]
import json, os, sys, time

NB = '.kernels/s1o/aimers_s1o.ipynb'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'cache'
STOP = 'BEST_PARAMS = _found'          # 이 문장이 있는 셀까지 실행하고 멈춘다

os.makedirs(OUT, exist_ok=True)
os.makedirs('model', exist_ok=True)

cells = [c for c in json.load(open(NB, encoding='utf-8'))['cells']
         if c['cell_type'] == 'code']
srcs = [''.join(c['source']) for c in cells]
last = next(i for i, s in enumerate(srcs) if STOP in s)
print(f'코드 셀 {len(srcs)}개 중 0~{last} 실행 (feature selection 까지)', flush=True)

g = {'__name__': '__main__'}
t0 = time.time()
for i, s in enumerate(srcs[:last + 1]):
    t = time.time()
    try:
        exec(compile(s, f'<cell {i}>', 'exec'), g)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(f'\n셀 {i} 에서 실패')
    print(f'  cell {i:>2}  {time.time()-t:6.1f}초', flush=True)

X, y = g['X_full'], g['y_full']
season = g['df_processed']['season'].to_numpy()
# 4-15: run_full_pipeline 이 행 순서를 바꾸므로 season 은 반드시 df_processed 에서 꺼낸다
assert len(X) == len(y) == len(season), '길이 불일치'

import pandas as pd
import numpy as np

X.to_pickle(f'{OUT}/X.pkl')
np.save(f'{OUT}/y.npy', y.to_numpy())
np.save(f'{OUT}/season.npy', season)
meta = {'cat_features': list(g['cat_features']),
        'best_params': {k: v for k, v in g['BEST_PARAMS'].items() if k != 'cat_features'},
        'n_rows': int(len(X)), 'n_feats': int(X.shape[1]),
        'season_counts': {int(s): int((season == s).sum()) for s in np.unique(season)}}
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=2, ensure_ascii=False)

print(f'\n캐시 저장: {OUT}/  ({time.time()-t0:.0f}초)')
print(f'  X {X.shape}  범주형 {len(meta["cat_features"])}개')
print(f'  시즌별 행수 {meta["season_counts"]}')

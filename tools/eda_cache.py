# 원본 CSV -> parquet 캐시. 이후 EDA 는 이걸 읽어서 반복이 빨라진다.
import pandas as pd, os, time
_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']   # 'None' 제외 (4-3)
os.makedirs('cache/raw', exist_ok=True)
for name, path in [('train', 'data/train.csv'), ('trackman', 'data/trackman_history.csv')]:
    out = f'cache/raw/{name}.parquet'
    if os.path.exists(out):
        print(f'{name}: 이미 있음'); continue
    t = time.time()
    df = pd.read_csv(path, keep_default_na=False, na_values=_NA)
    df.columns = [c.lstrip('﻿') for c in df.columns]
    df.to_parquet(out, index=False)
    print(f'{name}: {len(df):,}행 x {df.shape[1]}컬럼  ({time.time()-t:.0f}s)  -> {out}')
    del df

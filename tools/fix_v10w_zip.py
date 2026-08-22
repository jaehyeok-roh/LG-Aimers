# v10w 제출 zip 의 train_constants.json 을 고친다.
#
# 버그: 오프셋 셀(cell 10)이 train_constants.json 을 통째로 덮어써서
# cell 7 에서 넣은 ws_* 상수가 지워졌다. 그러면 script.py 가
# `_tc.get("ws_rates")` 를 None 으로 읽어 **당해 시즌 복원 블록을 건너뛴다** ->
# 모델은 w_* 로 학습됐는데 추론에서는 전부 NaN (트랙맨 exact 사고와 같은 유형).
#
# 모델과 pitcher_prior.csv 는 정상이므로 재학습은 필요 없다. 상수는 train.csv 에서
# 결정적으로 재계산되므로 노트북이 넣었어야 할 값과 정확히 같다.
import io, json, os, shutil, sys, zipfile
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else 'kaggle_output/v10w/submit_v10w.zip'
DST = sys.argv[2] if len(sys.argv) > 2 else 'out/submit_v10w_fixed.zip'
RATES = ['success', 'middle', 'reverse', 'ball', 'strike']
WS_C = 100.0

_NA = ['', 'NaN', 'nan', 'NULL', 'null', 'NA', 'N/A', 'n/a']
cols = ['season'] + [f'asof_pitcher_{k}_rate' for k in RATES]
df = pd.read_csv('data/train.csv', usecols=cols, keep_default_na=False, na_values=_NA)
means = {c: df.groupby('season')[c].mean().to_dict() for c in cols[1:]}
tgt = int(df['season'].max()) + 1

lg = {}
for c, d in means.items():
    ss = sorted(d)
    k = ss[-3:]
    a, b = np.polyfit(k, [d[s] for s in k], 1)
    lg[c] = float(np.clip(a * tgt + b, 0.0, 1.0))
    print(f'  {c:<34} 최근3시즌 ' + ' '.join(f'{d[s]:.4f}' for s in k)
          + f'  -> {tgt} 외삽 {lg[c]:.4f}')

with zipfile.ZipFile(SRC) as z:
    names = z.namelist()
    tc = json.loads(z.read('model/train_constants.json'))
    sf = json.loads(z.read('model/selected_features.json'))
    blobs = {n: z.read(n) for n in names}

wcols = [c for c in sf if c.startswith('w_')]
print(f'\n학습 피처 {len(sf)}개 | w_ 피처 {wcols}')
assert wcols, 'w_ 피처가 학습에 없다 — 이 zip 은 대상이 아니다'
if 'ws_league_mean' in tc:
    sys.exit('이미 ws 상수가 들어있다 — 고칠 필요 없음')

tc['ws_target_season'] = tgt
tc['ws_league_mean'] = lg
tc['ws_rates'] = RATES
tc['ws_C'] = WS_C
blobs['model/train_constants.json'] = json.dumps(tc).encode('utf-8')

os.makedirs(os.path.dirname(DST) or '.', exist_ok=True)
with zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in names:
        z.writestr(n, blobs[n])

print(f'\n{DST} 생성 — {os.path.getsize(DST)/1e6:.1f}MB, 파일 {len(names)}개')
print('  train_constants:', {k: (round(v, 5) if isinstance(v, float) else v)
                             for k, v in tc.items() if k != 'ws_league_mean'})

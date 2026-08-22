# cpua / cpub / cpuc (seed 별 10모델) 를 30모델 제출본 하나로 조립한다.
#
# ⚠️ 이 스크립트가 존재하는 이유 — 커널 산출물에는 버그가 있다:
#   v10w 계열 노트북의 오프셋 셀이 train_constants.json 을 **통째로 덮어써서**
#   ws_* 키(당해 시즌 복원 상수)가 날아간다. script.py 는 그 키가 없으면
#   wseason 블록을 통째로 건너뛰므로, **학습은 진짜 값으로 하고 추론은 전부 NaN**
#   이 된다 — 트랙맨 exact 모드(-36점)와 같은 실패이고 여기서는 61점짜리다.
#   그래서 조립 시 ws_* 를 다시 넣고 반드시 검증한다.
#
# 그리고 recenter_offset 은 커널마다 다른 seed 로 재므로(±13 노이즈) **셋을 평균**한다.
import json
import os
import shutil
import sys
import tempfile
import zipfile

SRC = ['kaggle_output/cpua/submit_cpu_a.zip',
       'kaggle_output/cpub/submit_cpu_b.zip',
       'kaggle_output/cpuc/submit_cpu_c.zip']
REF = 'out/submit_v10w_fixed.zip'          # ws_* 상수를 여기서 가져온다
OUT = 'out/submit_cpu30.zip'

missing = [p for p in SRC if not os.path.exists(p)]
if missing:
    sys.exit(f'아직 없는 zip: {missing}')

ws = {k: v for k, v in json.loads(
    zipfile.ZipFile(REF).read('model/train_constants.json')).items()
    if k.startswith('ws_')}
if not ws:
    sys.exit(f'{REF} 에 ws_* 가 없다 — 기준으로 쓸 수 없다')
print(f'기준 ws_* 키 {sorted(ws)}\n')

work = tempfile.mkdtemp(prefix='cpu30_')
root = os.path.join(work, 'sub')
offsets = []
models = set()

for i, p in enumerate(SRC):
    z = zipfile.ZipFile(p)
    tc = json.loads(z.read('model/train_constants.json'))
    offsets.append(float(tc['recenter_offset']))
    got = [n for n in z.namelist() if n.endswith('.cbm')]
    print(f'{os.path.basename(p):<22} 모델 {len(got):>2}개  offset {tc["recenter_offset"]:+.4f}')
    if i == 0:
        z.extractall(root)                  # 첫 zip 을 뼈대로 (script.py, 룩업 테이블 등)
    else:
        for n in z.namelist():              # 나머지는 모델·보정기만 합친다
            if n.endswith('.cbm') or ('isotonic' in n and n.endswith('.pkl')):
                z.extract(n, root)
    models |= set(got)

off = sum(offsets) / len(offsets)
print(f'\nrecenter_offset  {offsets} -> 평균 {off:+.4f}')

tcp = os.path.join(root, 'model', 'train_constants.json')
tc = json.load(open(tcp, encoding='utf-8'))
tc.update(ws)
tc['recenter_offset'] = off
json.dump(tc, open(tcp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

# ---- 검증 ----
md = os.path.join(root, 'model')
cbm = [f for f in os.listdir(md) if f.endswith('.cbm')]
iso = [f for f in os.listdir(md) if f.startswith('isotonic') and f.endswith('.pkl')]
tc2 = json.load(open(tcp, encoding='utf-8'))
REQ = ['script.py', 'requirements.txt', 'model/train_constants.json',
       'model/selected_features.json', 'model/feat_diff.csv', 'model/feat_speed.csv',
       'model/feat_rp.csv', 'model/pitcher_prior.csv']
bad = []
if len(cbm) != 30:
    bad.append(f'모델이 30개가 아니라 {len(cbm)}개')
if len(iso) != 30:
    bad.append(f'isotonic 이 30개가 아니라 {len(iso)}개')
for k in ('ws_league_mean', 'ws_rates', 'ws_C', 'ws_target_season'):
    if k not in tc2:
        bad.append(f'{k} 누락')
for r in REQ:
    if not os.path.exists(os.path.join(root, r)):
        bad.append(f'{r} 없음')
src = open(os.path.join(root, 'script.py'), encoding='utf-8').read()
if 'ws_league_mean' not in src:
    bad.append('script.py 에 wseason 추론 블록이 없다')

print(f'\n모델 {len(cbm)} | isotonic {len(iso)} | seed {sorted({f.split("_")[2] for f in cbm})}')
print(f'ws_* {sorted(k for k in tc2 if k.startswith("ws_"))}')
if bad:
    print('\n❌ 검증 실패:'); [print('   -', b) for b in bad]
    sys.exit(1)

os.makedirs('out', exist_ok=True)
if os.path.exists(OUT):
    os.remove(OUT)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for dp, _, fs in os.walk(root):
        for f in fs:
            fp = os.path.join(dp, f)
            z.write(fp, os.path.relpath(fp, root))
print(f'\n✅ {OUT}  {os.path.getsize(OUT)/1e6:.1f}MB  파일 {len(zipfile.ZipFile(OUT).namelist())}개')
print('\n다음: python tools/audit_independence.py out/submit_cpu30.zip')

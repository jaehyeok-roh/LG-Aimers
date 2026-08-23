# seed 별로 쪼개 돌린 CPU 커널 3개를 30모델 제출본 하나로 조립한다.
#
# ⚠️ 이 스크립트가 존재하는 이유 — 커널 산출물에 버그가 있었다:
#   v10w 계열 노트북의 오프셋 셀이 train_constants.json 을 **통째로 덮어써서**
#   ws_* 키(당해 시즌 복원 상수)가 날아갔다. script.py 는 그 키가 없으면
#   복원 블록을 통째로 건너뛰므로, **학습은 진짜 값으로 하고 추론은 전부 NaN**
#   이 된다 — 트랙맨 exact 모드(-36점)와 같은 실패이고 여기서는 61점짜리다.
#   빌더(mk_ws/mk_wsbat/mk_prevfix)에는 재주입 + assert 를 넣었지만,
#   옛 커널 산출물을 조립할 때를 대비해 여기서도 채우고 **반드시 검증한다.**
#
# recenter_offset 은 커널마다 다른 seed 로 재므로(±13 노이즈) **셋을 평균**한다.
#
# 사용:
#   CS_PREFIX=wbc python tools/assemble_cpu.py        # v10wb 묶음
#   CS_PREFIX=cpu CS_REF=out/submit_v10w_fixed.zip \
#       CS_OUT=out/submit_cpu30.zip python tools/assemble_cpu.py
import json
import os
import shutil
import sys
import tempfile
import zipfile

PFX = os.environ.get('CS_PREFIX', 'wbc')
SRC = [f'kaggle_output/{PFX}{t}/submit_cpu_{t}.zip' for t in 'abc']
REF = os.environ.get('CS_REF', 'out/submit_v10wb.zip')
OUT = os.environ.get('CS_OUT', f'out/submit_{PFX}30.zip')
PRE = ('ws_', 'wb_', 'pf_')          # 오프셋 셀이 날릴 수 있는 상수군 전부

missing = [p for p in SRC if not os.path.exists(p)]
if missing:
    sys.exit(f'아직 없는 zip: {missing}')

ref = {}
if os.path.exists(REF):
    ref = {k: v for k, v in json.loads(
        zipfile.ZipFile(REF).read('model/train_constants.json')).items()
        if k.startswith(PRE)}
print(f'커널 묶음 {PFX}a/b/c -> {OUT}')
print(f'참조본 {REF} 상수 {sorted(ref) if ref else "(없음)"}\n')

work = tempfile.mkdtemp(prefix=f'{PFX}30_')
root = os.path.join(work, 'sub')
offsets, models = [], set()

for i, p in enumerate(SRC):
    z = zipfile.ZipFile(p)
    tc = json.loads(z.read('model/train_constants.json'))
    offsets.append(float(tc['recenter_offset']))
    got = [n for n in z.namelist() if n.endswith('.cbm')]
    print(f'{os.path.basename(p):<22} 모델 {len(got):>2}개  offset {tc["recenter_offset"]:+.4f}')
    if i == 0:
        z.extractall(root)                  # 첫 zip 을 뼈대로 (script.py, 룩업 등)
    else:
        for n in z.namelist():              # 나머지는 모델·보정기만 합친다
            if n.endswith('.cbm') or ('isotonic' in n and n.endswith('.pkl')):
                z.extract(n, root)
    models |= set(got)

off = sum(offsets) / len(offsets)
print(f'\nrecenter_offset {[round(o, 4) for o in offsets]} -> 평균 {off:+.4f}')

tcp = os.path.join(root, 'model', 'train_constants.json')
tc = json.load(open(tcp, encoding='utf-8'))
filled = [k for k in ref if k not in tc]
for k in filled:
    tc[k] = ref[k]
print('참조본에서 채운 상수: ' + (str(filled) if filled else '없음 (커널 산출물이 온전함)'))
tc['recenter_offset'] = off
json.dump(tc, open(tcp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

# ---------------- 검증 ----------------
md = os.path.join(root, 'model')
cbm = [f for f in os.listdir(md) if f.endswith('.cbm')]
iso = [f for f in os.listdir(md) if f.startswith('isotonic') and f.endswith('.pkl')]
tc2 = json.load(open(tcp, encoding='utf-8'))
script = open(os.path.join(root, 'script.py'), encoding='utf-8').read()

REQ = ['script.py', 'requirements.txt', 'model/train_constants.json',
       'model/selected_features.json', 'model/feat_diff.csv', 'model/feat_speed.csv',
       'model/feat_rp.csv']
# script.py 가 실제로 쓰는 블록에 맞춰 필수 파일·상수를 요구한다
NEED = [('ws_league_mean', 'model/pitcher_prior.csv', 'ws_league_mean'),
        ('wb_league_mean', 'model/batter_prior.csv', 'wb_league_mean'),
        ('pf_league_cur', 'model/pitcher_appearance.csv', 'pf_league_cur')]
bad = []
for marker, need_file, need_key in NEED:
    if marker in script:
        REQ.append(need_file)
        if need_key not in tc2:
            bad.append(f'{need_key} 누락 — script.py 가 이 상수를 쓴다')

if len(cbm) != 30:
    bad.append(f'모델이 30개가 아니라 {len(cbm)}개')
if len(iso) != 30:
    bad.append(f'isotonic 이 30개가 아니라 {len(iso)}개')
for r in REQ:
    if not os.path.exists(os.path.join(root, r)):
        bad.append(f'{r} 없음')

print(f'\n모델 {len(cbm)} | isotonic {len(iso)} | seed {sorted({f.split("_")[2] for f in cbm})}')
print('상수: ' + ' '.join(sorted(k for k in tc2 if k.startswith(PRE))))
print('필수 파일: ' + ' '.join(os.path.basename(r) for r in REQ))
if bad:
    print('\n❌ 검증 실패:')
    for b in bad:
        print('   -', b)
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
print(f'\n다음:  python tools/audit_independence.py {OUT} 800 3')
print(f'       python tools/verify_ws_live.py {OUT} 600 wb_')

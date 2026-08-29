# 저장된 모델은 그대로 두고 **사용할 트리 수만** 줄인 제출 zip 을 만든다. 재학습 0.
#
#   python tools/mk_ntree.py out/submit_v10wzc.zip 700
#
# 왜: iterations=1000 은 **이진 타겟 시절** 리더보드로 포화를 확인한 값이다
# (500 -23.07 / 700 -4.74 / 1000 기준). v10wz 부터 타겟이 5분류라 반복당 용량이
# 다르다 -- claude.md 'optbest 절개' 의 교훈("다른 파라미터 영역에서 잰 효과를
# 더하지 말 것", 그때는 부호까지 뒤집혔다)이 정확히 이 모양이다.
#
# ⚠️ isotonic 은 1000트리 출력으로 적합돼 있다. 자르면 raw 분포가 조금 이동하므로
#    보정이 미세하게 어긋난다. v7 이 이진에서 쓴 방식과 같고, 그때도 유의미한
#    리더보드 차이가 나왔다. 결과가 음수여도 '트리 수' 와 '보정 어긋남' 이
#    섞여 있다는 점은 감안할 것.
import os
import re
import shutil
import sys
import zipfile

SRC = sys.argv[1]
N = int(sys.argv[2])
DST = sys.argv[3] if len(sys.argv) > 3 else re.sub(
    r'\.zip$', '_n%d.zip' % N, SRC)

OLD = 'raw = model.predict_proba(df_in[names])[:, _sc]'
NEW = ('raw = model.predict_proba(df_in[names], ntree_end=%d)[:, _sc]' % N)
# 다중분류가 success_cols(리스트)를 쓰는 판(mk_ptc/mk_j7)도 지원한다
OLD2 = 'raw = model.predict_proba(df_in[names])[:, _sc].sum(axis=1)'
NEW2 = ('raw = model.predict_proba(df_in[names], ntree_end=%d)[:, _sc].sum(axis=1)' % N)

zin = zipfile.ZipFile(SRC)
s = zin.read('script.py').decode('utf-8')
hit = 0
for o, n in ((OLD, NEW), (OLD2, NEW2)):
    if o in s:
        s = s.replace(o, n)
        hit += 1
if hit != 1:
    sys.exit('예측 앵커 %d개 -- 수동 확인 필요' % hit)

shutil.copy(SRC, DST)
# zip 안의 한 파일만 갈아끼우려면 다시 쓰는 게 제일 짧다
tmp = DST + '.tmp'
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zo:
    for it in zin.infolist():
        zo.writestr(it, s.encode('utf-8') if it.filename == 'script.py'
                    else zin.read(it.filename))
zin.close()
os.replace(tmp, DST)
print('%s -> %s  (ntree_end=%d)' % (SRC, DST, N))

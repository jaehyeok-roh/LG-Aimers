#!/bin/sh
# 로컬 30모델 학습이 끝나면 zip 을 캐글 데이터셋에 얹고 상관 측정 커널을 다시 띄운다.
# 로컬 CPU 는 학습이 끝나는 즉시 완전히 쉰다 — 여기서 하는 일은 업로드뿐이다.
#
# ⚠️ 캐글 CLI 는 Windows 에서 -p 경로에 '/' 가 들어가면 깨진다
#    ([Errno 2] ... '.kernels/_zips_submit_x.zip.json'). 반드시 단일 이름 폴더를 쓸 것.
cd /c/Users/nojh4/github/LG-Aimers || exit 1
export PYTHONUTF8=1
Z=out/submit_local_cpu.zip
DS=zipsds

echo "[$(date +%H:%M)] 대기 시작 — $Z"
n=0
while [ ! -f "$Z" ]; do
  n=$((n + 1))
  if [ $n -gt 240 ]; then echo "4시간 초과, 포기"; exit 1; fi
  sleep 60
done

# 쓰는 중에 올리면 깨진다 — 크기가 안정될 때까지 기다린다
prev=0
while true; do
  cur=$(stat -c%s "$Z")
  if [ "$cur" = "$prev" ] && [ "$cur" -gt 1000000 ]; then break; fi
  prev=$cur
  sleep 30
done
echo "[$(date +%H:%M)] zip 완성 ($cur 바이트)"
tail -20 out/train_local.log

cp "$Z" "$DS/submit_local_cpu.zip" || exit 1
ls -la "$DS"

echo "[$(date +%H:%M)] 데이터셋 새 버전 업로드"
kaggle datasets version -p "$DS" -m "add local_cpu $(date +%m%d-%H%M)" 2>&1 | tail -3

n=0
while true; do
  s=$(kaggle datasets status homekeggle/aimers-zips 2>&1 | tr -d '\r')
  echo "  데이터셋 상태: $s"
  case "$s" in ready*) break;; esac
  n=$((n + 1))
  if [ $n -gt 40 ]; then echo "처리 지연, 그래도 진행"; break; fi
  sleep 30
done

python tools/mk_kcorr.py submit_local_cpu.zip submit_v9m.zip submit_s1lr.zip || exit 1
cd .kernels/kcorr && kaggle kernels push -p . 2>&1 | tail -2
echo "[$(date +%H:%M)] 커널 push 완료 — homekeggle/aimers-kcorr"
